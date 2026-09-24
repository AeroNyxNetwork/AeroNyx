// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store/route_selection.rs
// ============================================================================
//! Live, fail-closed peer lookup and route candidate selection.
//!
//! [PEER-STORE-ROUTE-SELECTION-SPLIT 2026-09-25 by Codex] Exact node lookup,
//! capability/TTL gates, network anti-affinity, routeability scoring, and
//! strict multi-hop planning retain their existing public inherent API and
//! selection order. This child does not own route-domain policy mutation.

use super::*;

impl PeerStore {
    /// Returns a descriptor for a node id if present and valid at `now`.
    #[must_use]
    pub fn get_valid(&self, node_id: &[u8; 32], now: u64) -> Option<SignedNodeDescriptor> {
        self.peers
            .read()
            .get(node_id)
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| self.permissionless_gate_allows(descriptor, now, true))
            .cloned()
    }

    /// Returns an authentic cached descriptor without requiring time validity.
    ///
    /// [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] This narrow accessor
    /// breaks the strict-startup deadlock where every cached witness descriptor
    /// expires while a node is offline and witness reconciliation runs before
    /// gossip listeners can refresh it. Callers must independently pin the
    /// returned node identity, validate the endpoint, and verify the signed
    /// protocol response against that identity.
    ///
    /// This accessor must never be used for liveness, gossip export, relay
    /// selection, route planning, quorum, capacity, or public peer counts.
    #[must_use]
    pub(crate) fn get_signature_verified_cached(
        &self,
        node_id: &[u8; 32],
    ) -> Option<SignedNodeDescriptor> {
        self.peers
            .read()
            .get(node_id)
            .filter(|descriptor| descriptor.verify_signature().is_ok())
            .cloned()
    }

    /// Returns valid public descriptors for internal bounded peer selection.
    ///
    /// Unlike bootstrap export, this read-only accessor does not mutate export
    /// counters or audit history. Callers must still apply operation-specific
    /// endpoint safety and trust policy. In particular, public discovery is not
    /// authority membership, voting weight, consensus, or finality.
    #[must_use]
    pub fn valid_public_descriptors(&self, now: u64, limit: usize) -> Vec<SignedNodeDescriptor> {
        if limit == 0 {
            return Vec::new();
        }
        let mut descriptors = self
            .peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| self.permissionless_gate_allows(descriptor, now, true))
            .filter(|descriptor| descriptor.descriptor.policy.public_discovery)
            .cloned()
            .collect::<Vec<_>>();
        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        descriptors.truncate(limit);
        descriptors
    }

    /// Returns every valid public endpoint with its verified node identity.
    ///
    /// [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] Outbound gossip uses
    /// this complete, lightweight view to detect when multiple signed
    /// descriptors claim one canonical endpoint. The result is bounded by the
    /// store's configured peer capacity, does not mutate counters or audit
    /// history, and must remain internal control-plane input only.
    #[must_use]
    pub(crate) fn valid_public_endpoint_identities(&self, now: u64) -> Vec<([u8; 32], String)> {
        let mut identities = self
            .peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| self.permissionless_gate_allows(descriptor, now, true))
            .filter(|descriptor| descriptor.descriptor.policy.public_discovery)
            .filter_map(|descriptor| {
                descriptor
                    .descriptor
                    .public_endpoint
                    .as_ref()
                    .map(|endpoint| (descriptor.node_id(), endpoint.clone()))
            })
            .collect::<Vec<_>>();
        identities.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        identities
    }

    /// Returns valid descriptors that advertise a capability.
    #[must_use]
    pub fn peers_with_capability(
        &self,
        capability: NodeCapability,
        now: u64,
    ) -> Vec<SignedNodeDescriptor> {
        self.peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| self.permissionless_gate_allows(descriptor, now, true))
            .filter(|descriptor| descriptor.descriptor.capabilities.contains(&capability))
            .cloned()
            .collect()
    }

    fn endpoint_network_identity(
        descriptor: &SignedNodeDescriptor,
    ) -> Option<EndpointNetworkIdentity> {
        let endpoint = descriptor.descriptor.public_endpoint.as_deref()?.trim();
        if endpoint.is_empty() {
            return None;
        }
        let normalized = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("http://{endpoint}")
        };
        let url = reqwest::Url::parse(&normalized).ok()?;
        let host = url.host_str()?.trim_end_matches('.').to_ascii_lowercase();
        if host.is_empty() {
            return None;
        }

        let ip_host = host.trim_start_matches('[').trim_end_matches(']');
        match ip_host.parse::<IpAddr>() {
            Ok(IpAddr::V4(address)) => {
                let octets = address.octets();
                Some(EndpointNetworkIdentity::Ipv4([
                    octets[0], octets[1], octets[2],
                ]))
            }
            Ok(IpAddr::V6(address)) => {
                let octets = address.octets();
                Some(EndpointNetworkIdentity::Ipv6([
                    octets[0], octets[1], octets[2], octets[3], octets[4], octets[5],
                ]))
            }
            Err(_) => Some(EndpointNetworkIdentity::Dns(host)),
        }
    }

    /// Returns whether two signed route endpoints pass coarse anti-affinity.
    ///
    /// Literal IPv4 endpoints must differ at /24 and IPv6 endpoints at /48;
    /// DNS endpoints must have different normalized hostnames. Missing or
    /// malformed endpoints fail closed. This prevents obviously collocated
    /// hops, but it is not proof of distinct operators or autonomous systems.
    /// That stronger property requires separately audited routing-domain
    /// attestations and must not be inferred from this helper.
    #[must_use]
    pub fn route_endpoints_are_network_diverse(
        left: &SignedNodeDescriptor,
        right: &SignedNodeDescriptor,
    ) -> bool {
        match (
            Self::endpoint_network_identity(left),
            Self::endpoint_network_identity(right),
        ) {
            (Some(left), Some(right)) => left != right,
            _ => false,
        }
    }

    /// Returns whether a candidate is network-diverse from every selected hop.
    #[must_use]
    pub fn route_endpoint_is_network_diverse_from_all(
        candidate: &SignedNodeDescriptor,
        selected: &[SignedNodeDescriptor],
    ) -> bool {
        selected
            .iter()
            .all(|hop| Self::route_endpoints_are_network_diverse(candidate, hop))
    }

    /// Returns health-ranked route candidates for a capability.
    ///
    /// This method is the server-internal companion to the privacy-safe
    /// `PeerStoreStatus.route_candidates` payload. It sorts only by signed
    /// descriptor metadata and local discovery observation age. It never reads
    /// or derives from encrypted chat/media blobs, packet payloads, DNS data,
    /// destinations, client public IPs, voucher secrets, private keys, or
    /// wallet-level traffic.
    #[must_use]
    pub fn route_candidates_with_capability(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
    ) -> Vec<SignedNodeDescriptor> {
        self.route_candidates_with_capability_excluding(capability, now, limit, &[])
    }

    /// Returns health-ranked route candidates after excluding specific node ids.
    ///
    /// Exclusion happens before the limit is applied. This matters for fanout
    /// and future controlled multi-hop planning: self, already-used hops, or
    /// policy-excluded peers must not consume the limited candidate budget.
    ///
    /// The selection still uses only signed node descriptor metadata plus local
    /// node-to-node route health. It never reads encrypted blobs, plaintext,
    /// client IPs, destinations, DNS contents, voucher secrets, private keys,
    /// or wallet-level traffic.
    #[must_use]
    pub fn route_candidates_with_capability_excluding(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
        excluded_node_ids: &[[u8; 32]],
    ) -> Vec<SignedNodeDescriptor> {
        self.scored_route_candidates(capability, now, None, false)
            .into_iter()
            .filter(|candidate| {
                let node_id = candidate.descriptor.node_id();
                !excluded_node_ids
                    .iter()
                    .any(|excluded| *excluded == node_id)
            })
            .take(limit)
            .map(|candidate| candidate.descriptor)
            .collect()
    }

    /// Returns routeable peers that recently proved purpose-bound v2 receipt
    /// interoperability, after applying capability and exclusion policy.
    ///
    /// This gate is intentionally process-local and freshness-bounded. It
    /// enables gradual mixed-version rollout without advertising a descriptor
    /// enum that older nodes may reject. Selection never reads message data,
    /// sender/receiver identities, route ids, endpoints, or payload contents.
    #[must_use]
    pub fn delivery_receipt_route_candidates_with_capability_excluding(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
        excluded_node_ids: &[[u8; 32]],
    ) -> Vec<SignedNodeDescriptor> {
        let candidates = self.scored_route_candidates(capability, now, None, false);
        let capability_evidence = self.purpose_bound_delivery_receipt_capability.read();
        candidates
            .into_iter()
            .filter(|candidate| {
                let node_id = candidate.descriptor.node_id();
                candidate.summary.routeability_ready
                    && !excluded_node_ids
                        .iter()
                        .any(|excluded| *excluded == node_id)
                    && capability_evidence.get(&node_id).is_some_and(|evidence| {
                        Self::purpose_bound_delivery_receipt_evidence_matches_descriptor(
                            evidence,
                            &candidate.descriptor,
                            now,
                        )
                    })
            })
            .take(limit)
            .map(|candidate| candidate.descriptor)
            .collect()
    }

    /// Returns purpose-bound v2 receipt-capable candidates under multi-hop policy.
    ///
    /// When strict attestation is disabled this is behaviorally identical to
    /// [`Self::delivery_receipt_route_candidates_with_capability_excluding`].
    /// In strict mode, filtering happens before the limit so unproven peers
    /// cannot consume the candidate budget or cause a partial-path fallback.
    #[must_use]
    pub fn multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
        excluded_node_ids: &[[u8; 32]],
    ) -> Vec<SignedNodeDescriptor> {
        let candidates = self.scored_route_candidates(capability, now, None, false);
        let capability_evidence = self.purpose_bound_delivery_receipt_capability.read();
        candidates
            .into_iter()
            .filter(|candidate| {
                let node_id = candidate.descriptor.node_id();
                candidate.summary.routeability_ready
                    && !excluded_node_ids
                        .iter()
                        .any(|excluded| *excluded == node_id)
                    && capability_evidence.get(&node_id).is_some_and(|evidence| {
                        Self::purpose_bound_delivery_receipt_evidence_matches_descriptor(
                            evidence,
                            &candidate.descriptor,
                            now,
                        )
                    })
                    && self.route_domain_certificate_allows_multi_hop(&node_id, now)
            })
            .take(limit)
            .map(|candidate| candidate.descriptor)
            .collect()
    }

    /// Determines whether authenticated App traffic has one usable two-hop
    /// receipt path under the exact production candidate budgets.
    ///
    /// [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Counting two
    /// independently capable peers is insufficient: they may not expose the
    /// required middle/terminal capabilities after exclusions, may fail strict
    /// route-domain attestation, or may share one coarse endpoint network. This
    /// helper reuses the production receipt selector and returns only a stable
    /// aggregate reason. It never exports which peers or endpoints were tested.
    #[must_use]
    pub(crate) fn authenticated_delivery_path_readiness_excluding(
        &self,
        now: u64,
        excluded_node_ids: &[[u8; 32]],
    ) -> AuthenticatedDeliveryPathReadiness {
        let terminals = self.multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            AUTHENTICATED_CHAT_TERMINAL_FANOUT_LIMIT,
            excluded_node_ids,
        );
        if terminals.is_empty() {
            return AuthenticatedDeliveryPathReadiness {
                ready: false,
                reason: "no_receipt_capable_terminal",
            };
        }

        let mut middle_candidate_seen = false;
        for terminal in terminals {
            let terminal_node_id = terminal.node_id();
            let mut middle_exclusions = Vec::with_capacity(excluded_node_ids.len() + 1);
            middle_exclusions.extend_from_slice(excluded_node_ids);
            middle_exclusions.push(terminal_node_id);
            let middles = self
                .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
                    NodeCapability::OnionMiddle,
                    now,
                    AUTHENTICATED_CHAT_MIDDLE_CANDIDATE_LIMIT,
                    &middle_exclusions,
                );
            middle_candidate_seen |= !middles.is_empty();
            if middles
                .iter()
                .any(|middle| Self::route_endpoints_are_network_diverse(middle, &terminal))
            {
                return AuthenticatedDeliveryPathReadiness {
                    ready: true,
                    reason: "authenticated_receipt_path_ready",
                };
            }
        }

        AuthenticatedDeliveryPathReadiness {
            ready: false,
            reason: if middle_candidate_seen {
                "no_network_diverse_receipt_path"
            } else {
                "no_receipt_capable_middle"
            },
        }
    }

    /// Returns probe-only route candidates with cooled-down quarantine recovery.
    ///
    /// This method is intentionally narrower than the normal route candidate
    /// API: it exists only for local discovery probes that try to recover route
    /// health after restart or transient endpoint failures. User traffic and
    /// ordinary blind relay forwarding must continue using
    /// `route_candidates_with_capability_excluding()` so quarantined peers do
    /// not carry encrypted payloads until a successful probe clears quarantine.
    ///
    /// Normal unknown/stale candidates are intentionally allowed so cold-start
    /// probes can establish fresh routeability after restart. The recovery gate
    /// applies only to quarantined peers: it waits for a short cool-down inside
    /// the quarantine window, then allows one signed descriptor to prove
    /// reachability again. It still excludes self/already-used hops and never
    /// reads encrypted blobs, plaintext, client IPs, destinations, DNS contents,
    /// route ids, voucher secrets, private keys, or wallet-level traffic.
    #[must_use]
    pub fn route_probe_candidates_with_capability_excluding(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
        excluded_node_ids: &[[u8; 32]],
    ) -> Vec<SignedNodeDescriptor> {
        self.scored_route_candidates(capability, now, None, true)
            .into_iter()
            .filter(|candidate| {
                let node_id = candidate.descriptor.node_id();
                !excluded_node_ids
                    .iter()
                    .any(|excluded| *excluded == node_id)
            })
            .filter(|candidate| {
                !candidate.summary.route_quarantined
                    || Self::route_quarantine_recovery_probe_ready(
                        candidate.summary.route_quarantine_remaining_seconds,
                    )
            })
            .take(limit)
            .map(|candidate| candidate.descriptor)
            .collect()
    }

    /// Returns probe candidates under the current multi-hop attestation gate.
    ///
    /// Direct single-hop warmup continues using the legacy probe selector.
    /// Two/three-hop proofs call this method so strict mode cannot establish
    /// readiness through a peer lacking current quorum-valid evidence.
    #[must_use]
    pub fn multi_hop_route_probe_candidates_with_capability_excluding(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
        excluded_node_ids: &[[u8; 32]],
    ) -> Vec<SignedNodeDescriptor> {
        self.scored_route_candidates(capability, now, None, true)
            .into_iter()
            .filter(|candidate| {
                let node_id = candidate.descriptor.node_id();
                !excluded_node_ids
                    .iter()
                    .any(|excluded| *excluded == node_id)
                    && self.route_domain_certificate_allows_multi_hop(&node_id, now)
            })
            .filter(|candidate| {
                !candidate.summary.route_quarantined
                    || Self::route_quarantine_recovery_probe_ready(
                        candidate.summary.route_quarantine_remaining_seconds,
                    )
            })
            .take(limit)
            .map(|candidate| candidate.descriptor)
            .collect()
    }

    pub(super) fn scored_route_path_with_capabilities_excluding(
        &self,
        capabilities: &[NodeCapability],
        now: u64,
        excluded_node_ids: &[[u8; 32]],
    ) -> Option<Vec<ScoredPeerRouteCandidate>> {
        fn search_complete_path(
            candidates_by_hop: &[Vec<ScoredPeerRouteCandidate>],
            hop_index: usize,
            excluded: &mut Vec<[u8; 32]>,
            selected: &mut Vec<ScoredPeerRouteCandidate>,
        ) -> bool {
            if hop_index == candidates_by_hop.len() {
                return true;
            }

            for candidate in &candidates_by_hop[hop_index] {
                let node_id = candidate.descriptor.node_id();
                if excluded.iter().any(|excluded| *excluded == node_id)
                    || selected.iter().any(|hop| {
                        !PeerStore::route_endpoints_are_network_diverse(
                            &candidate.descriptor,
                            &hop.descriptor,
                        )
                    })
                {
                    continue;
                }

                excluded.push(node_id);
                selected.push(candidate.clone());
                if search_complete_path(candidates_by_hop, hop_index + 1, excluded, selected) {
                    return true;
                }
                selected.pop();
                excluded.pop();
            }
            false
        }

        let strict_multi_hop = capabilities.len() > 1;
        let candidates_by_hop = capabilities
            .iter()
            .map(|capability| {
                self.scored_route_candidates(*capability, now, None, false)
                    .into_iter()
                    .filter(|candidate| candidate.summary.routeability_ready)
                    .filter(|candidate| {
                        !strict_multi_hop
                            || self.route_domain_certificate_allows_multi_hop(
                                &candidate.descriptor.node_id(),
                                now,
                            )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut excluded = excluded_node_ids.to_vec();
        let mut selected = Vec::with_capacity(capabilities.len());
        search_complete_path(&candidates_by_hop, 0, &mut excluded, &mut selected)
            .then_some(selected)
    }

    /// Plans a complete controlled route for the requested capability sequence.
    ///
    /// The planner selects one unique healthy-ranked peer for each requested
    /// capability, excluding self or already-used hops before each hop is
    /// chosen. It returns `None` unless the full path can be satisfied. This
    /// keeps future multi-hop/onion relay callers from accidentally using a
    /// partial route that weakens the intended privacy boundary.
    ///
    /// This method never reads or derives from encrypted chat/media blobs,
    /// packet payloads, DNS data, destinations, client public IPs, voucher
    /// secrets, private keys, wallet-level traffic, or plaintext content.
    #[must_use]
    pub fn route_path_with_capabilities_excluding(
        &self,
        capabilities: &[NodeCapability],
        now: u64,
        excluded_node_ids: &[[u8; 32]],
    ) -> Option<Vec<SignedNodeDescriptor>> {
        self.scored_route_path_with_capabilities_excluding(capabilities, now, excluded_node_ids)
            .map(|path| {
                path.into_iter()
                    .map(|candidate| candidate.descriptor)
                    .collect()
            })
    }

    pub(super) fn capability_label(capability: NodeCapability) -> &'static str {
        match capability {
            NodeCapability::PrivacyRelay => "privacy_relay",
            NodeCapability::ChatRelay => "chat_relay",
            NodeCapability::EncryptedStorage => "encrypted_storage",
            NodeCapability::AgentRelay => "agent_relay",
            NodeCapability::OnionMiddle => "onion_middle",
            NodeCapability::DirectoryMirrorCarrier => "directory_mirror_carrier",
            NodeCapability::BlindVaultReplica => "blind_vault_replica",
        }
    }

    pub(super) fn descriptor_health(
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> (&'static str, Option<u64>) {
        if descriptor.verify_at(now).is_err() {
            return ("expired", None);
        }
        let remaining = descriptor.descriptor.expires_at.saturating_sub(now);
        if remaining <= PEER_DESCRIPTOR_STALE_WINDOW_SECS {
            ("stale", Some(remaining))
        } else {
            ("healthy", Some(remaining))
        }
    }

    fn source_score(source: &str) -> i64 {
        match source {
            "self" => 30,
            "gossip_announce" | "gossip_snapshot" => 25,
            "cache" | "cache_backup" => 18,
            "url" | "file" => 12,
            _ => 0,
        }
    }

    fn last_seen_score(last_seen_age_seconds: u64) -> i64 {
        if last_seen_age_seconds <= PEER_ROUTE_LAST_SEEN_FRESH_SECS {
            20
        } else if last_seen_age_seconds <= PEER_ROUTE_LAST_SEEN_ACCEPTABLE_SECS {
            10
        } else if last_seen_age_seconds <= PEER_ROUTE_LAST_SEEN_STALE_SECS {
            0
        } else {
            -20
        }
    }

    fn ttl_score(ttl_remaining_seconds: Option<u64>) -> i64 {
        ttl_remaining_seconds
            .map(|ttl| (ttl / PEER_DESCRIPTOR_STALE_WINDOW_SECS).min(20) as i64)
            .unwrap_or(-50)
    }

    fn capacity_score(descriptor: &SignedNodeDescriptor) -> i64 {
        let sessions = (descriptor.descriptor.capacity.max_sessions / 32).min(20) as i64;
        let bps = descriptor
            .descriptor
            .capacity
            .max_bps
            .map(|value| (value / 100_000_000).min(10) as i64)
            .unwrap_or(0);
        let pps = descriptor
            .descriptor
            .capacity
            .max_pps
            .map(|value| (value / 10_000).min(10) as i64)
            .unwrap_or(0);
        sessions + bps + pps
    }

    pub(super) fn route_health_bucket_and_score(
        route_health: Option<&PeerRouteHealth>,
        now: u64,
    ) -> (&'static str, i64) {
        let Some(route_health) = route_health else {
            return ("unknown", 0);
        };
        if Self::route_quarantine_remaining_seconds(route_health, now).is_some() {
            return ("quarantined", -300);
        }
        let recent_failure = route_health
            .last_failure_at
            .map(|failed_at| now.saturating_sub(failed_at) <= PEER_ROUTE_RECENT_FAILURE_SECS)
            .unwrap_or(false);
        let success_after_failure =
            match (route_health.last_success_at, route_health.last_failure_at) {
                (Some(success_at), Some(failure_at)) => success_at >= failure_at,
                (Some(_), None) => true,
                _ => false,
            };

        if recent_failure && !success_after_failure {
            let penalty = (route_health.consecutive_failures.min(4) as i64) * 35;
            if route_health.consecutive_failures >= 3 {
                ("failing", -140)
            } else {
                ("degraded", -penalty)
            }
        } else if route_health.success_count > 0 {
            ("healthy", 8)
        } else {
            ("unknown", 0)
        }
    }

    pub(super) fn routeability_probe_at(route_health: Option<&PeerRouteHealth>) -> Option<u64> {
        route_health.and_then(
            |health| match (health.last_success_at, health.last_failure_at) {
                (Some(success_at), Some(failure_at)) => Some(success_at.max(failure_at)),
                (Some(success_at), None) => Some(success_at),
                (None, Some(failure_at)) => Some(failure_at),
                (None, None) => None,
            },
        )
    }

    pub(super) fn routeability_state_and_ready(
        route_health: Option<&PeerRouteHealth>,
        now: u64,
    ) -> (&'static str, bool) {
        let Some(route_health) = route_health else {
            return ("unknown", false);
        };
        if Self::route_quarantine_remaining_seconds(route_health, now).is_some() {
            return ("quarantined", false);
        }

        match (route_health.last_success_at, route_health.last_failure_at) {
            (Some(success_at), Some(failure_at)) if failure_at > success_at => {
                if now.saturating_sub(failure_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS {
                    ("unreachable", false)
                } else {
                    ("stale", false)
                }
            }
            (Some(success_at), _) => {
                if now.saturating_sub(success_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS {
                    ("reachable", true)
                } else {
                    ("stale", false)
                }
            }
            (None, Some(failure_at)) => {
                if now.saturating_sub(failure_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS {
                    ("unreachable", false)
                } else {
                    ("stale", false)
                }
            }
            (None, None) => ("unknown", false),
        }
    }

    /// Returns whether a peer has fresh routeability evidence at `now`.
    ///
    /// This is intentionally stricter than descriptor validity. A peer may be
    /// signed, unexpired, and capability-compatible while still being unknown,
    /// stale, unreachable, or quarantined from the routing layer's point of
    /// view. Blind relay forwarding must use this helper so encrypted envelopes
    /// are sent only to nodes with fresh successful probe/forward evidence.
    #[must_use]
    pub fn is_routeable_now(&self, node_id: &[u8; 32], now: u64) -> bool {
        if self
            .peers
            .read()
            .get(node_id)
            .is_none_or(|descriptor| !self.permissionless_gate_allows(descriptor, now, true))
        {
            return false;
        }
        let route_health = self.route_health.read();
        let (_, ready) = Self::routeability_state_and_ready(route_health.get(node_id), now);
        ready
    }

    /// Returns whether a peer is currently isolated by route-health quarantine.
    ///
    /// This is narrower than `is_routeable_now`: unknown or stale peers return
    /// false here. Onion middle recovery uses this helper to allow a fresh signed
    /// descriptor to prove itself after restart while still refusing peers that
    /// recently crossed the local failure threshold.
    #[must_use]
    pub fn is_route_quarantined_now(&self, node_id: &[u8; 32], now: u64) -> bool {
        let route_health = self.route_health.read();
        route_health
            .get(node_id)
            .and_then(|value| Self::route_quarantine_remaining_seconds(value, now))
            .is_some()
    }

    pub(super) fn route_quarantine_remaining_seconds(
        route_health: &PeerRouteHealth,
        now: u64,
    ) -> Option<u64> {
        route_health.quarantine_until.and_then(|quarantine_until| {
            (now < quarantine_until).then_some(quarantine_until.saturating_sub(now))
        })
    }

    fn route_quarantine_recovery_probe_ready(remaining_seconds: Option<u64>) -> bool {
        let Some(remaining_seconds) = remaining_seconds else {
            return false;
        };
        let max_remaining_after_cooldown =
            PEER_ROUTE_FAILURE_QUARANTINE_SECS.saturating_sub(PEER_ROUTE_RECOVERY_PROBE_AFTER_SECS);
        remaining_seconds <= max_remaining_after_cooldown
    }

    fn route_score(
        descriptor: &SignedNodeDescriptor,
        source: &str,
        health: &str,
        last_seen_age_seconds: u64,
        ttl_remaining_seconds: Option<u64>,
        route_health_score: i64,
    ) -> i64 {
        let health_score = match health {
            "healthy" => 100,
            "stale" => 40,
            _ => -100,
        };
        health_score
            + Self::source_score(source)
            + Self::last_seen_score(last_seen_age_seconds)
            + Self::ttl_score(ttl_remaining_seconds)
            + Self::capacity_score(descriptor)
            + route_health_score
    }

    pub(super) fn scored_route_candidates(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: Option<usize>,
        include_route_quarantined: bool,
    ) -> Vec<ScoredPeerRouteCandidate> {
        let peers = self.peers.read();
        let metadata = self.peer_runtime.read();
        let capability_label = Self::capability_label(capability).to_string();
        let mut candidates = Vec::new();

        for (node_id, descriptor) in peers.iter() {
            if descriptor.verify_at(now).is_err()
                || !self.permissionless_gate_allows(descriptor, now, !include_route_quarantined)
                || !descriptor.descriptor.capabilities.contains(&capability)
                || descriptor.descriptor.public_endpoint.is_none()
            {
                continue;
            }

            let meta = metadata.get(node_id);
            let source = meta
                .map(|value| value.source.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let last_seen_age_seconds = meta
                .map(|value| now.saturating_sub(value.last_seen_at))
                .unwrap_or(u64::MAX);
            let (health, ttl_remaining_seconds) = Self::descriptor_health(descriptor, now);
            if health == "expired" {
                continue;
            }
            let route_health = self.route_health.read();
            let route_health_entry = route_health.get(node_id);
            let (route_health_bucket, route_health_score) =
                Self::route_health_bucket_and_score(route_health_entry, now);
            let (routeability_state, routeability_ready) =
                Self::routeability_state_and_ready(route_health_entry, now);
            let last_routeability_probe_at = Self::routeability_probe_at(route_health_entry);
            let last_routeability_probe_age_seconds =
                last_routeability_probe_at.map(|probe_at| now.saturating_sub(probe_at));
            if !include_route_quarantined && route_health_bucket == "quarantined" {
                continue;
            }
            let route_quarantine_remaining_seconds = route_health_entry
                .and_then(|value| Self::route_quarantine_remaining_seconds(value, now));
            let score = Self::route_score(
                descriptor,
                &source,
                health,
                last_seen_age_seconds,
                ttl_remaining_seconds,
                route_health_score,
            );

            candidates.push(ScoredPeerRouteCandidate {
                descriptor: descriptor.clone(),
                summary: PeerStoreRouteCandidate {
                    node_id_prefix: hex::encode(&node_id[..4]),
                    capability: capability_label.clone(),
                    score,
                    source,
                    health: health.to_string(),
                    route_health: route_health_bucket.to_string(),
                    routeability_state: routeability_state.to_string(),
                    routeability_ready,
                    last_routeability_probe_at,
                    last_routeability_probe_age_seconds,
                    route_failure_count: route_health_entry
                        .map(|value| value.failure_count)
                        .unwrap_or(0),
                    route_consecutive_failures: route_health_entry
                        .map(|value| value.consecutive_failures)
                        .unwrap_or(0),
                    last_route_success_at: route_health_entry
                        .and_then(|value| value.last_success_at),
                    last_route_failure_at: route_health_entry
                        .and_then(|value| value.last_failure_at),
                    last_route_failure_reason: route_health_entry
                        .and_then(|value| value.last_failure_reason.clone()),
                    route_quarantined: route_quarantine_remaining_seconds.is_some(),
                    route_quarantine_remaining_seconds,
                    route_quarantine_count: route_health_entry
                        .map(|value| value.quarantine_count)
                        .unwrap_or(0),
                    last_seen_age_seconds,
                    ttl_remaining_seconds,
                    endpoint_advertised: true,
                    public_discovery: descriptor.descriptor.policy.public_discovery,
                    region: descriptor.descriptor.policy.region.clone(),
                    max_sessions: descriptor.descriptor.capacity.max_sessions,
                    max_bps: descriptor.descriptor.capacity.max_bps,
                    max_pps: descriptor.descriptor.capacity.max_pps,
                },
            });
        }

        candidates.sort_by(|a, b| {
            b.summary
                .score
                .cmp(&a.summary.score)
                .then_with(|| {
                    a.summary
                        .last_seen_age_seconds
                        .cmp(&b.summary.last_seen_age_seconds)
                })
                .then_with(|| {
                    b.summary
                        .ttl_remaining_seconds
                        .unwrap_or(0)
                        .cmp(&a.summary.ttl_remaining_seconds.unwrap_or(0))
                })
                .then_with(|| a.descriptor.node_id().cmp(&b.descriptor.node_id()))
        });

        if let Some(limit) = limit {
            candidates.truncate(limit);
        }
        candidates
    }
}
