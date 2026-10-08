// [ARCH-SPLIT 2026-10-02]
// Verified descriptor import and discovery-message application.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    // [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Install the complete
    // queue pin set under one authority epoch. Capacity rejection must leave
    // neither a new route pair nor a partially installed source allowlist.
    pub(crate) fn pin_private_onion_queue_identities(
        &self, pins: &PrivateOnionQueueIdentityPins,
    ) -> Result<(), PeerStoreError> {
        let _epoch = self.private_onion_authority_gate.write();
        let mut routes = self.private_onion_route_identity_pins.write();
        let mut sources = self.private_onion_source_identity_pins.write();
        let pair = (pins.relay(), pins.recipient());
        let new_route = usize::from(!routes.contains(&pair));
        let new_sources = pins.sources().iter().filter(|id| !sources.contains(*id)).count();
        if routes.len().saturating_add(new_route) > 64
            || sources.len().saturating_add(new_sources) > 64
        {
            return Err(PeerStoreError::CapacityExceeded { max_peers: 64 });
        }
        routes.insert(pair);
        sources.extend(pins.sources().iter().copied());
        Ok(())
    }

    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub fn pin_private_onion_route_identities(
        &self,
        relay: [u8; 32],
        recipient: [u8; 32],
    ) -> Result<(), PeerStoreError> {
        let _authority_update = self.private_onion_authority_gate.write();
        if relay == [0; 32] || recipient == [0; 32] || relay == recipient {
            return Err(PeerStoreError::VerificationFailed);
        }
        let mut pins = self.private_onion_route_identity_pins.write();
        if pins.len() >= 64 && !pins.contains(&(relay, recipient)) {
            return Err(PeerStoreError::CapacityExceeded { max_peers: 64 });
        }
        pins.insert((relay, recipient));
        Ok(())
    }

    pub fn has_private_onion_route_identity_pin(
        &self,
        relay: &[u8; 32],
        recipient: &[u8; 32],
    ) -> bool {
        self.private_onion_route_identity_pins
            .read()
            .contains(&(*relay, *recipient))
    }

    // [REVERSE-ONION-STALE-SEED 2026-10-05 by Codex] Config authority is a
    // bootstrap hint. It may seed only the pinned R/P identities and can never
    // roll a newer descriptor backward; callers may continue to gossip refresh.
    pub fn seed_private_onion_route_descriptor(
        &self,
        relay: &[u8; 32],
        recipient: &[u8; 32],
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: &'static str,
    ) -> Result<bool, PeerStoreError> {
        let node_id = descriptor.node_id();
        if !self.has_private_onion_route_identity_pin(relay, recipient)
            || (node_id != *relay && node_id != *recipient)
            || (node_id == *relay
                && !descriptor.descriptor.public_endpoint.as_deref()
                    .is_some_and(crate::api::reverse_onion_endpoint_supported))
            || (node_id == *recipient
                && (descriptor.descriptor.public_endpoint.is_some()
                    || descriptor.descriptor.policy.public_discovery))
        {
            return Err(PeerStoreError::VerificationFailed);
        }
        match self.upsert_verified_from_source(descriptor, now, source) {
            Ok(changed) => Ok(changed),
            Err(PeerStoreError::StaleSequence { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub fn pin_private_onion_source_identity(
        &self,
        source: [u8; 32],
    ) -> Result<(), PeerStoreError> {
        // [REVERSE-ONION-AUTHORITY-FENCE 2026-10-05 by Codex] Source identity
        // pins share the same authority update epoch as descriptor/grant state.
        let _authority_update = self.private_onion_authority_gate.write();
        if source == [0; 32] {
            return Err(PeerStoreError::VerificationFailed);
        }
        let mut pins = self.private_onion_source_identity_pins.write();
        if pins.len() >= 64 && !pins.contains(&source) {
            return Err(PeerStoreError::CapacityExceeded { max_peers: 64 });
        }
        pins.insert(source);
        Ok(())
    }

    /// Imports a P-signed authority bundle at R or at a locally configured
    /// source. Sources may refresh only an operator-pinned identity pair.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub fn import_private_onion_authorization_bundle(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        relay_descriptor: SignedNodeDescriptor,
        recipient_descriptor: SignedNodeDescriptor,
        local_node: [u8; 32],
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        let relay_id = authorization.relay_node_id();
        let recipient_id = authorization.recipient_node_id();
        let Some(purpose) = authorization.canonical_purpose() else {
            return Err(PeerStoreError::VerificationFailed);
        };
        if relay_id == recipient_id
            || relay_descriptor.node_id() != relay_id
            || recipient_descriptor.node_id() != recipient_id
            || relay_descriptor.verify_at(now).is_err()
            || recipient_descriptor.verify_at(now).is_err()
            || authorization.verify_at(
                &relay_descriptor,
                &recipient_descriptor,
                purpose,
                now,
            ).is_err()
            // [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] Signed FQDNs
            // are valid only for the operator-pinned relay/recipient pair;
            // ordinary PeerStore admission remains public-IP-only.
            || !relay_descriptor.descriptor.public_endpoint.as_deref()
                .is_some_and(crate::api::reverse_onion_endpoint_supported)
            || recipient_descriptor.descriptor.public_endpoint.is_some()
            // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
            // Bundled authority must carry the same signed private-role policy
            // required by route construction, not only omit an endpoint.
            || recipient_descriptor.descriptor.policy.public_discovery
            // [REVERSE-ONION-DISCOVERY-BOOTSTRAP 2026-10-05 by Codex] The
            // relay and source accept grants only for an operator-pinned pair;
            // valid signatures alone must not let arbitrary P identities fill
            // the bounded authorization cache on R.
            || !self.has_private_onion_route_identity_pin(&relay_id, &recipient_id)
        {
            return Err(PeerStoreError::VerificationFailed);
        }

        // [REVERSE-ONION-AUTHORITY-ATOMIC-IMPORT 2026-10-05 by Codex]
        // Readers capture descriptors and grant under this same epoch. Keep
        // the paired refresh invisible until both descriptors and the grant
        // have passed their monotonicity/capacity checks.
        let _authority_update = self.private_onion_authority_gate.write();
        self.preflight_private_onion_grant_import(&authorization, now)?;

        if local_node == relay_id {
            let (current_relay, current_recipient) = self
                .get_valid_pair(&relay_id, &recipient_id, now)
                .ok_or(PeerStoreError::VerificationFailed)?;
            if current_relay.encode_canonical()
                .map_err(|_| PeerStoreError::VerificationFailed)?
                != relay_descriptor.encode_canonical()
                    .map_err(|_| PeerStoreError::VerificationFailed)?
                || current_recipient.encode_canonical()
                    .map_err(|_| PeerStoreError::VerificationFailed)?
                    != recipient_descriptor.encode_canonical()
                        .map_err(|_| PeerStoreError::VerificationFailed)?
            {
                return Err(PeerStoreError::VerificationFailed);
            }
        } else {
            if local_node == recipient_id {
                return Err(PeerStoreError::VerificationFailed);
            }
            self.preflight_private_onion_authority_import(
                &relay_descriptor,
                &recipient_descriptor,
            )?;
            let max_peers = self.max_peers();
            self.upsert_verified_from_source_under_authority_guard(
                relay_descriptor,
                now,
                "private_onion_authority".to_owned(),
                max_peers,
            )?;
            self.upsert_verified_from_source_under_authority_guard(
                recipient_descriptor,
                now,
                "private_onion_authority".to_owned(),
                max_peers,
            )?;
        }
        self.store_current_private_onion_authorization(authorization, relay_id, recipient_id, now)
    }

    // [REVERSE-ONION-AUTHORITY-ATOMIC-IMPORT 2026-10-05 by Codex]
    fn preflight_private_onion_authority_import(
        &self,
        relay: &SignedNodeDescriptor,
        recipient: &SignedNodeDescriptor,
    ) -> Result<(), PeerStoreError> {
        let peers = self.peers.read();
        for descriptor in [relay, recipient] {
            let incoming = descriptor.sequence();
            if let Some(current) = peers.get(&descriptor.node_id()) {
                let sequence = current.sequence();
                if incoming < sequence {
                    return Err(PeerStoreError::StaleSequence { current: sequence, incoming });
                }
                if incoming == sequence && current != descriptor {
                    return Err(PeerStoreError::VerificationFailed);
                }
            }
        }
        let missing = [relay, recipient]
            .iter()
            .filter(|descriptor| !peers.contains_key(&descriptor.node_id()))
            .count();
        if let Some(max_peers) = self.max_peers() {
            if peers.len().saturating_add(missing) > max_peers {
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }
        Ok(())
    }

    // [REVERSE-ONION-AUTHORITY-ATOMIC-IMPORT 2026-10-05 by Codex]
    fn preflight_private_onion_grant_import(
        &self,
        authorization: &SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<(), PeerStoreError> {
        let purpose = authorization
            .canonical_purpose()
            .ok_or(PeerStoreError::VerificationFailed)?;
        let key = (
            authorization.relay_node_id(),
            authorization.recipient_node_id(),
            purpose.to_owned(),
        );
        let grants = self.private_onion_authorizations.read();
        if let Some(previous) = grants.get(&key).filter(|grant| grant.expires_at() > now) {
            if authorization.issued_at() < previous.issued_at()
                || (authorization.issued_at() == previous.issued_at()
                    && authorization != previous)
            {
                return Err(PeerStoreError::VerificationFailed);
            }
        }
        let live_count = grants.values().filter(|grant| grant.expires_at() > now).count();
        let has_live_key = grants.get(&key).is_some_and(|grant| grant.expires_at() > now);
        if live_count >= 64 && !has_live_key {
            return Err(PeerStoreError::CapacityExceeded { max_peers: 64 });
        }
        Ok(())
    }

    /// Imports a P-signed grant only for this exact local relay and the
    /// currently verified R/P descriptor pair. It is not a peer-admission path.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub fn import_private_onion_authorization(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        local_relay: [u8; 32],
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        let relay_id = authorization.relay_node_id();
        let recipient_id = authorization.recipient_node_id();
        if relay_id != local_relay
            || relay_id == recipient_id
            || !self.has_private_onion_route_identity_pin(&relay_id, &recipient_id)
        {
            return Err(PeerStoreError::VerificationFailed);
        }
        self.cache_verified_private_onion_authorization(authorization, now)
    }

    /// Caches a public P-signed grant for a configured source process. It
    /// creates no relay admission; queue source allowlists remain authoritative.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub fn cache_verified_private_onion_authorization(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        // [REVERSE-ONION-AUTHORITY-FENCE 2026-10-05 by Codex] Verify and store
        // a grant against one descriptor epoch.
        let _authority_update = self.private_onion_authority_gate.write();
        self.cache_verified_private_onion_authorization_under_authority_guard(
            authorization,
            now,
        )
    }

    // [REVERSE-ONION-AUTHORITY-ATOMIC-IMPORT 2026-10-05 by Codex]
    fn cache_verified_private_onion_authorization_under_authority_guard(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        let relay_id = authorization.relay_node_id();
        let recipient_id = authorization.recipient_node_id();
        if relay_id == recipient_id {
            return Err(PeerStoreError::VerificationFailed);
        }
        let (relay, recipient) = self.get_valid_pair(&relay_id, &recipient_id, now)
            .ok_or(PeerStoreError::VerificationFailed)?;
        // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] Local and
        // remote grant-cache paths share the private recipient role boundary.
        if recipient.descriptor.public_endpoint.is_some()
            || recipient.descriptor.policy.public_discovery
        {
            return Err(PeerStoreError::VerificationFailed);
        }
        authorization.verify_at(
            &relay,
            &recipient,
            authorization
                .canonical_purpose()
                .ok_or(PeerStoreError::VerificationFailed)?,
            now,
        ).map_err(|_| PeerStoreError::VerificationFailed)?;
        self.store_current_private_onion_authorization(authorization, relay_id, recipient_id, now)
    }

    /// Retains a locally issued P grant only after it is still bound to the
    /// current signed R/P descriptors. This is separate from remote import:
    /// on P, the recipient identity is local; on R, the relay identity is.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub fn remember_issued_private_onion_authorization(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        local_recipient: [u8; 32],
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        let relay_id = authorization.relay_node_id();
        let recipient_id = authorization.recipient_node_id();
        if recipient_id != local_recipient || relay_id == recipient_id {
            return Err(PeerStoreError::VerificationFailed);
        }
        self.cache_verified_private_onion_authorization(authorization, now)
    }

    // [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex] ACK-time cache
    // admission is best effort. Never block the HTTP executor behind authority
    // writes or use a clock sampled before that epoch. No replacement grant is
    // minted here; a later gossip round may retry after a busy/expired ACK.
    pub(crate) fn try_remember_issued_private_onion_authorization_at(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        local_recipient: [u8; 32],
        floor: u64,
        clock: impl FnOnce() -> u64,
    ) -> Result<Option<bool>, PeerStoreError> {
        let relay = authorization.relay_node_id();
        let recipient = authorization.recipient_node_id();
        if recipient != local_recipient || relay == recipient {
            return Err(PeerStoreError::VerificationFailed);
        }
        let Some(_authority_update) = self.private_onion_authority_gate.try_write() else {
            return Ok(None);
        };
        let now = clock();
        if now == 0 || now < floor || !self.has_private_onion_route_identity_pin(&relay, &recipient) {
            return Err(PeerStoreError::VerificationFailed);
        }
        self.cache_verified_private_onion_authorization_under_authority_guard(authorization, now)
            .map(Some)
    }

    fn store_current_private_onion_authorization(
        &self,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        relay_id: [u8; 32],
        recipient_id: [u8; 32],
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        let purpose = authorization
            .canonical_purpose()
            .ok_or(PeerStoreError::VerificationFailed)?;
        let key = (relay_id, recipient_id, purpose.to_owned());
        let mut grants = self.private_onion_authorizations.write();
        grants.retain(|_, grant| grant.expires_at() > now);
        if let Some(previous) = grants.get(&key) {
            if authorization.issued_at() < previous.issued_at() {
                return Err(PeerStoreError::VerificationFailed);
            }
            if authorization.issued_at() == previous.issued_at() {
                return if authorization == *previous {
                    Ok(false)
                } else {
                    Err(PeerStoreError::VerificationFailed)
                };
            }
        }
        if grants.len() >= 64 && !grants.contains_key(&key) {
            return Err(PeerStoreError::CapacityExceeded { max_peers: 64 });
        }
        grants.insert(key, authorization);
        Ok(true)
    }

    /// Returns a current grant only while both exact signed descriptor pins
    /// and the grant's own validity interval remain live.
    // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
    pub fn current_private_onion_authorization(
        &self,
        relay: &[u8; 32],
        recipient: &[u8; 32],
        now: u64,
    ) -> Option<SignedPrivateOnionRecipientAuthorizationV1> {
        self.current_private_onion_authority_snapshot(relay, recipient, now)
            .map(|(_, _, authorization)| authorization)
    }

    // [PRIVATE-ONION-AUTHORITY-PURPOSES 2026-10-05 by Codex]
    /// Returns the current grant for one exact canonical private purpose.
    pub fn current_private_onion_authorization_for_purpose(
        &self,
        relay: &[u8; 32],
        recipient: &[u8; 32],
        purpose: &str,
        now: u64,
    ) -> Option<SignedPrivateOnionRecipientAuthorizationV1> {
        self.current_private_onion_authority_snapshot_for_purpose(
            relay, recipient, purpose, now,
        )
        .map(|(_, _, authorization)| authorization)
    }

    /// Verifies and inserts or refreshes a descriptor.
    ///
    /// Returns `Ok(true)` when the store changed, `Ok(false)` when the same
    /// sequence was already present, and an error when the descriptor is
    /// invalid or would roll the node backward to an older sequence.
    pub fn upsert_verified(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        self.upsert_verified_from_source(descriptor, now, "unknown")
    }

    /// Verifies and inserts or refreshes a descriptor with a source bucket.
    ///
    /// Source buckets are intentionally coarse (`cache`, `gossip_snapshot`,
    /// `self`, etc.). They let nodeboard distinguish restart recovery from
    /// live discovery without exposing peer URLs or transport metadata.
    pub fn upsert_verified_from_source(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: impl Into<String>,
    ) -> Result<bool, PeerStoreError> {
        let source = source.into();
        if descriptor.verify_at(now).is_err() {
            let node_id = descriptor.node_id();
            self.record_peer_event(
                now,
                "peer_rejected",
                "rejected",
                source,
                &node_id,
                Some(descriptor.sequence()),
                Some("verification_failed"),
            );
            return Err(PeerStoreError::VerificationFailed);
        }
        let _authority_update = self.private_onion_authority_gate.write();
        self.upsert_verified_from_source_under_authority_guard(
            descriptor,
            now,
            source,
            self.max_peers(),
        )
    }

    // [REVERSE-ONION-AUTHORITY-ATOMIC-IMPORT 2026-10-05 by Codex]
    fn upsert_verified_from_source_under_authority_guard(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: String,
        max_peers: Option<usize>,
    ) -> Result<bool, PeerStoreError> {
        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
        let incoming_route_fingerprint =
            Self::descriptor_routeability_surface_fingerprint(&descriptor);

        let mut peers = self.peers.write();
        let mut route_surface_changed = false;

        let is_existing_peer = if let Some(existing) = peers.get(&node_id) {
            let current = existing.sequence();
            if incoming_sequence < current {
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("stale_sequence"),
                );
                return Err(PeerStoreError::StaleSequence {
                    current,
                    incoming: incoming_sequence,
                });
            }
            if incoming_sequence == current {
                if existing != &descriptor {
                    drop(peers);
                    self.record_peer_event(
                        now,
                        "peer_rejected",
                        "rejected",
                        source,
                        &node_id,
                        Some(incoming_sequence),
                        Some("sequence_conflict"),
                    );
                    return Err(PeerStoreError::VerificationFailed);
                }
                drop(peers);
                self.record_peer_runtime(&descriptor, now, source.clone(), false);
                self.record_peer_event(
                    now,
                    "peer_refreshed",
                    "ignored",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("same_sequence"),
                );
                return Ok(false);
            }
            route_surface_changed = Self::descriptor_routeability_surface_fingerprint(existing)
                != incoming_route_fingerprint;
            true
        } else {
            false
        };

        if let Some(max_peers) = max_peers {
            if !is_existing_peer && peers.len() >= max_peers {
                self.counters
                    .capacity_rejected
                    .fetch_add(1, Ordering::Relaxed);
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("capacity_exceeded"),
                );
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }

        peers.insert(node_id, descriptor.clone());
        drop(peers);
        if route_surface_changed {
            self.invalidate_route_success_for_surface_change(&node_id, now);
        }
        self.record_peer_runtime(&descriptor, now, source.clone(), true);
        self.record_peer_event(
            now,
            if is_existing_peer {
                "peer_upgraded"
            } else {
                "peer_inserted"
            },
            "accepted",
            source,
            &node_id,
            Some(incoming_sequence),
            None,
        );
        Ok(true)
    }

    /// Invalidates positive route and receipt evidence after a signed
    /// route-surface change. Failure and quarantine history is retained so
    /// descriptor rotation cannot evade local abuse or failure isolation.
    ///
    /// [RECEIPT-EVIDENCE-LIFECYCLE 2026-08-10 by Codex] Purpose-bound receipt
    /// evidence proves one concrete endpoint/KEM/capability surface. Carrying
    /// it across a signed surface rotation would let a merely reachable new
    /// endpoint inherit cryptographic authority established by the old one.
    pub(super) fn invalidate_route_success_for_surface_change(&self, node_id: &[u8; 32], now: u64) {
        let routeability_invalidated = {
            let mut route_health = self.route_health.write();
            if let Some(health) = route_health.get_mut(node_id) {
                let invalidated = health.last_success_at.take().is_some()
                    || health
                        .last_success_route_fingerprint_sha256
                        .take()
                        .is_some();
                if invalidated {
                    health.success_count = 0;
                }
                invalidated
            } else {
                false
            }
        };
        let receipt_authority_invalidated = self
            .purpose_bound_delivery_receipt_capability
            .write()
            .remove(node_id)
            .is_some();

        if routeability_invalidated || receipt_authority_invalidated {
            self.record_audit_event(
                now,
                "routeability_surface_changed",
                "invalidated",
                format!(
                    "node_prefix={} result=reprobe_required routeability_invalidated={} receipt_authority_invalidated={}",
                    hex::encode(&node_id[..4]),
                    routeability_invalidated,
                    receipt_authority_invalidated,
                ),
            );
        }
    }

    pub(super) fn record_peer_runtime(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        source: String,
        inserted_or_upgraded: bool,
    ) {
        let node_id = descriptor.node_id();
        let mut metadata = self.peer_runtime.write();
        let entry = metadata
            .entry(node_id)
            .or_insert_with(|| PeerRuntimeMetadata {
                source: source.clone(),
                first_seen_at: now,
                last_seen_at: now,
                last_sequence: descriptor.sequence(),
                imported_count: 0,
                expired_degraded_at: None,
            });
        entry.source = source;
        entry.last_seen_at = now;
        entry.last_sequence = descriptor.sequence();
        entry.expired_degraded_at = None;
        if inserted_or_upgraded {
            entry.imported_count = entry.imported_count.saturating_add(1);
        }
    }

    /// Imports all descriptors from a validated bootstrap snapshot.
    ///
    /// Invalid descriptors are counted and skipped. This lets a node keep using
    /// the healthy part of a bootstrap snapshot while surfacing corruption or
    /// expiry through the returned report.
    pub fn load_bootstrap_snapshot(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
    ) -> PeerStoreImportReport {
        self.load_bootstrap_snapshot_from_source(snapshot, now, "unknown")
    }

    /// Imports descriptors from a snapshot and tags runtime metadata source.
    pub fn load_bootstrap_snapshot_from_source(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
        source: impl Into<String>,
    ) -> PeerStoreImportReport {
        let source = source.into();
        let mut report = PeerStoreImportReport {
            total: snapshot.peers.len(),
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };

        for descriptor in &snapshot.peers {
            match self.upsert_verified_from_source(descriptor.clone(), now, source.clone()) {
                Ok(true) => report.inserted += 1,
                Ok(false) => report.unchanged += 1,
                Err(PeerStoreError::StaleSequence { .. }) => report.stale += 1,
                Err(_) => report.rejected += 1,
            }
        }

        self.record_import_report(&report, now);
        report
    }

    /// Applies a discovery gossip message to this store.
    ///
    /// Snapshot requests are read-only and return an empty report; callers can
    /// use `build_snapshot_response()` to construct the actual response.
    pub fn apply_discovery_message(
        &self,
        message: &NodeDiscoveryMessage,
        now: u64,
    ) -> PeerStoreImportReport {
        if self
            .untrusted_discovery_candidate_mode
            .load(Ordering::Acquire)
        {
            // [PERMISSIONLESS-DISCOVERY-CANDIDATES 2026-09-14 by Codex] An
            // anonymous announce may refresh only an identity already held in
            // the receiver's locally established peer cache. This preserves
            // the pinned-witness reboot preflight without admitting a new
            // self-signed identity to live routing. The same receiver-local
            // lifetime limits apply before the normal verified upsert.
            if let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = message {
                if Self::untrusted_candidate_is_within_limits(descriptor, now)
                    && self.has_locally_established_identity(&descriptor.node_id())
                {
                    return self.apply_verified_descriptor_from_source(
                        descriptor.clone(),
                        now,
                        "local_identity_refresh",
                    );
                }
            }
            return self.apply_untrusted_discovery_message(message, now);
        }
        match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => PeerStoreImportReport::empty(),
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => {
                self.load_bootstrap_snapshot_from_source(snapshot, now, "gossip_snapshot")
            }
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => self
                .apply_verified_descriptor_from_source(descriptor.clone(), now, "gossip_announce"),
            // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] PeerStore has
            // no Directory replica trust anchor. Direct callers therefore fail
            // closed; the discovery API may use the dedicated locally anchored
            // admission function before calling the shared descriptor path.
            NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                self.record_rejected_directory_proof_import(now)
            }
            NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                Self::rejected_endpoint_attestation_report()
            }
            NodeDiscoveryMessage::PrivateOnionRecipientAuthorizationV1 { .. } => {
                PeerStoreImportReport { rejected: 1, ..PeerStoreImportReport::empty() }
            }
        }
    }

    /// Admits legacy gossip as a bounded, non-routeable candidate only.
    ///
    /// [PERMISSIONLESS-DISCOVERY-CANDIDATES 2026-09-14 by Codex] A valid
    /// self-signature proves control of one key, not endpoint possession,
    /// independent operation, or routing authority. This boundary must remain
    /// separate from `upsert_verified_from_source`, which is reserved for
    /// self/local cache or independently anchored imports.
    pub(crate) fn apply_untrusted_discovery_message(
        &self,
        message: &NodeDiscoveryMessage,
        now: u64,
    ) -> PeerStoreImportReport {
        let descriptors: Vec<SignedNodeDescriptor> = match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => return PeerStoreImportReport::empty(),
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => vec![descriptor.clone()],
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => snapshot.peers.clone(),
            NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                return self.record_rejected_directory_proof_import(now);
            }
            NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                return Self::rejected_endpoint_attestation_report();
            }
            NodeDiscoveryMessage::PrivateOnionRecipientAuthorizationV1 { .. } => {
                return PeerStoreImportReport { rejected: 1, ..PeerStoreImportReport::empty() };
            }
        };

        let admitted_limit = descriptors
            .len()
            .min(UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE);
        let mut report = PeerStoreImportReport {
            total: descriptors.len(),
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: descriptors.len().saturating_sub(admitted_limit),
        };
        for descriptor in descriptors.into_iter().take(admitted_limit) {
            match self.admit_untrusted_candidate(descriptor, now) {
                CandidateAdmissionOutcome::Candidate => report.candidates += 1,
                CandidateAdmissionOutcome::Unchanged => report.unchanged += 1,
                CandidateAdmissionOutcome::Stale => report.stale += 1,
                CandidateAdmissionOutcome::Conflict
                | CandidateAdmissionOutcome::Saturated
                | CandidateAdmissionOutcome::Rejected => report.rejected += 1,
            }
        }
        self.record_import_report(&report, now);
        report
    }

    // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex] PeerStore is not
    // an attestation verifier or promotion authority. Direct callers receive
    // a coarse rejection report without mutating peer, audit, or counters.
    pub(super) fn rejected_endpoint_attestation_report() -> PeerStoreImportReport {
        PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 1,
        }
    }

    /// Applies one descriptor through the normal verification, capacity, and
    /// anti-rollback path while producing the same aggregate import contract as
    /// snapshot and legacy gossip ingestion.
    ///
    /// This is crate-visible for the Directory-authenticated admission boundary.
    /// Callers must complete their own trust-anchor checks before invoking it.
    pub(crate) fn apply_verified_descriptor_from_source(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: &'static str,
    ) -> PeerStoreImportReport {
        let mut report = PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };
        match self.upsert_verified_from_source(descriptor, now, source) {
            Ok(true) => report.inserted = 1,
            Ok(false) => report.unchanged = 1,
            Err(PeerStoreError::StaleSequence { .. }) => report.stale = 1,
            Err(_) => report.rejected = 1,
        }
        self.record_import_report(&report, now);
        report
    }

    /// Records one fail-closed proof-gossip rejection without retaining any
    /// producer, descriptor, block, endpoint, route, or sender identity.
    pub(crate) fn record_rejected_directory_proof_import(&self, now: u64) -> PeerStoreImportReport {
        let report = PeerStoreImportReport {
            total: 1,
            inserted: 0,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 1,
        };
        self.record_import_report(&report, now);
        report
    }

    pub(super) fn record_import_report(&self, report: &PeerStoreImportReport, now: u64) {
        if report.total == 0 {
            return;
        }

        self.counters
            .total_imported
            .fetch_add(report.total as u64, Ordering::Relaxed);
        self.counters
            .inserted
            .fetch_add(report.inserted as u64, Ordering::Relaxed);
        self.counters
            .candidate_admitted
            .fetch_add(report.candidates as u64, Ordering::Relaxed);
        self.counters
            .unchanged
            .fetch_add(report.unchanged as u64, Ordering::Relaxed);
        self.counters
            .stale
            .fetch_add(report.stale as u64, Ordering::Relaxed);
        self.counters
            .rejected
            .fetch_add(report.rejected as u64, Ordering::Relaxed);
        self.counters.last_import_at.store(now, Ordering::Relaxed);

        let outcome = if report.rejected > 0 || report.stale > 0 {
            "warning"
        } else if report.inserted > 0 || report.candidates > 0 || report.unchanged > 0 {
            "accepted"
        } else {
            "ignored"
        };
        self.record_audit_event(
            now,
            "descriptor_import",
            outcome,
            format!(
                "total={} inserted={} candidates={} unchanged={} stale={} rejected={}",
                report.total,
                report.inserted,
                report.candidates,
                report.unchanged,
                report.stale,
                report.rejected
            ),
        );
    }
}
