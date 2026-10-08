// Split from crates/aeronyx-server/src/services/peer_store.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;
use aeronyx_core::protocol::onion::{OnionRoutePurpose, ONION_FORWARD_HOP_REQUIRED_CAPABILITIES};
use aeronyx_core::protocol::discovery::{NodeCapability, SignedNodeDescriptor};

// [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Authored, not executed.
#[test]
fn private_queue_identity_policy_is_bounded_ordered_and_not_live_authority() {
    let id = |seed| IdentityKeyPair::from_bytes(&[seed; 32]).unwrap().public_key_bytes();
    let (relay, recipient) = (id(200), id(201));
    let sources: Vec<_> = (1..=64).map(id).collect();
    let pins = PrivateOnionQueueIdentityPins::new(relay, recipient, sources.clone()).unwrap();
    assert_eq!(pins.sources(), sources.as_slice());
    for rejected in [Vec::new(), vec![[0; 32]], vec![relay], vec![recipient],
        vec![sources[0], sources[0]], (1..=65).map(id).collect()]
    {
        assert!(PrivateOnionQueueIdentityPins::new(relay, recipient, rejected).is_err());
    }
    assert!(PrivateOnionQueueIdentityPins::new([0; 32], recipient, vec![sources[0]]).is_err());
    assert!(PrivateOnionQueueIdentityPins::new(relay, relay, vec![sources[0]]).is_err());
    let store = PeerStore::new();
    store.pin_private_onion_queue_identities(&pins).unwrap();
    assert!(store.has_private_onion_route_identity_pin(&relay, &recipient));
    assert_eq!(store.private_onion_source_identity_pins.read().len(), 64);
    assert!(store.current_private_onion_pull_authority_snapshot(&relay, &recipient, 1_700_000_100).is_none());
    assert!(store.is_empty(), "operator pins alone must not install descriptors");
    assert!(store.private_onion_authorizations.read().is_empty());
    store.pin_private_onion_queue_identities(&pins).unwrap();
    assert_eq!(store.private_onion_source_identity_pins.read().len(), 64);
}

// [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Authored, not executed:
// both independent pin-table capacities reject before either table mutates.
#[test]
fn private_queue_pin_capacity_rejection_is_atomic_and_reapplication_is_idempotent() {
    let id = |seed| IdentityKeyPair::from_bytes(&[seed; 32]).unwrap().public_key_bytes();
    let (relay, recipient) = (id(200), id(201));
    let pins = PrivateOnionQueueIdentityPins::new(relay, recipient, vec![id(64), id(65)]).unwrap();
    let store = PeerStore::new();
    for seed in 1..=63 { store.pin_private_onion_source_identity(id(seed)).unwrap(); }
    let before = store.private_onion_source_identity_pins.read().clone();
    assert!(matches!(store.pin_private_onion_queue_identities(&pins),
        Err(PeerStoreError::CapacityExceeded { max_peers: 64 })));
    assert_eq!(*store.private_onion_source_identity_pins.read(), before);
    assert!(!store.has_private_onion_route_identity_pin(&relay, &recipient));

    let full_routes = PeerStore::new();
    for seed in 1..=64 { full_routes.pin_private_onion_route_identities(id(seed), recipient).unwrap(); }
    assert!(matches!(full_routes.pin_private_onion_queue_identities(&pins),
        Err(PeerStoreError::CapacityExceeded { max_peers: 64 })));
    assert!(full_routes.private_onion_source_identity_pins.read().is_empty());
    assert!(!full_routes.has_private_onion_route_identity_pin(&relay, &recipient));

    let full = PrivateOnionQueueIdentityPins::new(id(1), recipient, (2..=65).map(id).collect()).unwrap();
    full_routes.pin_private_onion_queue_identities(&full).unwrap();
    full_routes.pin_private_onion_queue_identities(&full).unwrap();
    assert_eq!(full_routes.private_onion_source_identity_pins.read().len(), 64);
    assert_eq!(full_routes.private_onion_route_identity_pins.read().len(), 64);
    let restarted = PeerStore::new();
    assert!(!restarted.has_private_onion_route_identity_pin(&id(1), &recipient));
    restarted.pin_private_onion_queue_identities(&full).unwrap();
    assert!(restarted.current_private_onion_pull_authority_snapshot(&id(1), &recipient, 1_700_000_100).is_none());
}

// [PHALA-PROMOTION-NETWORK-ADMISSION 2026-10-07 by Codex] Authored,
// not run: network preflight is exact Stage-A/capacity admission, never a new
// promotion authority, and a signed rotation invalidates the selected work.
#[test]
fn phala_promotion_network_preflight_requires_exact_candidate_and_capacity() {
    use super::super::permissionless_promotion::MAX_PERMISSIONLESS_PROMOTION_GATES;
    use aeronyx_core::protocol::NodeProtocolFeature;
    let now = 1_700_000_100;
    let target = IdentityKeyPair::from_bytes(&[185; 32]).unwrap();
    // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex] Reserved
    // .example cannot supply a positive public DNS admission fixture.
    let descriptor = permissionless_descriptor_for(&target, 1, now, "https://candidate.aeronyx.network");
    let descriptor = SignedNodeDescriptor::sign(descriptor.descriptor.with_protocol_features(
        [NodeProtocolFeature::PhalaNodeAttestationV1],
    ), &target).unwrap();
    let store = PeerStore::new();
    store.enable_untrusted_discovery_candidate_mode();
    assert!(!store.permissionless_candidate_probe_is_admitted(&descriptor, now));
    assert_eq!(store.admit_permissionless_descriptor(descriptor.clone(), now),
        PermissionlessNodeAdmissionOutcome::Admitted);
    assert!(store.permissionless_candidate_probe_is_admitted(&descriptor, now));
    assert!(store.get_valid(&descriptor.node_id(), now).is_none(),
        "preflight must not install a live route");
    assert!(!store.permissionless_candidate_probe_is_admitted(&descriptor, 0));
    assert!(!store.permissionless_candidate_probe_is_admitted(&descriptor, now + 601));
    let mut altered = descriptor.clone();
    altered.signature[0] ^= 1;
    assert!(!store.permissionless_candidate_probe_is_admitted(&altered, now));
    {
        let mut candidates = store.untrusted_discovery_candidates.write();
        candidates.candidates.get_mut(&descriptor.node_id()).unwrap().commitment[0] ^= 1;
    }
    assert!(!store.permissionless_candidate_probe_is_admitted(&descriptor, now));
    {
        let mut candidates = store.untrusted_discovery_candidates.write();
        candidates.candidates.get_mut(&descriptor.node_id()).unwrap().commitment =
            PeerStore::untrusted_candidate_commitment(&descriptor).unwrap();
    }
    let gate = PermissionlessPromotionGate {
        descriptor_hash: [9; 32], valid_until: now + 300,
        active: false, verified_control_probe: false, generation: 1,
    };
    {
        let mut gates = store.permissionless_promotions.write();
        for index in 0..MAX_PERMISSIONLESS_PROMOTION_GATES {
            let mut id = [0; 32];
            id[..8].copy_from_slice(&(index as u64).to_le_bytes());
            assert_ne!(id, descriptor.node_id());
            gates.insert(id, gate);
        }
    }
    assert!(!store.permissionless_candidate_probe_is_admitted(&descriptor, now),
        "unexpired historic gates reject a new identity before appraisal");
    let removed_id = [0; 32];
    store.permissionless_promotions.write().get_mut(&removed_id).unwrap().valid_until = now - 1;
    assert!(store.permissionless_candidate_probe_is_admitted(&descriptor, now),
        "expired historic capacity can be reclaimed without dropping a live deny gate");
    assert_eq!(store.permissionless_promotions.read().len(), MAX_PERMISSIONLESS_PROMOTION_GATES - 1);
    store.permissionless_promotions.write().insert(removed_id, gate);
    store.permissionless_promotions.write().remove(&removed_id).unwrap();
    assert!(store.permissionless_candidate_probe_is_admitted(&descriptor, now));
    store.permissionless_promotions.write().insert(descriptor.node_id(), gate);
    assert!(store.permissionless_candidate_probe_is_admitted(&descriptor, now),
        "existing identity does not consume an additional historic gate");
    let mut rotated = descriptor.descriptor.clone();
    rotated.sequence += 1;
    rotated.public_endpoint = Some("https://rotated-candidate.aeronyx.network".into());
    let rotated = SignedNodeDescriptor::sign(rotated, &target).unwrap();
    assert_eq!(store.admit_permissionless_descriptor(rotated.clone(), now),
        PermissionlessNodeAdmissionOutcome::Admitted);
    assert!(!store.permissionless_candidate_probe_is_admitted(&descriptor, now));
    assert!(store.permissionless_candidate_probe_is_admitted(&rotated, now));
    assert!(store.get_valid(&descriptor.node_id(), now).is_none());
    store.upsert_verified(rotated.clone(), now).unwrap();
    assert!(store.permissionless_candidate_probe_is_admitted(&rotated, now),
        "an exact live descriptor still permits the same candidate retry");
    let mut conflicting = rotated.descriptor.clone();
    conflicting.public_endpoint = Some("https://live-conflict.aeronyx.network".into());
    let conflicting = SignedNodeDescriptor::sign(conflicting, &target).unwrap();
    let conflict_store = PeerStore::new();
    assert_eq!(conflict_store.admit_permissionless_descriptor(rotated.clone(), now),
        PermissionlessNodeAdmissionOutcome::Admitted);
    conflict_store.upsert_verified(conflicting, now).unwrap();
    assert!(!conflict_store.permissionless_candidate_probe_is_admitted(&rotated, now),
        "same-sequence live conflict denies the old Stage-A transport");
    let mut newer = rotated.descriptor.clone();
    newer.sequence += 1;
    let newer = SignedNodeDescriptor::sign(newer, &target).unwrap();
    store.upsert_verified(newer, now).unwrap();
    assert!(!store.permissionless_candidate_probe_is_admitted(&rotated, now),
        "a newer live import must dominate an unchanged Stage-A table");
    let restarted = PeerStore::new();
    assert!(!restarted.permissionless_candidate_probe_is_admitted(&rotated, now),
        "restart cannot reconstruct Stage-A work from a caller's descriptor");
}

// [PHALA-APPRAISAL-EGRESS-PIN 2026-10-07 by Codex] Authored, not run:
// an authenticated cache, a fresh R appraisal, or signed endpoint renewal
// cannot broaden a private recipient's network scope, including after restart.
#[test]
fn phala_appraisal_keeps_private_egress_and_fixed_relay_origin() {
    use aeronyx_core::protocol::NodeProtocolFeature;
    let now = 1_700_000_100;
    let relay_key = IdentityKeyPair::from_bytes(&[181; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[182; 32]).unwrap();
    let public_key = IdentityKeyPair::from_bytes(&[183; 32]).unwrap();
    let make = |key: &IdentityKeyPair, public: bool, endpoint: Option<&str>| {
        let mut body = signed_descriptor_for(key, 1, now + 900).descriptor;
        body.policy.public_discovery = public;
        body.public_endpoint = endpoint.map(str::to_owned);
        SignedNodeDescriptor::sign(body.with_protocol_features(
            [NodeProtocolFeature::PhalaNodeAttestationV1],
        ), key).unwrap()
    };
    // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex]
    let relay = make(&relay_key, false, Some("https://fixed-relay.aeronyx.network:443"));
    let recipient = make(&recipient_key, false, None);
    let public = make(&public_key, true, Some("https://public-peer.aeronyx.network"));
    let store = PeerStore::new();
    store.configure_phala_attested_peer_routes(true, 300);
    for descriptor in [&relay, &recipient, &public] {
        store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    }
    store.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    let private = PhalaPeerAppraisalEgress::fixed_relay(
        relay.node_id(), "https://fixed-relay.aeronyx.network", true,
    ).unwrap();
    let source = PhalaPeerAppraisalEgress::fixed_relay(
        relay.node_id(), "https://fixed-relay.aeronyx.network", false,
    ).unwrap();
    for cursor in [0, 1, 2, usize::MAX] {
        let target = store.next_phala_peer_appraisal_target(now, 64, cursor, &private).unwrap();
        assert_eq!(target.descriptor(), &relay);
        assert!(target.permits_private_relay());
        assert!(target.transport_origin_is_permitted());
    }
    assert!(store.record_phala_peer_attestation(&relay, now));
    assert!(store.next_phala_peer_appraisal_target(now, 64, 0, &private).is_none(),
        "fresh R evidence must not authorize public-peer fallback");
    assert_eq!(store.next_phala_peer_appraisal_target(now, 64, 0, &source).unwrap().descriptor(), &public,
        "source roles retain ordinary public peer appraisal");

    for (sequence, endpoint) in [
        (2, "https://rotated-relay.aeronyx.network"),
        (3, "https://fixed-relay.aeronyx.network:8443"),
        (4, "http://fixed-relay.aeronyx.network"),
    ] {
        let mut body = relay.descriptor.clone();
        body.sequence = sequence;
        body.public_endpoint = Some(endpoint.to_owned());
        let rotated = SignedNodeDescriptor::sign(body, &relay_key).unwrap();
        store.upsert_verified_from_source(rotated, now + 1, "test_pin").unwrap();
        assert!(store.next_phala_peer_appraisal_target(now + 1, 64, 0, &private).is_none());
        assert_eq!(store.next_phala_peer_appraisal_target(now + 1, 64, 0, &source).unwrap().descriptor(), &public);
    }
    let mut renewed = relay.descriptor.clone();
    renewed.sequence = 5;
    renewed.public_endpoint = Some("https://fixed-relay.aeronyx.network".into());
    let renewed = SignedNodeDescriptor::sign(renewed, &relay_key).unwrap();
    store.upsert_verified_from_source(renewed.clone(), now + 1, "test_pin").unwrap();
    let target = store.next_phala_peer_appraisal_target(now + 1, 64, 0, &private).unwrap();
    assert_eq!(target.descriptor(), &renewed);
    assert!(!store.phala_peer_route_is_eligible(&renewed, now + 1));
    let restarted = PeerStore::new();
    restarted.configure_phala_attested_peer_routes(true, 300);
    for descriptor in [&renewed, &public] {
        restarted.upsert_verified_from_source(descriptor.clone(), now + 1, "test_pin").unwrap();
    }
    restarted.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    assert_eq!(restarted.next_phala_peer_appraisal_target(now + 1, 64, 1, &private).unwrap().descriptor(), &renewed);
    let missing = PhalaPeerAppraisalEgress::fixed_relay(
        recipient.node_id(), "https://fixed-relay.aeronyx.network", true,
    ).unwrap();
    assert!(restarted.next_phala_peer_appraisal_target(now + 1, 64, 0, &missing).is_none());
    assert!(PhalaPeerAppraisalEgress::fixed_relay([0; 32], "https://fixed-relay.aeronyx.network", true).is_none());
    assert!(PhalaPeerAppraisalEgress::fixed_relay(relay.node_id(), "http://fixed-relay.aeronyx.network", true).is_none());
}

// [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] Authored only:
// fixed-relay appraisal bootstraps independently of fresh P grants and public
// discovery membership, but selection itself never supplies route authority.
#[test]
fn phala_appraisal_prioritizes_known_pinned_relays_without_promoting_private_peers() {
    use aeronyx_core::protocol::NodeProtocolFeature;
    let now = 1_700_000_100;
    let relay_key = IdentityKeyPair::from_bytes(&[181; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[182; 32]).unwrap();
    let public_key = IdentityKeyPair::from_bytes(&[183; 32]).unwrap();
    let other_key = IdentityKeyPair::from_bytes(&[184; 32]).unwrap();
    let make = |key: &IdentityKeyPair, public: bool, endpoint: Option<&str>| {
        let mut body = signed_descriptor_for(key, 1, now + 900).descriptor;
        body.policy.public_discovery = public;
        body.public_endpoint = endpoint.map(str::to_owned);
        body = body.with_protocol_features([NodeProtocolFeature::PhalaNodeAttestationV1]);
        SignedNodeDescriptor::sign(body, key).unwrap()
    };
    // [PHALA-EXECUTED-PROFILE-FIXTURES 2026-10-08 by Codex]
    let relay = make(&relay_key, false, Some("https://private-relay.aeronyx.network"));
    let recipient = make(&recipient_key, false, None);
    let public = make(&public_key, true, Some("https://public-peer.aeronyx.network"));
    let other = make(&other_key, false, Some("https://other-private.aeronyx.network"));
    let store = PeerStore::new();
    for descriptor in [&relay, &recipient, &public, &other] {
        store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    }
    store.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    store.pin_private_onion_source_identity(other.node_id()).unwrap();
    assert!(store.next_phala_peer_appraisal_target(now, 64, 0, &PhalaPeerAppraisalEgress::discovery()).is_none());
    store.configure_phala_attested_peer_routes(true, 300);
    assert!(store.next_phala_peer_appraisal_target(now, 0, 0, &PhalaPeerAppraisalEgress::discovery()).is_none());
    let target = store.next_phala_peer_appraisal_target(now, 1, usize::MAX - 1, &PhalaPeerAppraisalEgress::discovery()).unwrap();
    assert_eq!(target.descriptor(), &relay);
    assert!(target.permits_private_relay());
    assert_eq!(store.next_phala_peer_appraisal_target(now, 64, 1, &PhalaPeerAppraisalEgress::discovery()).unwrap().descriptor(), &public,
        "a failed fixed relay must not starve ordinary peers");
    assert!(!store.phala_peer_route_is_eligible(&relay, now));
    assert!(store.current_private_onion_pull_authority_snapshot(&relay.node_id(), &recipient.node_id(), now).is_none());
    assert!(store.record_phala_peer_attestation(&relay, now));
    let ordinary = store.next_phala_peer_appraisal_target(now, 64, 0, &PhalaPeerAppraisalEgress::discovery()).unwrap();
    assert_eq!(ordinary.descriptor(), &public);
    assert!(!ordinary.permits_private_relay());
    assert!(store.record_phala_peer_attestation(&public, now));
    assert!(store.next_phala_peer_appraisal_target(now, 64, 0, &PhalaPeerAppraisalEgress::discovery()).is_none());
    assert_eq!(store.next_phala_peer_appraisal_target(now + 301, 64, 0, &PhalaPeerAppraisalEgress::discovery()).unwrap().descriptor(), &relay);
    let mut rotated = relay.descriptor.clone();
    rotated.sequence += 1;
    let rotated = SignedNodeDescriptor::sign(rotated, &relay_key).unwrap();
    store.upsert_verified_from_source(rotated.clone(), now + 301, "test_pin").unwrap();
    assert_eq!(store.next_phala_peer_appraisal_target(now + 301, 64, 0, &PhalaPeerAppraisalEgress::discovery()).unwrap().descriptor(), &rotated);
    assert!(store.next_phala_peer_appraisal_target(now + 901, 64, 0, &PhalaPeerAppraisalEgress::discovery()).is_none());
    // A new store cannot restore appraisal merely from a signed descriptor.
    let restarted = PeerStore::new();
    restarted.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    restarted.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
    restarted.configure_phala_attested_peer_routes(true, 300);
    assert_eq!(restarted.next_phala_peer_appraisal_target(now, 64, 0, &PhalaPeerAppraisalEgress::discovery()).unwrap().descriptor(), &relay);
    assert!(!restarted.phala_peer_route_is_eligible(&relay, now));
    let staged = PeerStore::new();
    staged.enable_untrusted_discovery_candidate_mode();
    let candidate = make(&relay_key, true, Some("https://candidate-relay.example"));
    assert_eq!(staged.admit_untrusted_candidate(candidate, now), CandidateAdmissionOutcome::Candidate);
    staged.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    staged.configure_phala_attested_peer_routes(true, 300);
    assert!(staged.next_phala_peer_appraisal_target(now, 64, 0, &PhalaPeerAppraisalEgress::discovery()).is_none(),
        "a pin cannot turn Stage-A input into an authenticated descriptor");
}

// [PHALA-AUTHORITY-DESCRIPTOR-EPOCH 2026-10-08 by Codex] Authored only:
// unchanged signed heartbeat bytes preserve a cached exact grant; even a
// temporal-only real renewal must invalidate it until P signs a replacement.
#[cfg(unix)]
#[test]
fn private_authority_grant_survives_identical_heartbeat_not_descriptor_renewal() {
    let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
    let (relay, recipient, _) = fixture.policy_parts();
    let now = fixture.now();
    let recipient_key = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
    let relay_key = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
    let grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), now, now + 100, &recipient_key,
    ).unwrap();
    let store = PeerStore::new();
    store.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    for descriptor in [&relay, &recipient] {
        store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    }
    store.remember_issued_private_onion_authorization(grant.clone(), recipient.node_id(), now).unwrap();
    for descriptor in [&relay, &recipient] {
        store.upsert_verified_from_source(descriptor.clone(), now + 1, "self").unwrap();
    }
    assert!(store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now + 1)
        .as_ref() == Some(&grant));
    let mut renewed = relay.descriptor.clone();
    renewed.sequence += 1;
    renewed.issued_at = now + 2;
    let renewed = SignedNodeDescriptor::sign(renewed, &relay_key).unwrap();
    store.upsert_verified_from_source(renewed.clone(), now + 2, "self").unwrap();
    assert!(store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now + 2).is_none(),
        "descriptor retention must not weaken exact commitment verification");
    let replacement = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &renewed, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), now + 2, now + 100, &recipient_key,
    ).unwrap();
    store.remember_issued_private_onion_authorization(replacement.clone(), recipient.node_id(), now + 2).unwrap();
    assert!(store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now + 3)
        .as_ref() == Some(&replacement));
}

// [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex] Authored, not run:
// exercise exact outbound role/pin/cache admission and ACK-time mutation under
// a known held reader. This uses signed fixtures only, not HTTP or attestation.
#[cfg(unix)]
#[test]
fn private_authority_gossip_and_ack_keep_current_roles_and_clock_epochs() {
    use std::cell::Cell;
    use aeronyx_core::protocol::NodeProtocolFeature;
    let fixture = crate::services::reverse_onion_source::tests::Fixture::new();
    let (relay, recipient, _) = fixture.policy_parts();
    let now = fixture.now();
    let relay_key = IdentityKeyPair::from_bytes(&[42; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[43; 32]).unwrap();
    let source_key = fixture.source_identity();
    let relay = SignedNodeDescriptor::sign(relay.descriptor.with_protocol_features(
        [NodeProtocolFeature::PrivateOnionAuthorizationGossipV1],
    ), &relay_key).unwrap();
    let mut source_body = NodeDescriptor::new(source_key.public_key_bytes(), 1, now - 1, now + 3_600, "test")
        .with_protocol_features([NodeProtocolFeature::PrivateOnionAuthorizationGossipV1]);
    source_body.public_endpoint = Some("https://8.8.8.8".to_owned());
    let source = SignedNodeDescriptor::sign(source_body, &source_key).unwrap();
    let grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), now, now + 10, &recipient_key,
    ).unwrap();
    let store = PeerStore::new();
    for descriptor in [&relay, &recipient, &source] {
        store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    }
    store.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    {
        let _guard = store.private_onion_authority_read_guard();
        assert!(store.private_onion_gossip_bundle_is_current_under_guard(
            recipient.node_id(), &relay, &grant, &relay, &recipient, now,
        ));
        assert!(!store.private_onion_gossip_bundle_is_current_under_guard(
            relay.node_id(), &source, &grant, &relay, &recipient, now,
        ));
    }
    store.pin_private_onion_source_identity(source.node_id()).unwrap();
    {
        let _guard = store.private_onion_authority_read_guard();
        // A pinned source does not authorize R to forward an uncached grant.
        assert!(!store.private_onion_gossip_bundle_is_current_under_guard(
            relay.node_id(), &source, &grant, &relay, &recipient, now,
        ));
        let sampled = Cell::new(false);
        assert!(matches!(store.try_remember_issued_private_onion_authorization_at(
            grant.clone(), recipient.node_id(), now, || { sampled.set(true); now },
        ), Ok(None)));
        assert!(!sampled.get());
    }
    assert!(matches!(store.try_remember_issued_private_onion_authorization_at(
        grant.clone(), recipient.node_id(), now, || now + 1,
    ), Ok(Some(true))));
    {
        let _guard = store.private_onion_authority_read_guard();
        assert!(store.private_onion_gossip_bundle_is_current_under_guard(
            relay.node_id(), &source, &grant, &relay, &recipient, now + 1,
        ));
        assert!(!store.private_onion_gossip_bundle_is_current_under_guard(
            [9; 32], &source, &grant, &relay, &recipient, now + 1,
        ));
        assert!(!store.private_onion_gossip_bundle_is_current_under_guard(
            recipient.node_id(), &relay, &grant, &relay, &recipient, now + 10,
        ));
    }
    assert!(matches!(store.try_remember_issued_private_onion_authorization_at(
        grant.clone(), recipient.node_id(), now + 2, || now + 1,
    ), Err(PeerStoreError::VerificationFailed)));
    assert!(matches!(store.try_remember_issued_private_onion_authorization_at(
        grant.clone(), recipient.node_id(), now, || 0,
    ), Err(PeerStoreError::VerificationFailed)));
    assert!(matches!(store.try_remember_issued_private_onion_authorization_at(
        grant.clone(), recipient.node_id(), now, || now + 10,
    ), Err(PeerStoreError::VerificationFailed)));
    assert!(store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now + 2)
        .as_ref() == Some(&grant));

    let newer = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, OnionRoutePurpose::BlindVaultPull.as_str(), now + 2, now + 12, &recipient_key,
    ).unwrap();
    store.remember_issued_private_onion_authorization(newer.clone(), recipient.node_id(), now + 2).unwrap();
    {
        let _guard = store.private_onion_authority_read_guard();
        for (local, target) in [(recipient.node_id(), &relay), (relay.node_id(), &source)] {
            assert!(!store.private_onion_gossip_bundle_is_current_under_guard(
                local, target, &grant, &relay, &recipient, now + 2,
            ));
            assert!(store.private_onion_gossip_bundle_is_current_under_guard(
                local, target, &newer, &relay, &recipient, now + 2,
            ));
        }
    }
    let mut rotated_source = source.descriptor.clone();
    rotated_source.sequence += 1;
    rotated_source.issued_at = now + 3;
    rotated_source.public_endpoint = Some("https://8.8.4.4".to_owned());
    let rotated_source = SignedNodeDescriptor::sign(rotated_source, &source_key).unwrap();
    store.upsert_verified_from_source(rotated_source.clone(), now + 3, "test_pin").unwrap();
    let _guard = store.private_onion_authority_read_guard();
    assert!(!store.private_onion_gossip_bundle_is_current_under_guard(
        relay.node_id(), &source, &newer, &relay, &recipient, now + 3,
    ));
    assert!(store.private_onion_gossip_bundle_is_current_under_guard(
        relay.node_id(), &rotated_source, &newer, &relay, &recipient, now + 3,
    ));
}

// [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Authored, not run:
// calibrate the real nonblocking read against a known held writer, then prove
// the returned read guard belongs to the same signed-authority epoch.
#[test]
fn final_http_authority_epoch_defers_behind_a_writer() {
    let store = PeerStore::new();
    let writer = store.private_onion_authority_gate.write();
    assert!(store.try_private_onion_authority_read_guard().is_none());
    drop(writer);
    let reader = store.try_private_onion_authority_read_guard().unwrap();
    assert!(store.private_onion_authority_gate.try_write().is_none());
    drop(reader);
    assert!(store.private_onion_authority_gate.try_write().is_some());
}

// [PHALA-PEER-ATTESTED-ROUTING 2026-10-06 by Codex] A cached appraisal is
// usable only for its exact descriptor and expires on the configured age.
#[test]
fn phala_route_gate_requires_fresh_appraisal_for_advertised_phala_peers() {
    let now = 1_700_000_100;
    let key = IdentityKeyPair::from_bytes(&[91; 32]).unwrap();
    let mut body = signed_descriptor_for(&key, 1, now + 900).descriptor;
    body.public_endpoint = Some("https://phala-peer.example".into());
    body = body.with_protocol_features([
        aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
    ]);
    let descriptor = SignedNodeDescriptor::sign(body, &key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    let legacy_key = IdentityKeyPair::from_bytes(&[92; 32]).unwrap();
    let mut legacy_body = signed_descriptor_for(&legacy_key, 1, now + 900).descriptor;
    legacy_body.public_endpoint = Some("https://legacy-peer.example".into());
    store.upsert_verified_from_source(
        SignedNodeDescriptor::sign(legacy_body, &legacy_key).unwrap(),
        now,
        "test_pin",
    ).unwrap();
    store.configure_phala_attested_peer_routes(true, 300);

    assert!(store
        .route_candidates_with_capability(NodeCapability::ChatRelay, now, 4)
        .is_empty());
    assert!(store.record_phala_peer_attestation(&descriptor, now));
    assert_eq!(
        store.route_candidates_with_capability(NodeCapability::ChatRelay, now, 4).len(),
        1,
    );
    assert!(store
        .route_candidates_with_capability(NodeCapability::ChatRelay, now + 301, 4)
        .is_empty());

    let mut renewed_body = descriptor.descriptor.clone();
    renewed_body.sequence += 1;
    renewed_body.issued_at += 1;
    renewed_body.expires_at += 1;
    let renewed = SignedNodeDescriptor::sign(renewed_body, &key).unwrap();
    store.upsert_verified_from_source(renewed, now + 1, "test_pin").unwrap();
    assert!(store
        .route_candidates_with_capability(NodeCapability::ChatRelay, now + 1, 4)
        .is_empty());
}

// [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Synthetic result
// fixtures exercise publication only, not quote acceptance or TEE residency.
// [PHALA-APPRAISAL-TASK-OWNERSHIP 2026-10-07 by Codex] Authored only:
// owner cancellation is rechecked after locks and cannot overwrite existing
// evidence. The positive branch calibrates against an always-rejecting gate.
#[test]
fn phala_verified_publication_owner_gate_runs_inside_publication_epoch() {
    use crate::api::discovery::VerifiedPhalaPeerAttestation;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let key = IdentityKeyPair::from_bytes(&[122; 32]).unwrap();
    let mut body = signed_descriptor_for(&key, 1, now + 3_600).descriptor;
    body.issued_at = now - 600;
    body.public_endpoint = Some("https://phala-peer.example".into());
    body = body.with_protocol_features([
        aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
    ]);
    let descriptor = SignedNodeDescriptor::sign(body, &key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    store.configure_phala_attested_peer_routes(true, 300);
    let appraisal = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now, now);
    for allowed in [false, true] {
        let sampled = std::cell::Cell::new(false);
        assert_eq!(store.record_verified_phala_peer_attestation_if(&descriptor, &appraisal, || {
            sampled.set(true);
            assert!(store.private_onion_authority_gate.try_write().is_none());
            assert!(store.phala_peer_attestations.try_write().is_none());
            allowed
        }), allowed);
        assert!(sampled.get());
        assert_eq!(store.phala_peer_attestations.read().contains_key(&descriptor.node_id()), allowed);
    }
    let published_start = store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().started;
    let refresh = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now, now);
    let stopped = AtomicBool::new(false);
    let entered = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let authority = store.private_onion_authority_gate.write();
        let writer = scope.spawn(|| {
            entered.wait();
            store.record_verified_phala_peer_attestation_if(&descriptor, &refresh,
                || !stopped.load(Ordering::SeqCst))
        });
        entered.wait();
        stopped.store(true, Ordering::SeqCst);
        drop(authority);
        assert!(!writer.join().unwrap());
    });
    assert_eq!(store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().started, published_start);
}

#[test]
fn phala_verified_publication_uses_challenge_age_and_exact_descriptor() {
    use crate::api::discovery::VerifiedPhalaPeerAttestation;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let key = IdentityKeyPair::from_bytes(&[119; 32]).unwrap();
    let mut body = signed_descriptor_for(&key, 1, now + 3_600).descriptor;
    body.issued_at = now - 600;
    body.public_endpoint = Some("https://phala-peer.example".into());
    body = body.with_protocol_features([
        aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
    ]);
    let descriptor = SignedNodeDescriptor::sign(body, &key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    store.configure_phala_attested_peer_routes(true, 300);
    for (challenge, completion) in [(now - 301, now), (now, now + 60), (now + 1, now)] {
        let invalid = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, challenge, completion);
        assert!(!store.record_verified_phala_peer_attestation(&descriptor, &invalid));
        assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now, 300));
    }
    let same_second_earlier = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now - 10, now);
    let fresh = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now - 10, now);
    assert!(store.record_verified_phala_peer_attestation(&descriptor, &fresh));
    // [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex]
    let published_start = store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().started;
    assert!(store.record_verified_phala_peer_attestation(&descriptor, &same_second_earlier));
    assert_eq!(store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().started, published_start);
    assert_eq!(store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().challenged_at, now - 10);
    assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now + 291, 300));
    let older = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now - 20, now);
    // An earlier publication clock must not erase the observed expiry above.
    assert!(!store.record_verified_phala_peer_attestation(&descriptor, &older));
    assert_eq!(store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().challenged_at, now - 10);
    assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now - 11, 300));
    // Wall time alone cannot extend an appraisal after monotonic expiry.
    store.phala_peer_attestations.write().get_mut(&descriptor.node_id()).unwrap().started =
        std::time::Instant::now().checked_sub(std::time::Duration::from_secs(301)).unwrap();
    assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now, 300));
    let mut renewed = descriptor.descriptor.clone();
    renewed.sequence += 1;
    let renewed = SignedNodeDescriptor::sign(renewed, &key).unwrap();
    assert!(!store.record_verified_phala_peer_attestation(&renewed, &fresh));
    store.upsert_verified_from_source(renewed.clone(), now, "test_pin").unwrap();
    assert!(!store.record_verified_phala_peer_attestation(&descriptor, &fresh));
    assert!(!store.phala_peer_attestation_is_fresh(&renewed, now, 300));
}

#[test]
fn phala_verified_publication_rechecks_age_after_authority_lock_wait() {
    use crate::api::discovery::VerifiedPhalaPeerAttestation;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let key = IdentityKeyPair::from_bytes(&[120; 32]).unwrap();
    let mut body = signed_descriptor_for(&key, 1, now + 3_600).descriptor;
    body.issued_at = now - 600;
    body.public_endpoint = Some("https://phala-peer.example".into());
    body = body.with_protocol_features([
        aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
    ]);
    let descriptor = SignedNodeDescriptor::sign(body, &key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    store.configure_phala_attested_peer_routes(true, 300);
    let appraisal = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now - 100, now);
    // Hold the same gate used by production publication, then revoke the
    // freshness budget before releasing it. No time sleeps or network mocks.
    std::thread::scope(|scope| {
        let guard = store.private_onion_authority_gate.write();
        let writer = scope.spawn(|| store.record_verified_phala_peer_attestation(&descriptor, &appraisal));
        store.phala_peer_attestation_max_age_secs.store(1, Ordering::Release);
        drop(guard);
        assert!(!writer.join().unwrap());
    });
    assert!(store.phala_peer_attestations.read().is_empty());
}

// [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex] Authored
// only: delayed publication and later observations cannot be rolled back by
// older queries/results. All timestamps below are local deterministic fixtures.
#[test]
fn phala_cached_appraisal_retains_publication_and_observation_floors() {
    let now = 1_700_100_000;
    let key = IdentityKeyPair::from_bytes(&[121; 32]).unwrap();
    let mut body = signed_descriptor_for(&key, 1, now + 300).descriptor;
    body.issued_at = now - 20;
    body.public_endpoint = Some("https://phala-peer.example".into());
    body = body.with_protocol_features([
        aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
    ]);
    let descriptor = SignedNodeDescriptor::sign(body, &key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    store.configure_phala_attested_peer_routes(true, 300);
    let started = std::time::Instant::now();
    assert!(store.record_phala_peer_appraisal(&descriptor, |_| Some((now - 10, started, now))));
    assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now - 1, 300));
    assert!(store.phala_peer_attestation_is_fresh(&descriptor, now, 300));
    assert!(store.phala_peer_attestation_is_fresh(&descriptor, now + 5, 300));
    assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now + 4, 300));
    assert!(!store.record_phala_peer_appraisal(&descriptor,
        |_| Some((now + 4, std::time::Instant::now(), now + 4))));
    assert_eq!(store.phala_peer_attestations.read().get(&descriptor.node_id()).unwrap().challenged_at, now - 10);
    assert!(store.phala_peer_route_is_eligible(&descriptor, now + 5));
    // A genuinely current new appraisal can renew, without restoring old time.
    assert!(store.record_phala_peer_appraisal(&descriptor,
        |_| Some((now + 5, std::time::Instant::now(), now + 5))));
    assert!(!store.phala_peer_route_is_eligible(&descriptor, now + 4));
    assert!(store.phala_peer_route_is_eligible(&descriptor, now + 5));
}

// [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex]
#[test]
fn phala_cached_descriptor_expiry_is_monotonic_and_older_results_cannot_revive_it() {
    let now = 1_700_100_000;
    let key = IdentityKeyPair::from_bytes(&[122; 32]).unwrap();
    let mut body = signed_descriptor_for(&key, 1, now + 30).descriptor;
    body.issued_at = now - 20;
    body.public_endpoint = Some("https://phala-peer.example".into());
    body = body.with_protocol_features([
        aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
    ]);
    let descriptor = SignedNodeDescriptor::sign(body, &key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(descriptor.clone(), now, "test_pin").unwrap();
    store.configure_phala_attested_peer_routes(true, 300);
    let older = std::time::Instant::now();
    assert!(store.record_phala_peer_attestation(&descriptor, now));
    assert!(store.phala_peer_route_is_eligible(&descriptor, now + 1));
    store.phala_peer_attestations.write().get_mut(&descriptor.node_id()).unwrap().elapsed_override =
        Some(std::time::Duration::from_secs(31));
    assert!(descriptor.verify_at(now + 1).is_ok());
    assert!(!store.phala_peer_route_is_eligible(&descriptor, now + 1));
    assert!(!store.record_phala_peer_appraisal(&descriptor,
        |_| Some((now + 1, std::time::Instant::now(), now + 1))));
    // Restore only the synthetic monotonic anchor to isolate the independent
    // wall observation floor. Actual cache anchors never move in production.
    store.phala_peer_attestations.write().get_mut(&descriptor.node_id()).unwrap().elapsed_override =
        Some(std::time::Duration::ZERO);
    assert!(!store.phala_peer_attestation_is_fresh(&descriptor, now + 31, 300));
    assert!(!store.phala_peer_route_is_eligible(&descriptor, now + 1));
    assert!(!store.record_phala_peer_appraisal(&descriptor, |_| Some((now, older, now + 1))));
    // A new process starts with no eligibility; restart never imports cache.
    let restarted = PeerStore::new();
    restarted.configure_phala_attested_peer_routes(true, 300);
    restarted.upsert_verified_from_source(descriptor.clone(), now + 1, "test_pin").unwrap();
    assert!(!restarted.phala_peer_route_is_eligible(&descriptor, now + 1));
}

// [PHALA-RECIPIENT-POLL-ROUTE-GATE 2026-10-06 by Codex] Authored, not run:
// a signed Pull grant cannot bypass strict attested-peer requirements.
#[test]
fn recipient_pull_authority_snapshot_requires_current_phala_appraisal() {
    let now = 1_700_100_000;
    let relay_key = IdentityKeyPair::from_bytes(&[111; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[112; 32]).unwrap();
    let purpose = OnionRoutePurpose::BlindVaultPull;
    let mut relay_body = signed_descriptor_for(&relay_key, 1, now + 3_600)
        .descriptor
        .with_x25519_kem(relay_key.x25519_public_key_bytes());
    relay_body.public_endpoint = Some("https://relay.example.net".to_owned());
    relay_body.capabilities.extend(ONION_FORWARD_HOP_REQUIRED_CAPABILITIES);
    relay_body = relay_body.with_protocol_features(
        purpose
            .required_path_protocol_features()
            .iter()
            .copied()
            .chain(std::iter::once(
                aeronyx_core::protocol::NodeProtocolFeature::PhalaNodeAttestationV1,
            )),
    );
    let relay = SignedNodeDescriptor::sign(relay_body, &relay_key).unwrap();

    let mut recipient_body = signed_descriptor_for(&recipient_key, 1, now + 3_600)
        .descriptor
        .with_x25519_kem(recipient_key.x25519_public_key_bytes());
    recipient_body.public_endpoint = None;
    recipient_body.policy.public_discovery = false;
    recipient_body.capabilities.push(NodeCapability::BlindVaultReplica);
    recipient_body = recipient_body.with_protocol_features(
        purpose.required_terminal_protocol_features().iter().copied(),
    );
    let recipient = SignedNodeDescriptor::sign(recipient_body, &recipient_key).unwrap();
    let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay,
        &recipient,
        purpose.as_str(),
        now,
        now + 600,
        &recipient_key,
    )
    .unwrap();

    let store = PeerStore::new();
    store.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
    store
        .upsert_verified_from_source(recipient.clone(), now, "test_pin")
        .unwrap();
    store
        .remember_issued_private_onion_authorization(
            authorization,
            recipient_key.public_key_bytes(),
            now,
        )
        .unwrap();
    store
        .pin_private_onion_route_identities(relay.node_id(), recipient.node_id())
        .unwrap();
    store.configure_phala_attested_peer_routes(true, 300);

    assert!(store
        .current_private_onion_pull_authority_snapshot(
            &relay.node_id(),
            &recipient.node_id(),
            now,
        )
        .is_none());
    assert!(store.record_phala_peer_attestation(&relay, now));
    assert!(store
        .current_private_onion_pull_authority_snapshot(
            &relay.node_id(),
            &recipient.node_id(),
            now,
        )
        .is_some());
    assert!(store
        .current_private_onion_pull_authority_snapshot(
            &relay.node_id(),
            &recipient.node_id(),
            now + 301,
        )
        .is_none());
    // [PHALA-FINAL-ROUTE-ADMISSION 2026-10-07 by Codex] Authored, not
    // run: appraisal renewal does not replace the signed R/P/grant snapshot;
    // tightening policy immediately rejects that otherwise-current snapshot.
    assert!(store.record_phala_peer_attestation(&relay, now + 301));
    let epoch = store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 301,
    ).unwrap();
    {
        let _guard = store.private_onion_authority_read_guard();
        assert!(store.current_private_onion_pull_authority_snapshot_under_guard(
            &relay.node_id(), &recipient.node_id(), now + 301,
        ).as_ref() == Some(&epoch));
        assert!(store.current_private_onion_relay_descriptor_under_guard(
            &relay.node_id(), now + 301,
        ).as_ref() == Some(&relay));
    }
    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Use a nonzero
    // budget: a zero-duration monotonic appraisal expires immediately.
    store.configure_phala_attested_peer_routes(true, 1);
    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 303,
    ).is_none());
    assert!(store.current_private_onion_relay_descriptor(&relay.node_id(), now + 303).is_none());
    assert!(store.record_phala_peer_attestation(&relay, now + 303));
    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 303,
    ).as_ref() == Some(&epoch));
}

// [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
#[test]
fn private_authorization_cache_requires_exact_local_current_descriptor_pair() {
    let now = 1_700_000_100;
    let relay_key = IdentityKeyPair::from_bytes(&[71; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[72; 32]).unwrap();
    let purpose = OnionRoutePurpose::BlindVaultPull;
    let admission_purpose = OnionRoutePurpose::BlindVaultLeaseAdmission;
    let mut relay_body = signed_descriptor_for(&relay_key, 1, now + 3_600)
        .descriptor
        .with_x25519_kem(relay_key.x25519_public_key_bytes());
    relay_body.public_endpoint = Some("https://8.8.8.8".to_owned());
    relay_body.capabilities.extend(ONION_FORWARD_HOP_REQUIRED_CAPABILITIES);
    relay_body = relay_body.with_protocol_features(purpose.required_path_protocol_features().iter().copied());
    let relay = SignedNodeDescriptor::sign(relay_body, &relay_key).unwrap();
    let mut recipient_body = signed_descriptor_for(&recipient_key, 1, now + 3_600)
        .descriptor
        .with_x25519_kem(recipient_key.x25519_public_key_bytes());
    recipient_body.public_endpoint = None;
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
    recipient_body.policy.public_discovery = false;
    recipient_body.capabilities.push(NodeCapability::BlindVaultReplica);
    recipient_body.capabilities.push(NodeCapability::ChatRelay);
    recipient_body = recipient_body.with_protocol_features(
        purpose.required_terminal_protocol_features().iter().copied()
            .chain(admission_purpose.required_terminal_protocol_features().iter().copied()),
    );
    let recipient = SignedNodeDescriptor::sign(recipient_body, &recipient_key).unwrap();
    let grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, purpose.as_str(), now, now + 600, &recipient_key,
    ).unwrap();
    // [PRIVATE-ONION-AUTHORITY-PURPOSES 2026-10-05 by Codex] Distinct
    // operation grants for one exact descriptor pair must coexist.
    let admission_grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, admission_purpose.as_str(), now, now + 600, &recipient_key,
    ).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
    store.upsert_verified_from_source(recipient.clone(), now, "test_pin").unwrap();

    assert!(store.remember_issued_private_onion_authorization(
        grant.clone(), recipient_key.public_key_bytes(), now,
    ).unwrap());
    assert!(store.remember_issued_private_onion_authorization(
        admission_grant.clone(), recipient_key.public_key_bytes(), now,
    ).unwrap());
    // [REVERSE-ONION-DISCOVERY-BOOTSTRAP 2026-10-05 by Codex] Relay-side
    // gossip must not cache a valid grant until its configured identity pair
    // is pinned, even when both descriptors are otherwise current.
    assert!(store.import_private_onion_authorization_bundle(
        grant.clone(), relay.clone(), recipient.clone(), relay.node_id(), now,
    ).is_err());
    // [PHALA-PEER-STORE-REGRESSION 2026-10-08 by Codex] Exercise the
    // unpinned rejection before installing the identity-pair admission.
    store.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    assert!(!store.import_private_onion_authorization_bundle(
        grant.clone(), relay.clone(), recipient.clone(), relay.node_id(), now,
    ).unwrap());
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] Signed endpoint
    // omission is insufficient if the recipient policy remains public.
    let mut public_recipient_body = recipient.descriptor.clone();
    public_recipient_body.policy.public_discovery = true;
    let public_recipient = SignedNodeDescriptor::sign(public_recipient_body, &recipient_key).unwrap();
    let public_grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &public_recipient, purpose.as_str(), now, now + 600, &recipient_key,
    ).unwrap();
    assert!(store.seed_private_onion_route_descriptor(
        &relay.node_id(), &recipient.node_id(), public_recipient.clone(), now,
        "test_public_recipient_seed",
    ).is_err());
    let public_cache = PeerStore::new();
    public_cache.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
    public_cache.upsert_verified_from_source(public_recipient.clone(), now, "test_pin").unwrap();
    public_cache.pin_private_onion_route_identities(
        relay.node_id(), public_recipient.node_id(),
    ).unwrap();
    assert!(public_cache.cache_verified_private_onion_authorization(
        public_grant.clone(), now,
    ).is_err());
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] The live
    // recipient authority snapshot is the poller's final admission boundary;
    // keep it fail-closed even if an invalid grant reaches the in-memory map.
    public_cache.private_onion_authorizations.write().insert(
        (relay.node_id(), public_recipient.node_id(), purpose.as_str().to_owned()),
        public_grant.clone(),
    );
    assert!(public_cache.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &public_recipient.node_id(), now,
    ).is_none());
    assert!(store.import_private_onion_authorization_bundle(
        public_grant, relay.clone(), public_recipient, relay.node_id(), now,
    ).is_err());
    assert!(!store.import_private_onion_authorization(
        grant.clone(), relay.node_id(), now,
    ).unwrap());
    assert_eq!(store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now), Some(grant.clone()));
    assert_eq!(
        store.current_private_onion_authorization_for_purpose(
            &relay.node_id(), &recipient.node_id(), admission_purpose.as_str(), now,
        ),
        Some(admission_grant.clone()),
    );
    assert!(store.current_private_onion_authorization_for_purpose(
        &relay.node_id(), &recipient.node_id(), "blind_vault_delete", now,
    ).is_none());
    // [REVERSE-ONION-SOURCE-AUTHORITY-SNAPSHOT 2026-10-05 by Codex]
    let (snapshot_relay, snapshot_recipient, snapshot_grant) = store
        .current_private_onion_authority_snapshot(&relay.node_id(), &recipient.node_id(), now)
        .unwrap();
    assert_eq!(snapshot_relay, relay);
    assert_eq!(snapshot_recipient, recipient);
    assert_eq!(snapshot_grant, grant);
    assert!(store.import_private_onion_authorization(grant.clone(), [99; 32], now).is_err());
    assert!(store.remember_issued_private_onion_authorization(
        store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now).unwrap(),
        relay.node_id(),
        now,
    ).is_err());

    let source_id = IdentityKeyPair::from_bytes(&[73; 32]).unwrap().public_key_bytes();
    let source_store = PeerStore::new();
    source_store.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
    assert!(source_store.import_private_onion_authorization_bundle(
        store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now).unwrap(),
        relay.clone(),
        recipient.clone(),
        source_id,
        now,
    ).unwrap());
    assert_eq!(
        source_store.current_private_onion_authorization(&relay.node_id(), &recipient.node_id(), now),
        Some(grant.clone()),
    );

    let mut rotated_body = signed_descriptor_for(&recipient_key, 2, now + 3_600)
        .descriptor
        .with_x25519_kem(recipient_key.x25519_public_key_bytes());
    rotated_body.public_endpoint = None;
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
    rotated_body.policy.public_discovery = false;
    rotated_body.capabilities.push(NodeCapability::BlindVaultReplica);
    rotated_body.capabilities.push(NodeCapability::ChatRelay);
    rotated_body = rotated_body.with_protocol_features(purpose.required_terminal_protocol_features().iter().copied());
    let rotated = SignedNodeDescriptor::sign(rotated_body, &recipient_key).unwrap();
    store.upsert_verified_from_source(rotated, now, "gossip").unwrap();
    assert!(store.current_private_onion_authorization(&relay.node_id(), &recipient_key.public_key_bytes(), now).is_none());
    let rotated = store.get_valid(&recipient_key.public_key_bytes(), now).unwrap();
    assert!(store.current_private_onion_authority_snapshot(
        &relay.node_id(), &recipient_key.public_key_bytes(), now,
    ).is_none());
    source_store.upsert_verified_from_source(rotated.clone(), now, "gossip").unwrap();
    // [REVERSE-ONION-STALE-SEED 2026-10-05 by Codex] An old pinned config
    // descriptor is a no-op after rotation, not a rollback or startup error.
    assert!(!source_store.seed_private_onion_route_descriptor(
        &relay.node_id(), &recipient.node_id(), recipient.clone(), now + 1,
        "reverse_onion_identity_seed",
    ).unwrap());
    assert!(source_store.upsert_verified_from_source(
        recipient.clone(), now + 1, "reverse_onion_identity_seed",
    ).is_err());
    assert_eq!(
        source_store.get_valid(&recipient_key.public_key_bytes(), now + 1), Some(rotated.clone()),
    );
    // [REVERSE-ONION-DISCOVERY-BOOTSTRAP 2026-10-05 by Codex] Descriptor/KEM
    // rotation invalidates the old grant without erasing the operator's stable
    // route identity pin; discovery may install a fresh descriptor-bound grant.
    assert!(source_store.has_private_onion_route_identity_pin(
        &relay.node_id(), &recipient_key.public_key_bytes(),
    ));
    let rotated_grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay,
        &rotated,
        purpose.as_str(),
        now + 1,
        now + 600,
        &recipient_key,
    ).unwrap();
    assert!(source_store.import_private_onion_authorization_bundle(
        rotated_grant.clone(),
        relay,
        rotated,
        source_id,
        now + 2,
    ).unwrap());
    assert_eq!(
        source_store.current_private_onion_authorization(
            &rotated_grant.relay_node_id(),
            &rotated_grant.recipient_node_id(),
            now + 2,
        ),
        Some(rotated_grant),
    );
}

// [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
#[test]
fn recipient_fresh_claim_authority_pauses_on_expiry_and_resumes_on_renewal() {
    let now = 1_700_100_000;
    let relay_key = IdentityKeyPair::from_bytes(&[74; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[75; 32]).unwrap();
    let purpose = OnionRoutePurpose::BlindVaultPull;
    let mut relay_body = signed_descriptor_for(&relay_key, 1, now + 3_600)
        .descriptor.with_x25519_kem(relay_key.x25519_public_key_bytes());
    relay_body.public_endpoint = Some("https://8.8.8.8".to_owned());
    relay_body.capabilities.extend(ONION_FORWARD_HOP_REQUIRED_CAPABILITIES);
    relay_body = relay_body.with_protocol_features(purpose.required_path_protocol_features().iter().copied());
    let relay = SignedNodeDescriptor::sign(relay_body, &relay_key).unwrap();
    let mut recipient_body = signed_descriptor_for(&recipient_key, 1, now + 3_600)
        .descriptor.with_x25519_kem(recipient_key.x25519_public_key_bytes());
    recipient_body.public_endpoint = None;
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
    recipient_body.policy.public_discovery = false;
    recipient_body.capabilities.push(NodeCapability::BlindVaultReplica);
    recipient_body = recipient_body.with_protocol_features(
        purpose.required_terminal_protocol_features().iter().copied(),
    );
    let recipient = SignedNodeDescriptor::sign(recipient_body, &recipient_key).unwrap();
    let store = PeerStore::new();
    store.upsert_verified_from_source(relay.clone(), now, "test_pin").unwrap();
    store.upsert_verified_from_source(recipient.clone(), now, "self").unwrap();
    let first = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, purpose.as_str(), now, now + 10, &recipient_key,
    ).unwrap();
    store.remember_issued_private_onion_authorization(
        first, recipient_key.public_key_bytes(), now,
    ).unwrap();

    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 9,
    ).is_some());
    // [REVERSE-ONION-AUTHORITY-EPOCH 2026-10-06 by Codex] The outbound
    // carrier receives one typed R/P/grant snapshot, not separately fetched pieces.
    let epoch = store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 9,
    ).unwrap();
    assert_eq!(epoch.relay, relay);
    assert_eq!(epoch.recipient, recipient);
    assert_eq!(epoch.authorization.expires_at(), now + 10);
    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 10,
    ).is_none());

    let renewed = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &recipient, purpose.as_str(), now + 10, now + 20, &recipient_key,
    ).unwrap();
    assert!(store.remember_issued_private_onion_authorization(
        renewed, recipient_key.public_key_bytes(), now + 10,
    ).unwrap());
    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 11,
    ).is_some());

    // [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
    // A descriptor/KEM rotation invalidates the old grant until Discovery
    // installs a signature bound to the new exact R/P descriptor pair.
    let mut rotated_body = signed_descriptor_for(&recipient_key, 2, now + 3_600)
        .descriptor.with_x25519_kem(recipient_key.x25519_public_key_bytes());
    rotated_body.public_endpoint = None;
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
    rotated_body.policy.public_discovery = false;
    rotated_body.capabilities.push(NodeCapability::BlindVaultReplica);
    rotated_body = rotated_body.with_protocol_features(
        purpose.required_terminal_protocol_features().iter().copied(),
    );
    let rotated = SignedNodeDescriptor::sign(rotated_body, &recipient_key).unwrap();
    store.upsert_verified_from_source(rotated.clone(), now + 12, "gossip").unwrap();
    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 12,
    ).is_none());
    let rotated_grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay, &rotated, purpose.as_str(), now + 13, now + 23, &recipient_key,
    ).unwrap();
    store.remember_issued_private_onion_authorization(
        rotated_grant, recipient_key.public_key_bytes(), now + 13,
    ).unwrap();
    assert!(store.current_private_onion_pull_authority_snapshot(
        &relay.node_id(), &recipient.node_id(), now + 14,
    ).is_some());
}

// [REVERSE-ONION-AUTHORITY-ATOMIC-IMPORT 2026-10-05 by Codex]
#[test]
fn rejected_authority_pair_refresh_does_not_leave_half_updated_descriptors() {
    let now = 1_700_000_100;
    let relay_key = IdentityKeyPair::from_bytes(&[81; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[82; 32]).unwrap();
    let purpose = OnionRoutePurpose::BlindVaultPull;
    let mut relay_v1_body = signed_descriptor_for(&relay_key, 1, now + 3_600)
        .descriptor
        .with_x25519_kem(relay_key.x25519_public_key_bytes());
    relay_v1_body.public_endpoint = Some("https://8.8.8.8".to_owned());
    relay_v1_body.capabilities.extend(ONION_FORWARD_HOP_REQUIRED_CAPABILITIES);
    relay_v1_body = relay_v1_body.with_protocol_features(
        purpose.required_path_protocol_features().iter().copied(),
    );
    let relay_v1 = SignedNodeDescriptor::sign(relay_v1_body, &relay_key).unwrap();

    let mut recipient_body = signed_descriptor_for(&recipient_key, 1, now + 3_600)
        .descriptor
        .with_x25519_kem(recipient_key.x25519_public_key_bytes());
    recipient_body.public_endpoint = None;
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex]
    recipient_body.policy.public_discovery = false;
    recipient_body.capabilities.extend([
        NodeCapability::BlindVaultReplica,
        NodeCapability::ChatRelay,
    ]);
    recipient_body = recipient_body.with_protocol_features(
        purpose.required_terminal_protocol_features().iter().copied(),
    );
    let recipient_v1 = SignedNodeDescriptor::sign(recipient_body.clone(), &recipient_key).unwrap();

    let source_id = IdentityKeyPair::from_bytes(&[83; 32]).unwrap().public_key_bytes();
    let store = PeerStore::new();
    store.pin_private_onion_route_identities(relay_v1.node_id(), recipient_v1.node_id()).unwrap();
    store.upsert_verified_from_source(relay_v1.clone(), now, "test_pin").unwrap();
    store.upsert_verified_from_source(recipient_v1.clone(), now, "test_pin").unwrap();

    let mut relay_v2_body = signed_descriptor_for(&relay_key, 2, now + 3_600)
        .descriptor
        .with_x25519_kem(relay_key.x25519_public_key_bytes());
    relay_v2_body.public_endpoint = Some("https://relay.example.net".to_owned());
    relay_v2_body.capabilities.extend(ONION_FORWARD_HOP_REQUIRED_CAPABILITIES);
    relay_v2_body = relay_v2_body.with_protocol_features(
        purpose.required_path_protocol_features().iter().copied(),
    );
    let relay_v2 = SignedNodeDescriptor::sign(relay_v2_body, &relay_key).unwrap();

    // Same sequence with different signed contents is a conflict, not a
    // partial-refresh opportunity for the otherwise valid relay descriptor.
    // [PHALA-PEER-STORE-REGRESSION 2026-10-08 by Codex] Mutate a signed
    // field without corrupting the exact SemVer protocol feature tokens.
    recipient_body.capacity.max_sessions += 1;
    let conflicting_recipient = SignedNodeDescriptor::sign(recipient_body, &recipient_key).unwrap();
    let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay_v2,
        &conflicting_recipient,
        purpose.as_str(),
        now + 1,
        now + 600,
        &recipient_key,
    ).unwrap();

    assert!(store.import_private_onion_authorization_bundle(
        authorization,
        relay_v2,
        conflicting_recipient,
        source_id,
        now + 1,
    ).is_err());
    assert_eq!(store.get_valid(&relay_v1.node_id(), now + 1), Some(relay_v1.clone()));
    assert_eq!(store.get_valid(&recipient_v1.node_id(), now + 1), Some(recipient_v1.clone()));
    assert!(store.current_private_onion_authorization(
        &relay_key.public_key_bytes(),
        &recipient_key.public_key_bytes(),
        now + 1,
    ).is_none());

    let capacity_store = PeerStore::with_max_peers(1);
    capacity_store.pin_private_onion_route_identities(
        relay_v1.node_id(), recipient_v1.node_id(),
    ).unwrap();
    let capacity_grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay_v1,
        &recipient_v1,
        purpose.as_str(),
        now,
        now + 600,
        &recipient_key,
    ).unwrap();
    assert!(capacity_store.import_private_onion_authorization_bundle(
        capacity_grant,
        relay_v1.clone(),
        recipient_v1.clone(),
        source_id,
        now,
    ).is_err());
    assert!(capacity_store.get_valid(&relay_v1.node_id(), now).is_none());
    assert!(capacity_store.get_valid(&recipient_v1.node_id(), now).is_none());
}

// [REVERSE-ONION-RECIPIENT-ROUTE-REFRESH 2026-10-05 by Codex]
#[test]
fn recipient_relay_lookup_tracks_only_current_signed_route_surface() {
    let now = 1_700_000_100;
    let relay_key = IdentityKeyPair::from_bytes(&[74; 32]).unwrap();
    let purpose = OnionRoutePurpose::BlindVaultPull;
    let store = PeerStore::new();
    let make_relay = |sequence, endpoint: &str, advertises_path: bool| {
        let mut descriptor = signed_descriptor_for(&relay_key, sequence, now + 3_600).descriptor
            .with_x25519_kem(relay_key.x25519_public_key_bytes());
        descriptor.public_endpoint = Some(endpoint.to_owned());
        descriptor.capabilities.extend(ONION_FORWARD_HOP_REQUIRED_CAPABILITIES);
        if advertises_path {
            descriptor = descriptor.with_protocol_features(
                purpose.required_path_protocol_features().iter().copied(),
            );
        }
        SignedNodeDescriptor::sign(descriptor, &relay_key).unwrap()
    };

    store.upsert_verified_from_source(make_relay(1, "https://8.8.8.8", true), now, "test_pin").unwrap();
    assert_eq!(
        store.current_private_onion_relay_descriptor(&relay_key.public_key_bytes(), now)
            .unwrap().descriptor.public_endpoint.as_deref(),
        Some("https://8.8.8.8"),
    );
    store.upsert_verified_from_source(make_relay(2, "https://9.9.9.9", true), now + 1, "test_pin").unwrap();
    assert_eq!(
        store.current_private_onion_relay_descriptor(&relay_key.public_key_bytes(), now + 1)
            .unwrap().descriptor.public_endpoint.as_deref(),
        Some("https://9.9.9.9"),
    );
    store.upsert_verified_from_source(
        make_relay(3, "https://9.9.9.9", false), now + 2, "test_pin",
    ).unwrap();
    assert!(store.current_private_onion_relay_descriptor(
        &relay_key.public_key_bytes(), now + 2,
    ).is_none());
    assert!(store.current_private_onion_relay_descriptor(
        &relay_key.public_key_bytes(), now + 3_601,
    ).is_none());
    // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] Generic peer admission
    // may retain an HTTP IP endpoint, but it cannot become a private relay.
    store.upsert_verified_from_source(
        make_relay(4, "http://8.8.8.8:8422", true), now + 4, "test_pin",
    ).unwrap();
    assert!(store.current_private_onion_relay_descriptor(
        &relay_key.public_key_bytes(), now + 4,
    ).is_none());
}

#[test]
fn endpoint_attestation_direct_peer_store_entries_reject_without_mutation() {
    // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex] Only the API
    // adapter owns verification. Trusted and untrusted PeerStore entry
    // points must reject this carrier without counters, audit, or peers.
    let now = 1_780_000_000;
    let store = PeerStore::new();
    let message = NodeDiscoveryMessage::EndpointEvidenceAttestationV1 {
        attestation_frame: vec![0x55; 289],
    };
    let before = store.status(now);
    let expected = PeerStoreImportReport {
        total: 1,
        inserted: 0,
        candidates: 0,
        unchanged: 0,
        stale: 0,
        rejected: 1,
    };

    assert_eq!(store.apply_discovery_message(&message, now), expected);
    assert_eq!(store.status(now), before);
    assert_eq!(
        store.apply_untrusted_discovery_message(&message, now),
        expected
    );
    assert_eq!(store.status(now), before);
}

#[test]
fn test_upsert_verified_stores_descriptor() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);

    assert_eq!(
        store.upsert_verified(descriptor, 1_700_000_100).unwrap(),
        true
    );
    assert_eq!(store.len(), 1);
    assert_eq!(store.snapshot(1_700_000_100).valid_peers, 1);
}

#[test]
fn test_same_sequence_is_idempotent() {
    let store = PeerStore::new();
    let kp = IdentityKeyPair::generate();
    let descriptor = signed_descriptor_for(&kp, 1, 1_700_001_000);
    let same = signed_descriptor_for(&kp, 1, 1_700_001_000);

    assert_eq!(
        store
            .upsert_verified(descriptor, 1_700_000_100)
            .expect("first insert"),
        true
    );
    assert_eq!(
        store
            .upsert_verified(same, 1_700_000_100)
            .expect("same sequence"),
        false
    );
}

#[test]
fn test_stale_sequence_rejected() {
    let store = PeerStore::new();
    let kp = IdentityKeyPair::generate();
    let newer = signed_descriptor_for(&kp, 2, 1_700_001_000);
    let older = signed_descriptor_for(&kp, 1, 1_700_001_000);

    store.upsert_verified(newer, 1_700_000_100).unwrap();
    let err = store.upsert_verified(older, 1_700_000_100).unwrap_err();

    assert!(matches!(err, PeerStoreError::StaleSequence { .. }));
}

#[test]
fn test_expired_descriptor_rejected() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_000_050);

    let err = store
        .upsert_verified(descriptor, 1_700_000_100)
        .unwrap_err();
    assert!(matches!(err, PeerStoreError::VerificationFailed));
    assert!(store.is_empty());
}

#[test]
fn test_max_peers_rejects_new_descriptor_but_allows_existing_update() {
    let store = PeerStore::with_max_peers(1);
    let kp = IdentityKeyPair::generate();
    let first = signed_descriptor_for(&kp, 1, 1_700_001_000);
    let first_update = signed_descriptor_for(&kp, 2, 1_700_001_000);
    let second = signed_descriptor(1, 1_700_001_000);

    assert!(store.upsert_verified(first, 1_700_000_100).unwrap());
    assert!(store.upsert_verified(first_update, 1_700_000_100).unwrap());

    let err = store.upsert_verified(second, 1_700_000_100).unwrap_err();
    assert!(matches!(err, PeerStoreError::CapacityExceeded { .. }));
    assert_eq!(store.len(), 1);
    assert_eq!(store.status(1_700_000_100).runtime.capacity_rejected, 1);
}

#[test]
fn test_capability_query_returns_only_valid_matching_peers() {
    let store = PeerStore::new();
    let matching = signed_descriptor(1, 1_700_001_000);
    let mut non_matching = signed_descriptor(1, 1_700_001_000);
    let kp = IdentityKeyPair::generate();
    non_matching.descriptor.node_id = kp.public_key_bytes();
    non_matching.descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
    non_matching = SignedNodeDescriptor::sign(non_matching.descriptor, &kp).unwrap();

    store.upsert_verified(matching, 1_700_000_100).unwrap();
    store.upsert_verified(non_matching, 1_700_000_100).unwrap();

    let peers = store.peers_with_capability(NodeCapability::ChatRelay, 1_700_000_100);
    assert_eq!(peers.len(), 1);
}

#[test]
fn test_verified_receipt_peers_do_not_imply_network_diverse_path() {
    // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Preserve
    // valid terminal receipt evidence while refusing to advertise a live
    // App path when both eligible hops share one endpoint network identity.
    let now = 1_700_000_100;
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut middle = signed_descriptor_for(&middle_identity, 7, now + 4_000);
    middle.descriptor.public_endpoint = Some("https://collocated.example:8422".to_string());
    middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_identity).unwrap();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://collocated.example:9422".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();

    let store = PeerStore::new();
    store.upsert_verified(middle.clone(), now).unwrap();
    store.upsert_verified(terminal.clone(), now).unwrap();
    assert!(store.record_verified_client_onion_route_delivery(&middle, &terminal, now + 1,));

    let quality = store.status(now + 2).blind_relay_quality;
    assert_eq!(quality.delivery_receipt_capable_peers, 2);
    assert!(!quality.authenticated_delivery_path_ready);
    assert_eq!(
        quality.authenticated_delivery_path_reason,
        "no_network_diverse_receipt_path"
    );
    assert!(!quality.real_relay_ready);
}

#[test]
fn test_verified_synthetic_probe_path_evidence_is_all_or_nothing() {
    let now = 1_700_000_100;
    let first_identity = IdentityKeyPair::generate();
    let second_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut first = signed_descriptor_for(&first_identity, 7, now + 4_000);
    first.descriptor.public_endpoint = Some("https://first.example".to_string());
    first.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    first = SignedNodeDescriptor::sign(first.descriptor, &first_identity).unwrap();

    let mut second = signed_descriptor_for(&second_identity, 7, now + 4_000);
    second.descriptor.public_endpoint = Some("https://second-a.example".to_string());
    second.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    second = SignedNodeDescriptor::sign(second.descriptor, &second_identity).unwrap();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://terminal.example".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();

    let first_node_id = first.node_id();
    let second_node_id = second.node_id();
    let terminal_node_id = terminal.node_id();
    let store = PeerStore::new();
    for descriptor in [first.clone(), second.clone(), terminal.clone()] {
        store.upsert_verified(descriptor, now).unwrap();
    }

    let mut rotated_second_body = second.descriptor.clone();
    rotated_second_body.sequence = 8;
    rotated_second_body.issued_at = now + 20;
    rotated_second_body.expires_at = now + 4_020;
    rotated_second_body.public_endpoint = Some("https://second-b.example".to_string());
    let rotated_second = SignedNodeDescriptor::sign(rotated_second_body, &second_identity).unwrap();
    store
        .upsert_verified(rotated_second.clone(), now + 20)
        .unwrap();

    // [ATOMIC-MULTIHOP-PROOF-EVIDENCE 2026-08-11 by Codex] A rotation of
    // any hop rejects the whole proof. No unchanged hop receives partial
    // health/capability credit and no success enters admission history.
    assert!(!store.record_verified_three_hop_probe_delivery(
        &first,
        &second,
        &terminal,
        now + 21,
        2,
        1,
    ));
    let rejected = store.status(now + 21);
    assert_eq!(rejected.three_hop_path_proof_history.attempted, 0);
    assert_eq!(
        rejected.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    for node_id in [first_node_id, second_node_id, terminal_node_id] {
        assert!(!store.is_routeable_now(&node_id, now + 21));
    }
    assert!(!store.take_peer_cache_dirty());

    assert!(store.record_verified_three_hop_probe_delivery(
        &first,
        &rotated_second,
        &terminal,
        now + 22,
        2,
        1,
    ));
    let accepted = store.status(now + 22);
    assert_eq!(accepted.three_hop_path_proof_history.attempted, 1);
    assert_eq!(accepted.three_hop_path_proof_history.succeeded, 1);
    assert_eq!(
        accepted
            .three_hop_path_proof_history
            .message_delivery_successes,
        1
    );
    assert_eq!(
        accepted.blind_relay_quality.delivery_receipt_capable_peers,
        3
    );
    for node_id in [first_node_id, second_node_id, terminal_node_id] {
        assert!(store.is_routeable_now(&node_id, now + 22));
    }
    assert!(store.take_peer_cache_dirty());
}

#[test]
fn test_legacy_control_probe_binds_both_surfaces_without_receipt_upgrade() {
    let now = 1_700_000_100;
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut middle = signed_descriptor_for(&middle_identity, 7, now + 4_000);
    middle.descriptor.public_endpoint = Some("https://middle.example".to_string());
    middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_identity).unwrap();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://terminal-a.example".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();

    let middle_node_id = middle.node_id();
    let terminal_node_id = terminal.node_id();
    let store = PeerStore::new();
    store.upsert_verified(middle.clone(), now).unwrap();
    store.upsert_verified(terminal.clone(), now).unwrap();

    let mut rotated_terminal_body = terminal.descriptor.clone();
    rotated_terminal_body.sequence = 8;
    rotated_terminal_body.issued_at = now + 20;
    rotated_terminal_body.expires_at = now + 4_020;
    rotated_terminal_body.public_endpoint = Some("https://terminal-b.example".to_string());
    let rotated_terminal =
        SignedNodeDescriptor::sign(rotated_terminal_body, &terminal_identity).unwrap();
    store
        .upsert_verified(rotated_terminal.clone(), now + 20)
        .unwrap();

    // [LEGACY-CONTROL-PROOF-SURFACE-BINDING 2026-08-11 by Codex] A stale
    // terminal invalidates the full compatibility proof before any middle
    // health or proof-history success is published.
    assert!(!store.record_verified_two_hop_control_probe(&middle, &terminal, now + 21, 2, 1,));
    let rejected = store.status(now + 21);
    assert_eq!(rejected.two_hop_path_proof_history.attempted, 0);
    assert_eq!(
        rejected.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    assert!(!store.is_routeable_now(&middle_node_id, now + 21));
    assert!(!store.take_peer_cache_dirty());

    assert!(store.record_verified_two_hop_control_probe(
        &middle,
        &rotated_terminal,
        now + 22,
        2,
        1,
    ));
    let accepted = store.status(now + 22);
    assert_eq!(accepted.two_hop_path_proof_history.succeeded, 1);
    assert_eq!(
        accepted
            .two_hop_path_proof_history
            .latest_reason_bucket
            .as_deref(),
        Some("legacy_control_forwarded")
    );
    assert_eq!(
        accepted.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    assert!(store.is_routeable_now(&middle_node_id, now + 22));
    assert!(!store.is_routeable_now(&terminal_node_id, now + 22));
    assert!(store.take_peer_cache_dirty());
}

#[test]
fn test_peer_health_reason_vocabularies_are_closed_and_protocol_complete() {
    for reason in [
        "http_100",
        "http_599",
        "onion_delivery_http_425",
        "peer_relay_http_502",
        "ack_response_too_large",
        "onion_ack_response_json_decode_failed",
        "peer_relay_ack_response_body_read_failed",
        "blind_relay_probe_timeout",
        "two_hop_onion_delivery_probe_connect",
        "three_hop_onion_delivery_probe_http_503",
        "onion_delivery_request_decode",
        "peer_relay_request_unknown",
        "peer_relay_receipt_signature_invalid",
    ] {
        assert!(is_route_failure_reason(reason), "rejected {reason}");
    }

    for reason in [
        "http_099",
        "http_600",
        "http_502_private",
        "peer_relay_http_502 endpoint=private",
        "ack_peer_body",
        "unknown_phase_timeout",
        "peer_relay_request_private_detail",
        "PEER_RELAY_HTTP_502",
        "",
    ] {
        assert!(!is_route_failure_reason(reason), "admitted {reason}");
    }

    assert!(is_blind_relay_rejection_reason("backpressure"));
    assert!(is_blind_relay_rejection_reason(
        "onion_terminal_delivery_failed"
    ));
    assert!(is_peer_relay_rejection_reason("duplicate_route"));
    assert!(is_quarantine_reason("failure_threshold"));
    assert!(!is_quarantine_reason("failure_threshold peer=private"));
}

#[test]
fn test_same_sequence_descriptor_conflict_is_rejected() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 7, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();

    let store = PeerStore::new();
    store.upsert_verified(descriptor.clone(), now).unwrap();
    let mut conflicting_body = descriptor.descriptor;
    conflicting_body.public_endpoint = Some("https://route-b.example".to_string());
    let conflicting = SignedNodeDescriptor::sign(conflicting_body, &peer_kp).unwrap();

    assert!(matches!(
        store.upsert_verified(conflicting, now + 1),
        Err(PeerStoreError::VerificationFailed)
    ));
    assert_eq!(store.len(), 1);
}

#[test]
fn test_network_story_reports_onion_ready_without_endpoint_or_full_node_id() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, true, true, 2);
    store.record_gossip_round(now + 20, 2, 2, 1, None);

    let middle_kp = IdentityKeyPair::generate();
    let relay_kp = IdentityKeyPair::generate();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint = Some("https://story-middle.example".to_string());
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle_descriptor.node_id();

    let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
    relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    relay_descriptor.descriptor.public_endpoint = Some("https://story-relay.example".to_string());
    relay_descriptor = SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
    let relay_node_id = relay_descriptor.node_id();

    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
        .unwrap();
    store.record_route_forward_success(&middle_node_id, now + 25);
    store.record_route_forward_success(&relay_node_id, now + 26);

    let story = store.status(now + 30).network_story;

    assert_eq!(story.status, "onion_ready");
    assert!(story.chat_single_hop_ready);
    assert!(story.chat_two_hop_onion_ready);
    assert_eq!(story.valid_nodes, 2);
    assert_eq!(story.routeable_chat_relays, 1);
    assert_eq!(story.routeable_onion_middle_hops, 1);
    assert!(story.relay_foundation_ready);
    assert!(story.restart_recovery_configured);

    let story_json = serde_json::to_string(&story).unwrap();
    assert!(!story_json.contains("story-middle.example"));
    assert!(!story_json.contains("story-relay.example"));
    assert!(!story_json.contains(&hex::encode(middle_node_id)));
    assert!(!story_json.contains(&hex::encode(relay_node_id)));
    assert!(!story_json.contains("encrypted_blob"));
    assert!(!story_json.contains("receiver_pubkey"));
}

#[test]
fn test_network_story_attention_overrides_peer_view_when_recovery_is_missing() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, false, false, 0);

    let first_kp = IdentityKeyPair::generate();
    let second_kp = IdentityKeyPair::generate();

    let mut first_descriptor = signed_descriptor_for(&first_kp, 1, now + 2_000);
    first_descriptor.descriptor.public_endpoint =
        Some("https://recovery-missing-a.example".to_string());
    first_descriptor = SignedNodeDescriptor::sign(first_descriptor.descriptor, &first_kp)
        .expect("descriptor should sign");
    let first_node_id = first_descriptor.node_id();

    let mut second_descriptor = signed_descriptor_for(&second_kp, 1, now + 2_000);
    second_descriptor.descriptor.public_endpoint =
        Some("https://recovery-missing-b.example".to_string());
    second_descriptor = SignedNodeDescriptor::sign(second_descriptor.descriptor, &second_kp)
        .expect("descriptor should sign");
    let second_node_id = second_descriptor.node_id();

    store
        .upsert_verified_from_source(first_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(second_descriptor, now, "gossip_announce")
        .unwrap();

    let story = store.status(now + 30).network_story;

    assert_eq!(story.status, "attention");
    assert_eq!(story.valid_nodes, 2);
    assert!(!story.relay_foundation_ready);
    assert!(!story.restart_recovery_configured);

    let story_json = serde_json::to_string(&story).unwrap();
    assert!(!story_json.contains("recovery-missing-a.example"));
    assert!(!story_json.contains("recovery-missing-b.example"));
    assert!(!story_json.contains(&hex::encode(first_node_id)));
    assert!(!story_json.contains(&hex::encode(second_node_id)));
    assert!(!story_json.contains("encrypted_blob"));
    assert!(!story_json.contains("receiver_pubkey"));
}

#[test]
fn test_cleanup_expired_degrades_and_retains_old_peers() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);

    store.upsert_verified(descriptor, 1_700_000_100).unwrap();
    assert_eq!(store.cleanup_expired(1_700_002_000), 1);
    assert_eq!(store.len(), 1);
    assert_eq!(store.snapshot(1_700_002_001).valid_peers, 0);
    assert_eq!(store.cleanup_expired(1_700_002_100), 0);

    let status = store.status(1_700_002_001);
    assert_eq!(status.runtime.expired_removed, 0);
    assert_eq!(status.runtime.expired_degraded, 1);
    assert_eq!(status.runtime.last_cleanup_at, Some(1_700_002_000));
    assert_eq!(status.peer_summary.expired_peers, 1);
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_expired"
            && event.outcome == "degraded"
            && event.source == "cleanup"
            && event.reason.as_deref() == Some("descriptor_expired_retained")
    }));
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "expired_peer_cleanup"
            && event.outcome == "accepted"
            && event.detail.contains("degraded=1")
            && event.detail.contains("removed=0")
    }));

    let public_snapshot =
        store.export_bootstrap_snapshot(1_700_002_002, 1_700_002_002, false, None);
    assert_eq!(public_snapshot.peers.len(), 0);
    let cache_snapshot = store.export_peer_cache_snapshot(1_700_002_002);
    assert_eq!(cache_snapshot.peers.len(), 1);
}

#[test]
fn untrusted_discovery_candidates_do_not_consume_live_routing_capacity() {
    let now = 1_700_000_100;
    let store = PeerStore::with_max_peers(1);
    store.enable_untrusted_discovery_candidate_mode();
    let candidate = signed_descriptor(1, now + 600);
    let candidate_id = candidate.node_id();

    let admitted = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: candidate,
        },
        now,
    );
    assert_eq!(admitted.candidates, 1);
    assert_eq!(admitted.inserted, 0);
    assert_eq!(store.len(), 0);
    assert!(store.get_valid(&candidate_id, now).is_none());
    assert_eq!(store.status(now).runtime.candidate_admitted, 1);

    assert!(store
        .upsert_verified(signed_descriptor(1, now + 600), now)
        .expect("independently anchored live import"));
    assert_eq!(store.len(), 1);
}

// [REVERSE-ONION-PINNED-IDENTITY-REFRESH 2026-10-06 by Codex] A configured
// route identity may refresh after cache expiry; an unrelated identity remains
// a non-routeable candidate.
#[test]
fn expired_private_route_descriptor_can_be_refreshed_only_for_pinned_identity() {
    let now = 1_700_000_100;
    let store = PeerStore::new();
    store.enable_untrusted_discovery_candidate_mode();
    let relay = IdentityKeyPair::from_bytes(&[0xA1; 32]).unwrap();
    let recipient = IdentityKeyPair::from_bytes(&[0xA2; 32]).unwrap();
    store.pin_private_onion_route_identities(
        relay.public_key_bytes(), recipient.public_key_bytes(),
    ).unwrap();

    let mut expired_body = signed_descriptor_for(&recipient, 1, now + 1).descriptor;
    expired_body.public_endpoint = None;
    expired_body.policy.public_discovery = false;
    let expired = SignedNodeDescriptor::sign(expired_body, &recipient).unwrap();
    let initial = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor: expired }, now,
    );
    assert_eq!(initial.inserted, 1);
    assert_eq!(store.cleanup_expired(now + 2), 1);

    let mut refreshed_body = signed_descriptor_for(&recipient, 2, now + 600).descriptor;
    refreshed_body.public_endpoint = None;
    refreshed_body.policy.public_discovery = false;
    let refreshed = SignedNodeDescriptor::sign(refreshed_body, &recipient).unwrap();
    let report = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor: refreshed.clone() }, now + 2,
    );
    assert_eq!(report.inserted, 1);
    assert_eq!(report.candidates, 0);
    assert_eq!(store.get_valid(&recipient.public_key_bytes(), now + 2), Some(refreshed));

    let unknown = signed_descriptor(1, now + 600);
    let unknown_id = unknown.node_id();
    let report = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor: unknown }, now + 2,
    );
    assert_eq!(report.candidates, 1);
    assert!(store.get_valid(&unknown_id, now + 2).is_none());
}

#[test]
fn untrusted_discovery_candidate_limits_conflicts_and_expiry_release_slots() {
    let now = 1_700_000_100;
    let store = PeerStore::new();
    store.enable_untrusted_discovery_candidate_mode();
    let identity = IdentityKeyPair::generate();
    let descriptor = signed_descriptor_for(&identity, 1, now + 1);

    let first = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: descriptor.clone(),
        },
        now,
    );
    assert_eq!(first.candidates, 1);
    let exact = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: descriptor.clone(),
        },
        now,
    );
    assert_eq!(exact.unchanged, 1);

    let mut conflicting_body = descriptor.descriptor.clone();
    conflicting_body.public_endpoint = Some("https://conflict.example".to_string());
    let conflict = SignedNodeDescriptor::sign(conflicting_body, &identity).unwrap();
    let conflicting = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: conflict,
        },
        now,
    );
    assert_eq!(conflicting.rejected, 1);

    assert_eq!(store.cleanup_expired(now + 2), 1);
    let rolled_back_exact = store.apply_untrusted_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
        now,
    );
    assert_eq!(rolled_back_exact.unchanged, 1);

    let overlong = signed_descriptor(1, now + UNTRUSTED_DISCOVERY_MAX_LIFETIME_SECS + 2);
    let rejected = store.apply_untrusted_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: overlong,
        },
        now,
    );
    assert_eq!(rejected.rejected, 1);
}

#[test]
fn untrusted_candidate_exhaustion_cannot_starve_verified_live_capacity() {
    let now = 1_700_000_100;
    let store = PeerStore::with_max_peers(1);
    store.enable_untrusted_discovery_candidate_mode();
    for _ in 0..UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY {
        let report = store.apply_untrusted_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: signed_descriptor(1, now + 600),
            },
            now,
        );
        assert_eq!(report.candidates, 1);
    }
    let saturated = store.apply_untrusted_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed_descriptor(1, now + 600),
        },
        now,
    );
    assert_eq!(saturated.rejected, 1);
    assert_eq!(store.len(), 0);
    assert!(store
        .upsert_verified(signed_descriptor(1, now + 600), now)
        .expect("candidate exhaustion cannot consume live capacity"));
    assert_eq!(store.len(), 1);
}

#[test]
fn test_valid_public_descriptors_is_bounded_and_filters_private_peers() {
    let store = PeerStore::new();
    let public_a = signed_descriptor(1, 1_700_001_000);
    let public_b = signed_descriptor(2, 1_700_001_000);
    let private_key = IdentityKeyPair::generate();
    let mut private = signed_descriptor_for(&private_key, 1, 1_700_001_000);
    private.descriptor.policy.public_discovery = false;
    private = SignedNodeDescriptor::sign(private.descriptor, &private_key).unwrap();
    for descriptor in [public_a, public_b, private] {
        store.upsert_verified(descriptor, 1_700_000_100).unwrap();
    }

    assert!(store.valid_public_descriptors(1_700_000_100, 0).is_empty());
    let selected = store.valid_public_descriptors(1_700_000_100, 1);
    assert_eq!(selected.len(), 1);
    assert!(selected[0].descriptor.policy.public_discovery);
}

// [REVERSE-ONION-AUTHORITY-FENCE 2026-10-05 by Codex] Source authored only;
// execution is intentionally deferred to the authorized verification phase.
#[test]
fn private_onion_authority_read_epoch_excludes_authority_replacement() {
    let store = PeerStore::new();
    let epoch = store.private_onion_authority_read_guard();
    assert!(store.private_onion_authority_gate.try_write().is_none());
    drop(epoch);
    assert!(store.private_onion_authority_gate.try_write().is_some());
}

#[test]
fn test_valid_public_endpoint_identities_is_complete_and_side_effect_free(
) -> Result<(), Box<dyn std::error::Error>> {
    let store = PeerStore::new();
    let now = 1_700_000_100;

    let public_key = IdentityKeyPair::generate();
    let mut public = signed_descriptor_for(&public_key, 1, 1_700_001_000);
    public.descriptor.public_endpoint = Some("https://public.example".to_string());
    public = SignedNodeDescriptor::sign(public.descriptor, &public_key)?;
    let public_node_id = public.node_id();

    let private_key = IdentityKeyPair::generate();
    let mut private = signed_descriptor_for(&private_key, 1, 1_700_001_000);
    private.descriptor.public_endpoint = Some("https://private.example".to_string());
    private.descriptor.policy.public_discovery = false;
    private = SignedNodeDescriptor::sign(private.descriptor, &private_key)?;

    let no_endpoint = signed_descriptor(1, 1_700_001_000);
    for descriptor in [public, private, no_endpoint] {
        store.upsert_verified(descriptor, now)?;
    }

    let identities = store.valid_public_endpoint_identities(now);

    assert_eq!(
        identities,
        vec![(public_node_id, "https://public.example".to_string())]
    );
    Ok(())
}

#[test]
fn test_heartbeat_signed_peer_records_export_only_verifiable_live_records() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let valid = signed_descriptor(1, 1_700_001_000);
    let valid_node_id = valid.node_id();
    let expired = signed_descriptor(1, 1_699_999_999);
    let mut tampered = signed_descriptor(1, 1_700_001_000);
    tampered.signature[0] ^= 0x01;

    store.upsert_verified(valid, now).unwrap();
    store.peers.write().insert(expired.node_id(), expired);
    store.peers.write().insert(tampered.node_id(), tampered);

    let signed_records = store.export_signed_peer_records_for_heartbeat(now, Some(8));

    assert_eq!(signed_records.total_retained_records, 3);
    assert_eq!(signed_records.valid_signed_records, 1);
    assert_eq!(signed_records.exported_signed_records, 1);
    assert_eq!(signed_records.records.generated_at, now);
    assert_eq!(signed_records.records.peers.len(), 1);
    assert_eq!(signed_records.records.peers[0].node_id(), valid_node_id);
    assert!(signed_records.records.peers[0].verify_at(now).is_ok());
    assert!(signed_records
        .verification_rule
        .contains("SignedNodeDescriptor::verify_at"));
    assert!(signed_records
        .privacy_boundary
        .contains("signed node discovery descriptors only"));
    assert!(store.recent_audit_events().iter().any(|event| {
        event.action == "heartbeat_signed_peer_records_export"
            && event.outcome == "accepted"
            && event.detail.contains("retained=3")
            && event.detail.contains("valid=1")
            && event.detail.contains("exported=1")
    }));
}

#[test]
fn test_apply_descriptor_announce_imports_peer() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);
    let message = NodeDiscoveryMessage::DescriptorAnnounce { descriptor };

    let report = store.apply_discovery_message(&message, 1_700_000_100);

    assert_eq!(report.inserted, 1);
    assert_eq!(store.len(), 1);
    let status = store.status(1_700_000_100);
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_inserted"
            && event.outcome == "accepted"
            && event.source == "gossip_announce"
            && event.sequence == Some(1)
            && event.reason.is_none()
    }));
}

#[test]
fn test_runtime_rejection_counters_are_recorded() {
    let store = PeerStore::new();

    store.record_policy_rejected(1_700_000_300, "allow_list_enabled=true");
    store.record_rate_limited(1_700_000_301, "global_limit_per_minute=1");
    store.mark_gossip_at(1_700_000_333);

    let status = store.status(1_700_000_400);
    assert_eq!(status.runtime.policy_rejected, 1);
    assert_eq!(status.runtime.rate_limited, 1);
    assert_eq!(status.runtime.last_gossip_at, Some(1_700_000_333));
    assert_eq!(status.recent_audit_events.len(), 2);
    assert_eq!(
        status.recent_audit_events[0].action,
        "gossip_policy_rejected"
    );
    assert_eq!(status.recent_audit_events[1].action, "gossip_rate_limited");
}

#[test]
fn startup_self_check_status_is_recorded_without_config_values() {
    let store = PeerStore::new();

    store.record_startup_self_check(
        1_700_000_350,
        "warning",
        "missing=peer_cache_path,seed_endpoints,public_endpoint",
    );

    let status = store.status(1_700_000_400);
    assert_eq!(
        status.bootstrap.startup_self_check_status.as_deref(),
        Some("warning")
    );
    assert_eq!(
        status.bootstrap.startup_self_check_detail.as_deref(),
        Some("missing=peer_cache_path,seed_endpoints,public_endpoint")
    );
    assert_eq!(status.bootstrap.startup_self_check_at, Some(1_700_000_350));
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "startup_self_check"
            && event.outcome == "warning"
            && !event.detail.contains("https://")
            && !event.detail.contains("/root/")
    }));
}

#[test]
fn test_audit_log_is_bounded() {
    let store = PeerStore::new();

    for i in 0..70 {
        store.record_audit_event(1_700_000_000 + i, "snapshot_export", "accepted", "test");
    }

    let events = store.recent_audit_events();
    assert_eq!(events.len(), MAX_AUDIT_EVENTS);
    assert_eq!(events[0].at, 1_700_000_006);
    assert_eq!(events[MAX_AUDIT_EVENTS - 1].at, 1_700_000_069);
}

#[test]
fn test_delivery_receipt_capability_is_verified_peer_and_freshness_bounded() {
    let now = 1_700_000_000;
    let identity = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&identity, 1, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://relay.example".to_string());
    descriptor.descriptor.capabilities =
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
    let node_id = descriptor.node_id();
    let store = PeerStore::new();
    store.upsert_verified(descriptor, now).unwrap();
    store.record_route_forward_success(&node_id, now + 1);
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 2);
    assert!(store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 3));

    let candidates = store.delivery_receipt_route_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        now + 3,
        4,
        &[],
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        store
            .status(now + 3)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        1
    );

    store.record_route_forward_failure(&node_id, now + 4, "request_failed");
    assert!(store
        .delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now + 5,
            4,
            &[],
        )
        .is_empty());

    let stale_at = now + 2 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1;
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, stale_at));
    assert!(store
        .delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            stale_at,
            4,
            &[],
        )
        .is_empty());
    assert_eq!(
        store
            .status(stale_at)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        0
    );
}

#[test]
fn test_delivery_receipt_capability_status_excludes_expired_peer() {
    let now = 1_700_000_000;
    let identity = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&identity, 1, now + 10);
    descriptor.descriptor.public_endpoint = Some("https://relay.example".to_string());
    descriptor.descriptor.capabilities =
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
    let node_id = descriptor.node_id();
    let store = PeerStore::new();
    store.upsert_verified(descriptor, now).unwrap();
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 1);

    assert_eq!(
        store
            .status(now + 2)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        1
    );
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 11));
    assert_eq!(
        store
            .status(now + 11)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        0
    );
}

#[test]
fn test_delivery_witness_status_counts_only_accepted_signed_outcomes() {
    let store = PeerStore::new();
    let now = 1_700_000_000;
    let status = store.record_client_delivery_witness_round(
        now,
        7,
        true,
        2,
        PeerStoreVerifiedDeliveryWitnessRound {
            configured: 3,
            attempted: 3,
            verified: 3,
            advanced: 1,
            idempotent: 1,
            stale: 1,
            ..PeerStoreVerifiedDeliveryWitnessRound::default()
        },
    );
    assert_eq!(status, "rollback_detected");
    let bootstrap = store.status(now).bootstrap;
    assert_eq!(
        bootstrap.last_client_delivery_witness_status.as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(bootstrap.last_client_delivery_witness_generation, 7);
    assert!(bootstrap.last_client_delivery_witness_required);
    assert_eq!(bootstrap.last_client_delivery_witness_minimum_verified, 2);
    assert_eq!(bootstrap.last_client_delivery_witness_configured, 3);
    assert_eq!(bootstrap.last_client_delivery_witness_verified, 3);
    assert_eq!(bootstrap.last_client_delivery_witness_stale, 1);
}

#[test]
fn test_external_witness_gate_clears_only_client_delivery_evidence() {
    let store = PeerStore::new();
    let now = 1_700_000_000;
    store.record_verified_client_onion_delivery(now);
    store.record_blind_relay_terminal(now + 1, 0, 32);
    assert_eq!(
        store
            .status(now + 2)
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        1
    );

    store.clear_restored_verified_client_delivery_evidence(now + 3, "external_witness_rollback");
    let status = store.status(now + 4);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(status.runtime.blind_relay.terminal, 1);
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert!(store.take_client_delivery_cache_dirty());
}

#[test]
fn test_verified_client_delivery_receipt_drives_real_relay_readiness_only_while_fresh() {
    let store = PeerStore::new();
    let now = 1_700_000_000;
    store.record_verified_client_onion_delivery(now);

    let unproven_mesh = store.status(now + 1).blind_relay_quality;
    assert!(!unproven_mesh.real_relay_ready);
    assert_eq!(unproven_mesh.delivery_receipt_capable_peers, 0);
    assert_eq!(unproven_mesh.status, "observing");
    assert_eq!(
        unproven_mesh.readiness_reason,
        "verified_client_onion_delivery_peer_revalidation_required"
    );

    for (offset, endpoint, capability) in [
        (
            1,
            "https://middle-receipt.example",
            NodeCapability::OnionMiddle,
        ),
        (
            2,
            "https://terminal-receipt.example",
            NodeCapability::ChatRelay,
        ),
    ] {
        let identity = IdentityKeyPair::generate();
        let mut descriptor = signed_descriptor_for(&identity, offset, now + 4_000);
        descriptor.descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.descriptor.capabilities = vec![capability];
        descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&node_id, now);
    }

    let fresh = store.status(now + 10).blind_relay_quality;
    assert!(fresh.real_relay_ready);
    assert_eq!(fresh.verified_client_onion_deliveries, 1);
    assert_eq!(
        fresh.last_verified_client_onion_delivery_age_seconds,
        Some(10)
    );
    assert_eq!(
        fresh.evidence_mode,
        "verified_client_onion_delivery_receipt"
    );
    assert_eq!(fresh.proof_scope, "client_message_delivery");
    assert_eq!(
        fresh.readiness_reason,
        "verified_client_onion_delivery_receipt_ready"
    );

    let stale = store
        .status(now + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        .blind_relay_quality;
    assert!(!stale.real_relay_ready);
    assert_eq!(stale.status, "stale");
    assert_eq!(
        stale.readiness_reason,
        "verified_client_onion_delivery_receipt_stale"
    );
}

#[test]
fn test_peer_summary_tracks_source_ttl_health_and_capabilities() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let healthy = signed_descriptor(1, now + 1_000);
    let stale = signed_descriptor(1, now + 120);

    store
        .upsert_verified_from_source(healthy.clone(), now, "cache")
        .unwrap();
    store
        .upsert_verified_from_source(stale.clone(), now, "gossip_snapshot")
        .unwrap();
    store
        .upsert_verified_from_source(healthy, now + 20, "gossip_announce")
        .unwrap();

    let status = store.status(now + 30);

    assert_eq!(status.peer_summary.total_peers, 2);
    assert_eq!(status.peer_summary.valid_peers, 2);
    assert_eq!(status.peer_summary.healthy_peers, 1);
    assert_eq!(status.peer_summary.stale_peers, 1);
    assert_eq!(status.peer_summary.expired_peers, 0);
    assert_eq!(status.peer_summary.chat_relay_peers, 2);
    assert_eq!(status.peer_summary.privacy_relay_peers, 2);
    assert_eq!(
        status.peer_summary.source_counts.get("gossip_announce"),
        Some(&1)
    );
    assert_eq!(
        status.peer_summary.source_counts.get("gossip_snapshot"),
        Some(&1)
    );
    assert!(status
        .peer_summary
        .peers
        .iter()
        .any(|peer| peer.health == "stale" && peer.ttl_remaining_seconds == Some(90)));
    assert!(status
        .peer_summary
        .peers
        .iter()
        .all(|peer| peer.capabilities.contains(&"chat_relay".to_string())));
}
