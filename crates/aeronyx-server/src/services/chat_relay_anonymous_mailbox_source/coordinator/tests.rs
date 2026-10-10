// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/coordinator/tests.rs
// ============================================
//! # Tests: exact-target source coordinator
//!
//! Unit tests for exact pinned-target resolution, ticket-target guards, tamper
//! rejection and descriptor drift, moved from the former inline `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};

use aeronyx_core::protocol::discovery::SignedNodeDescriptor;
use parking_lot::Mutex;
use rusqlite::params;

use super::super::storage::load_source_meta;
use super::super::test_support::{
    journal, journal_record, mailbox_descriptor, target, ticket_request, ExactOnlyResolver,
    MutableExactResolver, NOW,
};

#[test]
fn coordinator_resolves_one_pinned_target_without_fallback() {
    let target = target();
    let descriptor = mailbox_descriptor(&target);
    let commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("commitment");
    let resolver = Arc::new(ExactOnlyResolver {
        descriptor,
        calls: AtomicUsize::new(0),
    });
    let coordinator = AnonymousMailboxSourceCoordinator::new(
        Arc::new(IdentityKeyPair::from_bytes(&[0x72; 32]).expect("source")),
        resolver.clone(),
        Arc::new(journal()),
    );
    let prepared = coordinator
        .prepare(
            ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
            [0x51; 16],
            ticket_request(&target),
            NOW,
        )
        .expect("one target request");
    let replay = coordinator
        .prepare(
            ExactAnonymousMailboxTargetPin::new(
                target.public_key_bytes(),
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&resolver.descriptor)
                    .expect("same commitment"),
            ),
            [0x51; 16],
            ticket_request(&target),
            NOW + 1,
        )
        .expect("exact retry");
    assert!(!prepared.body().is_empty());
    assert_eq!(prepared.body(), replay.body(), "retry retains exact bytes");
    assert_eq!(
        resolver.calls.load(Ordering::Relaxed),
        1,
        "no candidate fallback"
    );
}

#[test]
fn wrong_target_ticket_never_enters_journal_or_dispatch_and_valid_retry_remains_exact() {
    let target = target();
    let other = IdentityKeyPair::from_bytes(&[0x75; 32]).expect("other target");
    let descriptor = mailbox_descriptor(&target);
    let commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("commitment");
    let resolver = Arc::new(ExactOnlyResolver {
        descriptor,
        calls: AtomicUsize::new(0),
    });
    let journal = Arc::new(journal());
    let coordinator = AnonymousMailboxSourceCoordinator::new(
        Arc::new(IdentityKeyPair::from_bytes(&[0x76; 32]).expect("source")),
        resolver.clone(),
        journal.clone(),
    );
    let route_id = [0x77; 16];
    let pin = ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment);
    assert!(matches!(
        coordinator.prepare(pin.clone(), route_id, ticket_request(&other), NOW),
        Err(AnonymousMailboxSourceError::Rejected)
    ));
    assert!(journal.load(&route_id).expect("journal read").is_none());
    assert!(matches!(
        coordinator.begin_dispatch(route_id, NOW),
        Err(AnonymousMailboxSourceError::Rejected)
    ));
    assert_eq!(resolver.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        load_source_meta(&*journal.connection.lock())
            .expect("journal totals")
            .entries,
        0,
        "wrong-target request cannot reserve durable quota"
    );

    let terminal = ticket_request(&target);
    let prepared = coordinator
        .prepare(pin.clone(), route_id, terminal.clone(), NOW)
        .expect("valid target prepare");
    let retry = coordinator
        .prepare(pin, route_id, terminal, NOW + 1)
        .expect("valid exact retry");
    assert_eq!(prepared.body(), retry.body());
    assert_eq!(resolver.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn previously_journaled_wrong_target_ticket_cannot_dispatch() {
    let journal = Arc::new(journal());
    let target = target();
    let other = IdentityKeyPair::from_bytes(&[0x78; 32]).expect("other target");
    let mut record = journal_record(0x79, 0x7a, AnonymousMailboxSourcePhase::Prepared);
    let terminal = ticket_request(&other);
    record.request_commitment = source_request_commitment(
        &record.route_id,
        &record.target_node_id,
        &record.descriptor_commitment,
        &terminal,
    );
    record.state = encode_state(&record.body, &terminal, None, None).expect("legacy state");
    journal
        .insert_or_exact(&record)
        .expect("simulate previously accepted row");
    journal
        .audit_startup()
        .expect("legacy journal stays readable");
    let resolver = Arc::new(ExactOnlyResolver {
        descriptor: mailbox_descriptor(&target),
        calls: AtomicUsize::new(0),
    });
    let coordinator = AnonymousMailboxSourceCoordinator::new(
        Arc::new(IdentityKeyPair::from_bytes(&[0x7b; 32]).expect("source")),
        resolver.clone(),
        journal.clone(),
    );
    assert!(matches!(
        coordinator.begin_dispatch(record.route_id, NOW),
        Err(AnonymousMailboxSourceError::Corrupt)
    ));
    assert_eq!(resolver.calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        coordinator.result(record.route_id),
        Ok(AnonymousMailboxSourceResult::Prepared)
    ));
}

#[test]
fn journal_body_tamper_fails_before_restart_dispatch() {
    let target = target();
    let terminal = ticket_request(&target);
    let body = vec![0x63; 96];
    let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
        node_id: target.public_key_bytes(),
        sequence: 11,
        descriptor_hash: [0x64; 32],
    };
    let record = SourceJournalRecord {
        route_id: [0x65; 16],
        request_commitment: source_request_commitment(
            &[0x65; 16],
            &target.public_key_bytes(),
            &descriptor_commitment,
            &terminal,
        ),
        target_node_id: target.public_key_bytes(),
        descriptor_commitment,
        body: body.clone(),
        phase: AnonymousMailboxSourcePhase::Prepared,
        retain_until: None,
        state: encode_state(&body, &terminal, None, None).expect("state"),
    };
    let journal = journal();
    journal.insert_or_exact(&record).expect("insert");
    journal
        .connection
        .lock()
        .execute(
            "UPDATE anonymous_mailbox_source_journal SET body = ?1 WHERE route_id = ?2",
            params![vec![0x66u8; 96], record.route_id.as_slice()],
        )
        .expect("tamper row");
    assert!(matches!(
        journal.load(&record.route_id),
        Err(AnonymousMailboxSourceError::Corrupt)
    ));
}

#[test]
fn descriptor_drift_keeps_prepared_record_off_network() {
    let target = target();
    let descriptor = mailbox_descriptor(&target);
    let commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("commitment");
    let resolver = Arc::new(MutableExactResolver {
        descriptor: Mutex::new(descriptor),
    });
    let coordinator = AnonymousMailboxSourceCoordinator::new(
        Arc::new(IdentityKeyPair::from_bytes(&[0x73; 32]).expect("source")),
        resolver.clone(),
        Arc::new(journal()),
    );
    let route_id = [0x74; 16];
    coordinator
        .prepare(
            ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
            route_id,
            ticket_request(&target),
            NOW,
        )
        .expect("prepare");
    let mut drifted = mailbox_descriptor(&target).descriptor;
    drifted.sequence = drifted.sequence.saturating_add(1);
    *resolver.descriptor.lock() = SignedNodeDescriptor::sign(drifted, &target).expect("drift");
    assert!(matches!(
        coordinator.begin_dispatch(route_id, NOW),
        Err(AnonymousMailboxSourceError::Unavailable)
    ));
    assert!(matches!(
        coordinator.result(route_id).expect("result"),
        AnonymousMailboxSourceResult::Prepared
    ));
}
