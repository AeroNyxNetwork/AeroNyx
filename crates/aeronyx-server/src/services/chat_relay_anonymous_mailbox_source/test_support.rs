// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/test_support.rs
// ============================================
//! # Source mailbox test fixtures
//!
//! Shared fixtures for the per-module source-journal and coordinator test
//! suites, moved from the former inline `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use std::sync::atomic::{AtomicUsize, Ordering};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    encode_anonymous_mailbox_terminal_frame, AnonymousMailboxTerminalFrameV1,
    AnonymousMailboxTicketIssueV1,
};
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, NodeDescriptor, NodeProtocolFeature,
    SignedNodeDescriptor,
};
use parking_lot::Mutex;
use rusqlite::Connection;

use crate::config_chat_relay::AnonymousMailboxSourceConfig;

use super::journal::SourceJournalRecord;
use super::state_codec::{decode_state, encode_state, source_request_commitment};
use super::{
    AnonymousMailboxSourcePhase, ExactAnonymousMailboxTargetResolver,
    SqliteAnonymousMailboxSourceJournal,
};

pub(super) const NOW: u64 = 1_800_000_000;

pub(super) fn target() -> IdentityKeyPair {
    IdentityKeyPair::from_bytes(&[0x71; 32]).expect("valid identity")
}

pub(super) fn ticket_request(target: &IdentityKeyPair) -> Vec<u8> {
    let request = AnonymousMailboxTicketIssueV1::new(
        [0x11; 16],
        [0x12; 16],
        target.public_key_bytes(),
        [0x13; 32],
        NOW,
        NOW + 30,
        0,
    )
    .expect("ticket request");
    encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::TicketIssue(request))
        .expect("terminal frame")
}

pub(super) fn journal_with_max_bytes(
    max_journal_bytes: u64,
) -> SqliteAnonymousMailboxSourceJournal {
    journal_with_limits(4, max_journal_bytes, 7 * 24 * 60 * 60, 256)
}

pub(super) fn journal_with_limits(
    max_journal_entries: usize,
    max_journal_bytes: u64,
    terminal_retention_secs: u64,
    cleanup_batch_size: usize,
) -> SqliteAnonymousMailboxSourceJournal {
    let config = AnonymousMailboxSourceConfig {
        enabled: true,
        max_journal_entries,
        max_journal_bytes,
        terminal_retention_secs,
        cleanup_batch_size,
        ..AnonymousMailboxSourceConfig::default()
    };
    SqliteAnonymousMailboxSourceJournal::new(
        Connection::open_in_memory().expect("memory sqlite"),
        [0x22; 32],
        &config,
    )
    .expect("journal")
}

pub(super) fn journal() -> SqliteAnonymousMailboxSourceJournal {
    journal_with_max_bytes(4096)
}

pub(super) fn journal_record(
    route_byte: u8,
    body_byte: u8,
    phase: AnonymousMailboxSourcePhase,
) -> SourceJournalRecord {
    let target = target();
    let terminal = ticket_request(&target);
    let body = vec![body_byte; 64];
    let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
        node_id: target.public_key_bytes(),
        sequence: u64::from(route_byte),
        descriptor_hash: [route_byte.wrapping_add(1); 32],
    };
    SourceJournalRecord {
        route_id: [route_byte; 16],
        request_commitment: source_request_commitment(
            &[route_byte; 16],
            &target.public_key_bytes(),
            &descriptor_commitment,
            &terminal,
        ),
        target_node_id: target.public_key_bytes(),
        descriptor_commitment,
        body: body.clone(),
        phase,
        retain_until: None,
        state: encode_state(&body, &terminal, None, None).expect("state"),
    }
}

pub(super) fn retain_terminal_record(
    journal: &SqliteAnonymousMailboxSourceJournal,
    record: &SourceJournalRecord,
    phase: AnonymousMailboxSourcePhase,
    transitioned_at: u64,
    completed: Option<&[u8]>,
) -> SourceJournalRecord {
    journal.insert_or_exact(record).expect("insert source row");
    let terminal = decode_state(&record.state)
        .expect("decode prepared state")
        .terminal_frame;
    journal
        .transition_at(
            record,
            AnonymousMailboxSourcePhase::Prepared,
            phase,
            encode_state(&record.body, &terminal, None, completed).expect("encode terminal state"),
            transitioned_at,
        )
        .expect("terminal transition");
    journal
        .load(&record.route_id)
        .expect("load terminal row")
        .expect("terminal row")
}

pub(super) struct ExactOnlyResolver {
    pub(super) descriptor: SignedNodeDescriptor,
    pub(super) calls: AtomicUsize,
}

pub(super) struct MutableExactResolver {
    pub(super) descriptor: Mutex<SignedNodeDescriptor>,
}

impl ExactAnonymousMailboxTargetResolver for MutableExactResolver {
    fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
        let descriptor = self.descriptor.lock().clone();
        (node_id == &descriptor.descriptor.node_id).then_some(descriptor)
    }
}

impl ExactAnonymousMailboxTargetResolver for ExactOnlyResolver {
    fn get_valid_exact(&self, node_id: &[u8; 32], _: u64) -> Option<SignedNodeDescriptor> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (node_id == &self.descriptor.descriptor.node_id).then(|| self.descriptor.clone())
    }
}

pub(super) fn mailbox_descriptor(target: &IdentityKeyPair) -> SignedNodeDescriptor {
    let mut descriptor =
        NodeDescriptor::new(target.public_key_bytes(), 9, NOW - 1, NOW + 60, "test")
            .with_x25519_kem(target.x25519_public_key_bytes())
            .with_protocol_features([
                NodeProtocolFeature::AnonymousMailboxV1,
                NodeProtocolFeature::OnionReplyV1,
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ]);
    descriptor.public_endpoint = Some("https://node.invalid".into());
    descriptor.capabilities = vec![NodeCapability::ChatRelay];
    SignedNodeDescriptor::sign(descriptor, target).expect("signed descriptor")
}
