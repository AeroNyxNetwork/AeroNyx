// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/state_codec.rs
// ============================================
//! # Source journal state codec
//!
//! Owns the fixed persisted-state envelope (encode/decode), the AEAD associated
//! data, the request and body commitments, and the cross-check between a decoded
//! record and its projected columns.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use aeronyx_core::protocol::anonymous_mailbox::{
    AnonymousMailboxSourceSealSessionV1, MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES,
};
use aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1;
use sha2::{Digest, Sha256};

use super::frame_validation::ensure_request_frame;
use super::journal::SourceJournalRecord;
use super::storage::validate_phase_retention;
use super::{
    AnonymousMailboxSourceError, JOURNAL_STATE_BODY_COMMITMENT_BYTES, JOURNAL_STATE_ENVELOPE_BYTES,
    MAX_JOURNAL_BODY_BYTES, MAX_JOURNAL_CLEAR_STATE_BYTES, MAX_JOURNAL_RESTART_STATE_BYTES,
};

const JOURNAL_STATE_VERSION: u8 = 2;
const JOURNAL_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-journal/v1\0";
const REQUEST_COMMITMENT_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-request/v1\0";
const BODY_COMMITMENT_DOMAIN: &[u8] = b"aeronyx/anonymous-mailbox/source-body/v1\0";

pub(super) struct DecodedSourceState {
    pub(super) body_commitment: [u8; 32],
    pub(super) terminal_frame: Vec<u8>,
    pub(super) restart: Option<AnonymousMailboxSourceSealSessionV1>,
    pub(super) completed: Option<Vec<u8>>,
}

pub(super) fn source_request_commitment(
    route_id: &[u8; 16],
    target: &[u8; 32],
    descriptor: &DirectoryDescriptorCommitmentV1,
    terminal_frame: &[u8],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(REQUEST_COMMITMENT_DOMAIN);
    hash.update(route_id);
    hash.update(target);
    hash.update(descriptor.hash());
    hash.update((terminal_frame.len() as u64).to_be_bytes());
    hash.update(terminal_frame);
    hash.finalize().into()
}

pub(super) fn source_body_commitment(body: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(BODY_COMMITMENT_DOMAIN);
    hash.update((body.len() as u64).to_be_bytes());
    hash.update(body);
    hash.finalize().into()
}

pub(super) fn validate_source_record_projection(
    record: &SourceJournalRecord,
) -> Result<(), AnonymousMailboxSourceError> {
    validate_phase_retention(record.phase, record.retain_until)?;
    let decoded = decode_state(&record.state)?;
    if decoded.body_commitment != source_body_commitment(&record.body)
        || source_request_commitment(
            &record.route_id,
            &record.target_node_id,
            &record.descriptor_commitment,
            &decoded.terminal_frame,
        ) != record.request_commitment
    {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    Ok(())
}

pub(super) fn journal_aad(
    route_id: &[u8; 16],
    commitment: &[u8; 32],
    target: &[u8; 32],
    body_commitment: &[u8; 32],
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(JOURNAL_DOMAIN.len() + 112);
    aad.extend_from_slice(JOURNAL_DOMAIN);
    aad.extend_from_slice(route_id);
    aad.extend_from_slice(commitment);
    aad.extend_from_slice(target);
    aad.extend_from_slice(body_commitment);
    aad
}

pub(super) fn encode_state(
    body: &[u8],
    terminal_frame: &[u8],
    session: Option<&AnonymousMailboxSourceSealSessionV1>,
    completed: Option<&[u8]>,
) -> Result<Vec<u8>, AnonymousMailboxSourceError> {
    if terminal_frame.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    let restart = session
        .map(|value| {
            value
                .encode_restart_state()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)
        })
        .transpose()?
        .map(|value| value.as_bytes().to_vec())
        .unwrap_or_default();
    if restart.len() > MAX_JOURNAL_RESTART_STATE_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let terminal_len =
        u32::try_from(terminal_frame.len()).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    let restart_len =
        u16::try_from(restart.len()).map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let completed = completed.unwrap_or_default();
    if completed.len() > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    let completed_len =
        u32::try_from(completed.len()).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    if body.is_empty() || body.len() > MAX_JOURNAL_BODY_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    let mut bytes = Vec::with_capacity(
        JOURNAL_STATE_ENVELOPE_BYTES + terminal_frame.len() + restart.len() + completed.len(),
    );
    bytes.push(JOURNAL_STATE_VERSION);
    bytes.extend_from_slice(&source_body_commitment(body));
    bytes.extend_from_slice(&terminal_len.to_le_bytes());
    bytes.extend_from_slice(terminal_frame);
    bytes.extend_from_slice(&restart_len.to_le_bytes());
    bytes.extend_from_slice(&restart);
    bytes.extend_from_slice(&completed_len.to_le_bytes());
    bytes.extend_from_slice(completed);
    if bytes.len() > MAX_JOURNAL_CLEAR_STATE_BYTES {
        return Err(AnonymousMailboxSourceError::Rejected);
    }
    Ok(bytes)
}

pub(super) fn decode_state(
    bytes: &[u8],
) -> Result<DecodedSourceState, AnonymousMailboxSourceError> {
    if bytes.len() > MAX_JOURNAL_CLEAR_STATE_BYTES
        || bytes.first().copied() != Some(JOURNAL_STATE_VERSION)
        || bytes.len() < JOURNAL_STATE_ENVELOPE_BYTES
    {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let body_commitment = fixed::<JOURNAL_STATE_BODY_COMMITMENT_BYTES>(
        &bytes[1..1 + JOURNAL_STATE_BODY_COMMITMENT_BYTES],
    )?;
    let mut offset = 1 + JOURNAL_STATE_BODY_COMMITMENT_BYTES;
    let terminal_len = u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    if terminal_len > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    offset += 4;
    let terminal_end = offset
        .checked_add(terminal_len)
        .ok_or(AnonymousMailboxSourceError::Corrupt)?;
    if terminal_end > bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let terminal = bytes[offset..terminal_end].to_vec();
    ensure_request_frame(&terminal)?;
    offset = terminal_end;
    if offset + 2 > bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let restart_len = u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    if restart_len > MAX_JOURNAL_RESTART_STATE_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    offset += 2;
    let restart_end = offset
        .checked_add(restart_len)
        .ok_or(AnonymousMailboxSourceError::Corrupt)?;
    if restart_end + 4 > bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let session = if restart_len == 0 {
        None
    } else {
        Some(
            AnonymousMailboxSourceSealSessionV1::decode_restart_state(&bytes[offset..restart_end])
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
        )
    };
    offset = restart_end;
    let completed_len = u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
    ) as usize;
    if completed_len > MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    offset += 4;
    let end = offset
        .checked_add(completed_len)
        .ok_or(AnonymousMailboxSourceError::Corrupt)?;
    if end != bytes.len() {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    let completed = if completed_len == 0 {
        None
    } else {
        Some(bytes[offset..end].to_vec())
    };
    Ok(DecodedSourceState {
        body_commitment,
        terminal_frame: terminal,
        restart: session,
        completed,
    })
}

pub(super) fn fixed<const N: usize>(bytes: &[u8]) -> Result<[u8; N], AnonymousMailboxSourceError> {
    bytes
        .try_into()
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)
}
