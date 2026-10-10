// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/frame.rs
// ============================================
//! # Blind Vault binary frames
//!
//! Owns the frame magic/header/kind constants, frame size ceilings, the
//! `BlindVaultFrame` enum and its redacted `Debug`, magic detection, and
//! the bounded `encode_blind_vault_frame` / `decode_blind_vault_frame`
//! codec with its per-kind limit and bincode body helpers.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use bincode::Options;
use serde::{Deserialize, Serialize};

use super::delete::{BlindVaultDeleteRequest, BlindVaultDeletedReceipt};
use super::error::BlindVaultError;
use super::issuer_directory::BlindVaultBlindIssuerDirectory;
use super::lease_admission::{
    BlindVaultBlindLeaseAcceptedReceipt, BlindVaultBlindLeaseAdmissionRequest,
    BlindVaultLeaseAdmissionRequest, BlindVaultLeaseCreateRequest,
};
use super::lease_inventory::{BlindVaultLeaseInventoryReceipt, BlindVaultLeaseInventoryRequest};
use super::lease_renewal::{
    BlindVaultBlindLeaseRenewalRequest, BlindVaultBlindLeaseRenewedReceipt,
};
use super::lease_retire::{BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt};
use super::lease_status::{BlindVaultLeaseStatusReceipt, BlindVaultLeaseStatusRequest};
use super::pull::{BlindVaultPullRequest, BlindVaultPullResponse};
use super::put::{BlindVaultPutRequest, BlindVaultStoredReceipt};
use super::terminal_failure::BlindVaultTerminalFailure;
use super::{require_version, BLIND_VAULT_PROTOCOL_VERSION};

const FRAME_MAGIC: [u8; 4] = *b"ANBV";
const FRAME_HEADER_BYTES: usize = 7;
pub(super) const FRAME_KIND_PUT: u8 = 1;
const FRAME_KIND_STORED_RECEIPT: u8 = 2;
const FRAME_KIND_LEASE_CREATE: u8 = 3;
const FRAME_KIND_DELETE: u8 = 4;
const FRAME_KIND_DELETED_RECEIPT: u8 = 5;
const FRAME_KIND_LEASE_ADMISSION: u8 = 6;
const FRAME_KIND_PULL_REQUEST: u8 = 7;
const FRAME_KIND_PULL_RESPONSE: u8 = 8;
pub(super) const FRAME_KIND_BLIND_LEASE_ADMISSION: u8 = 9;
pub(super) const FRAME_KIND_BLIND_ISSUER_DIRECTORY: u8 = 10;
const FRAME_KIND_BLIND_LEASE_ACCEPTED: u8 = 11;
const FRAME_KIND_LEASE_RETIRE: u8 = 12;
const FRAME_KIND_LEASE_RETIRED_RECEIPT: u8 = 13;
const FRAME_KIND_BLIND_LEASE_RENEWAL: u8 = 14;
const FRAME_KIND_BLIND_LEASE_RENEWED: u8 = 15;
const FRAME_KIND_LEASE_STATUS: u8 = 16;
const FRAME_KIND_LEASE_STATUS_RECEIPT: u8 = 17;
const FRAME_KIND_LEASE_INVENTORY: u8 = 18;
const FRAME_KIND_LEASE_INVENTORY_RECEIPT: u8 = 19;
const FRAME_KIND_TERMINAL_FAILURE: u8 = 20;

/// Returns whether an opaque payload declares the Blind Vault wire format.
///
/// [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] This intentionally checks
/// only the fixed magic prefix. Callers must still use
/// [`decode_blind_vault_frame`] and fail closed when the remainder is malformed;
/// falling back to another protocol after seeing this prefix would create a
/// parser-confusion boundary at an onion terminal.
#[must_use]
pub fn is_blind_vault_frame(bytes: &[u8]) -> bool {
    bytes.starts_with(&FRAME_MAGIC)
}

/// Maximum mutation/request frame. The largest v1 ciphertext class is 256 KiB;
/// the remaining space covers fixed metadata and framing.
pub const MAX_BLIND_VAULT_MUTATION_FRAME_BYTES: u64 = 272 * 1024;

/// Maximum signed pull-response frame: sixteen 256 KiB ciphertext classes plus
/// bounded metadata. Public request handlers must retain their narrower body
/// limits and must not apply this response ceiling to attacker-controlled puts.
pub const MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES: u64 = 5 * 1024 * 1024;

/// Absolute largest v1 frame accepted by the generic decoder.
pub const MAX_BLIND_VAULT_FRAME_BYTES: u64 = MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES;

/// Binary frame carrying either an immutable put request or a node receipt.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultFrame {
    /// Client request to persist one immutable encrypted object.
    Put(BlindVaultPutRequest),
    /// Node proof that the exact object was accepted for bounded retention.
    StoredReceipt(BlindVaultStoredReceipt),
    /// Client request to create one anonymous, replica-local lease.
    LeaseCreate(BlindVaultLeaseCreateRequest),
    /// Lease administrator request to remove one immutable object.
    Delete(BlindVaultDeleteRequest),
    /// Node proof that an exact object was removed or already absent.
    DeletedReceipt(BlindVaultDeletedReceipt),
    /// One-time bearer admission ticket plus a self-authenticating lease.
    LeaseAdmission(BlindVaultLeaseAdmissionRequest),
    /// Capability-authenticated request for one stable encrypted-object page.
    PullRequest(BlindVaultPullRequest),
    /// Node-signed stable encrypted-object page.
    PullResponse(BlindVaultPullResponse),
    /// RFC 9474 blind-issued one-time credential plus anonymous lease.
    BlindLeaseAdmission(BlindVaultBlindLeaseAdmissionRequest),
    /// Node-signed public blind-admission key epochs and coarse policy.
    BlindIssuerDirectory(BlindVaultBlindIssuerDirectory),
    /// Terminal-signed proof that one blind-issued lease was accepted.
    BlindLeaseAccepted(BlindVaultBlindLeaseAcceptedReceipt),
    /// Administration-key request to retire one complete replica lease.
    LeaseRetire(BlindVaultLeaseRetireRequest),
    /// Terminal-signed proof that one complete replica lease was retired.
    LeaseRetiredReceipt(BlindVaultLeaseRetiredReceipt),
    /// Blind-issued capacity authorization plus an administration-key renewal.
    BlindLeaseRenewal(BlindVaultBlindLeaseRenewalRequest),
    /// Terminal-signed proof that one authorized renewal was committed.
    BlindLeaseRenewed(BlindVaultBlindLeaseRenewedReceipt),
    /// Administration-key request for one encrypted replica-local status view.
    LeaseStatus(BlindVaultLeaseStatusRequest),
    /// Terminal-signed proof of one coherent lease status observation.
    LeaseStatusReceipt(BlindVaultLeaseStatusReceipt),
    /// Administration-key request for one private live-set commitment.
    LeaseInventory(BlindVaultLeaseInventoryRequest),
    /// Terminal-signed proof of one coherent encrypted-object inventory.
    LeaseInventoryReceipt(BlindVaultLeaseInventoryReceipt),
    /// Coarse workload failure visible only after source-side reply decryption.
    TerminalFailure(BlindVaultTerminalFailure),
}

impl std::fmt::Debug for BlindVaultFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // [BLIND-VAULT-PRIVACY-SAFE-FRAME-DEBUG 2026-08-30 by Codex] Never
        // recurse into wire values: capabilities, ciphertext, commitments,
        // signatures, and replica identifiers must not enter node logs.
        formatter.write_str(match self {
            Self::Put(_) => "BlindVaultFrame::Put(<redacted>)",
            Self::StoredReceipt(_) => "BlindVaultFrame::StoredReceipt(<redacted>)",
            Self::LeaseCreate(_) => "BlindVaultFrame::LeaseCreate(<redacted>)",
            Self::Delete(_) => "BlindVaultFrame::Delete(<redacted>)",
            Self::DeletedReceipt(_) => "BlindVaultFrame::DeletedReceipt(<redacted>)",
            Self::LeaseAdmission(_) => "BlindVaultFrame::LeaseAdmission(<redacted>)",
            Self::PullRequest(_) => "BlindVaultFrame::PullRequest(<redacted>)",
            Self::PullResponse(_) => "BlindVaultFrame::PullResponse(<redacted>)",
            Self::BlindLeaseAdmission(_) => "BlindVaultFrame::BlindLeaseAdmission(<redacted>)",
            Self::BlindIssuerDirectory(_) => "BlindVaultFrame::BlindIssuerDirectory(<redacted>)",
            Self::BlindLeaseAccepted(_) => "BlindVaultFrame::BlindLeaseAccepted(<redacted>)",
            Self::LeaseRetire(_) => "BlindVaultFrame::LeaseRetire(<redacted>)",
            Self::LeaseRetiredReceipt(_) => "BlindVaultFrame::LeaseRetiredReceipt(<redacted>)",
            Self::BlindLeaseRenewal(_) => "BlindVaultFrame::BlindLeaseRenewal(<redacted>)",
            Self::BlindLeaseRenewed(_) => "BlindVaultFrame::BlindLeaseRenewed(<redacted>)",
            Self::LeaseStatus(_) => "BlindVaultFrame::LeaseStatus(<redacted>)",
            Self::LeaseStatusReceipt(_) => "BlindVaultFrame::LeaseStatusReceipt(<redacted>)",
            Self::LeaseInventory(_) => "BlindVaultFrame::LeaseInventory(<redacted>)",
            Self::LeaseInventoryReceipt(_) => "BlindVaultFrame::LeaseInventoryReceipt(<redacted>)",
            Self::TerminalFailure(_) => "BlindVaultFrame::TerminalFailure(<redacted>)",
        })
    }
}

/// Stable, bounded binary encoding with an explicit frame kind outside bincode.
/// This avoids depending on serde enum discriminants for future evolution.
pub fn encode_blind_vault_frame(frame: &BlindVaultFrame) -> Result<Vec<u8>, BlindVaultError> {
    let (kind, body) = match frame {
        BlindVaultFrame::Put(value) => (
            FRAME_KIND_PUT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::StoredReceipt(value) => (
            FRAME_KIND_STORED_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseCreate(value) => (
            FRAME_KIND_LEASE_CREATE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::Delete(value) => (
            FRAME_KIND_DELETE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::DeletedReceipt(value) => (
            FRAME_KIND_DELETED_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseAdmission(value) => (
            FRAME_KIND_LEASE_ADMISSION,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::PullRequest(value) => (
            FRAME_KIND_PULL_REQUEST,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::PullResponse(value) => (
            FRAME_KIND_PULL_RESPONSE,
            serialize_body(value, MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseAdmission(value) => (
            FRAME_KIND_BLIND_LEASE_ADMISSION,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindIssuerDirectory(value) => (
            FRAME_KIND_BLIND_ISSUER_DIRECTORY,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseAccepted(value) => (
            FRAME_KIND_BLIND_LEASE_ACCEPTED,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseRetire(value) => (
            FRAME_KIND_LEASE_RETIRE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseRetiredReceipt(value) => (
            FRAME_KIND_LEASE_RETIRED_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseRenewal(value) => (
            FRAME_KIND_BLIND_LEASE_RENEWAL,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseRenewed(value) => (
            FRAME_KIND_BLIND_LEASE_RENEWED,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseStatus(value) => (
            FRAME_KIND_LEASE_STATUS,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseStatusReceipt(value) => (
            FRAME_KIND_LEASE_STATUS_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseInventory(value) => (
            FRAME_KIND_LEASE_INVENTORY,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseInventoryReceipt(value) => (
            FRAME_KIND_LEASE_INVENTORY_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::TerminalFailure(value) => (
            FRAME_KIND_TERMINAL_FAILURE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
    };

    let total = FRAME_HEADER_BYTES
        .checked_add(body.len())
        .ok_or(BlindVaultError::FrameTooLarge)?;
    if total as u64 > frame_limit_for_kind(kind)? {
        return Err(BlindVaultError::FrameTooLarge);
    }

    let mut encoded = Vec::with_capacity(total);
    encoded.extend_from_slice(&FRAME_MAGIC);
    encoded.extend_from_slice(&BLIND_VAULT_PROTOCOL_VERSION.to_be_bytes());
    encoded.push(kind);
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

/// Decodes one complete v1 frame and rejects unknown kinds/trailing bytes.
pub fn decode_blind_vault_frame(bytes: &[u8]) -> Result<BlindVaultFrame, BlindVaultError> {
    if bytes.len() < FRAME_HEADER_BYTES {
        return Err(BlindVaultError::TruncatedFrame);
    }
    if bytes.len() as u64 > MAX_BLIND_VAULT_FRAME_BYTES {
        return Err(BlindVaultError::FrameTooLarge);
    }
    if bytes[..4] != FRAME_MAGIC {
        return Err(BlindVaultError::InvalidMagic);
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    require_version(version)?;

    let kind = bytes[6];
    let frame_limit = frame_limit_for_kind(kind)?;
    if bytes.len() as u64 > frame_limit {
        return Err(BlindVaultError::FrameTooLarge);
    }
    let body = &bytes[FRAME_HEADER_BYTES..];
    match kind {
        FRAME_KIND_PUT => Ok(BlindVaultFrame::Put(deserialize_body(body, frame_limit)?)),
        FRAME_KIND_STORED_RECEIPT => Ok(BlindVaultFrame::StoredReceipt(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_CREATE => Ok(BlindVaultFrame::LeaseCreate(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_DELETE => Ok(BlindVaultFrame::Delete(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_DELETED_RECEIPT => Ok(BlindVaultFrame::DeletedReceipt(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_ADMISSION => Ok(BlindVaultFrame::LeaseAdmission(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_PULL_REQUEST => Ok(BlindVaultFrame::PullRequest(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_PULL_RESPONSE => Ok(BlindVaultFrame::PullResponse(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_BLIND_LEASE_ADMISSION => Ok(BlindVaultFrame::BlindLeaseAdmission(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_BLIND_ISSUER_DIRECTORY => Ok(BlindVaultFrame::BlindIssuerDirectory(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_BLIND_LEASE_ACCEPTED => Ok(BlindVaultFrame::BlindLeaseAccepted(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_LEASE_RETIRE => Ok(BlindVaultFrame::LeaseRetire(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_RETIRED_RECEIPT => Ok(BlindVaultFrame::LeaseRetiredReceipt(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_BLIND_LEASE_RENEWAL => Ok(BlindVaultFrame::BlindLeaseRenewal(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_BLIND_LEASE_RENEWED => Ok(BlindVaultFrame::BlindLeaseRenewed(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_STATUS => Ok(BlindVaultFrame::LeaseStatus(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_STATUS_RECEIPT => Ok(BlindVaultFrame::LeaseStatusReceipt(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_LEASE_INVENTORY => Ok(BlindVaultFrame::LeaseInventory(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_INVENTORY_RECEIPT => Ok(BlindVaultFrame::LeaseInventoryReceipt(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_TERMINAL_FAILURE => Ok(BlindVaultFrame::TerminalFailure(deserialize_body(
            body,
            frame_limit,
        )?)),
        kind => Err(BlindVaultError::UnknownFrameKind(kind)),
    }
}

fn frame_limit_for_kind(kind: u8) -> Result<u64, BlindVaultError> {
    match kind {
        FRAME_KIND_PUT
        | FRAME_KIND_STORED_RECEIPT
        | FRAME_KIND_LEASE_CREATE
        | FRAME_KIND_DELETE
        | FRAME_KIND_DELETED_RECEIPT
        | FRAME_KIND_LEASE_ADMISSION
        | FRAME_KIND_PULL_REQUEST
        | FRAME_KIND_BLIND_LEASE_ADMISSION
        | FRAME_KIND_BLIND_ISSUER_DIRECTORY
        | FRAME_KIND_BLIND_LEASE_ACCEPTED
        | FRAME_KIND_LEASE_RETIRE
        | FRAME_KIND_LEASE_RETIRED_RECEIPT
        | FRAME_KIND_BLIND_LEASE_RENEWAL
        | FRAME_KIND_BLIND_LEASE_RENEWED
        | FRAME_KIND_LEASE_STATUS
        | FRAME_KIND_LEASE_STATUS_RECEIPT
        | FRAME_KIND_LEASE_INVENTORY
        | FRAME_KIND_LEASE_INVENTORY_RECEIPT
        | FRAME_KIND_TERMINAL_FAILURE => Ok(MAX_BLIND_VAULT_MUTATION_FRAME_BYTES),
        FRAME_KIND_PULL_RESPONSE => Ok(MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES),
        unknown => Err(BlindVaultError::UnknownFrameKind(unknown)),
    }
}

pub(super) fn serialize_body<T: Serialize>(
    value: &T,
    limit: u64,
) -> Result<Vec<u8>, BlindVaultError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(limit)
        .serialize(value)
        .map_err(|_| BlindVaultError::Serialization)
}

pub(super) fn deserialize_body<T>(bytes: &[u8], limit: u64) -> Result<T, BlindVaultError>
where
    T: for<'de> Deserialize<'de>,
{
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(limit)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|_| BlindVaultError::Deserialization)
}

#[cfg(test)]
mod tests;
