// ============================================
// File: crates/aeronyx-blind-issuer/src/lib.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] Public surface for the isolated signer;
// storage-node state and private custody must never cross this boundary.
//! # `AeroNyx` Blind Admission Issuer
//!
//! ## Creation Reason
//! Blind Vault clients need an RFC 9474 signer whose private keys are
//! physically absent from decentralized storage nodes.
//!
//! ## Main Functionality
//! - Loads bounded rotating issuer epochs from secure local files.
//! - Signs only already-blinded messages under an active requested key.
//! - Keeps policy independent from software RSA, HSM, or KMS custody backends.
//! - Exposes a loopback-only, backend-authenticated HTTP interface.
//! - Provides public epoch material without account or issuance metadata.
//!
//! ## Dependencies
//! - `config`: fail-closed listener, key epoch, and secret-file policy.
//! - `signer`: private RSA operation and public epoch derivation.
//! - `api`: bounded internal HTTP protocol and pressure controls.
//! - `aeronyx-core`: canonical Blind Vault admission versions and bounds.
//!
//! ## Privacy Invariant
//! This crate must never receive or store wallet, account, device, lease,
//! storage-node, or client-network identifiers. The authenticated backend
//! submits only a blinded RSA message and public issuer key fingerprint.
//!
//! ## Important Note For The Next Developer
//! - Never merge this crate into `aeronyx-server` or copy private keys there.
//! - Never add request-body logging, tracing layers, or per-request audit rows.
//! - Entitlement and quota checks belong in the upstream backend before this
//!   process; the signer intentionally cannot correlate issuance to redemption.
//! - Production key custody should migrate behind a KMS/HSM implementation of
//!   the signer boundary without changing the internal wire contract.
//!
//! Last Modified: v0.3.0-BlindIssuer - Added bounded custody response deadlines.
//! ============================================

pub mod api;
pub mod config;
pub mod signer;

pub use api::{
    build_router, build_router_with_timeout, decode_epoch_snapshot, decode_sign_response,
    encode_epoch_snapshot, encode_sign_request, BlindIssuerEpochSnapshot,
    BLIND_ISSUER_CONTENT_TYPE, BLIND_ISSUER_EPOCH_CONTENT_TYPE,
};
pub use config::{
    BlindIssuerConfig, BlindIssuerKeyConfig, ConfigError, DEFAULT_SIGNING_TIMEOUT_MS,
};
pub use signer::{
    BlindSignError, BlindSignRequest, BlindSignResponse, BlindSigner, BlindSignerBuildError,
    BlindSigningBackend, BlindSigningBackendError, BlindSigningKey,
};
