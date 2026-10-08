// ============================================
// File: crates/aeronyx-mailbox-wire/src/lib.rs
// ============================================
//! # AeroNyx anonymous mailbox wire format
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude] One source of truth for the mailbox
//! protocol shared by the node and the Flutter client (vendored, hash-pinned):
//! terminal frames, recipient-sealed items, deposit invitations (AMDI v2),
//! the blind relay envelope, onion layers, and a client-side onion source.
//!
//! Design rules:
//! - bytes in, bytes out; no I/O, no logging, no async;
//! - one signature policy everywhere: Ed25519 `verify_strict`;
//! - byte-for-byte compatible with `aeronyx-core`, proven by golden vectors
//!   and node interop tests rather than assumed.

#![forbid(unsafe_code)]

pub mod chat;
mod codec;
pub mod crypto;
pub mod invitation;
pub mod mailbox;
pub mod onion;
pub mod recipient_seal;
pub mod source;
