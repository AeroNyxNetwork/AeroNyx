// ============================================
// File: crates/aeronyx-server/src/commands.rs
// ============================================
//! # `aeronyx-server` subcommand implementations
//!
//! Binary-only module tree: it is declared from `main.rs`, not from `lib.rs`,
//! like the existing `mailbox_probe` and `relay_smoke` binary modules. The CLI
//! argument types and the `main` dispatcher stay in `main.rs`; every
//! subcommand implementation and its private helpers live in a focused child
//! module, with its unit tests next to it in `<child>/tests.rs`.
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.
//! - `commands/register.rs`: `register` (bounded registration-code input,
//!   operator metadata normalization, CMS registration)
//! - `commands/node.rs`: `start`, `status`, `validate`, hidden `pubkey`
//! - `commands/memchain.rs`: `memchain verify-aof`
//! - `commands/live_smoke.rs`: `mailbox-probe`, `relay-smoke`,
//!   `v1-compatibility-smoke`
//! - `commands/relay_custody.rs`: `relay-custody` dispatcher and maintenance
//!   audit verification, with children:
//!   - `relay_custody/audit_anchor.rs`: audit anchor export and verification
//!   - `relay_custody/witness_receipt.rs`: witness countersigning and receipt
//!     verification
//!   - `relay_custody/witness_vault.rs`: producer receipt import, vault audit,
//!     operator collection
//!   - `relay_custody/restore_plan.rs`: restore plans and readiness output
//!   - `relay_custody/host_io.rs`: bounded artifact files and node
//!     config/secret/identity loading
//! - `commands/directory_replica.rs`: `directory-replica` dispatcher, with
//!   children:
//!   - `directory_replica/observation_certificate.rs`: certificate verify,
//!     import and pull
//!   - `directory_replica/operator_smoke.rs`: loopback operator API smoke calls
//!   - `directory_replica/quarantine.rs`: incident inspection and resolution
//! - `commands/helpers.rs`: logging, strict hex, clock, config and node
//!   identity helpers shared by several commands

// ============================================
// Commands
// ============================================

mod directory_replica;
mod helpers;
mod live_smoke;
mod memchain;
mod node;
mod register;
mod relay_custody;

// Entry points called by `main` are plain `pub`: this module is private, so
// they stay crate-visible only (`pub(crate)` here trips
// `clippy::redundant_pub_crate`).
pub use directory_replica::cmd_directory_replica;
pub use helpers::init_logging;
pub use live_smoke::{cmd_mailbox_probe, cmd_relay_smoke, cmd_v1_compatibility_smoke};
pub use memchain::cmd_memchain;
pub use node::{cmd_pubkey, cmd_start, cmd_status, cmd_validate};
pub use register::{cmd_register, resolve_registration_code};
pub use relay_custody::cmd_relay_custody;
