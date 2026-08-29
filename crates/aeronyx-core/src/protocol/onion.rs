// ============================================
// File: crates/aeronyx-core/src/protocol/onion.rs
// ============================================
//! # Onion Routing v1 — Layered Per-Hop Encryption
//!
//! ## Creation Reason
//! Upgrades the existing **blind relay** (a single opaque envelope forwarder)
//! into real **onion routing**: the source wraps the payload in one encrypted
//! layer per hop, and each relay peels exactly one layer. A relay learns only
//! the *immediate* next hop — never the original source, the final destination,
//! or the payload. This guarantees that no single honest-but-curious relay can
//! link source and destination together.
//!
//! ## Relationship to the transport
//! This module does NOT define a new wire frame. It restructures the opaque
//! `BlindRelayEnvelope::encrypted_blob` (see `chat.rs`). The envelope and all of
//! its hardened guards (Ed25519 per-hop signature, freshness window, replay
//! cache, abuse guard, routeability gate, TTL, loop detection, probes, counters)
//! are reused unchanged. `envelope.next_hop` always addresses the node that
//! receives *this* envelope; the privacy-sensitive forward target is hidden
//! inside the peeled layer.
//!
//! ## Construction (HPKE-style, RFC 9180 DHKEM shape)
//! Each layer is a single-shot seal to the hop's KEM public key:
//! ```text
//!   ephemeral X25519  ->  ECDH  ->  HKDF-SHA256  ->  XChaCha20-Poly1305
//! ```
//! All primitives are the already-audited ones in `crypto/{keys,kdf}.rs`; no new
//! cryptographic dependency is introduced. The KEM is deliberately abstracted
//! behind a versioned descriptor field (`kem_alg`) so a future release can move
//! to the hybrid post-quantum X-Wing KEM (X25519 + ML-KEM-768) without changing
//! this wire format.
//!
//! ## Layer wire format (content of `encrypted_blob`)
//! ```text
//!   magic:   [0xA0, 0x01]   (2B  — ONION_V1 marker)
//!   eph_pub: [u8; 32]       (client ephemeral X25519 public for THIS hop)
//!   nonce:   [u8; 24]       (random XChaCha20 nonce)
//!   ct:      [u8]           (XChaCha20-Poly1305 over the encoded OnionHopPayload)
//! ```
//! Key derivation (both sides identical):
//! `key = HKDF-SHA256(ikm = ECDH(eph, hop_kem), salt = ONION_SALT,
//!                    info = eph_pub || hop_kem_pub, len = 32)`.
//! No AEAD AAD is used; the key already binds `eph_pub` and `hop_kem_pub`.
//!
//! The decrypted plaintext is an `OnionHopPayload`, encoded as:
//! ```text
//!   flags:     u8        (bit0: 1 = forward / next_hop present, 0 = terminal)
//!   next_hop:  [u8; 32]  (present ONLY when flags bit0 == 1)
//!   inner_len: u32 LE
//!   inner:     [u8; inner_len]
//! ```
//!
//! ## Threat model (v1)
//! Honest-but-curious relays. v1 does NOT defend against a *global passive
//! observer* that correlates packet lengths/timing (the onion shrinks one layer
//! per hop). That property requires a constant-length Sphinx packet with
//! per-hop replay MACs and ephemeral blinding, which is the documented v2
//! upgrade. See `docs/onion-routing-v1-spec.md`.
//!
//! ## ⚠️ Important Notes for Next Developer
//! - Both byte layouts (the layer header below and the `OnionHopPayload` in
//!   `encode_payload`) are wire contracts shared with non-Rust clients. They are
//!   explicit byte layouts (no Rust serialization library). Do NOT reorder
//!   fields. Add new fields only with a versioned magic (e.g. `[0xA0, 0x02]`).
//! - `open_onion_layer` must never log plaintext or the peeled `inner` bytes.
//! - A relay's X25519 *public* key is NOT derivable from its Ed25519 `node_id`
//!   (the X25519 secret is `SHA512(ed_secret)[..32]`), so it MUST be published
//!   in the node descriptor. See `discovery::NodeDescriptor::kem_public`.
//! - [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Route-purpose strings are a
//!   public protocol contract. Parse them through [`OnionRoutePurpose`] rather
//!   than duplicating aliases in an App, SDK, agent, or server implementation.
//!   Unknown values must fail closed instead of silently becoming chat routes.
//!
//! ## Last Modified
//! v1.11.0-VerifiedRoutePlan — Added descriptor-authenticated, purpose-aware
//! source route admission with exact path-derived TTL
//! v1.10.0-TerminalFeatureContract — Centralized purpose-specific signed
//! terminal feature requirements for node, App, SDK, and agent route builders
//! v1.9.0-BlindVaultLeaseInventoryPurpose — Added private inventory commitment
//! v1.8.0-BlindVaultLeaseStatusPurpose — Added private signed lease observation
//! v1.7.0-BlindVaultLeaseRenewalPurpose — Added blind-authorized lease renewal
//! v1.6.0-BlindVaultLeaseRetirePurpose — Added anonymous complete lease retirement
//! v1.5.0-BlindVaultPutReceiptPurpose — Added receipt-capable anonymous writes
//! v1.4.0-BlindVaultLeaseAdmissionPurpose — Added blind-issued lease admission
//! v1.3.0-BlindVaultDeletePurpose — Added a distinct anonymous deletion purpose
//! v1.2.0-BlindVaultPullPurpose — Added a distinct anonymous recovery purpose
//! v1.1.0-RoutePurposeContract — Added stable, capability-aware route purposes
//! v1.0.0-OnionV1 — Initial layered onion construction over the blind relay frame

use hkdf::Hkdf;
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::Sha256;
use thiserror::Error;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

use crate::crypto::keys::{E2eSession, EphemeralKeyPair, IdentityKeyPair};
use crate::error::CoreError;
use crate::protocol::chat::BlindRelayEnvelope;
use crate::protocol::discovery::{NodeCapability, NodeProtocolFeature, SignedNodeDescriptor};

// ============================================
// Constants
// ============================================

/// Onion layer magic prefix (version 1). Marks `encrypted_blob` as an onion
/// layer to be peeled, distinguishing it from a legacy opaque blind-relay blob.
pub const ONION_MAGIC: [u8; 2] = [0xA0, 0x01];

/// HKDF domain-separation salt for onion layer keys.
pub const ONION_SALT: &[u8] = b"AeroNyx-Onion-v1";

/// KEM algorithm id: classical X25519 (the v1 default).
pub const KEM_ALG_X25519: u8 = 1;

/// KEM algorithm id reserved for the hybrid post-quantum X-Wing KEM
/// (X25519 + ML-KEM-768). Not implemented in v1; reserved so the descriptor
/// field and this module can adopt it without a wire break.
pub const KEM_ALG_XWING: u8 = 2;

/// Canonical route-purpose values supported by onion candidate contracts.
///
/// The order is stable for deterministic capability responses. Compatibility
/// aliases accepted by [`OnionRoutePurpose::from_wire_value`] are deliberately
/// absent so new integrations emit only canonical values.
pub const ONION_ROUTE_PURPOSE_VALUES: [&str; 10] = [
    "message_relay",
    "blind_vault_put",
    "blind_vault_pull",
    "blind_vault_delete",
    "blind_vault_lease_admission",
    "blind_vault_put_receipt",
    "blind_vault_lease_retire",
    "blind_vault_lease_renewal",
    "blind_vault_lease_status",
    "blind_vault_lease_inventory",
];

/// Path-wide proof contract for topology-hiding terminal replies.
const SOURCE_SEALED_REPLY_PATH_FEATURES: [NodeProtocolFeature; 2] = [
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Generic reply support plus encrypted workload failures and sealed proof.
const BLIND_VAULT_REPLY_FEATURES: [NodeProtocolFeature; 4] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Reply contract for blind-issued lease admission.
const BLIND_VAULT_LEASE_ADMISSION_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindLeaseAdmissionV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Reply contract for receipt-capable immutable writes.
const BLIND_VAULT_PUT_RECEIPT_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultPutReceiptV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Reply contract for administration-key lease retirement.
const BLIND_VAULT_LEASE_RETIRE_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultLeaseRetireV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Reply contract for blind-authorized lease renewal.
const BLIND_VAULT_LEASE_RENEWAL_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultLeaseRenewalV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Reply contract for private lease-status observations.
const BLIND_VAULT_LEASE_STATUS_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultLeaseStatusV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Reply contract for private lease-inventory commitments.
const BLIND_VAULT_LEASE_INVENTORY_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultLeaseInventoryV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Fixed layer header length: magic(2) + eph_pub(32) + nonce(24).
const LAYER_HEADER_LEN: usize = 2 + 32 + 24;

/// Upper bound for a decoded `OnionHopPayload` inner field (matches the blind
/// relay frame cap).
const MAX_ONION_PAYLOAD_BYTES: usize = 256 * 1024;

/// Maximum remote hops accepted by the descriptor-verified route planner.
///
/// This matches the public candidate contract. The legacy raw builder remains
/// available for wire compatibility and controlled protocol experiments.
pub const MAX_VERIFIED_ONION_ROUTE_HOPS: usize = 3;

/// Signed capabilities required from a relay that forwards another onion layer.
pub const ONION_FORWARD_HOP_REQUIRED_CAPABILITIES: [NodeCapability; 2] =
    [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];

/// Signed capabilities required from the terminal before purpose-specific roles.
pub const ONION_TERMINAL_REQUIRED_CAPABILITIES: [NodeCapability; 1] = [NodeCapability::ChatRelay];

// ============================================
// Types
// ============================================

/// Terminal workload carried by an onion route.
///
/// [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] This enum standardizes purpose
/// negotiation across nodes, Apps, SDKs, and autonomous agents. It is not
/// serialized with a Rust enum representation: callers must emit [`Self::as_str`]
/// and parse untrusted input with [`Self::from_wire_value`] so unknown future
/// purposes fail closed on older implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnionRoutePurpose {
    /// End-to-end encrypted message delivery through blind relay terminals.
    MessageRelay,
    /// Anonymous durable ciphertext write to a Blind Vault replica terminal.
    BlindVaultPut,
    /// Anonymous bounded ciphertext recovery from a Blind Vault replica.
    BlindVaultPull,
    /// Anonymous capability-authorized deletion at a Blind Vault terminal.
    BlindVaultDelete,
    /// Anonymous blind-issued lease admission at a Blind Vault terminal.
    BlindVaultLeaseAdmission,
    /// Anonymous durable ciphertext write with an encrypted storage receipt.
    BlindVaultPutReceipt,
    /// Anonymous administration-key retirement of one complete replica lease.
    BlindVaultLeaseRetire,
    /// Blind-authorized administration-key renewal of one live replica lease.
    BlindVaultLeaseRenewal,
    /// Administration-authorized private observation of one live replica lease.
    BlindVaultLeaseStatus,
    /// Administration-authorized commitment to one live replica inventory.
    BlindVaultLeaseInventory,
}

impl OnionRoutePurpose {
    /// Parses a canonical purpose or a backward-compatible legacy alias.
    ///
    /// Returns `None` for blank or unknown input. Callers must preserve that
    /// unsupported state rather than defaulting it to [`Self::MessageRelay`].
    #[must_use]
    pub fn from_wire_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "message" | "message_relay" | "message-relay" | "chat" => Some(Self::MessageRelay),
            "blind_vault" | "blind-vault" | "blind_vault_put" | "blind-vault-put" => {
                Some(Self::BlindVaultPut)
            }
            "blind_vault_pull" | "blind-vault-pull" => Some(Self::BlindVaultPull),
            "blind_vault_delete" | "blind-vault-delete" => Some(Self::BlindVaultDelete),
            "blind_vault_lease_admission" | "blind-vault-lease-admission" => {
                Some(Self::BlindVaultLeaseAdmission)
            }
            "blind_vault_put_receipt" | "blind-vault-put-receipt" => {
                Some(Self::BlindVaultPutReceipt)
            }
            "blind_vault_lease_retire" | "blind-vault-lease-retire" => {
                Some(Self::BlindVaultLeaseRetire)
            }
            "blind_vault_lease_renewal" | "blind-vault-lease-renewal" => {
                Some(Self::BlindVaultLeaseRenewal)
            }
            "blind_vault_lease_status" | "blind-vault-lease-status" => {
                Some(Self::BlindVaultLeaseStatus)
            }
            "blind_vault_lease_inventory" | "blind-vault-lease-inventory" => {
                Some(Self::BlindVaultLeaseInventory)
            }
            _ => None,
        }
    }

    /// Returns the canonical, language-neutral wire value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MessageRelay => ONION_ROUTE_PURPOSE_VALUES[0],
            Self::BlindVaultPut => ONION_ROUTE_PURPOSE_VALUES[1],
            Self::BlindVaultPull => ONION_ROUTE_PURPOSE_VALUES[2],
            Self::BlindVaultDelete => ONION_ROUTE_PURPOSE_VALUES[3],
            Self::BlindVaultLeaseAdmission => ONION_ROUTE_PURPOSE_VALUES[4],
            Self::BlindVaultPutReceipt => ONION_ROUTE_PURPOSE_VALUES[5],
            Self::BlindVaultLeaseRetire => ONION_ROUTE_PURPOSE_VALUES[6],
            Self::BlindVaultLeaseRenewal => ONION_ROUTE_PURPOSE_VALUES[7],
            Self::BlindVaultLeaseStatus => ONION_ROUTE_PURPOSE_VALUES[8],
            Self::BlindVaultLeaseInventory => ONION_ROUTE_PURPOSE_VALUES[9],
        }
    }

    /// Returns the additional signed capability required of the terminal.
    ///
    /// Every middle relay remains governed by the base onion capability
    /// contract. This method describes only workload-specific terminal role
    /// admission and never trusts flattened API projections.
    #[must_use]
    pub const fn specialized_terminal_capability(self) -> Option<NodeCapability> {
        match self {
            Self::MessageRelay => None,
            Self::BlindVaultPut
            | Self::BlindVaultPull
            | Self::BlindVaultDelete
            | Self::BlindVaultLeaseAdmission
            | Self::BlindVaultPutReceipt
            | Self::BlindVaultLeaseRetire
            | Self::BlindVaultLeaseRenewal
            | Self::BlindVaultLeaseStatus
            | Self::BlindVaultLeaseInventory => Some(NodeCapability::BlindVaultReplica),
        }
    }

    /// Returns the signed protocol features required from the terminal.
    ///
    /// [ONION-TERMINAL-FEATURE-CONTRACT 2026-08-28 by Codex] This mapping is
    /// part of the core route-purpose domain model so nodes, Apps, SDKs, and AI
    /// agents cannot silently choose different rolling-upgrade requirements.
    /// Coarse role admission remains in [`Self::specialized_terminal_capability`].
    /// A caller must require every returned feature from the terminal's signed
    /// descriptor and fail closed when any feature is absent.
    #[must_use]
    pub const fn required_terminal_protocol_features(self) -> &'static [NodeProtocolFeature] {
        match self {
            Self::MessageRelay | Self::BlindVaultPut => &[],
            Self::BlindVaultPull | Self::BlindVaultDelete => &BLIND_VAULT_REPLY_FEATURES,
            Self::BlindVaultLeaseAdmission => &BLIND_VAULT_LEASE_ADMISSION_FEATURES,
            Self::BlindVaultPutReceipt => &BLIND_VAULT_PUT_RECEIPT_FEATURES,
            Self::BlindVaultLeaseRetire => &BLIND_VAULT_LEASE_RETIRE_FEATURES,
            Self::BlindVaultLeaseRenewal => &BLIND_VAULT_LEASE_RENEWAL_FEATURES,
            Self::BlindVaultLeaseStatus => &BLIND_VAULT_LEASE_STATUS_FEATURES,
            Self::BlindVaultLeaseInventory => &BLIND_VAULT_LEASE_INVENTORY_FEATURES,
        }
    }

    /// Returns signed features required from every selected path hop.
    ///
    /// [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] A v2 reply cannot
    /// safely traverse a legacy middle: that node expects a relay-visible
    /// terminal receipt and cannot authenticate an opaque-only response.
    /// Sources must therefore verify these tokens on the entry, every middle,
    /// and the terminal before constructing a reply-capable path.
    #[must_use]
    pub const fn required_path_protocol_features(self) -> &'static [NodeProtocolFeature] {
        match self {
            Self::MessageRelay | Self::BlindVaultPut => &[],
            Self::BlindVaultPull
            | Self::BlindVaultDelete
            | Self::BlindVaultLeaseAdmission
            | Self::BlindVaultPutReceipt
            | Self::BlindVaultLeaseRetire
            | Self::BlindVaultLeaseRenewal
            | Self::BlindVaultLeaseStatus
            | Self::BlindVaultLeaseInventory => &SOURCE_SEALED_REPLY_PATH_FEATURES,
        }
    }
}

/// One hop on an onion path: the relay's node id plus its published KEM key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionHop {
    /// Relay Ed25519 node id (matches `NodeDescriptor::node_id`).
    pub node_id: [u8; 32],
    /// Relay KEM public key (X25519 for v1; from `NodeDescriptor::kem_public`).
    pub kem_pub: [u8; 32],
}

/// Fail-closed descriptor and construction errors for a verified onion route.
///
/// No variant contains a node id, endpoint, payload, route id, or key. These
/// errors are safe to aggregate as coarse local diagnostics without publishing
/// route topology.
#[derive(Debug, Error)]
pub enum OnionRoutePlanError {
    /// A route needs at least one remote terminal.
    #[error("onion route must contain at least one hop")]
    EmptyPath,
    /// The public protocol currently supports at most three remote hops.
    #[error("onion route exceeds the supported maximum of {max_hops} hops")]
    TooManyHops {
        /// Maximum remote hops accepted by this protocol version.
        max_hops: usize,
    },
    /// Signature, schema, or validity-window verification failed.
    #[error("onion route descriptor at hop {hop_number} is not authentic and current")]
    DescriptorRejected {
        /// One-based hop position; no node identity is exposed.
        hop_number: usize,
    },
    /// A route cannot pass through the same node more than once.
    #[error("onion route repeats a node at hop {hop_number}")]
    DuplicateNode {
        /// One-based position of the repeated node.
        hop_number: usize,
    },
    /// A source cannot also appear as one of its own remote hops.
    #[error("onion route includes its source at hop {hop_number}")]
    SourceIncluded {
        /// One-based position of the source-node loop.
        hop_number: usize,
    },
    /// A selected descriptor does not advertise a required relay role.
    #[error("onion route hop {hop_number} is missing required capability {capability:?}")]
    MissingCapability {
        /// One-based hop position.
        hop_number: usize,
        /// Missing public capability.
        capability: NodeCapability,
    },
    /// A selected descriptor cannot execute the negotiated wire contract.
    #[error("onion route hop {hop_number} is missing required protocol feature {feature:?}")]
    MissingProtocolFeature {
        /// One-based hop position.
        hop_number: usize,
        /// Missing signed protocol feature.
        feature: NodeProtocolFeature,
    },
    /// A relay has no compatible, non-zero per-hop X25519 public key.
    #[error("onion route hop {hop_number} has no compatible X25519 KEM key")]
    MissingX25519Kem {
        /// One-based hop position.
        hop_number: usize,
    },
    /// A descriptor cannot be contacted by the entry or preceding relay.
    #[error("onion route hop {hop_number} has no public peer endpoint")]
    MissingPublicEndpoint {
        /// One-based hop position.
        hop_number: usize,
    },
    /// The signing identity supplied at construction differs from the plan.
    #[error("onion route source identity does not match the verified plan")]
    SourceIdentityMismatch,
    /// The plan was built from descriptors that are no longer current.
    #[error("onion route plan is outside its verified validity window")]
    OutsideValidityWindow,
    /// A verified route passed admission but cryptographic wrapping failed.
    #[error("failed to construct the verified onion envelope")]
    EnvelopeConstruction {
        /// Underlying safe core error.
        #[source]
        source: CoreError,
    },
}

impl OnionRoutePlanError {
    /// Stable privacy-safe category for local recovery and aggregate metrics.
    ///
    /// [ONION-ROUTE-FAILURE-DISPOSITION 2026-08-29 by Codex] The bucket never
    /// includes node identity, endpoint, route id, payload, or key material.
    /// Adapters should persist this bounded value rather than `Display` text.
    #[must_use]
    pub const fn reason_bucket(&self) -> &'static str {
        match self {
            Self::EmptyPath => "empty_path",
            Self::TooManyHops { .. } => "too_many_hops",
            Self::DescriptorRejected { .. } => "descriptor_rejected",
            Self::DuplicateNode { .. } => "duplicate_node",
            Self::SourceIncluded { .. } => "source_included",
            Self::MissingCapability { .. } => "missing_capability",
            Self::MissingProtocolFeature { .. } => "missing_protocol_feature",
            Self::MissingX25519Kem { .. } => "missing_x25519_kem",
            Self::MissingPublicEndpoint { .. } => "missing_public_endpoint",
            Self::SourceIdentityMismatch => "source_identity_mismatch",
            Self::OutsideValidityWindow => "outside_validity_window",
            Self::EnvelopeConstruction { .. } => "envelope_construction_failed",
        }
    }
}

/// Descriptor-authenticated onion route ready for source-side construction.
///
/// [VERIFIED-ONION-ROUTE 2026-08-29 by Codex] This domain object closes the
/// gap between discovery and encryption: callers cannot derive onion hops
/// until every original signed descriptor passes schema/signature/freshness,
/// node uniqueness, capability, purpose-feature, endpoint, and KEM checks.
/// The exact minimum TTL is derived from the admitted path, eliminating a
/// caller-controlled TTL/path mismatch.
///
/// This object proves static descriptor eligibility only. Endpoint liveness,
/// recent relay evidence, network/operator diversity, capacity, and route
/// weighting remain policy inputs and must be checked before constructing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedOnionRoute {
    source_node_id: [u8; 32],
    purpose: OnionRoutePurpose,
    verified_at: u64,
    valid_until: u64,
    hops: Vec<OnionHop>,
}

impl VerifiedOnionRoute {
    /// Verifies a bounded sequence of signed descriptors into one route plan.
    ///
    /// `descriptors` are ordered entry to terminal. The terminal needs the
    /// base terminal role plus the purpose-specific capability/features;
    /// preceding hops additionally need the `OnionMiddle` forwarding role.
    ///
    /// # Errors
    /// Returns [`OnionRoutePlanError`] on the first unsafe or unsupported hop.
    pub fn from_signed_descriptors<'a>(
        source_node_id: [u8; 32],
        descriptors: impl IntoIterator<Item = &'a SignedNodeDescriptor>,
        purpose: OnionRoutePurpose,
        now: u64,
    ) -> Result<Self, OnionRoutePlanError> {
        let mut bounded = Vec::with_capacity(MAX_VERIFIED_ONION_ROUTE_HOPS);
        for descriptor in descriptors {
            if bounded.len() == MAX_VERIFIED_ONION_ROUTE_HOPS {
                return Err(OnionRoutePlanError::TooManyHops {
                    max_hops: MAX_VERIFIED_ONION_ROUTE_HOPS,
                });
            }
            bounded.push(descriptor);
        }
        if bounded.is_empty() {
            return Err(OnionRoutePlanError::EmptyPath);
        }

        let mut hops = Vec::with_capacity(bounded.len());
        let mut valid_until = u64::MAX;
        for (index, signed) in bounded.iter().enumerate() {
            let hop_number = index + 1;
            signed
                .verify_at(now)
                .map_err(|_| OnionRoutePlanError::DescriptorRejected { hop_number })?;
            let descriptor = &signed.descriptor;
            if descriptor.node_id == source_node_id {
                return Err(OnionRoutePlanError::SourceIncluded { hop_number });
            }
            if hops
                .iter()
                .any(|hop: &OnionHop| hop.node_id == descriptor.node_id)
            {
                return Err(OnionRoutePlanError::DuplicateNode { hop_number });
            }

            let is_terminal = index + 1 == bounded.len();
            let required_capabilities = if is_terminal {
                &ONION_TERMINAL_REQUIRED_CAPABILITIES[..]
            } else {
                &ONION_FORWARD_HOP_REQUIRED_CAPABILITIES[..]
            };
            for capability in required_capabilities {
                if !descriptor.capabilities.contains(capability) {
                    return Err(OnionRoutePlanError::MissingCapability {
                        hop_number,
                        capability: *capability,
                    });
                }
            }
            if is_terminal {
                if let Some(capability) = purpose.specialized_terminal_capability() {
                    if !descriptor.capabilities.contains(&capability) {
                        return Err(OnionRoutePlanError::MissingCapability {
                            hop_number,
                            capability,
                        });
                    }
                }
                for feature in purpose.required_terminal_protocol_features() {
                    if !descriptor.advertises_protocol_feature(*feature) {
                        return Err(OnionRoutePlanError::MissingProtocolFeature {
                            hop_number,
                            feature: *feature,
                        });
                    }
                }
            }
            for feature in purpose.required_path_protocol_features() {
                if !descriptor.advertises_protocol_feature(*feature) {
                    return Err(OnionRoutePlanError::MissingProtocolFeature {
                        hop_number,
                        feature: *feature,
                    });
                }
            }

            if descriptor
                .public_endpoint
                .as_deref()
                .map_or(true, |endpoint| endpoint.trim().is_empty())
            {
                return Err(OnionRoutePlanError::MissingPublicEndpoint { hop_number });
            }
            let kem_pub = descriptor
                .x25519_kem_public()
                .ok_or(OnionRoutePlanError::MissingX25519Kem { hop_number })?;
            valid_until = valid_until.min(descriptor.expires_at);
            hops.push(OnionHop {
                node_id: descriptor.node_id,
                kem_pub,
            });
        }

        Ok(Self {
            source_node_id,
            purpose,
            verified_at: now,
            valid_until,
            hops,
        })
    }

    /// Returns the workload contract used to admit this route.
    #[must_use]
    pub const fn purpose(&self) -> OnionRoutePurpose {
        self.purpose
    }

    /// Returns the number of admitted remote hops.
    #[must_use]
    pub fn hop_count(&self) -> usize {
        self.hops.len()
    }

    /// Returns the entry node id without exposing any endpoint or key material.
    #[must_use]
    pub fn entry_node_id(&self) -> [u8; 32] {
        self.hops[0].node_id
    }

    /// Returns when the first selected signed descriptor expires.
    #[must_use]
    pub const fn valid_until(&self) -> u64 {
        self.valid_until
    }

    /// Builds an onion envelope with an exact, path-derived TTL.
    ///
    /// # Errors
    /// Fails closed if the source identity changed, time moved backwards, any
    /// descriptor expired, or cryptographic envelope construction fails.
    pub fn build_envelope(
        &self,
        final_payload: &[u8],
        route_id: [u8; 16],
        now: u64,
        source: &IdentityKeyPair,
    ) -> Result<BlindRelayEnvelope, OnionRoutePlanError> {
        if source.public_key_bytes() != self.source_node_id {
            return Err(OnionRoutePlanError::SourceIdentityMismatch);
        }
        if now < self.verified_at || now >= self.valid_until {
            return Err(OnionRoutePlanError::OutsideValidityWindow);
        }
        let ttl = u8::try_from(self.hops.len()).map_err(|_| OnionRoutePlanError::TooManyHops {
            max_hops: MAX_VERIFIED_ONION_ROUTE_HOPS,
        })?;
        build_onion_envelope(&self.hops, final_payload, route_id, ttl, now, source)
            .map_err(|source| OnionRoutePlanError::EnvelopeConstruction { source })
    }
}

/// Result of peeling exactly one onion layer at a relay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionPeel {
    /// `Some(node_id)` → forward `inner` (the next layer) onward to that hop.
    /// `None` → this node is the terminal hop; `inner` is the delivered payload.
    pub next_hop: Option<[u8; 32]>,
    /// Next layer bytes (when forwarding) or the final payload (when terminal).
    pub inner: Vec<u8>,
}

/// Plaintext carried inside one onion layer. Encoded with an explicit,
/// language-neutral byte layout (see `encode_payload`) so non-Rust clients can
/// produce it without a Rust serialization library — the layout is a wire
/// contract.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OnionHopPayload {
    next_hop: Option<[u8; 32]>,
    inner: Vec<u8>,
}

// ============================================
// Public helpers
// ============================================

/// Returns true if `blob` begins with the onion v1 magic prefix.
///
/// Used by relays to decide whether to peel an onion layer or fall back to the
/// legacy opaque blind-relay forwarding path.
#[must_use]
pub fn is_onion_blob(blob: &[u8]) -> bool {
    blob.len() >= 2 && blob[0] == ONION_MAGIC[0] && blob[1] == ONION_MAGIC[1]
}

/// Peels one onion layer, trying each candidate secret in order (typically the
/// node's current onion key, then the previous one during a rotation grace
/// window). Returns the first successful peel.
///
/// This supports **forward secrecy via rotating onion keys**: a relay rotates
/// its onion keypair on a schedule and keeps the previous key only for a short
/// grace window, so an onion built against the just-rotated descriptor still
/// peels. See `aeronyx-server::services::onion_keys`.
///
/// # Errors
/// Returns `CoreError` if none of the candidate secrets peel the layer.
pub fn try_open_onion_layer(
    blob: &[u8],
    node_x25519_secrets: &[StaticSecret],
) -> Result<OnionPeel, CoreError> {
    let mut last_err = CoreError::malformed("onion layer: no candidate keys");
    for secret in node_x25519_secrets {
        match open_onion_layer(blob, secret) {
            Ok(peel) => return Ok(peel),
            Err(err) => last_err = err,
        }
    }
    Err(last_err)
}

/// Peels exactly one onion layer using this node's static X25519 secret.
///
/// # Errors
/// Returns `CoreError` if the blob is not a well-formed onion layer, if AEAD
/// authentication fails (wrong key / tampered bytes), or if the inner payload
/// fails to decode.
pub fn open_onion_layer(
    blob: &[u8],
    node_x25519_sk: &StaticSecret,
) -> Result<OnionPeel, CoreError> {
    if !is_onion_blob(blob) {
        return Err(CoreError::malformed("onion layer: missing magic prefix"));
    }
    if blob.len() < LAYER_HEADER_LEN {
        return Err(CoreError::malformed("onion layer: truncated header"));
    }

    let mut eph_pub = [0u8; 32];
    eph_pub.copy_from_slice(&blob[2..34]);
    let mut nonce = [0u8; 24];
    nonce.copy_from_slice(&blob[34..LAYER_HEADER_LEN]);
    let ciphertext = &blob[LAYER_HEADER_LEN..];

    // This hop's own KEM public key, recomputed from its secret, binds the key
    // derivation to this specific relay (same value the sender used).
    let hop_kem_pub = X25519PublicKey::from(node_x25519_sk).to_bytes();

    let shared = node_x25519_sk.diffie_hellman(&X25519PublicKey::from(eph_pub));
    let mut ecdh = *shared.as_bytes();
    let key = derive_layer_key(&ecdh, &eph_pub, &hop_kem_pub)?;
    ecdh.zeroize();

    // E2eSession uses the 32-byte key directly with XChaCha20-Poly1305 and
    // zeroizes it on drop. peer_public_key is for logging only.
    let session = E2eSession::new(key, eph_pub);
    let plaintext = session
        .decrypt_raw(ciphertext, &nonce)
        .map_err(|_| CoreError::malformed("onion layer: AEAD open failed"))?;

    let payload = decode_payload(&plaintext)?;
    Ok(OnionPeel {
        next_hop: payload.next_hop,
        inner: payload.inner,
    })
}

/// Builds a complete onion-wrapped `BlindRelayEnvelope` for `path`.
///
/// Layers are sealed innermost (exit) → outermost (entry). The returned
/// envelope is addressed to `path[0]` and signed by `source` (which becomes the
/// `previous_hop_node_id` on the wire, exactly as a normal blind relay send).
///
/// `now` is the Unix-seconds timestamp to stamp on the outer envelope (callers
/// pass a clock value; this crate stays clock-free for deterministic tests).
///
/// # Errors
/// Returns `CoreError` if `path` is empty or any layer fails to seal.
pub fn build_onion_envelope(
    path: &[OnionHop],
    final_payload: &[u8],
    route_id: [u8; 16],
    ttl: u8,
    now: u64,
    source: &IdentityKeyPair,
) -> Result<BlindRelayEnvelope, CoreError> {
    if path.is_empty() {
        return Err(CoreError::malformed("onion path: empty"));
    }

    // Start with the raw payload; wrap one layer per hop from the exit inward.
    let mut inner = final_payload.to_vec();
    for i in (0..path.len()).rev() {
        let next_hop = if i + 1 < path.len() {
            Some(path[i + 1].node_id)
        } else {
            None
        };
        let payload = OnionHopPayload { next_hop, inner };
        let encoded = encode_payload(&payload)?;
        inner = seal_layer(&path[i].kem_pub, &encoded)?;
    }

    let envelope = BlindRelayEnvelope {
        route_id,
        next_hop: path[0].node_id,
        ttl,
        encrypted_blob: inner,
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(source);

    Ok(envelope)
}

// ============================================
// Internal
// ============================================

/// Seals one onion layer to `hop_kem_pub`.
fn seal_layer(hop_kem_pub: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, CoreError> {
    let ephemeral = EphemeralKeyPair::generate();
    let eph_pub = ephemeral.public_key_bytes();
    let mut ecdh = ephemeral.exchange(hop_kem_pub);
    let key = derive_layer_key(&ecdh, &eph_pub, hop_kem_pub)?;
    ecdh.zeroize();

    let session = E2eSession::new(key, *hop_kem_pub);
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut nonce);
    let ciphertext = session
        .encrypt_raw(plaintext, &nonce)
        .map_err(|_| CoreError::key_generation("onion layer: AEAD seal failed"))?;

    let mut out = Vec::with_capacity(LAYER_HEADER_LEN + ciphertext.len());
    out.extend_from_slice(&ONION_MAGIC);
    out.extend_from_slice(&eph_pub);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// HKDF-SHA256 layer key derivation. `info = eph_pub || hop_kem_pub` binds the
/// key to both the ephemeral and the specific relay.
fn derive_layer_key(
    ecdh: &[u8; 32],
    eph_pub: &[u8; 32],
    hop_kem_pub: &[u8; 32],
) -> Result<[u8; 32], CoreError> {
    let mut info = [0u8; 64];
    info[..32].copy_from_slice(eph_pub);
    info[32..].copy_from_slice(hop_kem_pub);

    let hk = Hkdf::<Sha256>::new(Some(ONION_SALT), ecdh);
    let mut key = [0u8; 32];
    hk.expand(&info, &mut key)
        .map_err(|_| CoreError::key_generation("onion layer: HKDF expand failed"))?;
    Ok(key)
}

/// Encodes an `OnionHopPayload` with an explicit, language-neutral byte layout
/// (so non-Rust clients do not need a Rust serialization library):
/// ```text
///   flags:     u8        // bit0: 1 = forward (next_hop present), 0 = terminal
///   next_hop:  [u8; 32]  // present ONLY when flags bit0 == 1
///   inner_len: u32 LE    // length of inner, <= MAX_ONION_PAYLOAD_BYTES
///   inner:     [u8; inner_len]
/// ```
fn encode_payload(payload: &OnionHopPayload) -> Result<Vec<u8>, CoreError> {
    if payload.inner.len() > MAX_ONION_PAYLOAD_BYTES {
        return Err(CoreError::malformed("onion payload: inner too large"));
    }
    let mut out = Vec::with_capacity(1 + 32 + 4 + payload.inner.len());
    match &payload.next_hop {
        Some(next_hop) => {
            out.push(0x01);
            out.extend_from_slice(next_hop);
        }
        None => out.push(0x00),
    }
    out.extend_from_slice(&(payload.inner.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload.inner);
    Ok(out)
}

fn decode_payload(bytes: &[u8]) -> Result<OnionHopPayload, CoreError> {
    let mut cursor = 0usize;
    let flags = *bytes
        .get(cursor)
        .ok_or_else(|| CoreError::malformed("onion payload: missing flags"))?;
    cursor += 1;

    let next_hop = if flags & 0x01 == 0x01 {
        let slice = bytes
            .get(cursor..cursor + 32)
            .ok_or_else(|| CoreError::malformed("onion payload: truncated next_hop"))?;
        let mut next_hop = [0u8; 32];
        next_hop.copy_from_slice(slice);
        cursor += 32;
        Some(next_hop)
    } else {
        None
    };

    let len_slice = bytes
        .get(cursor..cursor + 4)
        .ok_or_else(|| CoreError::malformed("onion payload: truncated length"))?;
    let inner_len = u32::from_le_bytes(len_slice.try_into().expect("4-byte slice")) as usize;
    cursor += 4;
    if inner_len > MAX_ONION_PAYLOAD_BYTES {
        return Err(CoreError::malformed("onion payload: inner too large"));
    }

    let inner = bytes
        .get(cursor..cursor + inner_len)
        .ok_or_else(|| CoreError::malformed("onion payload: truncated inner"))?
        .to_vec();
    cursor += inner_len;
    if cursor != bytes.len() {
        return Err(CoreError::malformed("onion payload: trailing bytes"));
    }

    Ok(OnionHopPayload { next_hop, inner })
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_purpose_normalizes_canonical_values_and_legacy_aliases() {
        for value in ["message_relay", "message", "message-relay", "chat"] {
            assert_eq!(
                OnionRoutePurpose::from_wire_value(value),
                Some(OnionRoutePurpose::MessageRelay)
            );
        }
        for value in [
            "blind_vault_put",
            "blind_vault",
            "blind-vault",
            "blind-vault-put",
        ] {
            assert_eq!(
                OnionRoutePurpose::from_wire_value(value),
                Some(OnionRoutePurpose::BlindVaultPut)
            );
        }
        assert_eq!(
            OnionRoutePurpose::from_wire_value("  BLIND_VAULT_PUT  "),
            Some(OnionRoutePurpose::BlindVaultPut)
        );
        assert_eq!(OnionRoutePurpose::MessageRelay.as_str(), "message_relay");
        assert_eq!(OnionRoutePurpose::BlindVaultPut.as_str(), "blind_vault_put");
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-pull"),
            Some(OnionRoutePurpose::BlindVaultPull)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultPull.as_str(),
            "blind_vault_pull"
        );
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-lease-admission"),
            Some(OnionRoutePurpose::BlindVaultLeaseAdmission)
        );
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-put-receipt"),
            Some(OnionRoutePurpose::BlindVaultPutReceipt)
        );
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-lease-retire"),
            Some(OnionRoutePurpose::BlindVaultLeaseRetire)
        );
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-lease-renewal"),
            Some(OnionRoutePurpose::BlindVaultLeaseRenewal)
        );
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-lease-status"),
            Some(OnionRoutePurpose::BlindVaultLeaseStatus)
        );
        assert_eq!(
            OnionRoutePurpose::from_wire_value("blind-vault-lease-inventory"),
            Some(OnionRoutePurpose::BlindVaultLeaseInventory)
        );
    }

    #[test]
    fn route_purpose_rejects_unknown_values_and_declares_terminal_role() {
        assert_eq!(OnionRoutePurpose::from_wire_value(""), None);
        assert_eq!(OnionRoutePurpose::from_wire_value("future_workload"), None);
        assert_eq!(
            OnionRoutePurpose::MessageRelay.specialized_terminal_capability(),
            None
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultPut.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultPull.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultDelete.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultLeaseAdmission.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultPutReceipt.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultLeaseRetire.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultLeaseRenewal.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultLeaseStatus.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            OnionRoutePurpose::BlindVaultLeaseInventory.specialized_terminal_capability(),
            Some(NodeCapability::BlindVaultReplica)
        );
        assert_eq!(
            ONION_ROUTE_PURPOSE_VALUES,
            [
                "message_relay",
                "blind_vault_put",
                "blind_vault_pull",
                "blind_vault_delete",
                "blind_vault_lease_admission",
                "blind_vault_put_receipt",
                "blind_vault_lease_retire",
                "blind_vault_lease_renewal",
                "blind_vault_lease_status",
                "blind_vault_lease_inventory"
            ]
        );
    }

    fn hop_keypair() -> (IdentityKeyPair, OnionHop) {
        let identity = IdentityKeyPair::generate();
        let node_id = identity.public_key_bytes();
        let kem_pub = identity.x25519_public_key_bytes();
        (identity, OnionHop { node_id, kem_pub })
    }

    fn x25519_secret(identity: &IdentityKeyPair) -> StaticSecret {
        identity.to_x25519().0
    }

    #[test]
    fn two_hop_round_trip_delivers_payload() {
        let source = IdentityKeyPair::generate();
        let (entry_id, entry_hop) = hop_keypair();
        let (exit_id, exit_hop) = hop_keypair();
        let payload = b"the inner ChatEnvelope bytes".to_vec();

        let envelope = build_onion_envelope(
            &[entry_hop.clone(), exit_hop.clone()],
            &payload,
            [7u8; 16],
            4,
            1_700_000_000,
            &source,
        )
        .unwrap();

        // Outer envelope is addressed to the entry hop and is a valid onion blob.
        assert_eq!(envelope.next_hop, entry_hop.node_id);
        assert!(is_onion_blob(&envelope.encrypted_blob));

        // Entry peels → forward target is the exit hop, inner is the next layer.
        let entry_peel =
            open_onion_layer(&envelope.encrypted_blob, &x25519_secret(&entry_id)).unwrap();
        assert_eq!(entry_peel.next_hop, Some(exit_hop.node_id));
        assert!(is_onion_blob(&entry_peel.inner));

        // Exit peels → terminal, inner is the original payload.
        let exit_peel = open_onion_layer(&entry_peel.inner, &x25519_secret(&exit_id)).unwrap();
        assert_eq!(exit_peel.next_hop, None);
        assert_eq!(exit_peel.inner, payload);
    }

    #[test]
    fn three_hop_round_trip_delivers_payload() {
        let source = IdentityKeyPair::generate();
        let (a_id, a) = hop_keypair();
        let (b_id, b) = hop_keypair();
        let (c_id, c) = hop_keypair();
        let payload = b"three hop secret".to_vec();

        let env = build_onion_envelope(
            &[a.clone(), b.clone(), c.clone()],
            &payload,
            [1u8; 16],
            8,
            1_700_000_000,
            &source,
        )
        .unwrap();

        let p1 = open_onion_layer(&env.encrypted_blob, &x25519_secret(&a_id)).unwrap();
        assert_eq!(p1.next_hop, Some(b.node_id));
        let p2 = open_onion_layer(&p1.inner, &x25519_secret(&b_id)).unwrap();
        assert_eq!(p2.next_hop, Some(c.node_id));
        let p3 = open_onion_layer(&p2.inner, &x25519_secret(&c_id)).unwrap();
        assert_eq!(p3.next_hop, None);
        assert_eq!(p3.inner, payload);
    }

    #[test]
    fn wrong_hop_key_fails_to_peel() {
        let source = IdentityKeyPair::generate();
        let (_entry_id, entry_hop) = hop_keypair();
        let (_exit_id, exit_hop) = hop_keypair();
        let wrong = IdentityKeyPair::generate();

        let env =
            build_onion_envelope(&[entry_hop, exit_hop], b"x", [0u8; 16], 4, 1, &source).unwrap();

        assert!(open_onion_layer(&env.encrypted_blob, &x25519_secret(&wrong)).is_err());
    }

    #[test]
    fn tampered_ephemeral_or_ciphertext_fails() {
        let source = IdentityKeyPair::generate();
        let (entry_id, entry_hop) = hop_keypair();
        let (_exit_id, exit_hop) = hop_keypair();

        let env =
            build_onion_envelope(&[entry_hop, exit_hop], b"payload", [0u8; 16], 4, 1, &source)
                .unwrap();

        // Flip a byte inside the ephemeral public key region.
        let mut tampered_eph = env.encrypted_blob.clone();
        tampered_eph[3] ^= 0xFF;
        assert!(open_onion_layer(&tampered_eph, &x25519_secret(&entry_id)).is_err());

        // Flip a byte inside the ciphertext region.
        let mut tampered_ct = env.encrypted_blob.clone();
        let last = tampered_ct.len() - 1;
        tampered_ct[last] ^= 0xFF;
        assert!(open_onion_layer(&tampered_ct, &x25519_secret(&entry_id)).is_err());
    }

    #[test]
    fn single_hop_is_immediately_terminal() {
        let source = IdentityKeyPair::generate();
        let (exit_id, exit_hop) = hop_keypair();
        let payload = b"direct".to_vec();

        let env =
            build_onion_envelope(&[exit_hop.clone()], &payload, [0u8; 16], 2, 1, &source).unwrap();
        assert_eq!(env.next_hop, exit_hop.node_id);

        let peel = open_onion_layer(&env.encrypted_blob, &x25519_secret(&exit_id)).unwrap();
        assert_eq!(peel.next_hop, None);
        assert_eq!(peel.inner, payload);
    }

    #[test]
    fn non_onion_blob_is_detected() {
        assert!(!is_onion_blob(b""));
        assert!(!is_onion_blob(&[0x00, 0x01, 0x02]));
        assert!(is_onion_blob(&[0xA0, 0x01, 0x99]));
    }

    #[test]
    fn try_open_succeeds_with_previous_key_in_candidate_set() {
        let source = IdentityKeyPair::generate();
        let (exit_id, exit_hop) = hop_keypair();
        let wrong = IdentityKeyPair::generate();
        let payload = b"rotation grace".to_vec();

        let env = build_onion_envelope(&[exit_hop], &payload, [0u8; 16], 2, 1, &source).unwrap();

        // Correct key second in the list (simulates current=wrong, previous=correct).
        let candidates = [x25519_secret(&wrong), x25519_secret(&exit_id)];
        let peel = try_open_onion_layer(&env.encrypted_blob, &candidates).unwrap();
        assert_eq!(peel.next_hop, None);
        assert_eq!(peel.inner, payload);

        // No correct key → fail.
        let only_wrong = [x25519_secret(&wrong)];
        assert!(try_open_onion_layer(&env.encrypted_blob, &only_wrong).is_err());
    }

    #[test]
    fn payload_byte_layout_is_explicit() {
        // Terminal: flags=0x00, then inner_len(LE u32)=3, then inner.
        let terminal = OnionHopPayload {
            next_hop: None,
            inner: vec![1, 2, 3],
        };
        assert_eq!(
            encode_payload(&terminal).unwrap(),
            vec![0x00, 0x03, 0x00, 0x00, 0x00, 1, 2, 3]
        );

        // Forward: flags=0x01, next_hop(32B), inner_len=1, inner.
        let forward = OnionHopPayload {
            next_hop: Some([0xAB; 32]),
            inner: vec![9],
        };
        let encoded = encode_payload(&forward).unwrap();
        assert_eq!(encoded[0], 0x01);
        assert_eq!(&encoded[1..33], &[0xAB; 32]);
        assert_eq!(&encoded[33..37], &[0x01, 0x00, 0x00, 0x00]);
        assert_eq!(encoded[37], 9);

        // Round-trips, and rejects trailing garbage.
        assert_eq!(decode_payload(&encoded).unwrap(), forward);
        let mut trailing = encoded.clone();
        trailing.push(0xFF);
        assert!(decode_payload(&trailing).is_err());
    }

    #[test]
    fn empty_path_is_rejected() {
        let source = IdentityKeyPair::generate();
        assert!(build_onion_envelope(&[], b"x", [0u8; 16], 1, 1, &source).is_err());
    }
}
