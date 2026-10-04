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
//! The original onion layer restructures the opaque
//! `BlindRelayEnvelope::encrypted_blob` (see `chat.rs`). The envelope and all of
//! its hardened guards (Ed25519 per-hop signature, freshness window, replay
//! cache, abuse guard, routeability gate, TTL, loop detection, probes, counters)
//! are reused unchanged. `envelope.next_hop` always addresses the node that
//! receives *this* envelope; the privacy-sensitive forward target is hidden
//! inside the peeled layer.
//! [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] The additive `AXRD` contract
//! below transports that unchanged envelope through recipient-initiated
//! delivery. It defines no HTTP endpoint, queue, poller, or persistence backend.
//! [VERIFIED-ONION-FORWARD-EXPECTATION 2026-10-04 by Codex] Verified routes
//! can capture one source-local first-forward transcript commitment during
//! construction without changing the onion wire or repeating encryption.
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
//! - [ONION-ROUTE-FAILURE-DISPOSITION 2026-08-29 by Codex] Source adapters
//!   must consume [`OnionRoutePlanError::disposition`] instead of inventing
//!   independent retry policies for the same route-admission failure.
//! - [VERIFIED-ONION-TERMINAL-BINDING 2026-08-29 by Codex] Lifecycle-gated
//!   operations may inspect the authenticated terminal identity at source.
//!
//! ## Last Modified
//! [PRIVATE-BLIND-VAULT-PULL 2026-10-04 by Codex] Explicit private Pull
//! role admission reuses existing fixed-class reply features; no codec change.
//! v1.15.0-ForwardExpectationJournalBinding — Private captured blob hash.
//! [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] No wire changes.
//! v1.14.0-ReverseDeliveryContract — Bounded signed adjacent-hop claim/lease/
//! opaque-result frames and persistence transition contract (not runtime wiring)
//! v1.13.0-VerifiedTerminalBinding — Exposed source-only authenticated
//! terminal identity for lifecycle-authorized route enforcement
//! v1.12.0-RouteFailureDisposition — Centralized fail-closed route recovery
//! decisions for chat, vault, and operator-telemetry adapters
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

use crate::crypto::keys::{E2eSession, EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::protocol::chat::BlindRelayEnvelope;
use crate::protocol::discovery::{
    NodeCapability, NodeProtocolFeature, SignedNodeDescriptor,
    SignedPrivateOnionRecipientAuthorizationV1,
};

// [REVERSE-ONION-CONTRACT 2026-10-04 by Codex] Additive adjacent-hop contract.
pub mod reverse_delivery;

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
pub const ONION_ROUTE_PURPOSE_VALUES: [&str; 11] = [
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
    "anonymous_mailbox_v1",
];

/// [ANONYMOUS-MAILBOX-V1 2026-09-02 by Codex] Terminal contract for the
/// source-sealed request/response mailbox codec. Every path hop separately
/// carries the source-sealed reply contract below.
const ANONYMOUS_MAILBOX_FEATURES: [NodeProtocolFeature; 4] = [
    NodeProtocolFeature::AnonymousMailboxV1,
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// Path-wide proof contract for topology-hiding terminal replies.
const SOURCE_SEALED_REPLY_PATH_FEATURES: [NodeProtocolFeature; 2] = [
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// [BLIND-VAULT-LARGE-PULL-NEGOTIATION 2026-08-30 by Codex]
/// Path contract for the maximum fixed-size anonymous recovery response.
const BLIND_VAULT_LARGE_PULL_PATH_FEATURES: [NodeProtocolFeature; 3] = [
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
    NodeProtocolFeature::OnionBlindVaultLargePullV1,
];

/// Generic reply support plus encrypted workload failures and sealed proof.
const BLIND_VAULT_DELETE_FEATURES: [NodeProtocolFeature; 4] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
];

/// [BLIND-VAULT-LARGE-PULL-NEGOTIATION 2026-08-30 by Codex]
/// Generic reply support plus the path-wide large recovery carrier.
const BLIND_VAULT_PULL_FEATURES: [NodeProtocolFeature; 5] = [
    NodeProtocolFeature::OnionReplyV1,
    NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
    NodeProtocolFeature::BlindRelaySuccessReceiptV1,
    NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
    NodeProtocolFeature::OnionBlindVaultLargePullV1,
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
    /// Anonymous capability-key mailbox operations with sealed chat envelopes.
    AnonymousMailboxV1,
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
            "anonymous_mailbox" | "anonymous-mailbox" | "anonymous_mailbox_v1" => {
                Some(Self::AnonymousMailboxV1)
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
            Self::AnonymousMailboxV1 => ONION_ROUTE_PURPOSE_VALUES[10],
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
            Self::AnonymousMailboxV1 => Some(NodeCapability::ChatRelay),
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
            Self::BlindVaultPull => &BLIND_VAULT_PULL_FEATURES,
            Self::BlindVaultDelete => &BLIND_VAULT_DELETE_FEATURES,
            Self::BlindVaultLeaseAdmission => &BLIND_VAULT_LEASE_ADMISSION_FEATURES,
            Self::BlindVaultPutReceipt => &BLIND_VAULT_PUT_RECEIPT_FEATURES,
            Self::BlindVaultLeaseRetire => &BLIND_VAULT_LEASE_RETIRE_FEATURES,
            Self::BlindVaultLeaseRenewal => &BLIND_VAULT_LEASE_RENEWAL_FEATURES,
            Self::BlindVaultLeaseStatus => &BLIND_VAULT_LEASE_STATUS_FEATURES,
            Self::BlindVaultLeaseInventory => &BLIND_VAULT_LEASE_INVENTORY_FEATURES,
            Self::AnonymousMailboxV1 => &ANONYMOUS_MAILBOX_FEATURES,
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
            Self::BlindVaultPull => &BLIND_VAULT_LARGE_PULL_PATH_FEATURES,
            Self::BlindVaultDelete
            | Self::BlindVaultLeaseAdmission
            | Self::BlindVaultPutReceipt
            | Self::BlindVaultLeaseRetire
            | Self::BlindVaultLeaseRenewal
            | Self::BlindVaultLeaseStatus
            | Self::BlindVaultLeaseInventory => &SOURCE_SEALED_REPLY_PATH_FEATURES,
            Self::AnonymousMailboxV1 => &SOURCE_SEALED_REPLY_PATH_FEATURES,
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
    /// A private terminal route is missing its exact P-signed R/P authorization.
    #[error("private onion recipient authorization is missing or invalid")]
    MissingPrivateRecipientAuthorization,
    /// Private P must not advertise a public endpoint that could be confused
    /// with an ordinary reachable terminal role.
    #[error("private onion recipient has a public endpoint")]
    PrivateRecipientHasPublicEndpoint,
    /// Only explicit mailbox or fixed-class Pull private roles are admitted.
    #[error("private onion recipient route requires an explicit supported purpose")]
    PrivateRecipientPurposeRequired,
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

/// Recovery disposition shared by source-side onion route adapters.
///
/// [ONION-ROUTE-FAILURE-DISPOSITION 2026-08-29 by Codex] This keeps retry
/// policy in the protocol domain instead of duplicating variant groupings in
/// chat, vault, or operator-telemetry adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnionRouteFailureDisposition {
    /// Refresh discovery state or select a different current route.
    RefreshRoute,
    /// The requested path shape violates a local privacy or routing policy.
    PolicyRejected,
    /// Local identity binding or cryptographic envelope construction failed.
    LocalConstructionFailed,
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
            Self::MissingPrivateRecipientAuthorization => "missing_private_recipient_authorization",
            Self::PrivateRecipientHasPublicEndpoint => "private_recipient_public_endpoint",
            Self::PrivateRecipientPurposeRequired => "private_recipient_purpose_required",
            Self::SourceIdentityMismatch => "source_identity_mismatch",
            Self::OutsideValidityWindow => "outside_validity_window",
            Self::EnvelopeConstruction { .. } => "envelope_construction_failed",
        }
    }

    /// Returns the fail-closed recovery decision for this route failure.
    #[must_use]
    pub const fn disposition(&self) -> OnionRouteFailureDisposition {
        match self {
            Self::EmptyPath
            | Self::DescriptorRejected { .. }
            | Self::MissingCapability { .. }
            | Self::MissingProtocolFeature { .. }
            | Self::MissingX25519Kem { .. }
            | Self::MissingPublicEndpoint { .. }
            | Self::MissingPrivateRecipientAuthorization
            | Self::PrivateRecipientHasPublicEndpoint
            | Self::PrivateRecipientPurposeRequired
            | Self::OutsideValidityWindow => OnionRouteFailureDisposition::RefreshRoute,
            Self::TooManyHops { .. } | Self::DuplicateNode { .. } | Self::SourceIncluded { .. } => {
                OnionRouteFailureDisposition::PolicyRejected
            }
            Self::SourceIdentityMismatch | Self::EnvelopeConstruction { .. } => {
                OnionRouteFailureDisposition::LocalConstructionFailed
            }
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

    /// Verifies the explicit two-hop R -> private-P route role.
    ///
    /// This opt-in builder is separate from [`Self::from_signed_descriptors`]:
    /// R remains publicly reachable and SSRF-validated by its transport, while
    /// P must have no public endpoint and is admitted only by P's standalone
    /// descriptor-bound authorization. The generic public route builder is
    /// unchanged and never infers this role from a missing endpoint.
    pub fn from_signed_private_recipient_descriptors(
        source_node_id: [u8; 32],
        relay: &SignedNodeDescriptor,
        recipient: &SignedNodeDescriptor,
        authorization: &SignedPrivateOnionRecipientAuthorizationV1,
        purpose: OnionRoutePurpose,
        now: u64,
    ) -> Result<Self, OnionRoutePlanError> {
        // [PRIVATE-BLIND-VAULT-PULL 2026-10-04 by Codex] Explicit allowlist;
        // this does not authorize generic private Put/Delete/message routing.
        if !matches!(purpose, OnionRoutePurpose::AnonymousMailboxV1 | OnionRoutePurpose::BlindVaultPull) {
            return Err(OnionRoutePlanError::PrivateRecipientPurposeRequired);
        }
        relay
            .verify_at(now)
            .map_err(|_| OnionRoutePlanError::DescriptorRejected { hop_number: 1 })?;
        recipient
            .verify_at(now)
            .map_err(|_| OnionRoutePlanError::DescriptorRejected { hop_number: 2 })?;
        if relay.node_id() == source_node_id {
            return Err(OnionRoutePlanError::SourceIncluded { hop_number: 1 });
        }
        if recipient.node_id() == source_node_id {
            return Err(OnionRoutePlanError::SourceIncluded { hop_number: 2 });
        }
        if relay.node_id() == recipient.node_id() {
            return Err(OnionRoutePlanError::DuplicateNode { hop_number: 2 });
        }
        for capability in ONION_FORWARD_HOP_REQUIRED_CAPABILITIES {
            if !relay.descriptor.capabilities.contains(&capability) {
                return Err(OnionRoutePlanError::MissingCapability {
                    hop_number: 1,
                    capability,
                });
            }
        }
        if !recipient
            .descriptor
            .capabilities
            .contains(&NodeCapability::ChatRelay)
        {
            return Err(OnionRoutePlanError::MissingCapability {
                hop_number: 2,
                capability: NodeCapability::ChatRelay,
            });
        }
        // Pull requires BlindVaultReplica in addition to the base ChatRelay.
        // For mailbox this is the already-required ChatRelay role.
        if let Some(capability) = purpose.specialized_terminal_capability() {
            if !recipient.descriptor.capabilities.contains(&capability) {
                return Err(OnionRoutePlanError::MissingCapability { hop_number: 2, capability });
            }
        }
        for feature in purpose.required_terminal_protocol_features() {
            if !recipient.descriptor.advertises_protocol_feature(*feature) {
                return Err(OnionRoutePlanError::MissingProtocolFeature {
                    hop_number: 2,
                    feature: *feature,
                });
            }
        }
        for feature in purpose.required_path_protocol_features() {
            if !relay.descriptor.advertises_protocol_feature(*feature) {
                return Err(OnionRoutePlanError::MissingProtocolFeature {
                    hop_number: 1,
                    feature: *feature,
                });
            }
        }
        if relay
            .descriptor
            .public_endpoint
            .as_deref()
            .map_or(true, |endpoint| endpoint.trim().is_empty())
        {
            return Err(OnionRoutePlanError::MissingPublicEndpoint { hop_number: 1 });
        }
        if recipient.descriptor.public_endpoint.is_some() {
            return Err(OnionRoutePlanError::PrivateRecipientHasPublicEndpoint);
        }
        authorization
            .verify_at(relay, recipient, purpose.as_str(), now)
            .map_err(|_| OnionRoutePlanError::MissingPrivateRecipientAuthorization)?;
        let relay_kem = relay
            .descriptor
            .x25519_kem_public()
            .ok_or(OnionRoutePlanError::MissingX25519Kem { hop_number: 1 })?;
        let recipient_kem = recipient
            .descriptor
            .x25519_kem_public()
            .ok_or(OnionRoutePlanError::MissingX25519Kem { hop_number: 2 })?;
        Ok(Self {
            source_node_id,
            purpose,
            verified_at: now,
            valid_until: relay
                .descriptor
                .expires_at
                .min(recipient.descriptor.expires_at)
                .min(authorization.expires_at()),
            hops: vec![
                OnionHop {
                    node_id: relay.node_id(),
                    kem_pub: relay_kem,
                },
                OnionHop {
                    node_id: recipient.node_id(),
                    kem_pub: recipient_kem,
                },
            ],
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

    /// Returns the descriptor-authenticated terminal node identity.
    ///
    /// [VERIFIED-ONION-TERMINAL-BINDING 2026-08-29 by Codex] Source-side
    /// lifecycle authorization may require one exact terminal without
    /// exposing that identity to intermediate relays or opaque I/O adapters.
    #[must_use]
    pub fn terminal_node_id(&self) -> [u8; 32] {
        self.hops[self.hops.len() - 1].node_id
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
        build_onion_envelope_with_forward_expectation(
            &self.hops,
            final_payload,
            route_id,
            ttl,
            now,
            source,
        )
        .map(|(envelope, _)| envelope)
        .map_err(|source| OnionRoutePlanError::EnvelopeConstruction { source })
    }

    /// Builds one envelope and captures the exact first-forward expectation
    /// from the same single-pass encryption loop.
    ///
    /// The expectation is source-local metadata only. It contains no inner
    /// payload, relay secret, endpoint, or wire field and is `None` for a
    /// single-hop route. The caller must persist it before sending the outer
    /// envelope; it cannot be regenerated from a later randomized rebuild.
    pub fn build_envelope_with_forward_expectation(
        &self,
        final_payload: &[u8],
        route_id: [u8; 16],
        now: u64,
        source: &IdentityKeyPair,
    ) -> Result<
        (
            BlindRelayEnvelope,
            Option<VerifiedOnionForwardExpectation>,
        ),
        OnionRoutePlanError,
    > {
        // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] The new
        // expectation API must not manufacture an unusable journal binding.
        // Legacy envelope-only construction keeps its existing contract.
        if route_id == [0; 16] || now == 0
            || now.checked_add(reverse_delivery::REVERSE_ONION_ENVELOPE_LIFETIME_SECS).is_none()
        {
            return Err(OnionRoutePlanError::EnvelopeConstruction {
                source: CoreError::malformed("onion forward expectation: invalid route context"),
            });
        }
        if source.public_key_bytes() != self.source_node_id {
            return Err(OnionRoutePlanError::SourceIdentityMismatch);
        }
        if now < self.verified_at || now >= self.valid_until {
            return Err(OnionRoutePlanError::OutsideValidityWindow);
        }
        let ttl = u8::try_from(self.hops.len()).map_err(|_| OnionRoutePlanError::TooManyHops {
            max_hops: MAX_VERIFIED_ONION_ROUTE_HOPS,
        })?;
        build_onion_envelope_with_forward_expectation(
            &self.hops,
            final_payload,
            route_id,
            ttl,
            now,
            source,
        )
        .map_err(|source| OnionRoutePlanError::EnvelopeConstruction { source })
    }
}

/// Source-local proof of the exact envelope a first relay must produce when it
/// peels a multi-hop request. Raw payloads and relay secrets are deliberately
/// absent, and the type has no `Debug`/deserialization surface.
pub struct VerifiedOnionForwardExpectation {
    route_id: [u8; 16],
    first_relay_node_id: [u8; 32],
    next_hop_node_id: [u8; 32],
    ttl: u8,
    timestamp: u64,
    // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] Captured from
    // the very same inner bytes as signing_data_commitment, not a rebuild.
    encrypted_blob_hash: [u8; 32],
    signing_data_commitment: [u8; 32],
}

impl VerifiedOnionForwardExpectation {
    /// Verifies one relay-produced forwarded envelope against this pinned
    /// source-local expectation.
    pub fn verify_relay_produced_envelope(
        &self,
        envelope: &BlindRelayEnvelope,
    ) -> Result<(), CoreError> {
        if self.route_id == [0u8; 16]
            || self.first_relay_node_id == [0u8; 32]
            || self.next_hop_node_id == [0u8; 32]
            || self.first_relay_node_id == self.next_hop_node_id
            || self.ttl == 0
            || self.timestamp == 0
            || self.timestamp.checked_add(reverse_delivery::REVERSE_ONION_ENVELOPE_LIFETIME_SECS).is_none()
            || self.encrypted_blob_hash == [0u8; 32]
            || self.signing_data_commitment == [0u8; 32]
        {
            return Err(CoreError::malformed(
                "onion forward expectation: invalid pinned fields",
            ));
        }
        if envelope.route_id != self.route_id
            || envelope.next_hop != self.next_hop_node_id
            || envelope.ttl != self.ttl
            || envelope.timestamp != self.timestamp
            || Sha256::digest(&envelope.encrypted_blob).as_slice() != self.encrypted_blob_hash
        {
            return Err(CoreError::malformed(
                "onion forward expectation: field mismatch",
            ));
        }
        let commitment = Sha256::digest(envelope.signing_data());
        if commitment.as_slice() != self.signing_data_commitment {
            return Err(CoreError::malformed(
                "onion forward expectation: signing commitment mismatch",
            ));
        }
        let first_relay = IdentityPublicKey::from_bytes(&self.first_relay_node_id)
            .map_err(|_| CoreError::malformed("onion forward expectation: invalid relay"))?;
        envelope.verify_signature_from(&first_relay)
    }

    #[must_use]
    pub const fn route_id(&self) -> [u8; 16] {
        self.route_id
    }

    #[must_use]
    pub const fn first_relay_node_id(&self) -> [u8; 32] {
        self.first_relay_node_id
    }

    #[must_use]
    pub const fn next_hop_node_id(&self) -> [u8; 32] {
        self.next_hop_node_id
    }

    #[must_use]
    pub const fn ttl(&self) -> u8 {
        self.ttl
    }

    #[must_use]
    pub const fn timestamp(&self) -> u64 {
        self.timestamp
    }

    #[must_use]
    pub const fn signing_data_commitment(&self) -> [u8; 32] {
        self.signing_data_commitment
    }

    /// Source-private journal projection, captured before first-hop wrapping.
    /// Does not expose the encrypted inner frame or permit raw reconstruction.
    #[must_use]
    pub const fn encrypted_blob_hash(&self) -> [u8; 32] {
        self.encrypted_blob_hash
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
    build_onion_envelope_with_forward_expectation(
        path,
        final_payload,
        route_id,
        ttl,
        now,
        source,
    )
    .map(|(envelope, _)| envelope)
}

/// Shared one-pass construction for the legacy builder and the verified-route
/// source expectation. The expectation is computed after the first relay's
/// encrypted inner layer exists and before the outer layer is sealed.
fn build_onion_envelope_with_forward_expectation(
    path: &[OnionHop],
    final_payload: &[u8],
    route_id: [u8; 16],
    ttl: u8,
    now: u64,
    source: &IdentityKeyPair,
) -> Result<(BlindRelayEnvelope, Option<VerifiedOnionForwardExpectation>), CoreError> {
    if path.is_empty() {
        return Err(CoreError::malformed("onion path: empty"));
    }

    // Start with the raw payload; wrap one layer per hop from the exit inward.
    let mut inner = final_payload.to_vec();
    let mut expectation = None;
    for i in (0..path.len()).rev() {
        let next_hop = if i + 1 < path.len() {
            Some(path[i + 1].node_id)
        } else {
            None
        };
        if i == 0 && path.len() > 1 {
            if let Some(forward_ttl) = ttl.checked_sub(1) {
                let forward = BlindRelayEnvelope {
                    route_id,
                    next_hop: path[1].node_id,
                    ttl: forward_ttl,
                    encrypted_blob: inner.clone(),
                    timestamp: now,
                    signature: [0u8; 64],
                };
                let digest = Sha256::digest(forward.signing_data());
                let mut signing_data_commitment = [0u8; 32];
                signing_data_commitment.copy_from_slice(&digest);
                expectation = Some(VerifiedOnionForwardExpectation {
                    route_id,
                    first_relay_node_id: path[0].node_id,
                    next_hop_node_id: path[1].node_id,
                    ttl: forward_ttl,
                    timestamp: now,
                    encrypted_blob_hash: Sha256::digest(&forward.encrypted_blob).into(),
                    signing_data_commitment,
                });
            }
        }
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

    Ok((envelope, expectation))
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
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine as _;

    use super::*;
    use crate::protocol::anonymous_mailbox::{
        encode_anonymous_mailbox_terminal_frame, AnonymousMailboxPutV1,
        AnonymousMailboxRouteRequestV1, AnonymousMailboxSourceTerminalCarrierV1,
        AnonymousMailboxTerminalFrameV1, MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES,
        MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES,
    };
    use crate::protocol::chat::{
        encode_blind_relay_envelope, validate_blind_relay_envelope_size, BlindRelayDeliveryReceipt,
    };
    use crate::protocol::discovery::NodeDescriptor;
    use crate::protocol::memchain::{encode_memchain, MemChainMessage};

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
        assert_eq!(
            OnionRoutePurpose::from_wire_value("anonymous-mailbox"),
            Some(OnionRoutePurpose::AnonymousMailboxV1)
        );
        assert_eq!(
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            "anonymous_mailbox_v1"
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
            OnionRoutePurpose::AnonymousMailboxV1.specialized_terminal_capability(),
            Some(NodeCapability::ChatRelay)
        );
        assert!(OnionRoutePurpose::AnonymousMailboxV1
            .required_terminal_protocol_features()
            .contains(&NodeProtocolFeature::AnonymousMailboxV1));
        assert_ne!(
            BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
                b"sealed",
                OnionRoutePurpose::AnonymousMailboxV1
            ),
            BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
                b"sealed",
                OnionRoutePurpose::MessageRelay
            ),
            "wrong purpose must not verify as an anonymous mailbox route"
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
                "blind_vault_lease_inventory",
                "anonymous_mailbox_v1"
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

    fn private_route_descriptor(
        identity: &IdentityKeyPair,
        endpoint: Option<&str>,
        capabilities: Vec<NodeCapability>,
        features: &[NodeProtocolFeature],
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            7,
            1_700_000_000,
            1_700_003_600,
            "test",
        )
        .with_x25519_kem(identity.x25519_public_key_bytes());
        descriptor.public_endpoint = endpoint.map(str::to_owned);
        descriptor.capabilities = capabilities;
        descriptor = descriptor.with_protocol_features(features.iter().copied());
        SignedNodeDescriptor::sign(descriptor, identity).unwrap()
    }

    // [PRIVATE-BLIND-VAULT-PULL 2026-10-04 by Codex] Authored, unexecuted.
    #[test]
    fn private_pull_builder_enforces_each_role_feature_and_public_boundary() {
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let relay = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
        let r = private_route_descriptor(&relay, Some("relay.example:443"),
            ONION_FORWARD_HOP_REQUIRED_CAPABILITIES.to_vec(), &BLIND_VAULT_LARGE_PULL_PATH_FEATURES);
        let p = private_route_descriptor(&recipient, None,
            vec![NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica], &BLIND_VAULT_PULL_FEATURES);
        let auth = SignedPrivateOnionRecipientAuthorizationV1::new_signed(&r, &p,
            "blind_vault_pull", 1_700_000_100, 1_700_001_000, &recipient).unwrap();
        let build = |r: &SignedNodeDescriptor, p: &SignedNodeDescriptor, purpose| {
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(source.public_key_bytes(),
                r, p, &auth, purpose, 1_700_000_500)
        };
        let route = build(&r, &p, OnionRoutePurpose::BlindVaultPull).unwrap();
        assert_eq!(route.purpose(), OnionRoutePurpose::BlindVaultPull);
        assert_eq!(route.hop_count(), 2);
        assert_eq!(route.valid_until(), 1_700_001_000);
        assert!(VerifiedOnionRoute::from_signed_descriptors(source.public_key_bytes(),
            [&r, &p], OnionRoutePurpose::BlindVaultPull, 1_700_000_500).is_err());
        for removed in BLIND_VAULT_PULL_FEATURES {
            let features: Vec<_> = BLIND_VAULT_PULL_FEATURES.into_iter().filter(|f| *f != removed).collect();
            let missing = private_route_descriptor(&recipient, None,
                vec![NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica], &features);
            assert!(matches!(build(&r, &missing, OnionRoutePurpose::BlindVaultPull),
                Err(OnionRoutePlanError::MissingProtocolFeature { hop_number: 2, feature }) if feature == removed));
        }
        for removed in BLIND_VAULT_LARGE_PULL_PATH_FEATURES {
            let features: Vec<_> = BLIND_VAULT_LARGE_PULL_PATH_FEATURES.into_iter().filter(|f| *f != removed).collect();
            let missing = private_route_descriptor(&relay, Some("relay.example:443"),
                ONION_FORWARD_HOP_REQUIRED_CAPABILITIES.to_vec(), &features);
            assert!(matches!(build(&missing, &p, OnionRoutePurpose::BlindVaultPull),
                Err(OnionRoutePlanError::MissingProtocolFeature { hop_number: 1, feature }) if feature == removed));
        }
        for removed in [NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica] {
            let caps = [NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica].into_iter()
                .filter(|c| *c != removed).collect();
            let missing = private_route_descriptor(&recipient, None, caps, &BLIND_VAULT_PULL_FEATURES);
            assert!(matches!(build(&r, &missing, OnionRoutePurpose::BlindVaultPull),
                Err(OnionRoutePlanError::MissingCapability { hop_number: 2, capability }) if capability == removed));
        }
        for removed in ONION_FORWARD_HOP_REQUIRED_CAPABILITIES {
            let caps = ONION_FORWARD_HOP_REQUIRED_CAPABILITIES.into_iter().filter(|c| *c != removed).collect();
            let missing = private_route_descriptor(&relay, Some("relay.example:443"), caps, &BLIND_VAULT_LARGE_PULL_PATH_FEATURES);
            assert!(matches!(build(&missing, &p, OnionRoutePurpose::BlindVaultPull),
                Err(OnionRoutePlanError::MissingCapability { hop_number: 1, capability }) if capability == removed));
        }
        for purpose in [OnionRoutePurpose::MessageRelay, OnionRoutePurpose::BlindVaultPut,
            OnionRoutePurpose::BlindVaultDelete, OnionRoutePurpose::BlindVaultLeaseAdmission] {
            assert!(matches!(build(&r, &p, purpose), Err(OnionRoutePlanError::PrivateRecipientPurposeRequired)));
        }
        let public_p = private_route_descriptor(&recipient, Some("recipient.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica], &BLIND_VAULT_PULL_FEATURES);
        assert!(matches!(build(&r, &public_p, OnionRoutePurpose::BlindVaultPull),
            Err(OnionRoutePlanError::PrivateRecipientHasPublicEndpoint)));
    }

    #[test]
    fn private_pull_fixed_class_chain_verifies_page_signature_and_source_seal() {
        use crate::crypto::IdentityPublicKey;
        use crate::protocol::blind_vault::{BlindVaultOnionPullSession, BlindVaultPullRequest,
            BlindVaultPullResponse, BlindVaultRecoveredObject, BlindVaultFrame,
            encode_blind_vault_frame, BLIND_VAULT_PROTOCOL_VERSION, BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES};
        use crate::protocol::onion_reply::{decode_onion_reply_request, encode_onion_sealed_response,
            seal_onion_reply, OnionReplyProofMode};
        use reverse_delivery::ReverseOnionFrameV1;
        use sha2::{Digest, Sha256};
        const NOW: u64 = 1_700_000_500;
        const ROUTE: [u8; 16] = [0x31; 16];
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let relay = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
        let r = private_route_descriptor(&relay, Some("relay.example:443"),
            ONION_FORWARD_HOP_REQUIRED_CAPABILITIES.to_vec(), &BLIND_VAULT_LARGE_PULL_PATH_FEATURES);
        let p = private_route_descriptor(&recipient, None,
            vec![NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica], &BLIND_VAULT_PULL_FEATURES);
        let auth = SignedPrivateOnionRecipientAuthorizationV1::new_signed(&r, &p,
            "blind_vault_pull", NOW, NOW + 500, &recipient).unwrap();
        let route = VerifiedOnionRoute::from_signed_private_recipient_descriptors(source.public_key_bytes(),
            &r, &p, &auth, OnionRoutePurpose::BlindVaultPull, NOW).unwrap();
        let (request, session) = BlindVaultOnionPullSession::prepare(ROUTE, recipient.public_key_bytes(),
            BlindVaultPullRequest { version: BLIND_VAULT_PROTOCOL_VERSION, lease_id: [7; 32],
                read_capability: [8; 32], continuation_cursor: vec![], limit: 1 }).unwrap();
        let restart = session.seal_restart(&source, ROUTE, recipient.public_key_bytes(), &request).unwrap();
        let (outer, expectation) = route.build_envelope_with_forward_expectation(&request, ROUTE, NOW, &source).unwrap();
        outer.verify_signature_from(&IdentityPublicKey::from_bytes(&source.public_key_bytes()).unwrap()).unwrap();
        assert_eq!(outer.ttl, 2);
        let peeled_r = open_onion_layer(&outer.encrypted_blob, &x25519_secret(&relay)).unwrap();
        assert_eq!(peeled_r.next_hop, Some(recipient.public_key_bytes()));
        let retained = crate::protocol::chat::BlindRelayEnvelope { route_id: ROUTE,
            next_hop: recipient.public_key_bytes(), ttl: outer.ttl - 1, timestamp: outer.timestamp,
            encrypted_blob: peeled_r.inner, signature: [0; 64] }.sign_with(&relay);
        expectation.unwrap().verify_relay_produced_envelope(&retained).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [9; 16], NOW, NOW + 30, &recipient).unwrap();
        let lease = ReverseOnionFrameV1::lease(&claim, &retained, [10; 16], route.valid_until(), NOW, &relay).unwrap();
        lease.verify_recipient_lease(&claim, relay.public_key_bytes(), recipient.public_key_bytes(), NOW).unwrap();
        let peeled_p = open_onion_layer(&retained.encrypted_blob, &x25519_secret(&recipient)).unwrap();
        assert!(peeled_p.next_hop.is_none());
        assert_eq!(peeled_p.inner, request);
        let reply_request = decode_onion_reply_request(&request).unwrap();
        assert_eq!(reply_request.proof_mode(), OnionReplyProofMode::SourceSealedTerminalProof);
        let ciphertext = vec![0x5a; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0]];
        let object = BlindVaultRecoveredObject { object_id: [11; 32],
            ciphertext_commitment: Sha256::digest(&ciphertext).into(), ciphertext,
            expires_at_ms: (NOW + 1000) * 1000 };
        let mut page = BlindVaultPullResponse::new([7; 32], vec![object], vec![],
            (NOW + 1) * 1000, recipient.public_key_bytes());
        page.sign(&recipient).unwrap();
        page.validate_and_verify(&IdentityPublicKey::from_bytes(&recipient.public_key_bytes()).unwrap()).unwrap();
        let encode_reply = |page: BlindVaultPullResponse| {
            let payload = encode_blind_vault_frame(&BlindVaultFrame::PullResponse(page)).unwrap();
            encode_onion_sealed_response(&seal_onion_reply(ROUTE, &reply_request, &payload, &recipient).unwrap()).unwrap()
        };
        let sealed = encode_reply(page.clone());
        let result = ReverseOnionFrameV1::result(&claim, &lease, &sealed, route.valid_until(), NOW + 1, &recipient).unwrap();
        let verified_payload = result.verify_result(&claim, &lease, route.valid_until(), NOW + 1).unwrap();
        assert_eq!(session.open(verified_payload).unwrap(), page);

        // Test-only restores permit independent negative cases; production
        // must first commit its one-shot Opening CAS before any restore/open.
        let restore = || BlindVaultOnionPullSession::restore_restart(&source, &restart,
            ROUTE, recipient.public_key_bytes(), &request).unwrap();
        let mut bad_page = page;
        bad_page.signature[0] ^= 1;
        // Valid outer source seal does not excuse an invalid inner page.
        assert!(restore().open(&encode_reply(bad_page)).is_err());
        let mut bad_seal = sealed;
        *bad_seal.last_mut().unwrap() ^= 1;
        assert!(restore().open(&bad_seal).is_err());
    }

    #[test]
    fn private_recipient_route_requires_signed_role_and_keeps_p_endpointless() {
        let source = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
        let relay = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
        let relay_descriptor = private_route_descriptor(
            &relay,
            Some("relay.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            &[
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ],
        );
        let recipient_descriptor = private_route_descriptor(
            &recipient,
            None,
            vec![NodeCapability::ChatRelay],
            &ANONYMOUS_MAILBOX_FEATURES,
        );
        let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay_descriptor,
            &recipient_descriptor,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_001_000,
            &recipient,
        )
        .unwrap();
        let route = VerifiedOnionRoute::from_signed_private_recipient_descriptors(
            source.public_key_bytes(),
            &relay_descriptor,
            &recipient_descriptor,
            &authorization,
            OnionRoutePurpose::AnonymousMailboxV1,
            1_700_000_500,
        )
        .unwrap();
        assert_eq!(route.hop_count(), 2);
        assert_eq!(route.entry_node_id(), relay.public_key_bytes());
        assert_eq!(route.terminal_node_id(), recipient.public_key_bytes());
        assert_eq!(route.valid_until(), 1_700_001_000);
        assert!(route
            .build_envelope(b"AMST", [7; 16], 1_700_000_500, &source)
            .is_ok());

        let public_recipient_descriptor = private_route_descriptor(
            &recipient,
            Some("recipient.example:443"),
            vec![NodeCapability::ChatRelay],
            &ANONYMOUS_MAILBOX_FEATURES,
        );
        let public_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay_descriptor,
            &public_recipient_descriptor,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_001_000,
            &recipient,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &relay_descriptor,
                &public_recipient_descriptor,
                &public_authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "private_recipient_public_endpoint"
        );
    }

    #[test]
    fn private_recipient_route_rejects_generic_and_stale_or_substituted_inputs() {
        let source = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
        let relay = IdentityKeyPair::from_bytes(&[0x65; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[0x66; 32]).unwrap();
        let relay_descriptor = private_route_descriptor(
            &relay,
            Some("relay.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            &[
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ],
        );
        let recipient_descriptor = private_route_descriptor(
            &recipient,
            None,
            vec![NodeCapability::ChatRelay],
            &ANONYMOUS_MAILBOX_FEATURES,
        );
        let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay_descriptor,
            &recipient_descriptor,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_001_000,
            &recipient,
        )
        .unwrap();

        assert_eq!(
            VerifiedOnionRoute::from_signed_descriptors(
                source.public_key_bytes(),
                [&relay_descriptor, &recipient_descriptor],
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_public_endpoint"
        );

        let mut authorization_bytes = authorization.encode_canonical().unwrap();
        *authorization_bytes.last_mut().unwrap() ^= 1;
        let tampered = SignedPrivateOnionRecipientAuthorizationV1::decode_canonical(
            &authorization_bytes,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &relay_descriptor,
                &recipient_descriptor,
                &tampered,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_private_recipient_authorization"
        );

        let mut rotated_body = relay_descriptor.descriptor.clone();
        rotated_body.sequence += 1;
        let rotated_relay = SignedNodeDescriptor::sign(rotated_body, &relay).unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &rotated_relay,
                &recipient_descriptor,
                &authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_private_recipient_authorization"
        );

        let other_recipient = IdentityKeyPair::from_bytes(&[0x67; 32]).unwrap();
        let other_descriptor = private_route_descriptor(
            &other_recipient,
            None,
            vec![NodeCapability::ChatRelay],
            &ANONYMOUS_MAILBOX_FEATURES,
        );
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &relay_descriptor,
                &other_descriptor,
                &authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_private_recipient_authorization"
        );

        let expired_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay_descriptor,
            &recipient_descriptor,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_000_400,
            &recipient,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &relay_descriptor,
                &recipient_descriptor,
                &expired_authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_private_recipient_authorization"
        );

        let mut expired_body = recipient_descriptor.descriptor.clone();
        expired_body.expires_at = 1_700_000_400;
        let expired_recipient = SignedNodeDescriptor::sign(expired_body, &recipient).unwrap();
        let expired_descriptor_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay_descriptor,
            &expired_recipient,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_000_300,
            &recipient,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &relay_descriptor,
                &expired_recipient,
                &expired_descriptor_authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "descriptor_rejected"
        );

        let source_relay = private_route_descriptor(
            &source,
            Some("relay.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            &[
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ],
        );
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &source_relay,
                &recipient_descriptor,
                &authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "source_included"
        );
    }

    #[test]
    fn private_recipient_route_rejects_same_node_kem_and_feature_gaps() {
        let source = IdentityKeyPair::from_bytes(&[0x68; 32]).unwrap();
        let relay = IdentityKeyPair::from_bytes(&[0x69; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[0x6a; 32]).unwrap();
        let relay_descriptor = private_route_descriptor(
            &relay,
            Some("relay.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            &[
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ],
        );
        let recipient_descriptor = private_route_descriptor(
            &recipient,
            None,
            vec![NodeCapability::ChatRelay],
            &ANONYMOUS_MAILBOX_FEATURES,
        );
        let mut no_kem_body = relay_descriptor.descriptor.clone();
        no_kem_body.kem_alg = 0;
        no_kem_body.kem_public = [0; 32];
        let no_kem_relay = SignedNodeDescriptor::sign(no_kem_body, &relay).unwrap();
        let no_kem_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &no_kem_relay,
            &recipient_descriptor,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_001_000,
            &recipient,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &no_kem_relay,
                &recipient_descriptor,
                &no_kem_authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_x25519_kem"
        );

        let no_feature_relay = private_route_descriptor(
            &relay,
            Some("relay.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            &[],
        );
        let no_feature_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &no_feature_relay,
            &recipient_descriptor,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_001_000,
            &recipient,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &no_feature_relay,
                &recipient_descriptor,
                &no_feature_authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "missing_protocol_feature"
        );

        let same = IdentityKeyPair::from_bytes(&[0x6b; 32]).unwrap();
        let same_relay = private_route_descriptor(
            &same,
            Some("relay.example:443"),
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            &[
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ],
        );
        let mut same_recipient_body = same_relay.descriptor.clone();
        same_recipient_body.public_endpoint = None;
        same_recipient_body = same_recipient_body.with_protocol_features(ANONYMOUS_MAILBOX_FEATURES);
        let same_recipient = SignedNodeDescriptor::sign(same_recipient_body, &same).unwrap();
        let same_authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &same_relay,
            &same_recipient,
            OnionRoutePurpose::AnonymousMailboxV1.as_str(),
            1_700_000_100,
            1_700_001_000,
            &same,
        )
        .unwrap();
        assert_eq!(
            VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                source.public_key_bytes(),
                &same_relay,
                &same_recipient,
                &same_authorization,
                OnionRoutePurpose::AnonymousMailboxV1,
                1_700_000_500,
            )
            .unwrap_err()
            .reason_bucket(),
            "duplicate_node"
        );
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
    fn verified_route_captures_and_verifies_first_forward_expectation() {
        let source = IdentityKeyPair::from_bytes(&[0x71; 32]).expect("source");
        let (entry_id, entry_hop) = hop_keypair();
        let (_exit_id, exit_hop) = hop_keypair();
        let route = VerifiedOnionRoute {
            source_node_id: source.public_key_bytes(),
            purpose: OnionRoutePurpose::MessageRelay,
            verified_at: 100,
            valid_until: 200,
            hops: vec![entry_hop.clone(), exit_hop.clone()],
        };
        let route_id = [0x72; 16];
        let now = 150;
        let (outer, expectation) = route
            .build_envelope_with_forward_expectation(b"opaque", route_id, now, &source)
            .expect("verified route envelope");
        let expectation = expectation.expect("multi-hop expectation");
        assert_eq!(expectation.route_id(), route_id);
        assert_eq!(expectation.first_relay_node_id(), entry_hop.node_id);
        assert_eq!(expectation.next_hop_node_id(), exit_hop.node_id);
        assert_eq!(expectation.ttl(), outer.ttl - 1);
        assert_eq!(expectation.timestamp(), now);
        assert_ne!(expectation.signing_data_commitment(), [0; 32]);

        let peeled = open_onion_layer(&outer.encrypted_blob, &x25519_secret(&entry_id))
            .expect("entry peel");
        let forwarded = BlindRelayEnvelope {
            route_id,
            next_hop: exit_hop.node_id,
            ttl: outer.ttl - 1,
            encrypted_blob: peeled.inner,
            timestamp: now,
            signature: [0; 64],
        }
        .sign_with(&entry_id);
        // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] Proves the
        // getter refers to THIS encryption pass, not fresh randomized wrapping.
        assert_eq!(expectation.encrypted_blob_hash(),
            <[u8; 32]>::from(Sha256::digest(&forwarded.encrypted_blob)));
        expectation
            .verify_relay_produced_envelope(&forwarded)
            .expect("forward expectation");

        for tampered in [
            {
                let mut value = forwarded.clone();
                value.route_id[0] ^= 1;
                value
            },
            {
                let mut value = forwarded.clone();
                value.next_hop[0] ^= 1;
                value
            },
            {
                let mut value = forwarded.clone();
                value.ttl = value.ttl.saturating_sub(1);
                value
            },
            {
                let mut value = forwarded.clone();
                value.timestamp += 1;
                value
            },
            {
                let mut value = forwarded.clone();
                value.encrypted_blob[0] ^= 1;
                value
            },
            {
                let mut value = forwarded.clone();
                value.signature[0] ^= 1;
                value
            },
        ] {
            assert!(expectation.verify_relay_produced_envelope(&tampered).is_err());
        }
    }

    #[test]
    fn verified_route_single_hop_has_no_forward_expectation_and_rejects_invalid_context() {
        let source = IdentityKeyPair::from_bytes(&[0x73; 32]).expect("source");
        let (entry_id, entry_hop) = hop_keypair();
        let route = VerifiedOnionRoute {
            source_node_id: source.public_key_bytes(),
            purpose: OnionRoutePurpose::MessageRelay,
            verified_at: 100,
            valid_until: 200,
            hops: vec![entry_hop],
        };
        let (outer, expectation) = route
            .build_envelope_with_forward_expectation(b"opaque", [0x74; 16], 150, &source)
            .expect("single-hop route");
        assert!(expectation.is_none());
        assert!(route
            .build_envelope_with_forward_expectation(b"opaque", [0x74; 16], 99, &source)
            .is_err());
        let wrong_source = IdentityKeyPair::from_bytes(&[0x75; 32]).expect("wrong source");
        assert!(route
            .build_envelope_with_forward_expectation(b"opaque", [0x74; 16], 150, &wrong_source)
            .is_err());
        assert!(open_onion_layer(&outer.encrypted_blob, &x25519_secret(&entry_id)).is_ok());
    }

    // [REVERSE-ONION-TYPED-EXPECTATION 2026-10-04 by Codex] Authored only.
    #[test]
    fn forward_expectation_rejects_zero_and_overflow_context_without_changing_legacy_builder() {
        let source = IdentityKeyPair::from_bytes(&[0x76; 32]).unwrap();
        let (_, entry) = hop_keypair(); let (_, exit) = hop_keypair();
        let route = VerifiedOnionRoute {
            source_node_id: source.public_key_bytes(), purpose: OnionRoutePurpose::MessageRelay,
            verified_at: 0, valid_until: u64::MAX, hops: vec![entry, exit],
        };
        for (id, now) in [([0; 16], 150), ([1; 16], 0), ([1; 16], u64::MAX - 1)] {
            assert!(route.build_envelope_with_forward_expectation(b"opaque", id, now, &source).is_err());
        }
        assert!(route.build_envelope(b"legacy", [0; 16], 0, &source).is_ok());
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
    fn maximum_mailbox_route_fits_three_onion_layers_and_legacy_relay_caps() {
        let source = IdentityKeyPair::from_bytes(&[0x91; 32]).expect("source");
        let depositor = IdentityKeyPair::from_bytes(&[0x94; 32]).expect("depositor");
        let (_, entry) = hop_keypair();
        let (_, middle) = hop_keypair();
        let (_, terminal) = hop_keypair();
        let maximum_put = AnonymousMailboxPutV1::new(
            [0x95; 32],
            [0x96; 16],
            vec![0xa5; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES],
            1_800_000_000,
            1_800_000_100,
            &depositor,
        )
        .expect("maximum Put");
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(
            &AnonymousMailboxTerminalFrameV1::Put(maximum_put),
        )
        .expect("maximum terminal frame");
        let (carrier, _source_session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            [0x92; 16],
            terminal.node_id,
            terminal_frame,
        )
        .expect("maximum source terminal carrier");
        let sealed_terminal_frame = carrier.encode().expect("maximum carrier bytes");
        assert!(sealed_terminal_frame.len() < MAX_ANONYMOUS_MAILBOX_SEALED_TERMINAL_BYTES);
        let request = AnonymousMailboxRouteRequestV1::signed(
            [0x92; 16],
            terminal.node_id,
            sealed_terminal_frame,
            1_800_000_000,
            &source,
        )
        .expect("maximum canonical route");
        let payload = encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(request))
            .expect("MemChain route");
        assert!(payload.len() < MAX_ONION_PAYLOAD_BYTES);

        let envelope = build_onion_envelope(
            &[entry, middle, terminal],
            &payload,
            [0x93; 16],
            3,
            1_800_000_000,
            &source,
        )
        .expect("maximum three-hop route");
        validate_blind_relay_envelope_size(&envelope).expect("legacy relay blob cap");
        let encoded = encode_blind_relay_envelope(&envelope).expect("legacy relay frame cap");
        assert!(encoded.len() < 256 * 1024);
        // [ANONYMOUS-MAILBOX-SOURCE-CARRIER 2026-09-02 by Codex] This is the
        // real largest admitted Put wrapped by the canonical request carrier,
        // then the frozen MemChain, three-hop onion, blind-relay, and base64
        // codecs. It must remain below their unchanged production caps.
        assert_eq!(payload.len(), 163_168);
        assert_eq!(envelope.encrypted_blob.len(), 163_469);
        assert_eq!(encoded.len(), 163_598);
        assert_eq!(STANDARD.encode(encoded).len(), 218_132);
        assert!(matches!(
            AnonymousMailboxPutV1::new(
                [0x95; 32],
                [0x96; 16],
                vec![0xa5; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES + 1],
                1_800_000_000,
                1_800_000_100,
                &depositor,
            ),
            Err(crate::protocol::anonymous_mailbox::AnonymousMailboxProtocolError::TooLarge)
        ));
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
