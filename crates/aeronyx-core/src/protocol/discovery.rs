// ============================================================================
// File: crates/aeronyx-core/src/protocol/discovery.rs
// ============================================================================
//! # Node Discovery Protocol Types
//!
//! ## Creation Reason
//! Provides the signed node descriptor types used by AeroNyx nodes to advertise
//! capabilities, endpoints, capacity hints, and expiry windows before any
//! cross-node gossip or encrypted relay logic is enabled.
//!
//! ## Main Functionality
//! - `NodeDescriptor`: canonical node metadata signed by the node identity key
//! - `SignedNodeDescriptor`: descriptor plus Ed25519 signature
//! - `NodeCapability`: protocol-level capability flags
//! - `NodeProtocolFeature`: backward-compatible signed wire-feature negotiation
//! - `NodePolicy`: public relay policy hints, including no-exit default
//! - `NodeCapacity`: coarse capacity hints for peer selection
//! - `NodeBootstrapSnapshot`: JSON-friendly bootstrap list of signed descriptors
//! - `NodeDiscoveryMessage`: bounded gossip message envelope for peer sync
//! - Signature-only descriptor verification for local peer-cache retention
//! - `DirectoryCommitmentBlockV1`: signed, hash-linked commitments to public
//!   node descriptor events without embedding endpoint or operator metadata
//! - `DirectoryDescriptorInclusionProofV1`: a compact producer-signed Merkle
//!   path proving one exact authenticated descriptor commitment is in one
//!   independently selected Directory block
//! - `DirectoryObservationCheckpointV1`: observer-signed, hash-linked evidence
//!   binding exact producer tips to a recomputable multi-source overlap root
//! - `DirectoryObservationCertificateV1`: a bounded portable package combining
//!   one checkpoint with independently signed accepted witness receipts
//! - `RouteDomainAttestationCertificateV1`: pinned-attestor evidence for one
//!   opaque node-to-route-domain assignment with bounded validity
//! - `DirectorySyncMessage`: authenticated, bounded node-to-node transport for
//!   serving one producer's tip, block ranges, descriptor objects, and
//!   exact descriptor-inclusion proofs plus independently recomputed
//!   observation-checkpoint witness receipts
//! - Opaque policy-head anchor frames that let independent pinned witnesses
//!   retain rollback evidence without receiving policy members or endpoints
//! - Authenticated observation-certificate exchange frames that let pinned
//!   nodes transport exact portable evidence without making it public
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Bounded witness-carrier frames that
//!   preserve the exact observer and witness signatures across one transport hop
//! - Replica-carrier frames that transport already audited producer evidence
//!   without allowing the carrier to replace the producer's signatures,
//!   including compact exact-block descriptor inclusion proofs
//! - Shared bounded fixed-integer codec policy for canonical control-plane
//!   frames and descriptor signing bytes
//!
//! ## Dependencies
//! - crates/aeronyx-core/src/crypto/keys.rs: IdentityKeyPair / IdentityPublicKey
//! - bincode: deterministic descriptor bytes for signing
//! - serde: JSON/bincode compatibility for future bootstrap snapshots
//!
//! ## Main Logical Flow
//! 1. Node builds a `NodeDescriptor` with its public identity and capabilities
//! 2. Node signs `descriptor.signing_bytes()` with `IdentityKeyPair`
//! 3. Peers call `SignedNodeDescriptor::verify_at(now)` before treating it as live
//! 4. Server-side `PeerStore` may use `verify_signature()` only to retain
//!    expired cache records as non-routeable history
//! 5. Bootstrap snapshots carry a bounded list of signed descriptors for
//!    first-contact peer discovery
//! 6. Gossip messages exchange snapshot requests/responses and descriptor
//!    announcements without depending on a specific transport
//! 7. Directory blocks commit to authenticated descriptors using stable hashes
//! 8. Observation checkpoints bind complete configured producer-tip sets to a
//!    locally recomputable overlap root without claiming consensus or finality
//! 9. A pinned peer may witness an exact checkpoint only after independently
//!    recomputing its producer prefixes and overlap root from local replicas
//! 10. A portable observation certificate may aggregate those exact signed
//!     receipts for offline verification without claiming consensus or finality
//! 11. A pinned carrier may serve an audited producer replica when direct
//!     producer admission is unavailable; receivers still verify both layers
//! 12. A pinned witness may retain one monotonic opaque policy head per
//!     observer and return a signed receipt without learning policy members
//! 13. A pinned node may request the latest portable observation certificate;
//!     the response binds the exact certificate bytes and SHA-256 digest
//! 14. A pinned carrier may forward one exact observer-signed witness request
//!     and return one exact witness-signed response without becoming authority
//! 15. A light verifier may validate one descriptor inclusion path against an
//!     independently trusted producer and exact Directory block hash
//! 16. A pinned peer may request that exact proof without downloading every
//!     commitment or descriptor object from the producer's chain
//! 17. A verified recovery peer may request the same proof from an audited
//!     carrier when the original producer is unavailable; the receiver still
//!     verifies the original producer signature and its selected block hash
//! 18. [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] A peer may announce
//!     one exact descriptor proof, but receivers admit it only against an
//!     independently retained producer/block anchor
//! 19. [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] An operator may verify a
//!     bounded portable certificate against its own pinned attestor quorum
//!     before using one opaque route-domain assignment for path diversity
//!
//! ## Important Note for Next Developer
//! - Do not put private keys, client IPs, destination metadata, DNS contents,
//!   packet payloads, browsing history, voucher secrets, or wallet-level
//!   traffic in this descriptor.
//! - `bincode` field order is part of the signing contract. Add new fields
//!   only at the end and keep backward compatibility in mind.
//! - Default public policy is no-exit. Future onion routing must opt into any
//!   exit behavior through a separate reviewed policy.
//! - Directory blocks are integrity evidence, not financial consensus. Never
//!   add user identities, traffic facts, routes, message ids, payloads, memory
//!   records, or client metadata to a directory commitment.
//! - [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] Inclusion proofs establish
//!   only that one producer signed one exact block containing one authenticated
//!   descriptor commitment. The caller must independently pin the producer and
//!   exact block hash. A proof is not canonical-chain selection, transaction
//!   inclusion, quorum, consensus, global finality, or user-activity evidence.
//!   Its peer wire variants must remain append-only and pinned-peer-only until
//!   a separate privacy and abuse review explicitly widens admission.
//! - Observation checkpoints are signed local evidence, not votes, fork choice,
//!   quorum certificates, global consensus, or finality.
//! - A checkpoint witness receipt proves one external node independently
//!   recomputed one exact checkpoint. It is not a vote, quorum, or finality.
//! - [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] A portable
//!   certificate contains public node identities required to verify signatures.
//!   Keep distribution operator-scoped until an explicit privacy review allows
//!   broader publication. Never label receipt thresholds as consensus/finality.
//! - [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] Certificate transport is
//!   restricted to authenticated pinned peers. Append wire variants only;
//!   changing the existing enum order breaks mixed-version bincode peers.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] A carrier signature authenticates
//!   one bounded transport envelope only. It must never replace the exact target
//!   witness signature or authorize recursive forwarding.
//! - A policy-head anchor proves only that one witness retained an opaque
//!   observer-signed epoch/digest at a time. It is not policy approval, a vote,
//!   validator membership, consensus, governance, or finality.
//! - A replica carrier proves transport of its audited copy. It cannot author,
//!   rewrite, finalize, or select the producer's signed chain.
//! - [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] A replica proof response
//!   has two independent signature layers: the original producer-signed block
//!   inside the proof and the carrier-signed transport envelope. The carrier
//!   signature grants availability only and must never become producer,
//!   checkpoint, witness, policy, consensus, fork-choice, or finality authority.
//! - [MIRROR-CAPABILITY 2026-07-24 by Codex] New capability variants must be
//!   appended, never reordered. Advertise `DirectoryMirrorCarrier` only after
//!   the operator has enabled the staged mixed-version rollout gate.
//! - [BLIND-VAULT-REPLICA-CAPABILITY 2026-08-10 by Codex] `BlindVaultReplica`
//!   is append-only and rollout-gated. It means the signed peer endpoint can
//!   accept admitted anonymous ciphertext replicas; it does not assert data
//!   possession, availability, trust, operator independence, or consensus.
//! - [BOUNDED-DISCOVERY-CODEC 2026-07-24 by Codex] Discovery and Directory
//!   Sync frames are canonical control-plane messages. Keep strict trailing
//!   rejection and the complete-input size preflight in the shared codec.
//! - [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] A valid certificate proves
//!   only that the verifier's pinned identities signed one opaque assignment
//!   for a bounded interval. It does not prove operator independence, ASN,
//!   geography, legal ownership, honest behavior, consensus, or Sybil
//!   resistance. Keep attestor pins local and never publish domain mappings as
//!   general discovery metadata.
//! - [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] Fine-grained wire features
//!   use exact SemVer build-metadata tokens inside the already signed
//!   `software_version` field. Do not add them to `NodeCapability`: older
//!   bincode decoders reject unknown enum variants and would partition a
//!   mixed-version fleet. Feature tokens negotiate response contracts only;
//!   they never grant routing, trust, consensus, or finality authority.
//! - [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] Direct relay node
//!   authentication is advertised as one signed feature token so upgraded
//!   senders cannot be redirected to a weaker endpoint by an HTTP response.
//! - [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Target-signed direct relay
//!   receipts use a separate feature token, preserving rolling compatibility
//!   with nodes that authenticate requests but do not yet sign durable ACKs.
//! - [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Target-bound direct
//!   relay authentication uses another signed feature token. A sender must not
//!   infer support from an endpoint response or send one v3 request to multiple
//!   targets because the selected target identity is part of the signature.
//!
//! ## Last Modified
//! v0.35.0-OnionBlindVaultEncryptedFailure - Added signed negotiation for
//! source-only authenticated terminal failure replies
//! v0.34.0-OnionBlindVaultLeaseInventory - Added signed negotiation for
//! private streaming encrypted-object inventory commitments
//! v0.33.0-OnionBlindVaultLeaseStatus - Added signed negotiation for private
//! administration-authorized lease observations
//! v0.32.0-OnionBlindVaultLeaseRenewal - Added signed negotiation for
//! blind-authorized administration-key lease renewal
//! v0.31.0-OnionBlindVaultLeaseRetire - Added signed negotiation for complete
//! administration-key lease retirement over fixed-size onion replies
//! v0.30.0-OnionBlindVaultPutReceipt - Added signed negotiation for anonymous
//! ciphertext writes returning request-bound storage receipts
//! v0.29.0-OnionBlindLeaseAdmission - Added signed negotiation for RFC 9474
//! blind-issued lease admission through the final onion layer
//! v0.28.0-OnionReplyNegotiation - Added signed rolling-upgrade negotiation
//! for fixed-size encrypted onion terminal responses
//! v0.27.0-DirectRelayTargetBindingV3 - Added signed negotiation for direct
//! relay requests bound to one selected target node identity
//! v0.26.0-DirectRelayReceiptV2 - Added signed negotiation for target-authored
//! direct encrypted relay durable-custody receipts
//! v0.25.0-DirectRelayAuthV2 - Added signed rolling-upgrade negotiation for
//! immediate-node-authenticated direct encrypted relay requests
//! v0.24.0-SignedReceiptNegotiation - Added a descriptor-bound purpose-receipt
//! probe feature while retaining unsigned-summary fallback for legacy nodes
//! v0.23.0-SignedProtocolFeatures - Added backward-compatible signed feature
//! negotiation without changing descriptor schema or capability discriminants
//! v0.22.0-BlindVaultReplicaCapability - Added an append-only, rollout-gated
//! anonymous ciphertext replica capability without changing prior discriminants
//! v0.21.0-RouteDomainAttestation - Added bounded portable route-domain
//! attestations, pinned-quorum verification, and strict framed codecs
//! v0.20.0-DirectoryAuthenticatedGossipWire - Added an append-only compact
//! descriptor-proof announcement without changing prior wire discriminants
//! v0.19.0-ReplicaDirectoryInclusionProofWire - Added append-only audited
//! carrier request/response frames for exact producer descriptor proofs
//! v0.18.0-DirectoryInclusionProofWire - Added append-only pinned-peer request
//! and response frames binding one exact descriptor proof to one selected block
//! v0.17.0-DirectoryDescriptorInclusionProof - Added compact, count-bound
//! producer-signed Merkle proofs for one authenticated descriptor commitment
//! v0.16.0-BoundedWitnessCarrier - Added append-only exact-frame carrier
//! request/response contracts without granting the carrier witness authority
//! v0.15.0-AuthenticatedCertificateExchange - Added pinned-peer request and
//! response frames binding exact portable certificate bytes
//! v0.14.0-PortableObservationCertificate - Added bounded offline-verifiable
//! checkpoint and external-recomputation receipt packages
//! v0.13.0-DirectoryMirrorCarrierCapability - Added a signed, rollout-gated
//! Directory Mirror carrier capability without changing prior discriminants
//! v0.12.0-BoundedControlPlaneCodec - Unified bounded discovery and directory wire encoding
//! v0.11.0-DirectoryPolicyHeadAnchor - Added privacy-bounded external policy-head anchor frames
//! v0.10.0-DirectoryEvidenceCarrier - Added producer-bound audited replica transport frames
//! v0.9.0-DirectoryObservationWitness - Added bounded independently recomputed checkpoint witness frames
//! v0.8.0-DirectoryObservationCheckpoint - Added canonical signed observation checkpoints
//! v0.7.0-DirectorySyncWire - Added signed bounded Directory Chain peer frames
//! v0.6.0-DirectoryCommitmentBlock - Added deterministic signed Directory Chain protocol primitives
//! v0.5.0-DescriptorKemBackwardCompatibility - Accept schema v1 descriptors without KEM fields
//! v0.4.0-DiscoverySignatureOnlyVerify - Added signature-only verification for expired peer-cache retention
//! v0.1.0-DiscoveryPhase1 - Initial signed descriptor primitives
//! v0.2.0-DiscoveryPhase2 - Added bounded bootstrap snapshot type
//! v0.3.0-DiscoveryPhase4 - Added bounded discovery gossip messages
// ============================================================================

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::ledger::{build_merkle_inclusion_proof, merkle_root, verify_merkle_inclusion_proof};
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

// ============================================
// Serialization constants
// ============================================

/// Maximum accepted serialized descriptor size.
///
/// Descriptors are intended to be small control-plane objects. Keeping a
/// strict cap prevents unbounded memory allocation when reading bootstrap
/// snapshots or future gossip payloads.
const MAX_DESCRIPTOR_BYTES: u64 = 16 * 1024;

/// Current signed descriptor schema version.
pub const NODE_DESCRIPTOR_SCHEMA_VERSION: u16 = 2;

/// Current bootstrap snapshot schema version.
pub const NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION: u16 = 1;

/// Maximum accepted JSON bootstrap snapshot size.
const MAX_BOOTSTRAP_SNAPSHOT_BYTES: usize = 512 * 1024;

/// Maximum accepted binary discovery gossip message size.
const MAX_DISCOVERY_MESSAGE_BYTES: u64 = 512 * 1024;

/// One-byte discriminator prepended to every Directory Sync V1 frame.
pub const DIRECTORY_SYNC_MAGIC: u8 = 0xd3;

/// Maximum encoded Directory Sync frame payload, excluding the magic byte.
const MAX_DIRECTORY_SYNC_MESSAGE_BYTES: u64 = 512 * 1024;

/// Maximum blocks returned by one Directory Sync range response.
pub const MAX_DIRECTORY_SYNC_BLOCKS_V1: u16 = 8;

/// Maximum content-addressed descriptors returned by one object response.
pub const MAX_DIRECTORY_SYNC_OBJECTS_V1: usize = 16;

/// Witness accepted the exact checkpoint after independent local recomputation.
pub const DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1: u8 = 1;
/// Witness lacks one or more exact retained producer prefixes.
pub const DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1: u8 = 2;
/// Witness has conflicting retained evidence or recomputed a different root.
pub const DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1: u8 = 3;

/// Witness durably retained the exact opaque observer policy head.
pub const DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1: u8 = 1;
/// Witness has a newer retained epoch and rejects observer rollback.
pub const DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1: u8 = 2;
/// Witness retained a different digest for the same observer epoch.
pub const DIRECTORY_POLICY_ANCHOR_CONFLICT_V1: u8 = 3;
/// Witness cannot connect the requested epoch to its retained policy head.
pub const DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1: u8 = 4;

/// Stable production chain identifier for public node-directory commitments.
///
/// This is `SHA-256("AeroNyx-Directory-Mainnet-v1")`. Changing it creates a
/// different directory chain and requires an explicit protocol migration.
pub const AERONYX_DIRECTORY_MAINNET_CHAIN_ID: [u8; 32] = [
    0xa0, 0x4a, 0x2f, 0xdf, 0xc8, 0x32, 0x07, 0x08, 0x30, 0x66, 0x2d, 0x43, 0x5a, 0xfc, 0x9e, 0x1e,
    0x78, 0x32, 0xda, 0xde, 0x2f, 0xd5, 0x95, 0x6b, 0xe7, 0x78, 0x28, 0x36, 0xca, 0x61, 0xd2, 0x2f,
];

/// First stable Directory Chain hashing and signature contract.
pub const DIRECTORY_COMMITMENT_BLOCK_VERSION_V1: u16 = 1;

/// Maximum descriptor commitments accepted in one directory block.
///
/// At 72 bytes of canonical commitment data per entry, this keeps the payload
/// bounded while matching the existing maximum discovery snapshot page size.
pub const MAX_DIRECTORY_COMMITMENTS_PER_BLOCK: usize = 256;

/// Current compact Directory descriptor-inclusion proof contract version.
pub const DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1: u16 = 1;

/// Maximum sibling hashes for a 256-leaf Directory commitment tree.
///
/// [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] This is a wire and
/// allocation bound. The contract test fails if the block limit changes
/// without a reviewed proof-version update.
pub const MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1: usize = 8;

/// Maximum producer clock lead accepted by a directory verifier.
///
/// Without this bound, a malicious producer could timestamp one validly signed
/// block far in the future and force every later block to follow that clock.
pub const MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS: u64 = 120;

/// First stable Directory observation-checkpoint hashing contract.
pub const DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1: u16 = 1;

/// Maximum producer tips bound into one observation checkpoint.
pub const MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1: usize = 16;

/// Stable portable observation-certificate contract version.
pub const DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1: u16 = 1;

/// One-byte discriminator prepended to portable observation certificates.
pub const DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC: u8 = 0xc7;

/// Maximum encoded certificate payload, excluding its magic byte.
///
/// The checkpoint and at most sixteen fixed-size receipts are substantially
/// smaller than this bound. The headroom preserves forward codec compatibility
/// without allowing attacker-controlled allocations.
const MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES: u64 = 64 * 1024;

/// Maximum complete portable observation-certificate frame size.
///
/// [PORTABLE-CERTIFICATE-VERIFIER 2026-07-26 by Codex] File, stdin, and
/// transport adapters must enforce this bound before allocating or decoding.
/// Keeping the complete-frame limit beside the canonical codec prevents each
/// adapter from inventing a subtly different allocation policy.
pub const MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES: usize =
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES as usize + 1;

/// Stable route-domain attestation statement version.
pub const ROUTE_DOMAIN_ATTESTATION_VERSION_V1: u16 = 1;

/// Stable portable route-domain certificate version.
pub const ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1: u16 = 1;

/// Maximum distinct attestations retained in one portable certificate.
pub const MAX_ROUTE_DOMAIN_ATTESTATIONS_V1: usize = 16;

/// Maximum lifetime of one route-domain attestation.
///
/// [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] Short-lived evidence limits
/// stale infrastructure/operator mappings without requiring revocation lists
/// in the first protocol version.
pub const MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1: u64 = 30 * 24 * 60 * 60;

/// One-byte discriminator prepended to portable route-domain certificates.
pub const ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC: u8 = 0xc8;

/// Maximum encoded route-domain certificate payload, excluding its magic byte.
const MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES: u64 = 16 * 1024;

/// Maximum complete portable route-domain certificate frame size.
pub const MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES: usize =
    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES as usize + 1;

// ============================================
// Serde helper for [u8; 64]
// ============================================

mod serde_bytes64 {
    use super::*;

    pub fn serialize<S: Serializer>(v: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        let (lo, hi) = v.split_at(32);
        let lo: [u8; 32] = lo.try_into().unwrap();
        let hi: [u8; 32] = hi.try_into().unwrap();
        (lo, hi).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let (lo, hi): ([u8; 32], [u8; 32]) = Deserialize::deserialize(d)?;
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&lo);
        out[32..].copy_from_slice(&hi);
        Ok(out)
    }
}

// ============================================
// NodeProtocolFeature
// ============================================

/// Fine-grained peer wire features negotiated through signed descriptors.
///
/// [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] These values deliberately do
/// not serialize as `NodeCapability` variants. Each feature maps to an exact,
/// valid SemVer build-metadata identifier inside `software_version`, preserving
/// the schema-v2 bincode layout for old nodes while keeping the advertisement
/// covered by the descriptor signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeProtocolFeature {
    /// Handled blind-relay protocol failures carry an immediate-hop signed
    /// `BlindRelayFailureReceipt` bound to the exact request and reason.
    BlindRelayFailureReceiptV1,
    /// The node can return purpose-bound version-2 terminal delivery receipts.
    /// This claim authorizes a probe only; route authority still requires a
    /// successfully verified receipt from the selected terminal.
    PurposeBoundDeliveryReceiptV2,
    /// The node accepts direct encrypted chat relay requests authenticated by
    /// the immediate previous hop's Ed25519 node identity.
    ///
    /// [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] This is advertised through
    /// signed SemVer metadata so upgraded senders can select the authenticated
    /// endpoint without breaking rolling compatibility with legacy nodes.
    DirectPeerRelayAuthV2,
    /// The node returns a target-signed direct relay v2 receipt bound to the
    /// exact authenticated request after durable ciphertext acceptance.
    ///
    /// [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] This is intentionally
    /// separate from request authentication so mixed-version v2 fleets can
    /// upgrade the response contract without treating HTTP claims as trust.
    DirectPeerRelayReceiptV2,
    /// The node accepts direct relay requests whose previous-hop signature is
    /// bound to the exact selected target node identity.
    ///
    /// [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] This prevents one
    /// valid authenticated request from being replayed across different relay
    /// nodes. It remains separately negotiated so v1/v2 peers keep working
    /// during a rolling fleet upgrade.
    DirectPeerRelayTargetBindingV3,
    /// The node supports fixed-size encrypted terminal responses propagated
    /// through blind relay acknowledgements.
    ///
    /// [ONION-REPLY-NEGOTIATION 2026-08-28 by Codex] This token gates the
    /// additive response surface during rolling upgrades; it does not reveal
    /// whether any route carries a request/response workload.
    OnionReplyV1,
    /// The node accepts RFC 9474 blind-issued lease admission inside the final
    /// onion layer and returns a request-bound terminal-signed receipt.
    ///
    /// [ONION-BLIND-LEASE-ADMISSION 2026-08-28 by Codex] This remains separate
    /// from `OnionReplyV1`: supporting the generic carrier never implies that a
    /// rolling-upgrade peer executes this sensitive workload.
    OnionBlindLeaseAdmissionV1,
    /// The node accepts an immutable Blind Vault Put inside the final onion
    /// layer and returns a request-bound signed storage receipt.
    ///
    /// [ONION-BLIND-VAULT-PUT-RECEIPT 2026-08-28 by Codex] Legacy one-way Put
    /// remains available under its existing purpose; clients request this
    /// feature only when they require cryptographic custody evidence.
    OnionBlindVaultPutReceiptV1,
    /// The node accepts administration-key retirement of a complete Blind
    /// Vault lease and returns a request-bound signed aggregate receipt.
    ///
    /// [ONION-BLIND-VAULT-LEASE-RETIRE 2026-08-28 by Codex] This token gates
    /// destructive lease-wide mutation independently from generic reply and
    /// object-deletion support during rolling upgrades.
    OnionBlindVaultLeaseRetireV1,
    /// The node consumes a fresh blind credential while atomically extending
    /// one administration-key-controlled live Blind Vault lease.
    OnionBlindVaultLeaseRenewalV1,
    /// The node returns an encrypted terminal-signed observation for one
    /// administration-key-controlled live Blind Vault lease.
    OnionBlindVaultLeaseStatusV1,
    /// The node returns an encrypted terminal-signed commitment to the live
    /// object inventory of one administration-key-controlled lease.
    OnionBlindVaultLeaseInventoryV1,
    /// Valid Blind Vault workload failures are sealed into the same fixed-size
    /// source-only reply instead of escaping through relay-visible status.
    ///
    /// [ONION-BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] This token is
    /// separate from generic `OnionReplyV1` so upgraded sources never assume
    /// typed encrypted failures from a mixed-version terminal.
    OnionBlindVaultEncryptedFailureV1,
}

impl NodeProtocolFeature {
    /// Features understood by this binary, in stable negotiation order.
    pub const ALL: [Self; 13] = [
        Self::BlindRelayFailureReceiptV1,
        Self::PurposeBoundDeliveryReceiptV2,
        Self::DirectPeerRelayAuthV2,
        Self::DirectPeerRelayReceiptV2,
        Self::DirectPeerRelayTargetBindingV3,
        Self::OnionReplyV1,
        Self::OnionBlindLeaseAdmissionV1,
        Self::OnionBlindVaultPutReceiptV1,
        Self::OnionBlindVaultLeaseRetireV1,
        Self::OnionBlindVaultLeaseRenewalV1,
        Self::OnionBlindVaultLeaseStatusV1,
        Self::OnionBlindVaultLeaseInventoryV1,
        Self::OnionBlindVaultEncryptedFailureV1,
    ];

    /// Exact SemVer build-metadata identifier used on the signed wire.
    #[must_use]
    pub const fn semver_build_token(self) -> &'static str {
        match self {
            Self::BlindRelayFailureReceiptV1 => "anpf1-brfr1",
            Self::PurposeBoundDeliveryReceiptV2 => "anpf1-pbdr2",
            Self::DirectPeerRelayAuthV2 => "anpf1-dpra2",
            Self::DirectPeerRelayReceiptV2 => "anpf1-dprr2",
            Self::DirectPeerRelayTargetBindingV3 => "anpf1-dprtb3",
            Self::OnionReplyV1 => "anpf1-or1",
            Self::OnionBlindLeaseAdmissionV1 => "anpf1-obla1",
            Self::OnionBlindVaultPutReceiptV1 => "anpf1-obpr1",
            Self::OnionBlindVaultLeaseRetireV1 => "anpf1-oblr1",
            Self::OnionBlindVaultLeaseRenewalV1 => "anpf1-oblw1",
            Self::OnionBlindVaultLeaseStatusV1 => "anpf1-obls1",
            Self::OnionBlindVaultLeaseInventoryV1 => "anpf1-obli1",
            Self::OnionBlindVaultEncryptedFailureV1 => "anpf1-obef1",
        }
    }
}

// ============================================
// NodeCapability
// ============================================

/// Public capability flags a node can advertise for peer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeCapability {
    /// AeroNyx privacy protocol packet relay.
    PrivacyRelay,
    /// End-to-end encrypted chat envelope relay.
    ChatRelay,
    /// Encrypted MemChain storage and query support.
    EncryptedStorage,
    /// Agent-to-agent encrypted protocol relay.
    AgentRelay,
    /// Future no-exit onion middle-hop relay.
    OnionMiddle,
    /// Audited non-authoritative Directory replica carrier.
    ///
    /// [MIRROR-CAPABILITY 2026-07-24 by Codex] This variant is appended to
    /// preserve every existing bincode discriminant. Mixed-version fleets must
    /// upgrade decoders before operators enable advertisement because an old
    /// binary cannot decode a capability variant it does not know.
    DirectoryMirrorCarrier,
    /// Admitted node-blind ciphertext replica reachable through the peer API.
    ///
    /// This transport hint does not prove that any particular lease or object
    /// exists and grants no producer, witness, consensus, or finality role.
    BlindVaultReplica,
}

// ============================================
// NodePolicy
// ============================================

/// Public policy hints for routing and peer selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodePolicy {
    /// Whether this node allows public exit behavior.
    ///
    /// AeroNyx protocol default is `false`; independent operators must not be
    /// treated as public exits unless a future reviewed policy explicitly says so.
    pub allows_public_exit: bool,
    /// Whether the node is visible to public bootstrap snapshots.
    pub public_discovery: bool,
    /// Optional operator-defined region label, for example `us-central`.
    pub region: Option<String>,
}

impl Default for NodePolicy {
    fn default() -> Self {
        Self {
            allows_public_exit: false,
            public_discovery: true,
            region: None,
        }
    }
}

// ============================================
// NodeCapacity
// ============================================

/// Coarse capacity hints advertised by a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeCapacity {
    /// Maximum concurrent privacy protocol sessions the node is willing to serve.
    pub max_sessions: u32,
    /// Optional bandwidth policy in bytes per second.
    pub max_bps: Option<u64>,
    /// Optional packet-rate policy in packets per second.
    pub max_pps: Option<u64>,
}

impl Default for NodeCapacity {
    fn default() -> Self {
        Self {
            max_sessions: 0,
            max_bps: None,
            max_pps: None,
        }
    }
}

// ============================================
// NodeDescriptor
// ============================================

/// Canonical signed metadata for one AeroNyx node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeDescriptor {
    /// Descriptor schema version.
    pub schema_version: u16,
    /// Node Ed25519 identity public key.
    pub node_id: [u8; 32],
    /// Monotonic descriptor sequence number from this node.
    pub sequence: u64,
    /// Unix timestamp in seconds when the descriptor was issued.
    pub issued_at: u64,
    /// Unix timestamp in seconds when the descriptor expires.
    pub expires_at: u64,
    /// Optional public control-plane endpoint for node-to-node traffic.
    pub public_endpoint: Option<String>,
    /// Running software version reported by the node.
    pub software_version: String,
    /// Public capability flags.
    pub capabilities: Vec<NodeCapability>,
    /// Coarse capacity hints.
    pub capacity: NodeCapacity,
    /// Public policy hints.
    pub policy: NodePolicy,
    /// KEM algorithm id for the onion-routing per-hop key (schema v2+).
    ///
    /// `0` = none (node is not an onion hop), `1` = X25519 (`KEM_ALG_X25519`),
    /// `2` = reserved for the hybrid post-quantum X-Wing KEM. A node's X25519
    /// public key is NOT derivable from its Ed25519 `node_id`, so it must be
    /// published here for clients to build onion layers addressed to this node.
    #[serde(default)]
    pub kem_alg: u8,
    /// KEM public key bytes for onion layer encryption (schema v2+).
    ///
    /// All-zero when `kem_alg == 0`. For `kem_alg == 1` this is the node's
    /// X25519 public key (`IdentityKeyPair::x25519_public_key_bytes()`).
    #[serde(default)]
    pub kem_public: [u8; 32],
}

/// Legacy schema-v1 descriptor layout used before onion KEM fields existed.
///
/// This is intentionally private and used only to verify old signed peer-cache
/// and bootstrap records. The public descriptor type keeps v2 fields so new
/// nodes publish onion KEM material, while schema-v1 signatures remain
/// verifiable after serde fills missing KEM fields with safe defaults.
#[derive(Debug, Serialize)]
struct LegacyNodeDescriptorV1<'a> {
    schema_version: u16,
    node_id: &'a [u8; 32],
    sequence: u64,
    issued_at: u64,
    expires_at: u64,
    public_endpoint: &'a Option<String>,
    software_version: &'a String,
    capabilities: &'a Vec<NodeCapability>,
    capacity: &'a NodeCapacity,
    policy: &'a NodePolicy,
}

fn legacy_descriptor_v1_signing_bytes(descriptor: &NodeDescriptor) -> Result<Vec<u8>, CoreError> {
    let legacy = LegacyNodeDescriptorV1 {
        schema_version: descriptor.schema_version,
        node_id: &descriptor.node_id,
        sequence: descriptor.sequence,
        issued_at: descriptor.issued_at,
        expires_at: descriptor.expires_at,
        public_endpoint: &descriptor.public_endpoint,
        software_version: &descriptor.software_version,
        capabilities: &descriptor.capabilities,
        capacity: &descriptor.capacity,
        policy: &descriptor.policy,
    };

    encode_bincode_bounded(&legacy, MAX_DESCRIPTOR_BYTES)
        .map_err(|err| CoreError::malformed(format!("legacy node descriptor serialization: {err}")))
}

impl NodeDescriptor {
    /// Creates a descriptor with the current schema version.
    #[must_use]
    pub fn new(
        node_id: [u8; 32],
        sequence: u64,
        issued_at: u64,
        expires_at: u64,
        software_version: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: NODE_DESCRIPTOR_SCHEMA_VERSION,
            node_id,
            sequence,
            issued_at,
            expires_at,
            public_endpoint: None,
            software_version: software_version.into(),
            capabilities: Vec::new(),
            capacity: NodeCapacity::default(),
            policy: NodePolicy::default(),
            kem_alg: 0,
            kem_public: [0u8; 32],
        }
    }

    /// Publishes an X25519 KEM public key so this node can serve as an onion
    /// hop. Sets `kem_alg = 1` (`KEM_ALG_X25519`).
    #[must_use]
    pub fn with_x25519_kem(mut self, kem_public: [u8; 32]) -> Self {
        self.kem_alg = 1;
        self.kem_public = kem_public;
        self
    }

    /// Adds signed, backward-compatible protocol feature advertisements.
    ///
    /// Existing SemVer build metadata is preserved. Feature tokens are sorted
    /// and deduplicated so the same feature set has one stable representation.
    /// Old nodes continue to decode the unchanged descriptor schema and simply
    /// treat the result as an opaque software-version string.
    #[must_use]
    pub fn with_protocol_features(
        mut self,
        features: impl IntoIterator<Item = NodeProtocolFeature>,
    ) -> Self {
        let mut requested = features
            .into_iter()
            .map(NodeProtocolFeature::semver_build_token)
            .collect::<Vec<_>>();
        requested.sort_unstable();
        requested.dedup();
        if requested.is_empty() {
            return self;
        }

        let (release, existing_build) = self.software_version.split_once('+').map_or_else(
            || (self.software_version.clone(), None),
            |(release, build)| (release.to_string(), Some(build.to_string())),
        );
        let mut build_identifiers = existing_build
            .as_deref()
            .into_iter()
            .flat_map(|metadata| metadata.split('.'))
            .filter(|identifier| !identifier.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for token in requested {
            if !build_identifiers
                .iter()
                .any(|identifier| identifier == token)
            {
                build_identifiers.push(token.to_string());
            }
        }
        self.software_version = format!("{release}+{}", build_identifiers.join("."));
        self
    }

    /// Returns whether this signed descriptor advertises one exact wire feature.
    #[must_use]
    pub fn advertises_protocol_feature(&self, feature: NodeProtocolFeature) -> bool {
        self.software_version
            .split_once('+')
            .map(|(_, metadata)| {
                metadata
                    .split('.')
                    .any(|identifier| identifier == feature.semver_build_token())
            })
            .unwrap_or(false)
    }

    /// Returns the published X25519 KEM key if this node advertises one
    /// (`kem_alg == 1` and the key is non-zero), else `None`.
    #[must_use]
    pub fn x25519_kem_public(&self) -> Option<[u8; 32]> {
        if self.kem_alg == 1 && self.kem_public != [0u8; 32] {
            Some(self.kem_public)
        } else {
            None
        }
    }

    /// Returns `true` when `now` is within the descriptor validity window.
    #[must_use]
    pub const fn is_valid_at(&self, now: u64) -> bool {
        self.issued_at <= now && now < self.expires_at
    }

    /// Returns the canonical bytes signed by the node identity key.
    ///
    /// # Errors
    /// Returns a `CoreError` if serialization fails.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, CoreError> {
        if self.schema_version == 1 {
            return legacy_descriptor_v1_signing_bytes(self);
        }

        encode_bincode_bounded(self, MAX_DESCRIPTOR_BYTES)
            .map_err(|err| CoreError::malformed(format!("node descriptor serialization: {err}")))
    }
}

// ============================================
// SignedNodeDescriptor
// ============================================

/// A node descriptor plus Ed25519 signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNodeDescriptor {
    /// Signed descriptor body.
    pub descriptor: NodeDescriptor,
    /// Ed25519 signature over `descriptor.signing_bytes()`.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl SignedNodeDescriptor {
    /// Signs a descriptor with the node identity key.
    ///
    /// # Errors
    /// Returns a `CoreError` if descriptor serialization fails.
    pub fn sign(descriptor: NodeDescriptor, keypair: &IdentityKeyPair) -> Result<Self, CoreError> {
        let bytes = descriptor.signing_bytes()?;
        let signature = keypair.sign(&bytes);
        Ok(Self {
            descriptor,
            signature,
        })
    }

    /// Verifies the descriptor signature and expiry at `now`.
    ///
    /// # Errors
    /// Returns `CoreError::SignatureVerification` if the descriptor is expired,
    /// not yet valid, has an unsupported schema version, or signature
    /// verification fails.
    pub fn verify_at(&self, now: u64) -> Result<(), CoreError> {
        // Accept any known schema (1 = pre-onion, 2 = onion KEM key). A v1
        // descriptor simply advertises no onion KEM key. Reject unknown/newer.
        if self.descriptor.schema_version == 0
            || self.descriptor.schema_version > NODE_DESCRIPTOR_SCHEMA_VERSION
        {
            return Err(CoreError::SignatureVerification);
        }
        if !self.descriptor.is_valid_at(now) {
            return Err(CoreError::SignatureVerification);
        }

        self.verify_signature()
    }

    /// Verifies only the descriptor schema version and Ed25519 signature.
    ///
    /// This method deliberately does not check `issued_at` / `expires_at`.
    /// It exists so a local peer cache can retain expired-but-authentic node
    /// records as non-routeable history after restart. Callers must still use
    /// `verify_at(now)` before counting a descriptor as live, valid, routeable,
    /// gossip-exportable, or relay-eligible.
    ///
    /// # Errors
    /// Returns `CoreError::SignatureVerification` if the schema version is
    /// unsupported or signature verification fails.
    pub fn verify_signature(&self) -> Result<(), CoreError> {
        // Accept any known schema (1 = pre-onion, 2 = onion KEM key). A v1
        // descriptor simply advertises no onion KEM key. Reject unknown/newer.
        if self.descriptor.schema_version == 0
            || self.descriptor.schema_version > NODE_DESCRIPTOR_SCHEMA_VERSION
        {
            return Err(CoreError::SignatureVerification);
        }

        let pk = IdentityPublicKey::from_bytes(&self.descriptor.node_id)?;
        let bytes = self.descriptor.signing_bytes()?;
        pk.verify(&bytes, &self.signature)
    }

    /// Returns the descriptor node id.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.descriptor.node_id
    }

    /// Returns the descriptor sequence number.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.descriptor.sequence
    }
}

// ============================================
// Directory Chain V1
// ============================================

/// Opaque, content-addressed commitment to one authenticated node descriptor.
///
/// The commitment identifies the public node and monotonic descriptor sequence
/// needed for deterministic replay, while the digest binds the complete signed
/// descriptor. Endpoint, region, capacity, policy, and capability fields are
/// not duplicated into the directory block payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DirectoryDescriptorCommitmentV1 {
    /// Public Ed25519 identity of the node that signed the descriptor.
    pub node_id: [u8; 32],
    /// Monotonic sequence copied from the authenticated descriptor.
    pub sequence: u64,
    /// Domain-separated digest of descriptor signing bytes and signature.
    pub descriptor_hash: [u8; 32],
}

impl DirectoryDescriptorCommitmentV1 {
    /// Creates a commitment after verifying the descriptor schema and signature.
    ///
    /// Expiry is deliberately not checked here: an authenticated descriptor may
    /// remain part of immutable directory history after it stops being routeable.
    ///
    /// # Errors
    /// Returns a `CoreError` when the descriptor schema, key, signature, or
    /// canonical serialization is invalid.
    pub fn from_signed_descriptor(descriptor: &SignedNodeDescriptor) -> Result<Self, CoreError> {
        descriptor.verify_signature()?;
        Ok(Self {
            node_id: descriptor.node_id(),
            sequence: descriptor.sequence(),
            descriptor_hash: signed_descriptor_commitment_hash(descriptor)?,
        })
    }

    /// Returns the domain-separated Merkle leaf for this commitment.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryDescriptorCommitment-v1");
        hasher.update(self.node_id);
        hasher.update(self.sequence.to_le_bytes());
        hasher.update(self.descriptor_hash);
        hasher.finalize().into()
    }

    /// Checks whether this commitment binds the supplied signed descriptor.
    ///
    /// # Errors
    /// Returns a `CoreError` when the supplied descriptor is not authentic or
    /// cannot be canonically serialized.
    pub fn matches_signed_descriptor(
        &self,
        descriptor: &SignedNodeDescriptor,
    ) -> Result<bool, CoreError> {
        let candidate = Self::from_signed_descriptor(descriptor)?;
        Ok(self == &candidate)
    }

    fn structurally_valid(&self) -> bool {
        self.node_id != [0u8; 32] && self.sequence > 0 && self.descriptor_hash != [0u8; 32]
    }
}

/// Computes the stable digest committed by [`DirectoryDescriptorCommitmentV1`].
///
/// The descriptor signature is included so the commitment proves exactly which
/// authenticated descriptor object was observed. A length prefix keeps the
/// canonical field boundary explicit for future schema versions.
fn signed_descriptor_commitment_hash(
    descriptor: &SignedNodeDescriptor,
) -> Result<[u8; 32], CoreError> {
    let signing_bytes = descriptor.descriptor.signing_bytes()?;
    let signing_bytes_len = u32::try_from(signing_bytes.len()).map_err(|_| {
        CoreError::malformed("signed node descriptor canonical bytes exceed u32 length")
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-SignedNodeDescriptorCommitment-v1");
    hasher.update(signing_bytes_len.to_le_bytes());
    hasher.update(signing_bytes);
    hasher.update(descriptor.signature);
    Ok(hasher.finalize().into())
}

/// Canonical signed header for one Directory Chain block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryCommitmentHeaderV1 {
    /// Stable hashing and signature contract version.
    pub protocol_version: u16,
    /// Prevents replay between production, test, and private directories.
    pub chain_id: [u8; 32],
    /// One-based block height.
    pub height: u64,
    /// Producer timestamp in Unix epoch seconds.
    pub timestamp: u64,
    /// Hash of the previous V1 header, or all zeroes at height one.
    pub prev_block_hash: [u8; 32],
    /// Merkle root of canonically sorted descriptor commitment leaves.
    pub commitment_root: [u8; 32],
    /// Number of commitments carried by the block.
    pub commitment_count: u32,
    /// Ed25519 identity of the node producing this block.
    pub producer: [u8; 32],
}

impl DirectoryCommitmentHeaderV1 {
    /// Computes the domain-separated canonical block identity.
    ///
    /// Field order and little-endian integer encoding are stable protocol
    /// contracts and must not change within V1.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryCommitmentBlock-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.height.to_le_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.prev_block_hash);
        hasher.update(self.commitment_root);
        hasher.update(self.commitment_count.to_le_bytes());
        hasher.update(self.producer);
        hasher.finalize().into()
    }

    /// Returns the canonical block hash as lowercase hexadecimal.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash())
    }
}

/// Signed, hash-linked directory block containing no client or traffic data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryCommitmentBlockV1 {
    /// Signed chain header.
    pub header: DirectoryCommitmentHeaderV1,
    /// Canonically sorted descriptor commitments.
    pub commitments: Vec<DirectoryDescriptorCommitmentV1>,
    /// Ed25519 signature by `header.producer` over `header.hash()`.
    #[serde(with = "serde_bytes64")]
    pub producer_signature: [u8; 64],
}

/// Compact proof that one authenticated descriptor commitment is included in
/// one exact producer-signed Directory block.
///
/// The proof intentionally carries no user, traffic, message, route, Memory
/// Chain, DNS, destination, or wallet data. It is useful only when the verifier
/// independently trusts `expected_producer` and `expected_block_hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryDescriptorInclusionProofV1 {
    /// Stable proof contract version.
    pub proof_version: u16,
    /// Exact producer-signed block header containing the commitment root.
    pub block_header: DirectoryCommitmentHeaderV1,
    /// Producer signature over `block_header.hash()`.
    #[serde(with = "serde_bytes64")]
    pub producer_signature: [u8; 64],
    /// Exact descriptor commitment used as the Merkle leaf.
    pub commitment: DirectoryDescriptorCommitmentV1,
    /// Zero-based commitment position in the canonical block payload.
    pub commitment_index: u32,
    /// Sibling hashes ordered from the leaf level toward the root.
    pub sibling_hashes: Vec<[u8; 32]>,
    /// Exact authenticated descriptor object bound by `commitment`.
    pub descriptor: SignedNodeDescriptor,
}

/// Fail-closed validation outcomes for a compact Directory inclusion proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryDescriptorInclusionProofError {
    /// The inclusion-proof contract version is unsupported.
    UnsupportedVersion,
    /// The proof belongs to another Directory chain.
    WrongChain,
    /// The producer differs from the verifier's independently pinned producer.
    WrongProducer,
    /// The signed block differs from the verifier's independently selected block.
    WrongBlockHash,
    /// The signed block header is structurally or cryptographically invalid.
    InvalidBlock,
    /// The descriptor object is malformed or has an invalid signature.
    InvalidDescriptor,
    /// The commitment does not bind the included descriptor object.
    DescriptorMismatch,
    /// The leaf index or declared block commitment count is invalid.
    InvalidPosition,
    /// The sibling path exceeds or differs from the exact tree depth.
    InvalidProofLength,
    /// The sibling path does not reconstruct the signed commitment root.
    InvalidMerkleProof,
}

impl std::fmt::Display for DirectoryDescriptorInclusionProofError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory inclusion proof version",
            Self::WrongChain => "directory inclusion proof belongs to another chain",
            Self::WrongProducer => "directory inclusion proof producer is not trusted",
            Self::WrongBlockHash => "directory inclusion proof block hash is not selected",
            Self::InvalidBlock => "directory inclusion proof block is invalid",
            Self::InvalidDescriptor => "directory inclusion proof descriptor is invalid",
            Self::DescriptorMismatch => {
                "directory inclusion proof commitment does not bind descriptor"
            }
            Self::InvalidPosition => "directory inclusion proof position is invalid",
            Self::InvalidProofLength => "directory inclusion proof path length is invalid",
            Self::InvalidMerkleProof => "directory inclusion proof path is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryDescriptorInclusionProofError {}

impl DirectoryDescriptorInclusionProofV1 {
    /// Builds a compact proof from one complete, valid Directory block.
    ///
    /// The constructor validates the block payload and signature at
    /// `observed_at`, authenticates `descriptor`, and requires its exact
    /// commitment to be present. Chain-continuity selection remains the
    /// caller's responsibility; this constructor cannot choose a canonical
    /// producer history.
    ///
    /// # Errors
    /// Returns a fail-closed proof error when the block, descriptor,
    /// commitment lookup, or bounded Merkle path is invalid.
    pub fn from_block_at(
        block: &DirectoryCommitmentBlockV1,
        descriptor: &SignedNodeDescriptor,
        observed_at: u64,
    ) -> Result<Self, DirectoryDescriptorInclusionProofError> {
        block
            .verify_at(
                &block.header.chain_id,
                block.header.height,
                &block.header.prev_block_hash,
                0,
                observed_at,
            )
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)?;
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidDescriptor)?;
        let commitment_index = block
            .commitments
            .binary_search(&commitment)
            .map_err(|_| DirectoryDescriptorInclusionProofError::DescriptorMismatch)?;
        let commitment_hashes = block
            .commitments
            .iter()
            .map(DirectoryDescriptorCommitmentV1::hash)
            .collect::<Vec<_>>();
        let sibling_hashes = build_merkle_inclusion_proof(&commitment_hashes, commitment_index)
            .ok_or(DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        if sibling_hashes.len() > MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1 {
            return Err(DirectoryDescriptorInclusionProofError::InvalidProofLength);
        }
        let commitment_index = u32::try_from(commitment_index)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        let proof = Self {
            proof_version: DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1,
            block_header: block.header.clone(),
            producer_signature: block.producer_signature,
            commitment,
            commitment_index,
            sibling_hashes,
            descriptor: descriptor.clone(),
        };
        proof.verify_at(
            &block.header.chain_id,
            &block.header.producer,
            &block.hash(),
            observed_at,
        )?;
        Ok(proof)
    }

    /// Verifies one proof against an independently selected producer and block.
    ///
    /// A successful result proves producer-signed inclusion only. It does not
    /// choose a canonical chain or establish voting, quorum, consensus,
    /// finality, transaction inclusion, or user activity.
    ///
    /// # Errors
    /// Returns a stable fail-closed proof error when any expected trust anchor,
    /// block signature, descriptor binding, position, or sibling hash fails.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        expected_producer: &[u8; 32],
        expected_block_hash: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), DirectoryDescriptorInclusionProofError> {
        if self.proof_version != DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1 {
            return Err(DirectoryDescriptorInclusionProofError::UnsupportedVersion);
        }
        if &self.block_header.chain_id != expected_chain_id {
            return Err(DirectoryDescriptorInclusionProofError::WrongChain);
        }
        if &self.block_header.producer != expected_producer {
            return Err(DirectoryDescriptorInclusionProofError::WrongProducer);
        }
        if &self.block_header.hash() != expected_block_hash {
            return Err(DirectoryDescriptorInclusionProofError::WrongBlockHash);
        }
        verify_directory_inclusion_header_at(
            &self.block_header,
            &self.producer_signature,
            observed_at,
        )?;
        let commitment_count = usize::try_from(self.block_header.commitment_count)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        let commitment_index = usize::try_from(self.commitment_index)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        if commitment_count == 0
            || commitment_count > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK
            || commitment_index >= commitment_count
        {
            return Err(DirectoryDescriptorInclusionProofError::InvalidPosition);
        }
        let expected_depth = directory_inclusion_proof_depth(commitment_count);
        if self.sibling_hashes.len() != expected_depth
            || self.sibling_hashes.len() > MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1
        {
            return Err(DirectoryDescriptorInclusionProofError::InvalidProofLength);
        }
        if !self.commitment.structurally_valid() {
            return Err(DirectoryDescriptorInclusionProofError::InvalidDescriptor);
        }
        match self.commitment.matches_signed_descriptor(&self.descriptor) {
            Ok(true) => {}
            Ok(false) => {
                return Err(DirectoryDescriptorInclusionProofError::DescriptorMismatch);
            }
            Err(_) => {
                return Err(DirectoryDescriptorInclusionProofError::InvalidDescriptor);
            }
        }
        if !verify_merkle_inclusion_proof(
            &self.block_header.commitment_root,
            &self.commitment.hash(),
            commitment_index,
            commitment_count,
            &self.sibling_hashes,
        ) {
            return Err(DirectoryDescriptorInclusionProofError::InvalidMerkleProof);
        }
        Ok(())
    }

    /// Returns the exact producer-signed block identity bound by this proof.
    #[must_use]
    pub fn block_hash(&self) -> [u8; 32] {
        self.block_header.hash()
    }
}

/// Validation failures for the V1 Directory Chain contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryCommitmentValidationError {
    /// The block hashing/signature contract version is unsupported.
    UnsupportedVersion,
    /// The block belongs to another directory chain.
    WrongChain,
    /// The block height is zero or does not continue the expected chain.
    InvalidHeight,
    /// The previous block hash does not match the expected chain tip.
    InvalidPreviousHash,
    /// The block timestamp is zero or regresses behind its predecessor.
    InvalidTimestamp,
    /// A directory block must carry at least one commitment.
    EmptyBlock,
    /// The block exceeds the commitment count bound.
    TooManyCommitments,
    /// Header and payload commitment counts differ.
    CommitmentCountMismatch,
    /// A commitment contains a sentinel identity, sequence, or digest.
    InvalidCommitment,
    /// Commitments are not in canonical lexicographic order.
    NonCanonicalOrder,
    /// The same descriptor commitment appears more than once.
    DuplicateCommitment,
    /// The payload does not match the signed Merkle root.
    InvalidMerkleRoot,
    /// The producer public key is malformed.
    InvalidProducer,
    /// The producer signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for DirectoryCommitmentValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory block protocol version",
            Self::WrongChain => "directory block belongs to another chain",
            Self::InvalidHeight => "directory block height does not continue the chain",
            Self::InvalidPreviousHash => "directory block previous hash does not match the tip",
            Self::InvalidTimestamp => "directory block timestamp is invalid",
            Self::EmptyBlock => "directory block is empty",
            Self::TooManyCommitments => "directory block exceeds the commitment limit",
            Self::CommitmentCountMismatch => "directory header count does not match payload",
            Self::InvalidCommitment => "directory descriptor commitment is invalid",
            Self::NonCanonicalOrder => "directory commitments are not canonically ordered",
            Self::DuplicateCommitment => "directory descriptor commitment is duplicated",
            Self::InvalidMerkleRoot => "directory commitment Merkle root is invalid",
            Self::InvalidProducer => "directory block producer public key is invalid",
            Self::InvalidSignature => "directory block producer signature is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryCommitmentValidationError {}

impl DirectoryCommitmentBlockV1 {
    /// Builds and signs one deterministic production directory block.
    ///
    /// Input commitments are sorted before hashing. The constructor rejects
    /// empty, oversized, duplicated, sentinel, or impossible genesis inputs so
    /// invalid local blocks are never signed accidentally.
    ///
    /// # Errors
    /// Returns a [`DirectoryCommitmentValidationError`] for invalid block or
    /// commitment inputs.
    pub fn new_signed(
        height: u64,
        timestamp: u64,
        prev_block_hash: [u8; 32],
        mut commitments: Vec<DirectoryDescriptorCommitmentV1>,
        identity: &IdentityKeyPair,
    ) -> Result<Self, DirectoryCommitmentValidationError> {
        validate_directory_block_position(height, timestamp, &prev_block_hash, 0)?;
        commitments.sort_unstable();
        validate_directory_commitments(&commitments)?;
        let commitment_hashes = commitments
            .iter()
            .map(DirectoryDescriptorCommitmentV1::hash)
            .collect::<Vec<_>>();
        let commitment_count = u32::try_from(commitments.len())
            .map_err(|_| DirectoryCommitmentValidationError::TooManyCommitments)?;
        let header = DirectoryCommitmentHeaderV1 {
            protocol_version: DIRECTORY_COMMITMENT_BLOCK_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            height,
            timestamp,
            prev_block_hash,
            commitment_root: merkle_root(&commitment_hashes),
            commitment_count,
            producer: identity.public_key_bytes(),
        };
        let producer_signature = identity.sign(&header.hash());
        Ok(Self {
            header,
            commitments,
            producer_signature,
        })
    }

    /// Returns the canonical block identity.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.header.hash()
    }

    /// Validates contract, chain continuity, canonical payload, Merkle root,
    /// and producer authenticity.
    ///
    /// `previous_timestamp` is zero for genesis and the prior block timestamp
    /// otherwise. Equal timestamps are accepted to tolerate one-second clocks.
    /// `observed_at` is the verifier's current Unix time and enforces a bounded
    /// future-clock lead.
    ///
    /// # Errors
    /// Returns a [`DirectoryCommitmentValidationError`] when the block breaks
    /// the V1 contract, expected chain position, canonical payload, Merkle
    /// commitment, timestamp bound, producer identity, or signature.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        expected_height: u64,
        expected_prev_hash: &[u8; 32],
        previous_timestamp: u64,
        observed_at: u64,
    ) -> Result<(), DirectoryCommitmentValidationError> {
        if self.header.protocol_version != DIRECTORY_COMMITMENT_BLOCK_VERSION_V1 {
            return Err(DirectoryCommitmentValidationError::UnsupportedVersion);
        }
        if &self.header.chain_id != expected_chain_id {
            return Err(DirectoryCommitmentValidationError::WrongChain);
        }
        if self.header.height != expected_height {
            return Err(DirectoryCommitmentValidationError::InvalidHeight);
        }
        if &self.header.prev_block_hash != expected_prev_hash {
            return Err(DirectoryCommitmentValidationError::InvalidPreviousHash);
        }
        validate_directory_block_position(
            self.header.height,
            self.header.timestamp,
            &self.header.prev_block_hash,
            previous_timestamp,
        )?;
        if self.header.timestamp > observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryCommitmentValidationError::InvalidTimestamp);
        }
        validate_directory_commitments(&self.commitments)?;
        if self.header.commitment_count as usize != self.commitments.len() {
            return Err(DirectoryCommitmentValidationError::CommitmentCountMismatch);
        }
        let commitment_hashes = self
            .commitments
            .iter()
            .map(DirectoryDescriptorCommitmentV1::hash)
            .collect::<Vec<_>>();
        if merkle_root(&commitment_hashes) != self.header.commitment_root {
            return Err(DirectoryCommitmentValidationError::InvalidMerkleRoot);
        }
        let producer = IdentityPublicKey::from_bytes(&self.header.producer)
            .map_err(|_| DirectoryCommitmentValidationError::InvalidProducer)?;
        producer
            .verify(&self.header.hash(), &self.producer_signature)
            .map_err(|_| DirectoryCommitmentValidationError::InvalidSignature)
    }
}

impl std::fmt::Display for DirectoryCommitmentBlockV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "DirectoryCommitmentBlockV1(height={}, commitments={}, hash={}..)",
            self.header.height,
            self.commitments.len(),
            &self.header.hash_hex()[..8],
        )
    }
}

fn verify_directory_inclusion_header_at(
    header: &DirectoryCommitmentHeaderV1,
    producer_signature: &[u8; 64],
    observed_at: u64,
) -> Result<(), DirectoryDescriptorInclusionProofError> {
    if header.protocol_version != DIRECTORY_COMMITMENT_BLOCK_VERSION_V1 {
        return Err(DirectoryDescriptorInclusionProofError::InvalidBlock);
    }
    validate_directory_block_position(header.height, header.timestamp, &header.prev_block_hash, 0)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)?;
    if header.timestamp > observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS) {
        return Err(DirectoryDescriptorInclusionProofError::InvalidBlock);
    }
    let commitment_count = usize::try_from(header.commitment_count)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
    if commitment_count == 0 || commitment_count > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK {
        return Err(DirectoryDescriptorInclusionProofError::InvalidPosition);
    }
    let producer = IdentityPublicKey::from_bytes(&header.producer)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)?;
    producer
        .verify(&header.hash(), producer_signature)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)
}

fn directory_inclusion_proof_depth(mut commitment_count: usize) -> usize {
    let mut depth = 0usize;
    while commitment_count > 1 {
        commitment_count = commitment_count.saturating_add(1) / 2;
        depth = depth.saturating_add(1);
    }
    depth
}

fn validate_directory_block_position(
    height: u64,
    timestamp: u64,
    prev_block_hash: &[u8; 32],
    previous_timestamp: u64,
) -> Result<(), DirectoryCommitmentValidationError> {
    if height == 0 {
        return Err(DirectoryCommitmentValidationError::InvalidHeight);
    }
    let genesis_position_valid = if height == 1 {
        prev_block_hash == &[0u8; 32]
    } else {
        prev_block_hash != &[0u8; 32]
    };
    if !genesis_position_valid {
        return Err(DirectoryCommitmentValidationError::InvalidPreviousHash);
    }
    if timestamp == 0 || timestamp < previous_timestamp {
        return Err(DirectoryCommitmentValidationError::InvalidTimestamp);
    }
    Ok(())
}

fn validate_directory_commitments(
    commitments: &[DirectoryDescriptorCommitmentV1],
) -> Result<(), DirectoryCommitmentValidationError> {
    if commitments.is_empty() {
        return Err(DirectoryCommitmentValidationError::EmptyBlock);
    }
    if commitments.len() > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK {
        return Err(DirectoryCommitmentValidationError::TooManyCommitments);
    }
    if commitments.iter().any(|entry| !entry.structurally_valid()) {
        return Err(DirectoryCommitmentValidationError::InvalidCommitment);
    }
    if commitments.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(DirectoryCommitmentValidationError::NonCanonicalOrder);
    }
    if commitments.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(DirectoryCommitmentValidationError::DuplicateCommitment);
    }
    Ok(())
}

// ============================================
// Directory Observation Checkpoint V1
// ============================================

/// One exact producer prefix included in an observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DirectoryObservationTipV1 {
    /// Producer identity whose independently signed chain was observed.
    pub producer: [u8; 32],
    /// Accepted one-based prefix height.
    pub tip_height: u64,
    /// Producer-signed block hash at `tip_height`.
    pub tip_hash: [u8; 32],
}

impl DirectoryObservationTipV1 {
    fn structurally_valid(&self) -> bool {
        self.producer != [0u8; 32] && self.tip_height > 0 && self.tip_hash != [0u8; 32]
    }
}

/// Observer-signed evidence for one complete configured producer-tip set.
///
/// The overlap root is recomputable from retained producer blocks and public
/// descriptor commitments. This object records what one observer verified; it
/// is not consensus, finality, fork choice, or a quorum certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryObservationCheckpointV1 {
    /// Stable hashing/signature contract version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// One-based observer-local checkpoint sequence.
    pub sequence: u64,
    /// Observer timestamp in Unix epoch seconds.
    pub observed_at: u64,
    /// Prior checkpoint hash, or zero for sequence one.
    pub previous_checkpoint_hash: [u8; 32],
    /// Node identity that created this observation.
    pub observer: [u8; 32],
    /// Number of configured producers represented by this complete checkpoint.
    pub configured_producer_count: u16,
    /// Canonically sorted exact producer tips.
    pub producer_tips: Vec<DirectoryObservationTipV1>,
    /// Deterministic overlap root recomputed from the represented prefixes.
    pub observation_root: [u8; 32],
    /// Ed25519 observer signature over [`Self::hash`].
    #[serde(with = "serde_bytes64")]
    pub observer_signature: [u8; 64],
}

/// Validation failures for Directory Observation Checkpoint V1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationCheckpointValidationError {
    /// The hashing/signature contract version is unsupported.
    UnsupportedVersion,
    /// The checkpoint belongs to another Directory Chain.
    WrongChain,
    /// Sequence or prior-checkpoint linkage is invalid.
    InvalidPosition,
    /// Timestamp is zero, regressed, or too far in the future.
    InvalidTimestamp,
    /// The configured producer count is outside the V1 bound.
    InvalidProducerCount,
    /// A producer tip contains a sentinel or duplicates the observer.
    InvalidProducerTip,
    /// Producer tips are not in canonical ascending order.
    NonCanonicalProducerOrder,
    /// The same producer occurs more than once.
    DuplicateProducer,
    /// The overlap root uses the zero sentinel.
    InvalidObservationRoot,
    /// The observer public key is malformed.
    InvalidObserver,
    /// The observer signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for DirectoryObservationCheckpointValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory observation checkpoint version",
            Self::WrongChain => "directory observation checkpoint belongs to another chain",
            Self::InvalidPosition => "directory observation checkpoint position is invalid",
            Self::InvalidTimestamp => "directory observation checkpoint timestamp is invalid",
            Self::InvalidProducerCount => {
                "directory observation checkpoint producer count is invalid"
            }
            Self::InvalidProducerTip => "directory observation checkpoint producer tip is invalid",
            Self::NonCanonicalProducerOrder => {
                "directory observation checkpoint producers are not canonically ordered"
            }
            Self::DuplicateProducer => {
                "directory observation checkpoint contains a duplicate producer"
            }
            Self::InvalidObservationRoot => {
                "directory observation checkpoint overlap root is invalid"
            }
            Self::InvalidObserver => "directory observation checkpoint observer is invalid",
            Self::InvalidSignature => "directory observation checkpoint signature is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryObservationCheckpointValidationError {}

impl DirectoryObservationCheckpointV1 {
    /// Builds and signs one complete, canonical observation checkpoint.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCheckpointValidationError`] when sequence,
    /// timestamp, producer tips, root, or observer identity is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new_signed(
        sequence: u64,
        observed_at: u64,
        previous_checkpoint_hash: [u8; 32],
        configured_producer_count: u16,
        mut producer_tips: Vec<DirectoryObservationTipV1>,
        observation_root: [u8; 32],
        identity: &IdentityKeyPair,
    ) -> Result<Self, DirectoryObservationCheckpointValidationError> {
        producer_tips.sort_unstable();
        let mut checkpoint = Self {
            protocol_version: DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            sequence,
            observed_at,
            previous_checkpoint_hash,
            observer: identity.public_key_bytes(),
            configured_producer_count,
            producer_tips,
            observation_root,
            observer_signature: [0u8; 64],
        };
        checkpoint.validate_structure()?;
        checkpoint.observer_signature = identity.sign(&checkpoint.hash());
        Ok(checkpoint)
    }

    /// Computes the domain-separated canonical checkpoint identity.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationCheckpoint-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.sequence.to_le_bytes());
        hasher.update(self.observed_at.to_le_bytes());
        hasher.update(self.previous_checkpoint_hash);
        hasher.update(self.observer);
        hasher.update(self.configured_producer_count.to_le_bytes());
        hasher.update(
            u64::try_from(self.producer_tips.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for tip in &self.producer_tips {
            hasher.update(tip.producer);
            hasher.update(tip.tip_height.to_le_bytes());
            hasher.update(tip.tip_hash);
        }
        hasher.update(self.observation_root);
        hasher.finalize().into()
    }

    /// Verifies structure, position, timestamp, observer identity, and signature.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCheckpointValidationError`] on any
    /// contract, continuity, time, canonicalization, identity, or signature
    /// mismatch.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        expected_sequence: u64,
        expected_previous_hash: &[u8; 32],
        previous_observed_at: u64,
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCheckpointValidationError> {
        self.validate_structure()?;
        if &self.chain_id != expected_chain_id {
            return Err(DirectoryObservationCheckpointValidationError::WrongChain);
        }
        if self.sequence != expected_sequence
            || &self.previous_checkpoint_hash != expected_previous_hash
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidPosition);
        }
        if self.observed_at < previous_observed_at
            || self.observed_at
                > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidTimestamp);
        }
        IdentityPublicKey::from_bytes(&self.observer)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidObserver)?
            .verify(&self.hash(), &self.observer_signature)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidSignature)
    }

    /// Verifies a standalone checkpoint's structure, chain, time, and observer
    /// signature without claiming knowledge of the observer's prior sequence.
    ///
    /// An external witness uses this before independently recomputing every
    /// referenced producer prefix and the observation root from its own store.
    /// Sequence linkage remains the observer's local append-only invariant and
    /// is deliberately not inferred from a single transported checkpoint.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCheckpointValidationError`] when the
    /// checkpoint is malformed, belongs to another chain, is too far in the
    /// future, or has an invalid observer identity/signature.
    pub fn verify_standalone_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCheckpointValidationError> {
        self.validate_structure()?;
        if &self.chain_id != expected_chain_id {
            return Err(DirectoryObservationCheckpointValidationError::WrongChain);
        }
        if self.observed_at
            > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidTimestamp);
        }
        IdentityPublicKey::from_bytes(&self.observer)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidObserver)?
            .verify(&self.hash(), &self.observer_signature)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidSignature)
    }

    fn validate_structure(&self) -> Result<(), DirectoryObservationCheckpointValidationError> {
        if self.protocol_version != DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1 {
            return Err(DirectoryObservationCheckpointValidationError::UnsupportedVersion);
        }
        let producer_count = usize::from(self.configured_producer_count);
        if !(2..=MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).contains(&producer_count)
            || producer_count != self.producer_tips.len()
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidProducerCount);
        }
        let genesis_position_valid = if self.sequence == 1 {
            self.previous_checkpoint_hash == [0u8; 32]
        } else {
            self.sequence > 1 && self.previous_checkpoint_hash != [0u8; 32]
        };
        if !genesis_position_valid {
            return Err(DirectoryObservationCheckpointValidationError::InvalidPosition);
        }
        if self.observed_at == 0 {
            return Err(DirectoryObservationCheckpointValidationError::InvalidTimestamp);
        }
        if self.observer == [0u8; 32]
            || self
                .producer_tips
                .iter()
                .any(|tip| !tip.structurally_valid() || tip.producer == self.observer)
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidProducerTip);
        }
        if self.producer_tips.windows(2).any(|tips| tips[0] > tips[1]) {
            return Err(DirectoryObservationCheckpointValidationError::NonCanonicalProducerOrder);
        }
        if self
            .producer_tips
            .windows(2)
            .any(|tips| tips[0].producer == tips[1].producer)
        {
            return Err(DirectoryObservationCheckpointValidationError::DuplicateProducer);
        }
        if self.observation_root == [0u8; 32] {
            return Err(DirectoryObservationCheckpointValidationError::InvalidObservationRoot);
        }
        Ok(())
    }
}

// ============================================
// Portable Directory Observation Certificate V1
// ============================================

/// One independently signed accepted receipt carried by a portable certificate.
///
/// The receipt is a stable projection of
/// [`DirectorySyncMessage::ObservationCheckpointWitnessResponseV1`]. Keeping a
/// standalone representation lets offline verifiers validate a certificate
/// without interpreting an open-ended transport enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryObservationWitnessReceiptV1 {
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Original request identifier bound by the witness signature.
    pub request_id: [u8; 16],
    /// Observer identity copied from the witnessed checkpoint.
    pub observer: [u8; 32],
    /// Observer-local checkpoint sequence.
    pub checkpoint_sequence: u64,
    /// Exact canonical checkpoint hash evaluated by the witness.
    pub checkpoint_hash: [u8; 32],
    /// Independent witness identity.
    pub responder: [u8; 32],
    /// Witness response time in Unix epoch seconds.
    pub response_timestamp: u64,
    /// Stable witness outcome. Portable certificates accept only `accepted`.
    pub outcome: u8,
    /// Witness signature over every preceding field.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl DirectoryObservationWitnessReceiptV1 {
    /// Extracts a standalone receipt from the existing Directory Sync frame.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCertificateValidationError::InvalidReceiptContract`]
    /// when `message` is not an observation-witness response.
    pub fn from_sync_message(
        message: &DirectorySyncMessage,
    ) -> Result<Self, DirectoryObservationCertificateValidationError> {
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = message
        else {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptContract);
        };
        Ok(Self {
            chain_id: *chain_id,
            request_id: *request_id,
            observer: *observer,
            checkpoint_sequence: *checkpoint_sequence,
            checkpoint_hash: *checkpoint_hash,
            responder: *responder,
            response_timestamp: *response_timestamp,
            outcome: *outcome,
            signature: *signature,
        })
    }

    /// Recreates the backward-compatible Directory Sync response frame.
    #[must_use]
    pub const fn to_sync_message(self) -> DirectorySyncMessage {
        DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: self.chain_id,
            request_id: self.request_id,
            observer: self.observer,
            checkpoint_sequence: self.checkpoint_sequence,
            checkpoint_hash: self.checkpoint_hash,
            responder: self.responder,
            response_timestamp: self.response_timestamp,
            outcome: self.outcome,
            signature: self.signature,
        }
    }

    /// Computes a stable receipt identity including its witness signature.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &self.chain_id,
            &self.request_id,
            &self.observer,
            self.checkpoint_sequence,
            &self.checkpoint_hash,
            &self.responder,
            self.response_timestamp,
            self.outcome,
        );
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationWitnessReceipt-v1");
        hasher.update(signing_bytes);
        hasher.update(self.signature);
        hasher.finalize().into()
    }

    fn verify_for_checkpoint_at(
        &self,
        checkpoint: &DirectoryObservationCheckpointV1,
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCertificateValidationError> {
        let expected_checkpoint_hash = checkpoint.hash();
        if self.chain_id != checkpoint.chain_id
            || self.observer != checkpoint.observer
            || self.checkpoint_sequence != checkpoint.sequence
            || self.checkpoint_hash != expected_checkpoint_hash
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptContract);
        }
        if self.outcome != DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1
            || self.responder == [0u8; 32]
            || self.responder == checkpoint.observer
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptContract);
        }
        if self.response_timestamp < checkpoint.observed_at
            || self.response_timestamp
                > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptTimestamp);
        }
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &self.chain_id,
            &self.request_id,
            &self.observer,
            self.checkpoint_sequence,
            &self.checkpoint_hash,
            &self.responder,
            self.response_timestamp,
            self.outcome,
        );
        IdentityPublicKey::from_bytes(&self.responder)
            .map_err(|_| DirectoryObservationCertificateValidationError::InvalidWitness)?
            .verify(&signing_bytes, &self.signature)
            .map_err(|_| DirectoryObservationCertificateValidationError::InvalidReceiptSignature)
    }
}

/// Portable, bounded evidence that independent nodes recomputed one checkpoint.
///
/// This package is not signed by an aggregator: every contained statement is
/// verified against the observer or witness identity that authored it. A
/// threshold records the exporting operator's evidence target only. It is not
/// validator membership, voting weight, fork choice, consensus, or finality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryObservationCertificateV1 {
    /// Stable certificate contract version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Exact observer-signed checkpoint covered by every receipt.
    pub checkpoint: DirectoryObservationCheckpointV1,
    /// Minimum distinct receipts required by the exporting operator.
    pub minimum_witnesses: u16,
    /// Canonically responder-sorted accepted witness receipts.
    pub receipts: Vec<DirectoryObservationWitnessReceiptV1>,
}

/// Fail-closed portable observation-certificate validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationCertificateValidationError {
    /// The certificate contract version is unsupported.
    UnsupportedVersion,
    /// Certificate and checkpoint chain identifiers do not match the verifier.
    WrongChain,
    /// The embedded observer checkpoint is invalid.
    InvalidCheckpoint,
    /// The configured receipt threshold is zero or outside the protocol bound.
    InvalidThreshold,
    /// The receipt set is below threshold or outside the protocol bound.
    InvalidReceiptCount,
    /// Receipts are not sorted by witness identity.
    NonCanonicalWitnessOrder,
    /// The same witness appears more than once.
    DuplicateWitness,
    /// A receipt does not bind the exact accepted checkpoint.
    InvalidReceiptContract,
    /// A witness timestamp predates the checkpoint or is too far in the future.
    InvalidReceiptTimestamp,
    /// A witness public key is malformed.
    InvalidWitness,
    /// A receipt signature is invalid.
    InvalidReceiptSignature,
}

impl std::fmt::Display for DirectoryObservationCertificateValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory observation certificate version",
            Self::WrongChain => "directory observation certificate belongs to another chain",
            Self::InvalidCheckpoint => "directory observation certificate checkpoint is invalid",
            Self::InvalidThreshold => {
                "directory observation certificate witness threshold is invalid"
            }
            Self::InvalidReceiptCount => {
                "directory observation certificate receipt count is invalid"
            }
            Self::NonCanonicalWitnessOrder => {
                "directory observation certificate witnesses are not canonically ordered"
            }
            Self::DuplicateWitness => {
                "directory observation certificate contains a duplicate witness"
            }
            Self::InvalidReceiptContract => {
                "directory observation certificate receipt contract is invalid"
            }
            Self::InvalidReceiptTimestamp => {
                "directory observation certificate receipt timestamp is invalid"
            }
            Self::InvalidWitness => "directory observation certificate witness identity is invalid",
            Self::InvalidReceiptSignature => {
                "directory observation certificate receipt signature is invalid"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryObservationCertificateValidationError {}

impl DirectoryObservationCertificateV1 {
    /// Builds a canonical certificate and verifies every contained signature.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCertificateValidationError`] when the
    /// threshold, checkpoint, receipt binding, order, timestamp, identity, or
    /// signature is invalid.
    pub fn new_verified(
        checkpoint: DirectoryObservationCheckpointV1,
        minimum_witnesses: u16,
        mut receipts: Vec<DirectoryObservationWitnessReceiptV1>,
        verifier_observed_at: u64,
    ) -> Result<Self, DirectoryObservationCertificateValidationError> {
        receipts.sort_unstable_by_key(|receipt| receipt.responder);
        let certificate = Self {
            protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
            chain_id: checkpoint.chain_id,
            checkpoint,
            minimum_witnesses,
            receipts,
        };
        certificate.verify_at(&certificate.chain_id, verifier_observed_at)?;
        Ok(certificate)
    }

    /// Computes the stable identity of the checkpoint and exact receipt set.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationCertificate-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.checkpoint.hash());
        hasher.update(self.minimum_witnesses.to_le_bytes());
        hasher.update(
            u64::try_from(self.receipts.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for receipt in &self.receipts {
            hasher.update(receipt.hash());
        }
        hasher.finalize().into()
    }

    /// Verifies the complete portable evidence package.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCertificateValidationError`] for any
    /// version, chain, checkpoint, threshold, canonical order, binding,
    /// timestamp, identity, or signature failure.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCertificateValidationError> {
        if self.protocol_version != DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1 {
            return Err(DirectoryObservationCertificateValidationError::UnsupportedVersion);
        }
        if &self.chain_id != expected_chain_id || self.checkpoint.chain_id != self.chain_id {
            return Err(DirectoryObservationCertificateValidationError::WrongChain);
        }
        self.checkpoint
            .verify_standalone_at(expected_chain_id, verifier_observed_at)
            .map_err(|_| DirectoryObservationCertificateValidationError::InvalidCheckpoint)?;
        let threshold = usize::from(self.minimum_witnesses);
        if !(1..=MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).contains(&threshold) {
            return Err(DirectoryObservationCertificateValidationError::InvalidThreshold);
        }
        if self.receipts.len() < threshold
            || self.receipts.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptCount);
        }
        if self
            .receipts
            .windows(2)
            .any(|receipts| receipts[0].responder > receipts[1].responder)
        {
            return Err(DirectoryObservationCertificateValidationError::NonCanonicalWitnessOrder);
        }
        if self
            .receipts
            .windows(2)
            .any(|receipts| receipts[0].responder == receipts[1].responder)
        {
            return Err(DirectoryObservationCertificateValidationError::DuplicateWitness);
        }
        for receipt in &self.receipts {
            receipt.verify_for_checkpoint_at(&self.checkpoint, verifier_observed_at)?;
        }
        Ok(())
    }
}

// ============================================
// Portable Route-Domain Attestation V1
// ============================================

/// One independently signed, time-bounded opaque route-domain assignment.
///
/// [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] The 128-bit domain token is
/// deliberately opaque. A verifier learns only that one pinned attestor signed
/// the exact node/token pair for this interval; the statement does not encode
/// an operator, provider, ASN, geography, ownership claim, or public label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDomainAttestationV1 {
    /// Stable statement contract version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Node identity assigned to the opaque route domain.
    pub subject_node_id: [u8; 32],
    /// Opaque 128-bit route-domain token.
    pub route_domain: [u8; 16],
    /// Independent attestor identity.
    pub attestor_node_id: [u8; 32],
    /// Statement creation time in Unix epoch seconds.
    pub issued_at: u64,
    /// Exclusive statement expiry in Unix epoch seconds.
    pub expires_at: u64,
    /// Attestor signature over every preceding field.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

/// Fail-closed route-domain attestation validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDomainAttestationValidationError {
    /// The statement contract version is unsupported.
    UnsupportedVersion,
    /// The statement belongs to another Directory Chain.
    WrongChain,
    /// The subject identity is malformed.
    InvalidSubject,
    /// The route-domain token is the reserved zero value.
    InvalidRouteDomain,
    /// The attestor is malformed or self-attests the subject.
    InvalidAttestor,
    /// The issue/expiry interval is malformed, too long, or too far ahead.
    InvalidTimestamp,
    /// The statement is expired at the verifier's current time.
    Expired,
    /// The attestor signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for RouteDomainAttestationValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported route-domain attestation version",
            Self::WrongChain => "route-domain attestation belongs to another chain",
            Self::InvalidSubject => "route-domain attestation subject is invalid",
            Self::InvalidRouteDomain => "route-domain attestation token is invalid",
            Self::InvalidAttestor => "route-domain attestation signer is invalid",
            Self::InvalidTimestamp => "route-domain attestation interval is invalid",
            Self::Expired => "route-domain attestation is expired",
            Self::InvalidSignature => "route-domain attestation signature is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RouteDomainAttestationValidationError {}

impl RouteDomainAttestationV1 {
    /// Builds and signs one bounded opaque route-domain assignment.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationValidationError`] for a malformed
    /// subject, zero token, self-attestation, or invalid validity interval.
    pub fn new_signed(
        subject_node_id: [u8; 32],
        route_domain: [u8; 16],
        issued_at: u64,
        expires_at: u64,
        attestor: &IdentityKeyPair,
    ) -> Result<Self, RouteDomainAttestationValidationError> {
        let mut attestation = Self {
            protocol_version: ROUTE_DOMAIN_ATTESTATION_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            subject_node_id,
            route_domain,
            attestor_node_id: attestor.public_key_bytes(),
            issued_at,
            expires_at,
            signature: [0u8; 64],
        };
        attestation.validate_unsigned_at(&attestation.chain_id, issued_at)?;
        attestation.signature = attestor.sign(&attestation.signing_bytes());
        Ok(attestation)
    }

    /// Returns the canonical digest signed by the attestor.
    #[must_use]
    pub fn signing_bytes(&self) -> [u8; 32] {
        route_domain_attestation_signing_bytes(
            &self.chain_id,
            &self.subject_node_id,
            &self.route_domain,
            &self.attestor_node_id,
            self.issued_at,
            self.expires_at,
        )
    }

    /// Computes a stable identity including the exact signature.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-RouteDomainAttestationHash-v1");
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }

    /// Verifies structure, bounded validity, identity keys, and signature.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationValidationError`] when any field,
    /// timestamp, chain binding, or signature is invalid.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), RouteDomainAttestationValidationError> {
        self.validate_unsigned_at(expected_chain_id, verifier_observed_at)?;
        IdentityPublicKey::from_bytes(&self.attestor_node_id)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidAttestor)?
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidSignature)
    }

    fn validate_unsigned_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), RouteDomainAttestationValidationError> {
        if self.protocol_version != ROUTE_DOMAIN_ATTESTATION_VERSION_V1 {
            return Err(RouteDomainAttestationValidationError::UnsupportedVersion);
        }
        if &self.chain_id != expected_chain_id {
            return Err(RouteDomainAttestationValidationError::WrongChain);
        }
        IdentityPublicKey::from_bytes(&self.subject_node_id)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidSubject)?;
        if self.route_domain == [0u8; 16] {
            return Err(RouteDomainAttestationValidationError::InvalidRouteDomain);
        }
        IdentityPublicKey::from_bytes(&self.attestor_node_id)
            .map_err(|_| RouteDomainAttestationValidationError::InvalidAttestor)?;
        if self.attestor_node_id == self.subject_node_id {
            return Err(RouteDomainAttestationValidationError::InvalidAttestor);
        }
        let lifetime = self
            .expires_at
            .checked_sub(self.issued_at)
            .ok_or(RouteDomainAttestationValidationError::InvalidTimestamp)?;
        if self.issued_at == 0
            || lifetime == 0
            || lifetime > MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1
            || self.issued_at
                > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(RouteDomainAttestationValidationError::InvalidTimestamp);
        }
        if self.expires_at <= verifier_observed_at {
            return Err(RouteDomainAttestationValidationError::Expired);
        }
        Ok(())
    }
}

/// Portable, bounded evidence for one opaque route-domain assignment.
///
/// Every statement retains its original attestor signature. The package has no
/// aggregator signature and carries no network-wide threshold; each verifier
/// must apply its own local pinned-attestor policy with
/// [`Self::verify_with_policy_at`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDomainAttestationCertificateV1 {
    /// Stable portable certificate version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Node identity covered by every statement.
    pub subject_node_id: [u8; 32],
    /// Opaque route-domain token covered by every statement.
    pub route_domain: [u8; 16],
    /// Canonically attestor-sorted signed statements.
    pub attestations: Vec<RouteDomainAttestationV1>,
}

/// Fail-closed portable route-domain certificate validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDomainAttestationCertificateValidationError {
    /// The certificate contract version is unsupported.
    UnsupportedVersion,
    /// The certificate belongs to another Directory Chain.
    WrongChain,
    /// The certificate subject is malformed.
    InvalidSubject,
    /// The certificate route-domain token is malformed.
    InvalidRouteDomain,
    /// The statement count is zero or exceeds the protocol bound.
    InvalidAttestationCount,
    /// Statements are not canonically sorted by attestor identity.
    NonCanonicalAttestorOrder,
    /// The same attestor appears more than once.
    DuplicateAttestor,
    /// A statement does not bind the exact certificate subject and token.
    InvalidAttestationContract,
    /// One statement has an invalid time, identity, chain, or signature.
    InvalidAttestation,
    /// The verifier's local pin set or threshold is malformed.
    InvalidPolicy,
    /// Too few currently valid statements came from locally pinned attestors.
    InsufficientTrustedAttestations,
}

impl std::fmt::Display for RouteDomainAttestationCertificateValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported route-domain certificate version",
            Self::WrongChain => "route-domain certificate belongs to another chain",
            Self::InvalidSubject => "route-domain certificate subject is invalid",
            Self::InvalidRouteDomain => "route-domain certificate token is invalid",
            Self::InvalidAttestationCount => "route-domain certificate statement count is invalid",
            Self::NonCanonicalAttestorOrder => {
                "route-domain certificate attestors are not canonically ordered"
            }
            Self::DuplicateAttestor => "route-domain certificate contains a duplicate attestor",
            Self::InvalidAttestationContract => {
                "route-domain certificate statement contract is invalid"
            }
            Self::InvalidAttestation => "route-domain certificate statement is invalid",
            Self::InvalidPolicy => "route-domain attestor policy is invalid",
            Self::InsufficientTrustedAttestations => {
                "route-domain certificate does not satisfy the pinned attestor threshold"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RouteDomainAttestationCertificateValidationError {}

impl RouteDomainAttestationCertificateV1 {
    /// Builds a canonical certificate and verifies every statement signature.
    ///
    /// This does not establish trust in the attestors. Call
    /// [`Self::verify_with_policy_at`] with local pins before routing.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationCertificateValidationError`] for an
    /// empty/oversized set, invalid binding, timestamp, identity, or signature.
    pub fn new_verified(
        subject_node_id: [u8; 32],
        route_domain: [u8; 16],
        mut attestations: Vec<RouteDomainAttestationV1>,
        verifier_observed_at: u64,
    ) -> Result<Self, RouteDomainAttestationCertificateValidationError> {
        attestations.sort_unstable_by_key(|attestation| attestation.attestor_node_id);
        let certificate = Self {
            protocol_version: ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            subject_node_id,
            route_domain,
            attestations,
        };
        certificate.verify_at(&certificate.chain_id, verifier_observed_at)?;
        Ok(certificate)
    }

    /// Computes the stable identity of the exact canonical statement set.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-RouteDomainAttestationCertificate-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.subject_node_id);
        hasher.update(self.route_domain);
        hasher.update(
            u64::try_from(self.attestations.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for attestation in &self.attestations {
            hasher.update(attestation.hash());
        }
        hasher.finalize().into()
    }

    /// Verifies the certificate structure and every independent signature.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationCertificateValidationError`] for any
    /// invalid version, chain, binding, ordering, duplicate, time, or signature.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), RouteDomainAttestationCertificateValidationError> {
        if self.protocol_version != ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1 {
            return Err(RouteDomainAttestationCertificateValidationError::UnsupportedVersion);
        }
        if &self.chain_id != expected_chain_id {
            return Err(RouteDomainAttestationCertificateValidationError::WrongChain);
        }
        IdentityPublicKey::from_bytes(&self.subject_node_id)
            .map_err(|_| RouteDomainAttestationCertificateValidationError::InvalidSubject)?;
        if self.route_domain == [0u8; 16] {
            return Err(RouteDomainAttestationCertificateValidationError::InvalidRouteDomain);
        }
        if !(1..=MAX_ROUTE_DOMAIN_ATTESTATIONS_V1).contains(&self.attestations.len()) {
            return Err(RouteDomainAttestationCertificateValidationError::InvalidAttestationCount);
        }
        if self
            .attestations
            .windows(2)
            .any(|items| items[0].attestor_node_id > items[1].attestor_node_id)
        {
            return Err(
                RouteDomainAttestationCertificateValidationError::NonCanonicalAttestorOrder,
            );
        }
        if self
            .attestations
            .windows(2)
            .any(|items| items[0].attestor_node_id == items[1].attestor_node_id)
        {
            return Err(RouteDomainAttestationCertificateValidationError::DuplicateAttestor);
        }
        for attestation in &self.attestations {
            if attestation.chain_id != self.chain_id
                || attestation.subject_node_id != self.subject_node_id
                || attestation.route_domain != self.route_domain
            {
                return Err(
                    RouteDomainAttestationCertificateValidationError::InvalidAttestationContract,
                );
            }
            attestation
                .verify_at(expected_chain_id, verifier_observed_at)
                .map_err(|_| {
                    RouteDomainAttestationCertificateValidationError::InvalidAttestation
                })?;
        }
        Ok(())
    }

    /// Applies one verifier-local pinned-attestor policy.
    ///
    /// The returned count includes only valid statements whose identities are
    /// present in `allowed_attestors`. Unpinned statements remain portable but
    /// do not contribute to the local threshold.
    ///
    /// # Errors
    /// Returns [`RouteDomainAttestationCertificateValidationError`] when the
    /// certificate is invalid, pins/threshold are malformed, or too few pinned
    /// attestors signed the exact assignment.
    pub fn verify_with_policy_at(
        &self,
        expected_chain_id: &[u8; 32],
        allowed_attestors: &[[u8; 32]],
        minimum_attestors: usize,
        verifier_observed_at: u64,
    ) -> Result<usize, RouteDomainAttestationCertificateValidationError> {
        self.verify_at(expected_chain_id, verifier_observed_at)?;
        let mut canonical_allowed = allowed_attestors.to_vec();
        canonical_allowed.sort_unstable();
        let original_count = canonical_allowed.len();
        canonical_allowed.dedup();
        if canonical_allowed.len() != original_count
            || !(1..=MAX_ROUTE_DOMAIN_ATTESTATIONS_V1).contains(&canonical_allowed.len())
            || minimum_attestors == 0
            || minimum_attestors > canonical_allowed.len()
            || canonical_allowed.iter().any(|attestor| {
                *attestor == [0u8; 32]
                    || *attestor == self.subject_node_id
                    || IdentityPublicKey::from_bytes(attestor).is_err()
            })
        {
            return Err(RouteDomainAttestationCertificateValidationError::InvalidPolicy);
        }
        let trusted = self
            .attestations
            .iter()
            .filter(|attestation| {
                canonical_allowed
                    .binary_search(&attestation.attestor_node_id)
                    .is_ok()
            })
            .count();
        if trusted < minimum_attestors {
            return Err(
                RouteDomainAttestationCertificateValidationError::InsufficientTrustedAttestations,
            );
        }
        Ok(trusted)
    }
}

// ============================================
// Directory Sync V1
// ============================================

/// Authenticated, bounded wire messages for one producer's Directory Chain.
///
/// A responder serves only the chain signed by its own node identity. Requests
/// are separately signed by an admitted peer. Descriptor objects remain public
/// node metadata; no user, route, traffic, or encrypted payload data belongs in
/// this protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectorySyncMessage {
    /// Requests the responder's current locally audited chain tip.
    TipRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over canonical tip-request bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns the responder's current locally audited chain tip.
    TipResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for this chain.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Current tip height, or zero for an empty chain.
        tip_height: u64,
        /// Current tip hash, or all zeroes for an empty chain.
        tip_hash: [u8; 32],
        /// Current tip block timestamp, or zero for an empty chain.
        tip_timestamp: u64,
        /// Responder signature over canonical tip-response bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests a contiguous bounded range from the responder's own chain.
    BlockRangeRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// First one-based block height requested.
        from_height: u64,
        /// Maximum number of blocks requested.
        limit: u16,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over canonical range-request bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns a contiguous bounded block range and a signed current tip.
    BlockRangeResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for every returned block.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Contiguous blocks in ascending height order.
        blocks: Vec<DirectoryCommitmentBlockV1>,
        /// Whether the signed tip extends beyond this page.
        has_more: bool,
        /// Current responder tip height.
        tip_height: u64,
        /// Current responder tip hash.
        tip_hash: [u8; 32],
        /// Responder signature over request binding, block hashes, and tip.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests exact content-addressed signed descriptor objects.
    DescriptorObjectsRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Descriptor commitment hashes, in required response order.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over canonical object-request bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns exact signed descriptor objects in requested hash order.
    DescriptorObjectsResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for the source chain.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Requested hashes in the exact order represented by `objects`.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Authenticated public node descriptors committed by those hashes.
        objects: Vec<SignedNodeDescriptor>,
        /// Responder signature over request binding and ordered object hashes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests an independent witness decision for one exact signed checkpoint.
    ///
    /// This variant is appended to preserve every existing bincode enum index.
    /// The responder must recompute producer-prefix evidence from its own
    /// replica store; validating only the observer signature is insufficient.
    ObservationCheckpointWitnessRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Requester identity; must equal `checkpoint.observer`.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Canonical observer-signed checkpoint to recompute independently.
        checkpoint: DirectoryObservationCheckpointV1,
        /// Requester signature binding the exact checkpoint hash.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one signed external decision for an exact checkpoint.
    ObservationCheckpointWitnessResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Observer identity copied from the witnessed checkpoint.
        observer: [u8; 32],
        /// Observer-local sequence copied from the witnessed checkpoint.
        checkpoint_sequence: u64,
        /// Exact canonical checkpoint hash evaluated by the witness.
        checkpoint_hash: [u8; 32],
        /// Independent witness identity.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// One stable `DIRECTORY_OBSERVATION_WITNESS_*_V1` outcome code.
        outcome: u8,
        /// Responder signature over every response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests a bounded producer range from an audited evidence carrier.
    ///
    /// This variant is appended to preserve every existing bincode enum index.
    /// Returned blocks remain signed by `producer`; `carrier` only transports
    /// evidence that it has already imported and audited.
    ReplicaBlockRangeRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Producer whose signed replica prefix is requested.
        producer: [u8; 32],
        /// First one-based block height requested.
        from_height: u64,
        /// Maximum number of blocks requested.
        limit: u16,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns a bounded producer-signed range through an audited carrier.
    ReplicaBlockRangeResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer identity carried by every returned block.
        producer: [u8; 32],
        /// Independent node transporting its audited replica evidence.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Contiguous producer-signed blocks in ascending height order.
        blocks: Vec<DirectoryCommitmentBlockV1>,
        /// Whether the audited producer tip extends beyond this page.
        has_more: bool,
        /// Audited producer tip height at the carrier.
        tip_height: u64,
        /// Audited producer tip hash at the carrier.
        tip_hash: [u8; 32],
        /// Carrier signature binding request, producer, block hashes, and tip.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests exact producer descriptor objects from an audited carrier.
    ReplicaDescriptorObjectsRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Producer namespace containing every requested descriptor object.
        producer: [u8; 32],
        /// Descriptor commitment hashes, in required response order.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns exact producer descriptor objects through an audited carrier.
    ReplicaDescriptorObjectsResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer namespace represented by `objects`.
        producer: [u8; 32],
        /// Independent node transporting its audited replica evidence.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Requested hashes in the exact order represented by `objects`.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Signed public descriptors committed by the producer blocks.
        objects: Vec<SignedNodeDescriptor>,
        /// Carrier signature binding request, producer, and ordered hashes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests durable external retention of one opaque witness-policy head.
    ///
    /// This variant is appended to preserve every existing bincode enum index.
    /// Policy member identities and endpoints are deliberately absent. The
    /// witness validates observer authentication and monotonic continuity, but
    /// does not approve the operator's policy or interpret its opaque digest.
    ObservationWitnessPolicyAnchorRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Node whose local witness policy is being externally anchored.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Positive observer-local policy epoch.
        policy_epoch: u64,
        /// Previous policy digest, or zero only for epoch one.
        previous_policy_digest: [u8; 32],
        /// Opaque digest of the observer-signed complete local policy object.
        policy_digest: [u8; 32],
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one signed external policy-head retention decision.
    ObservationWitnessPolicyAnchorResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Observer identity copied from the anchor request.
        observer: [u8; 32],
        /// Exact observer-local policy epoch evaluated by the witness.
        policy_epoch: u64,
        /// Exact opaque policy digest evaluated by the witness.
        policy_digest: [u8; 32],
        /// Independent witness identity.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// One stable `DIRECTORY_POLICY_ANCHOR_*_V1` outcome code.
        outcome: u8,
        /// Responder signature over every response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests the responder's latest portable observation certificate.
    ///
    /// [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] This variant is appended to
    /// preserve every existing bincode enum index. The server must admit only
    /// authenticated pinned peers and must never expose this frame publicly.
    ObservationCertificateRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting pinned peer.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one exact portable observation certificate to a pinned peer.
    ///
    /// The responder authenticates transport only. Receivers must still verify
    /// the observer checkpoint, every witness receipt, local witness pins, and
    /// their own certificate-age policy before importing the evidence.
    ObservationCertificateResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Requester identity copied from the authenticated request.
        requester: [u8; 32],
        /// Pinned responder transporting its locally verified certificate.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// SHA-256 of the exact canonical certificate frame.
        certificate_sha256: [u8; 32],
        /// Canonical bounded portable observation-certificate frame.
        #[serde(with = "serde_bytes")]
        certificate_frame: Vec<u8>,
        /// Responder signature binding metadata, digest, and frame length.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Relays one exact observer-signed witness request through a carrier.
    ///
    /// [WITNESS-CARRIER 2026-07-26 by Codex] This variant is appended to
    /// preserve every existing bincode enum index. The carrier is transport
    /// only: it cannot alter the inner request, choose another witness, or
    /// produce an accepted witness receipt.
    ObservationCheckpointWitnessCarrierRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random carrier-request identifier used for replay protection.
        request_id: [u8; 16],
        /// Observer that signed both this envelope and the inner request.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Exact pinned witness that must evaluate the inner request.
        witness: [u8; 32],
        /// SHA-256 of the exact canonical inner witness-request frame.
        witness_request_sha256: [u8; 32],
        /// Canonical observer-signed witness-request frame.
        #[serde(with = "serde_bytes")]
        witness_request_frame: Vec<u8>,
        /// Observer signature binding target, digest, and frame length.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one exact witness-signed response through a carrier.
    ///
    /// The carrier signature authenticates only the bounded transport
    /// envelope. Receivers must independently verify the inner response
    /// against the original observer request and the exact pinned witness.
    ObservationCheckpointWitnessCarrierResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Carrier-request identifier copied from the outer request.
        request_id: [u8; 16],
        /// Observer identity copied from the authenticated outer request.
        requester: [u8; 32],
        /// Exact witness that signed the inner response.
        witness: [u8; 32],
        /// Independent carrier transporting the exact response frame.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// SHA-256 of the exact inner request frame.
        witness_request_sha256: [u8; 32],
        /// SHA-256 of the exact inner response frame.
        witness_response_sha256: [u8; 32],
        /// Canonical witness-signed response frame.
        #[serde(with = "serde_bytes")]
        witness_response_frame: Vec<u8>,
        /// Carrier signature binding request, target, digests, and frame length.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests one compact proof for a descriptor in one exact selected block.
    ///
    /// [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] This variant is
    /// appended to preserve every existing bincode enum index. Server
    /// admission remains restricted to authenticated pinned peers.
    DescriptorInclusionProofRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Independently selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact content-addressed descriptor requested.
        descriptor_hash: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting pinned node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one producer-signed descriptor inclusion proof.
    ///
    /// The responder signature authenticates request/response transport. A
    /// receiver must separately call
    /// [`DirectoryDescriptorInclusionProofV1::verify_at`] with its own pinned
    /// producer and independently selected `block_hash`.
    DescriptorInclusionProofResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for the source block.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Exact selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact requested descriptor content hash.
        descriptor_hash: [u8; 32],
        /// Compact producer-signed inclusion evidence.
        proof: DirectoryDescriptorInclusionProofV1,
        /// Responder signature binding request, proof digest, and response time.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests one exact producer descriptor proof from an audited carrier.
    ///
    /// [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] This variant is appended
    /// to preserve every existing bincode enum index. `producer` is the
    /// original block author; the responding carrier never replaces it.
    ReplicaDescriptorInclusionProofRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Original producer whose signed replica block is selected.
        producer: [u8; 32],
        /// Independently selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact content-addressed descriptor requested.
        descriptor_hash: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the authenticated recovery requester.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one original producer-signed proof through an audited carrier.
    ///
    /// The carrier signature authenticates transport only. Receivers must
    /// independently verify `proof` against `producer` and `block_hash`.
    ReplicaDescriptorInclusionProofResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Original producer whose signed block is represented by `proof`.
        producer: [u8; 32],
        /// Independent node transporting its audited replica evidence.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Exact selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact requested descriptor content hash.
        descriptor_hash: [u8; 32],
        /// Compact original producer-signed inclusion evidence.
        proof: DirectoryDescriptorInclusionProofV1,
        /// Carrier signature binding request, producer, proof, and response.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
}

fn directory_sync_signing_digest<'a>(
    domain: &[u8],
    fields: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update(u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

/// Canonical digest signed by a Directory Sync tip request.
#[must_use]
pub fn directory_tip_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-TipRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a Directory Sync tip response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_tip_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    tip_height: u64,
    tip_hash: &[u8; 32],
    tip_timestamp: u64,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let tip_height = tip_height.to_le_bytes();
    let tip_timestamp = tip_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-TipResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            tip_height.as_slice(),
            tip_hash.as_slice(),
            tip_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a Directory Sync block-range request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_block_range_request_signing_bytes(
    chain_id: &[u8; 32],
    from_height: u64,
    limit: u16,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let from_height = from_height.to_le_bytes();
    let limit = limit.to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-BlockRangeRequest-v1",
        [
            chain_id.as_slice(),
            from_height.as_slice(),
            limit.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a Directory Sync block-range response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_block_range_response_signing_bytes(
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let block_count = u64::try_from(blocks.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let has_more = [u8::from(has_more)];
    let tip_height = tip_height.to_le_bytes();
    let block_hashes = blocks
        .iter()
        .map(DirectoryCommitmentBlockV1::hash)
        .collect::<Vec<_>>();
    let mut fields = Vec::<&[u8]>::with_capacity(block_hashes.len() + 7);
    fields.extend([
        request_id.as_slice(),
        responder.as_slice(),
        response_timestamp.as_slice(),
        block_count.as_slice(),
    ]);
    fields.extend(block_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        has_more.as_slice(),
        tip_height.as_slice(),
        tip_hash.as_slice(),
    ]);
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-BlockRangeResponse-v1", fields)
}

/// Canonical digest signed by a Directory Sync object request.
#[must_use]
pub fn directory_descriptor_objects_request_signing_bytes(
    chain_id: &[u8; 32],
    descriptor_hashes: &[[u8; 32]],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 5);
    fields.extend([chain_id.as_slice(), count.as_slice()]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        request_id.as_slice(),
        requester.as_slice(),
        request_timestamp.as_slice(),
    ]);
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ObjectsRequest-v1", fields)
}

/// Canonical digest signed by a Directory Sync object response.
#[must_use]
pub fn directory_descriptor_objects_response_signing_bytes(
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    descriptor_hashes: &[[u8; 32]],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 4);
    fields.extend([
        request_id.as_slice(),
        responder.as_slice(),
        response_timestamp.as_slice(),
        count.as_slice(),
    ]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ObjectsResponse-v1", fields)
}

/// Canonical digest signed by an exact descriptor-inclusion proof request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_descriptor_inclusion_proof_request_signing_bytes(
    chain_id: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-DescriptorInclusionProofRequest-v1",
        [
            chain_id.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

fn directory_descriptor_inclusion_proof_transport_digest(
    proof: &DirectoryDescriptorInclusionProofV1,
) -> [u8; 32] {
    let proof_version = proof.proof_version.to_le_bytes();
    let block_hash = proof.block_hash();
    let commitment_hash = proof.commitment.hash();
    let commitment_index = proof.commitment_index.to_le_bytes();
    let sibling_count = u64::try_from(proof.sibling_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(proof.sibling_hashes.len() + 6);
    fields.extend([
        proof_version.as_slice(),
        block_hash.as_slice(),
        proof.producer_signature.as_slice(),
        commitment_hash.as_slice(),
        commitment_index.as_slice(),
        sibling_count.as_slice(),
    ]);
    fields.extend(proof.sibling_hashes.iter().map(<[u8; 32]>::as_slice));
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-DescriptorInclusionProof-v1", fields)
}

/// Canonical digest signed by an exact descriptor-inclusion proof response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_descriptor_inclusion_proof_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    proof: &DirectoryDescriptorInclusionProofV1,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let proof_digest = directory_descriptor_inclusion_proof_transport_digest(proof);
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-DescriptorInclusionProofResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            proof_digest.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica descriptor-proof request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_descriptor_inclusion_proof_request_signing_bytes(
    chain_id: &[u8; 32],
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaDescriptorInclusionProofRequest-v1",
        [
            chain_id.as_slice(),
            producer.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica descriptor-proof response carrier.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_descriptor_inclusion_proof_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    producer: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    proof: &DirectoryDescriptorInclusionProofV1,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let proof_digest = directory_descriptor_inclusion_proof_transport_digest(proof);
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaDescriptorInclusionProofResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            producer.as_slice(),
            carrier.as_slice(),
            response_timestamp.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            proof_digest.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica-carrier block-range request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_block_range_request_signing_bytes(
    chain_id: &[u8; 32],
    producer: &[u8; 32],
    from_height: u64,
    limit: u16,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let from_height = from_height.to_le_bytes();
    let limit = limit.to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaBlockRangeRequest-v1",
        [
            chain_id.as_slice(),
            producer.as_slice(),
            from_height.as_slice(),
            limit.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica-carrier block-range response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_block_range_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    producer: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let block_count = u64::try_from(blocks.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let has_more = [u8::from(has_more)];
    let tip_height = tip_height.to_le_bytes();
    let block_hashes = blocks
        .iter()
        .map(DirectoryCommitmentBlockV1::hash)
        .collect::<Vec<_>>();
    let mut fields = Vec::<&[u8]>::with_capacity(block_hashes.len() + 10);
    fields.extend([
        chain_id.as_slice(),
        request_id.as_slice(),
        producer.as_slice(),
        carrier.as_slice(),
        response_timestamp.as_slice(),
        block_count.as_slice(),
    ]);
    fields.extend(block_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        has_more.as_slice(),
        tip_height.as_slice(),
        tip_hash.as_slice(),
    ]);
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaBlockRangeResponse-v1",
        fields,
    )
}

/// Canonical digest signed by a replica-carrier object request.
#[must_use]
pub fn directory_replica_descriptor_objects_request_signing_bytes(
    chain_id: &[u8; 32],
    producer: &[u8; 32],
    descriptor_hashes: &[[u8; 32]],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 7);
    fields.extend([chain_id.as_slice(), producer.as_slice(), count.as_slice()]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        request_id.as_slice(),
        requester.as_slice(),
        request_timestamp.as_slice(),
    ]);
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ReplicaObjectsRequest-v1", fields)
}

/// Canonical digest signed by a replica-carrier object response.
#[must_use]
pub fn directory_replica_descriptor_objects_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    producer: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    descriptor_hashes: &[[u8; 32]],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 7);
    fields.extend([
        chain_id.as_slice(),
        request_id.as_slice(),
        producer.as_slice(),
        carrier.as_slice(),
        response_timestamp.as_slice(),
        count.as_slice(),
    ]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ReplicaObjectsResponse-v1", fields)
}

/// Canonical digest signed by one opaque route-domain attestation.
#[must_use]
pub fn route_domain_attestation_signing_bytes(
    chain_id: &[u8; 32],
    subject_node_id: &[u8; 32],
    route_domain: &[u8; 16],
    attestor_node_id: &[u8; 32],
    issued_at: u64,
    expires_at: u64,
) -> [u8; 32] {
    let issued_at = issued_at.to_le_bytes();
    let expires_at = expires_at.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-RouteDomainAttestation-v1",
        [
            chain_id.as_slice(),
            subject_node_id.as_slice(),
            route_domain.as_slice(),
            attestor_node_id.as_slice(),
            issued_at.as_slice(),
            expires_at.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-checkpoint witness request.
#[must_use]
pub fn directory_observation_witness_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    checkpoint_hash: &[u8; 32],
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
            checkpoint_hash.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-checkpoint witness response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_witness_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    observer: &[u8; 32],
    checkpoint_sequence: u64,
    checkpoint_hash: &[u8; 32],
    responder: &[u8; 32],
    response_timestamp: u64,
    outcome: u8,
) -> [u8; 32] {
    let checkpoint_sequence = checkpoint_sequence.to_le_bytes();
    let response_timestamp = response_timestamp.to_le_bytes();
    let outcome = [outcome];
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            observer.as_slice(),
            checkpoint_sequence.as_slice(),
            checkpoint_hash.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            outcome.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-witness carrier request.
///
/// The digest binds the exact inner frame by both SHA-256 and byte length. The
/// carrier must independently recompute both before forwarding the frame.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_witness_carrier_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    witness: &[u8; 32],
    witness_request_sha256: &[u8; 32],
    witness_request_frame_bytes: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    let witness_request_frame_bytes = witness_request_frame_bytes.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessCarrierRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
            witness.as_slice(),
            witness_request_sha256.as_slice(),
            witness_request_frame_bytes.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-witness carrier response.
///
/// A carrier authenticates bounded transport only. The caller must recompute
/// both frame digests and verify the inner witness response independently.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_witness_carrier_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    witness: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    witness_request_sha256: &[u8; 32],
    witness_response_sha256: &[u8; 32],
    witness_response_frame_bytes: u64,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let witness_response_frame_bytes = witness_response_frame_bytes.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessCarrierResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            witness.as_slice(),
            carrier.as_slice(),
            response_timestamp.as_slice(),
            witness_request_sha256.as_slice(),
            witness_response_sha256.as_slice(),
            witness_response_frame_bytes.as_slice(),
        ],
    )
}

/// Canonical digest signed by an opaque witness-policy anchor request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_policy_anchor_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    policy_epoch: u64,
    previous_policy_digest: &[u8; 32],
    policy_digest: &[u8; 32],
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    let policy_epoch = policy_epoch.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-PolicyAnchorRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
            policy_epoch.as_slice(),
            previous_policy_digest.as_slice(),
            policy_digest.as_slice(),
        ],
    )
}

/// Canonical digest signed by an opaque witness-policy anchor response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_policy_anchor_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    observer: &[u8; 32],
    policy_epoch: u64,
    policy_digest: &[u8; 32],
    responder: &[u8; 32],
    response_timestamp: u64,
    outcome: u8,
) -> [u8; 32] {
    let policy_epoch = policy_epoch.to_le_bytes();
    let response_timestamp = response_timestamp.to_le_bytes();
    let outcome = [outcome];
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-PolicyAnchorResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            observer.as_slice(),
            policy_epoch.as_slice(),
            policy_digest.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            outcome.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-certificate request.
#[must_use]
pub fn directory_observation_certificate_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationCertificateRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-certificate response.
///
/// The digest binds both the SHA-256 digest and exact byte length. The caller
/// must independently recompute the digest from `certificate_frame` before
/// accepting the responder signature.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_certificate_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    responder: &[u8; 32],
    response_timestamp: u64,
    certificate_sha256: &[u8; 32],
    certificate_frame_bytes: u64,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let certificate_frame_bytes = certificate_frame_bytes.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationCertificateResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            certificate_sha256.as_slice(),
            certificate_frame_bytes.as_slice(),
        ],
    )
}

/// Encodes a canonical bounded Directory Sync frame including its magic byte.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization fails.
pub fn encode_directory_sync_message(message: &DirectorySyncMessage) -> Result<Vec<u8>, CoreError> {
    let payload = encode_bincode_bounded(message, MAX_DIRECTORY_SYNC_MESSAGE_BYTES)
        .map_err(|error| CoreError::malformed(format!("directory sync encode: {error}")))?;
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(DIRECTORY_SYNC_MAGIC);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one canonical bounded Directory Sync frame.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for a wrong magic byte, trailing data,
/// oversized payload, or malformed message.
pub fn decode_directory_sync_message(bytes: &[u8]) -> Result<DirectorySyncMessage, CoreError> {
    if bytes.first().copied() != Some(DIRECTORY_SYNC_MAGIC) {
        return Err(CoreError::malformed("directory sync magic mismatch"));
    }
    decode_bincode_bounded(
        &bytes[1..],
        MAX_DIRECTORY_SYNC_MESSAGE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| CoreError::malformed(format!("directory sync decode: {error}")))
}

/// Encodes one canonical bounded portable observation certificate.
///
/// Callers should run [`DirectoryObservationCertificateV1::verify_at`] with
/// their own current time before distributing a stored certificate. Encoding
/// itself enforces only the stable allocation bound.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization exceeds the bound
/// or fails.
pub fn encode_directory_observation_certificate(
    certificate: &DirectoryObservationCertificateV1,
) -> Result<Vec<u8>, CoreError> {
    let payload = encode_bincode_bounded(certificate, MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES)
        .map_err(|error| {
            CoreError::malformed(format!("directory observation certificate encode: {error}"))
        })?;
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one canonical bounded portable observation certificate.
///
/// Decoding does not establish trust. Call
/// [`DirectoryObservationCertificateV1::verify_at`] before using the evidence.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for a wrong magic byte, trailing data,
/// oversized payload, or malformed certificate.
pub fn decode_directory_observation_certificate(
    bytes: &[u8],
) -> Result<DirectoryObservationCertificateV1, CoreError> {
    if bytes.first().copied() != Some(DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC) {
        return Err(CoreError::malformed(
            "directory observation certificate magic mismatch",
        ));
    }
    decode_bincode_bounded(
        &bytes[1..],
        MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| {
        CoreError::malformed(format!("directory observation certificate decode: {error}"))
    })
}

/// Encodes one canonical bounded portable route-domain certificate.
///
/// Encoding does not establish attestor trust. Call
/// [`RouteDomainAttestationCertificateV1::verify_with_policy_at`] before using
/// a certificate for path admission.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization exceeds the bound
/// or fails.
pub fn encode_route_domain_attestation_certificate(
    certificate: &RouteDomainAttestationCertificateV1,
) -> Result<Vec<u8>, CoreError> {
    let payload =
        encode_bincode_bounded(certificate, MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES)
            .map_err(|error| {
                CoreError::malformed(format!("route-domain certificate encode: {error}"))
            })?;
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one canonical bounded portable route-domain certificate.
///
/// Decoding does not establish trust. Call
/// [`RouteDomainAttestationCertificateV1::verify_with_policy_at`] before using
/// the evidence.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for a wrong magic byte, trailing data,
/// oversized payload, or malformed certificate.
pub fn decode_route_domain_attestation_certificate(
    bytes: &[u8],
) -> Result<RouteDomainAttestationCertificateV1, CoreError> {
    if bytes.first().copied() != Some(ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC) {
        return Err(CoreError::malformed(
            "route-domain attestation certificate magic mismatch",
        ));
    }
    decode_bincode_bounded(
        &bytes[1..],
        MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| CoreError::malformed(format!("route-domain certificate decode: {error}")))
}

// ============================================
// NodeBootstrapSnapshot
// ============================================

/// JSON-friendly bootstrap list of signed node descriptors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeBootstrapSnapshot {
    /// Snapshot schema version.
    pub schema_version: u16,
    /// Unix timestamp in seconds when the snapshot was generated.
    pub generated_at: u64,
    /// Signed node descriptors included in this snapshot.
    pub peers: Vec<SignedNodeDescriptor>,
}

impl NodeBootstrapSnapshot {
    /// Creates a bootstrap snapshot with the current schema version.
    #[must_use]
    pub fn new(generated_at: u64, peers: Vec<SignedNodeDescriptor>) -> Self {
        Self {
            schema_version: NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION,
            generated_at,
            peers,
        }
    }

    /// Parses a bounded JSON bootstrap snapshot.
    ///
    /// # Errors
    /// Returns `CoreError::MessageTooLarge` when input exceeds the bootstrap
    /// cap, or `CoreError::MalformedMessage` when JSON/schema parsing fails.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, CoreError> {
        if bytes.len() > MAX_BOOTSTRAP_SNAPSHOT_BYTES {
            return Err(CoreError::MessageTooLarge {
                max: MAX_BOOTSTRAP_SNAPSHOT_BYTES,
                actual: bytes.len(),
            });
        }

        let snapshot: Self = serde_json::from_slice(bytes)
            .map_err(|err| CoreError::malformed(format!("bootstrap snapshot json: {err}")))?;
        snapshot.validate_schema()?;
        Ok(snapshot)
    }

    /// Serializes this snapshot to pretty JSON for operator-readable bootstrap files.
    ///
    /// # Errors
    /// Returns `CoreError::MalformedMessage` if JSON serialization fails.
    pub fn to_json_pretty(&self) -> Result<Vec<u8>, CoreError> {
        serde_json::to_vec_pretty(self)
            .map_err(|err| CoreError::malformed(format!("bootstrap snapshot json: {err}")))
    }

    /// Validates the snapshot schema version.
    ///
    /// # Errors
    /// Returns `CoreError::MalformedMessage` for unsupported schema versions.
    pub fn validate_schema(&self) -> Result<(), CoreError> {
        if self.schema_version != NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION {
            return Err(CoreError::malformed(format!(
                "unsupported bootstrap snapshot schema version: {}",
                self.schema_version
            )));
        }
        Ok(())
    }

    /// Counts descriptors that verify at `now`.
    ///
    /// This is intentionally non-mutating. Server-side stores decide whether
    /// to reject, keep, or report invalid descriptors.
    #[must_use]
    pub fn verified_count_at(&self, now: u64) -> usize {
        self.peers
            .iter()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .count()
    }
}

// ============================================
// NodeDiscoveryMessage
// ============================================

/// Bounded peer discovery gossip message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeDiscoveryMessage {
    /// Requests a valid descriptor snapshot from a peer.
    SnapshotRequest {
        /// Unix timestamp in seconds when the request was sent.
        requested_at: u64,
        /// Optional maximum number of descriptors requested.
        limit: Option<u16>,
    },
    /// Responds with a bounded descriptor snapshot.
    SnapshotResponse {
        /// Descriptor snapshot generated by the responding peer.
        snapshot: NodeBootstrapSnapshot,
    },
    /// Announces a single descriptor update.
    DescriptorAnnounce {
        /// Signed descriptor being announced.
        descriptor: SignedNodeDescriptor,
    },
    /// Announces one descriptor together with exact producer-signed Directory
    /// inclusion evidence.
    ///
    /// [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] This append-only
    /// variant is transport evidence, not an authority token. Receivers must
    /// independently retain and audit the exact `producer` / `block_hash`
    /// anchor before admitting `proof.descriptor`. The duplicated
    /// `descriptor_hash` is an explicit request/response contract boundary and
    /// must match the proof commitment.
    DirectoryDescriptorAnnounceV1 {
        /// Original Directory block producer selected by the receiver's policy.
        producer: [u8; 32],
        /// Exact producer-signed block hash selected by the receiver.
        block_hash: [u8; 32],
        /// Exact authenticated descriptor commitment digest.
        descriptor_hash: [u8; 32],
        /// Compact producer-signed inclusion proof carrying the descriptor.
        proof: DirectoryDescriptorInclusionProofV1,
    },
}

/// Encodes a discovery gossip message using bounded bincode.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization fails.
pub fn encode_discovery_message(message: &NodeDiscoveryMessage) -> Result<Vec<u8>, CoreError> {
    encode_bincode_bounded(message, MAX_DISCOVERY_MESSAGE_BYTES)
        .map_err(|err| CoreError::malformed(format!("discovery message encode: {err}")))
}

/// Decodes a discovery gossip message using bounded bincode.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when decoding fails.
pub fn decode_discovery_message(bytes: &[u8]) -> Result<NodeDiscoveryMessage, CoreError> {
    decode_bincode_bounded(
        bytes,
        MAX_DISCOVERY_MESSAGE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|err| CoreError::malformed(format!("discovery message decode: {err}")))
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::Options;

    fn descriptor_for(kp: &IdentityKeyPair) -> NodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            7,
            1_700_000_000,
            1_700_003_600,
            "test",
        );
        descriptor.public_endpoint = Some("node.example:443".to_string());
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 256,
            max_bps: Some(1_000_000_000),
            max_pps: Some(250_000),
        };
        descriptor.policy = NodePolicy {
            allows_public_exit: false,
            public_discovery: true,
            region: Some("test-region".to_string()),
        };
        descriptor
    }

    #[test]
    fn test_signed_descriptor_roundtrip_verifies() {
        let kp = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();

        assert!(signed.verify_at(1_700_000_100).is_ok());
        assert_eq!(signed.node_id(), kp.public_key_bytes());
        assert_eq!(signed.sequence(), 7);
    }

    #[test]
    fn staged_capabilities_are_appended_and_signature_bound() {
        // [MIRROR-CAPABILITY 2026-07-24 by Codex] Existing enum positions are
        // part of the bincode wire contract. Appending the staged capability
        // keeps OnionMiddle at 4 and assigns the new carrier role to 5.
        assert_eq!(
            bincode::serialize(&NodeCapability::OnionMiddle).unwrap(),
            4u32.to_le_bytes()
        );
        assert_eq!(
            bincode::serialize(&NodeCapability::DirectoryMirrorCarrier).unwrap(),
            5u32.to_le_bytes()
        );
        assert_eq!(
            bincode::serialize(&NodeCapability::BlindVaultReplica).unwrap(),
            6u32.to_le_bytes()
        );

        let identity = IdentityKeyPair::generate();
        let mut descriptor = descriptor_for(&identity);
        descriptor
            .capabilities
            .push(NodeCapability::DirectoryMirrorCarrier);
        descriptor
            .capabilities
            .push(NodeCapability::BlindVaultReplica);
        let signed = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
        assert!(signed.verify_at(1_700_000_100).is_ok());

        let encoded = encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed,
        })
        .unwrap();
        let decoded = decode_discovery_message(&encoded).unwrap();
        let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = decoded else {
            panic!("unexpected discovery message variant");
        };
        assert!(descriptor
            .descriptor
            .capabilities
            .contains(&NodeCapability::DirectoryMirrorCarrier));
        assert!(descriptor
            .descriptor
            .capabilities
            .contains(&NodeCapability::BlindVaultReplica));
        assert!(descriptor.verify_at(1_700_000_100).is_ok());
    }

    #[test]
    fn signed_protocol_features_preserve_schema_and_detect_downgrade() {
        let identity = IdentityKeyPair::generate();
        let failure_receipt = NodeProtocolFeature::BlindRelayFailureReceiptV1;
        let purpose_receipt = NodeProtocolFeature::PurposeBoundDeliveryReceiptV2;
        let direct_relay_auth = NodeProtocolFeature::DirectPeerRelayAuthV2;
        let direct_relay_receipt = NodeProtocolFeature::DirectPeerRelayReceiptV2;
        let direct_relay_target_binding = NodeProtocolFeature::DirectPeerRelayTargetBindingV3;
        let onion_reply = NodeProtocolFeature::OnionReplyV1;
        let onion_blind_admission = NodeProtocolFeature::OnionBlindLeaseAdmissionV1;
        let onion_put_receipt = NodeProtocolFeature::OnionBlindVaultPutReceiptV1;
        let onion_lease_retire = NodeProtocolFeature::OnionBlindVaultLeaseRetireV1;
        let onion_lease_renewal = NodeProtocolFeature::OnionBlindVaultLeaseRenewalV1;
        let onion_lease_status = NodeProtocolFeature::OnionBlindVaultLeaseStatusV1;
        let onion_lease_inventory = NodeProtocolFeature::OnionBlindVaultLeaseInventoryV1;
        let descriptor = descriptor_for(&identity).with_protocol_features([
            purpose_receipt,
            failure_receipt,
            direct_relay_auth,
            direct_relay_receipt,
            direct_relay_target_binding,
            onion_reply,
            onion_blind_admission,
            onion_put_receipt,
            onion_lease_retire,
            onion_lease_renewal,
            onion_lease_status,
            onion_lease_inventory,
            purpose_receipt,
        ]);

        assert_eq!(descriptor.schema_version, NODE_DESCRIPTOR_SCHEMA_VERSION);
        assert_eq!(
            descriptor.software_version,
            "test+anpf1-brfr1.anpf1-dpra2.anpf1-dprr2.anpf1-dprtb3.anpf1-obla1.anpf1-obli1.anpf1-oblr1.anpf1-obls1.anpf1-oblw1.anpf1-obpr1.anpf1-or1.anpf1-pbdr2"
        );
        assert!(descriptor.advertises_protocol_feature(failure_receipt));
        assert!(descriptor.advertises_protocol_feature(purpose_receipt));
        assert!(descriptor.advertises_protocol_feature(direct_relay_auth));
        assert!(descriptor.advertises_protocol_feature(direct_relay_receipt));
        assert!(descriptor.advertises_protocol_feature(direct_relay_target_binding));
        assert!(descriptor.advertises_protocol_feature(onion_reply));
        assert!(descriptor.advertises_protocol_feature(onion_blind_admission));
        assert!(descriptor.advertises_protocol_feature(onion_put_receipt));
        assert!(descriptor.advertises_protocol_feature(onion_lease_retire));
        assert!(descriptor.advertises_protocol_feature(onion_lease_renewal));
        assert!(descriptor.advertises_protocol_feature(onion_lease_status));
        assert!(descriptor.advertises_protocol_feature(onion_lease_inventory));

        let signed = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
        let encoded = encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed.clone(),
        })
        .unwrap();
        let decoded = decode_discovery_message(&encoded).unwrap();
        let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = decoded else {
            panic!("unexpected discovery message variant");
        };
        assert!(descriptor.verify_at(1_700_000_100).is_ok());
        assert!(descriptor
            .descriptor
            .advertises_protocol_feature(failure_receipt));
        assert!(descriptor
            .descriptor
            .advertises_protocol_feature(purpose_receipt));
        assert!(descriptor
            .descriptor
            .advertises_protocol_feature(direct_relay_auth));
        assert!(descriptor
            .descriptor
            .advertises_protocol_feature(direct_relay_receipt));
        assert!(descriptor
            .descriptor
            .advertises_protocol_feature(direct_relay_target_binding));
        assert!(descriptor
            .descriptor
            .advertises_protocol_feature(onion_reply));

        let mut stripped = signed;
        stripped.descriptor.software_version = "test".to_string();
        assert!(stripped.verify_at(1_700_000_100).is_err());
    }

    #[test]
    fn protocol_features_preserve_existing_semver_build_metadata() {
        let identity = IdentityKeyPair::generate();
        let descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            1_700_000_000,
            1_700_003_600,
            "1.2.3-rc.1+git.abc",
        )
        .with_protocol_features([NodeProtocolFeature::BlindRelayFailureReceiptV1]);

        assert_eq!(
            descriptor.software_version,
            "1.2.3-rc.1+git.abc.anpf1-brfr1"
        );
    }

    #[test]
    fn test_descriptor_publishes_x25519_kem_key() {
        let kp = IdentityKeyPair::generate();
        let kem = kp.x25519_public_key_bytes();
        let descriptor = descriptor_for(&kp).with_x25519_kem(kem);
        assert_eq!(descriptor.schema_version, NODE_DESCRIPTOR_SCHEMA_VERSION);
        assert_eq!(descriptor.kem_alg, 1);
        assert_eq!(descriptor.x25519_kem_public(), Some(kem));

        // KEM key is covered by the signature and survives encode/decode.
        let signed = SignedNodeDescriptor::sign(descriptor, &kp).unwrap();
        assert!(signed.verify_at(1_700_000_100).is_ok());
        let bytes = encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed.clone(),
        })
        .unwrap();
        let decoded = decode_discovery_message(&bytes).unwrap();
        if let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = decoded {
            assert_eq!(descriptor.descriptor.x25519_kem_public(), Some(kem));
            assert!(descriptor.verify_at(1_700_000_100).is_ok());
        } else {
            panic!("unexpected discovery message variant");
        }
    }

    #[test]
    fn test_descriptor_without_kem_reports_none() {
        let kp = IdentityKeyPair::generate();
        let descriptor = descriptor_for(&kp);
        assert_eq!(descriptor.kem_alg, 0);
        assert_eq!(descriptor.x25519_kem_public(), None);
    }

    #[test]
    fn test_schema_v1_descriptor_without_kem_fields_still_verifies() {
        let kp = IdentityKeyPair::generate();
        let mut descriptor = descriptor_for(&kp);
        descriptor.schema_version = 1;
        descriptor.kem_alg = 0;
        descriptor.kem_public = [0u8; 32];
        let signature = kp.sign(&legacy_descriptor_v1_signing_bytes(&descriptor).unwrap());
        let signed = SignedNodeDescriptor {
            descriptor,
            signature,
        };

        let mut json = serde_json::to_value(&signed).unwrap();
        let descriptor_json = json
            .get_mut("descriptor")
            .and_then(serde_json::Value::as_object_mut)
            .expect("descriptor json object");
        descriptor_json.remove("kem_alg");
        descriptor_json.remove("kem_public");

        let decoded: SignedNodeDescriptor = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.descriptor.schema_version, 1);
        assert_eq!(decoded.descriptor.kem_alg, 0);
        assert_eq!(decoded.descriptor.kem_public, [0u8; 32]);
        assert_eq!(decoded.descriptor.x25519_kem_public(), None);
        assert!(decoded.verify_at(1_700_000_100).is_ok());
    }

    #[test]
    fn test_tampered_descriptor_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
        signed.descriptor.sequence += 1;

        assert!(signed.verify_at(1_700_000_100).is_err());
    }

    #[test]
    fn test_expired_descriptor_rejected() {
        let kp = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();

        assert!(signed.verify_at(1_700_004_000).is_err());
    }

    #[test]
    fn test_signature_only_verification_keeps_expired_records_non_live() {
        let kp = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();

        assert!(signed.verify_at(1_700_004_000).is_err());
        assert!(signed.verify_signature().is_ok());

        let mut tampered = signed.clone();
        tampered.signature[0] ^= 0x01;
        assert!(tampered.verify_signature().is_err());
    }

    #[test]
    fn test_descriptor_bincode_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
        let bytes = bincode::options()
            .with_fixint_encoding()
            .serialize(&signed)
            .unwrap();
        let restored: SignedNodeDescriptor = bincode::options()
            .with_fixint_encoding()
            .deserialize(&bytes)
            .unwrap();

        assert_eq!(restored, signed);
        assert!(restored.verify_at(1_700_000_100).is_ok());
    }

    #[test]
    fn test_bootstrap_snapshot_json_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
        let snapshot = NodeBootstrapSnapshot::new(1_700_000_010, vec![signed]);

        let json = snapshot.to_json_pretty().unwrap();
        let restored = NodeBootstrapSnapshot::from_json_bytes(&json).unwrap();

        assert_eq!(restored, snapshot);
        assert_eq!(restored.verified_count_at(1_700_000_100), 1);
    }

    #[test]
    fn test_bootstrap_snapshot_rejects_unsupported_schema() {
        let snapshot = NodeBootstrapSnapshot {
            schema_version: NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION + 1,
            generated_at: 1_700_000_010,
            peers: Vec::new(),
        };
        let json = serde_json::to_vec(&snapshot).unwrap();

        assert!(NodeBootstrapSnapshot::from_json_bytes(&json).is_err());
    }

    #[test]
    fn test_bootstrap_snapshot_rejects_oversized_json() {
        let too_large = vec![b' '; MAX_BOOTSTRAP_SNAPSHOT_BYTES + 1];

        assert!(matches!(
            NodeBootstrapSnapshot::from_json_bytes(&too_large),
            Err(CoreError::MessageTooLarge { .. })
        ));
    }

    #[test]
    fn test_discovery_message_snapshot_request_roundtrip() {
        let message = NodeDiscoveryMessage::SnapshotRequest {
            requested_at: 0x0102_0304_0506_0708,
            limit: Some(0x090a),
        };

        let bytes = encode_discovery_message(&message).unwrap();
        let decoded = decode_discovery_message(&bytes).unwrap();

        assert_eq!(decoded, message);
        assert_eq!(
            bytes,
            [
                0x00, 0x00, 0x00, 0x00, // enum variant
                0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // timestamp
                0x01, // Some
                0x0a, 0x09, // limit
            ],
            "the bounded codec must preserve the established discovery wire bytes"
        );

        let mut trailing = bytes;
        trailing.push(0);
        assert!(
            decode_discovery_message(&trailing).is_err(),
            "canonical discovery messages must reject trailing bytes"
        );

        let padded = vec![0; MAX_DISCOVERY_MESSAGE_BYTES as usize + 1];
        assert!(
            decode_discovery_message(&padded).is_err(),
            "the complete discovery input must obey the protocol ceiling"
        );
    }

    #[test]
    fn test_discovery_message_snapshot_response_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
        let snapshot = NodeBootstrapSnapshot::new(1_700_000_010, vec![signed]);
        let message = NodeDiscoveryMessage::SnapshotResponse { snapshot };

        let bytes = encode_discovery_message(&message).unwrap();
        let decoded = decode_discovery_message(&bytes).unwrap();

        assert_eq!(decoded, message);
    }

    #[test]
    fn test_discovery_message_descriptor_announce_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let descriptor = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
        let message = NodeDiscoveryMessage::DescriptorAnnounce { descriptor };

        let bytes = encode_discovery_message(&message).unwrap();
        let decoded = decode_discovery_message(&bytes).unwrap();

        assert_eq!(decoded, message);
    }

    #[test]
    fn directory_descriptor_announce_is_append_only_and_roundtrips() {
        // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] The enum index is
        // part of the mixed-version bincode contract. Existing variants remain
        // 0/1/2 and the proof-carrying announcement is appended at index 3.
        let producer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let descriptor = SignedNodeDescriptor::sign(descriptor_for(&subject), &subject).unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![commitment],
            &producer,
        )
        .unwrap();
        let block_hash = block.hash();
        let proof =
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, 1_700_000_100)
                .unwrap();
        let message = NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
            producer: producer.public_key_bytes(),
            block_hash,
            descriptor_hash: proof.commitment.descriptor_hash,
            proof,
        };

        assert_eq!(
            &encode_discovery_message(&NodeDiscoveryMessage::SnapshotRequest {
                requested_at: 1,
                limit: None,
            })
            .unwrap()[..4],
            &0u32.to_le_bytes()
        );
        assert_eq!(
            &encode_discovery_message(&NodeDiscoveryMessage::SnapshotResponse {
                snapshot: NodeBootstrapSnapshot::new(1_700_000_100, Vec::new()),
            })
            .unwrap()[..4],
            &1u32.to_le_bytes()
        );
        assert_eq!(
            &encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: descriptor.clone(),
            })
            .unwrap()[..4],
            &2u32.to_le_bytes()
        );
        let encoded = encode_discovery_message(&message).unwrap();
        assert_eq!(&encoded[..4], &3u32.to_le_bytes());
        assert_eq!(decode_discovery_message(&encoded).unwrap(), message);
    }

    #[test]
    fn directory_sync_tip_frame_is_canonical_and_domain_bound() {
        let requester = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let request_id = [0x92; 16];
        let timestamp = 1_700_000_123;
        let signing_bytes = directory_tip_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester.public_key_bytes(),
            timestamp,
        );
        let message = DirectorySyncMessage::TipRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp: timestamp,
            signature: requester.sign(&signing_bytes),
        };

        let encoded = encode_directory_sync_message(&message).unwrap();
        assert_eq!(encoded.first().copied(), Some(DIRECTORY_SYNC_MAGIC));
        assert_eq!(decode_directory_sync_message(&encoded).unwrap(), message);
        assert_ne!(
            signing_bytes,
            directory_tip_request_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &requester.public_key_bytes(),
                timestamp + 1,
            )
        );

        let mut trailing = encoded;
        trailing.push(0);
        assert!(decode_directory_sync_message(&trailing).is_err());
    }

    #[test]
    fn directory_sync_range_and_object_digests_bind_order_and_tip() {
        let producer = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
        let first_peer = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
        let second_peer = IdentityKeyPair::from_bytes(&[0x95; 32]).unwrap();
        let first = SignedNodeDescriptor::sign(descriptor_for(&first_peer), &first_peer).unwrap();
        let second =
            SignedNodeDescriptor::sign(descriptor_for(&second_peer), &second_peer).unwrap();
        let first_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&first).unwrap();
        let second_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&second).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_200,
            [0u8; 32],
            vec![first_commitment, second_commitment],
            &producer,
        )
        .unwrap();
        let request_id = [0x96; 16];
        let forward = directory_block_range_response_signing_bytes(
            &request_id,
            &producer.public_key_bytes(),
            1_700_000_201,
            std::slice::from_ref(&block),
            false,
            1,
            &block.hash(),
        );
        let different_tip = directory_block_range_response_signing_bytes(
            &request_id,
            &producer.public_key_bytes(),
            1_700_000_201,
            std::slice::from_ref(&block),
            false,
            2,
            &block.hash(),
        );
        assert_ne!(forward, different_tip);

        let hashes = [
            first_commitment.descriptor_hash,
            second_commitment.descriptor_hash,
        ];
        let reversed = [hashes[1], hashes[0]];
        assert_ne!(
            directory_descriptor_objects_request_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &hashes,
                &request_id,
                &producer.public_key_bytes(),
                1_700_000_201,
            ),
            directory_descriptor_objects_request_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &reversed,
                &request_id,
                &producer.public_key_bytes(),
                1_700_000_201,
            )
        );
    }

    #[test]
    fn test_directory_descriptor_commitment_binds_authenticated_descriptor() {
        let identity = IdentityKeyPair::generate();
        let signed = SignedNodeDescriptor::sign(descriptor_for(&identity), &identity).unwrap();
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&signed).unwrap();

        assert_eq!(commitment.node_id, identity.public_key_bytes());
        assert_eq!(commitment.sequence, signed.sequence());
        assert!(commitment.matches_signed_descriptor(&signed).unwrap());
        assert_ne!(commitment.hash(), [0u8; 32]);

        let mut next_descriptor = descriptor_for(&identity);
        next_descriptor.sequence += 1;
        let next_signed = SignedNodeDescriptor::sign(next_descriptor, &identity).unwrap();
        assert!(!commitment.matches_signed_descriptor(&next_signed).unwrap());

        let mut forged = signed;
        forged.signature[0] ^= 0x01;
        assert!(commitment.matches_signed_descriptor(&forged).is_err());
    }

    #[test]
    fn test_directory_block_is_deterministic_and_roundtrips() {
        let producer = IdentityKeyPair::generate();
        let first_identity = IdentityKeyPair::generate();
        let second_identity = IdentityKeyPair::generate();
        let first = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(descriptor_for(&first_identity), &first_identity).unwrap(),
        )
        .unwrap();
        let second = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(descriptor_for(&second_identity), &second_identity)
                .unwrap(),
        )
        .unwrap();

        let forward = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![first, second],
            &producer,
        )
        .unwrap();
        let reverse = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![second, first],
            &producer,
        )
        .unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(forward.header.commitment_count, 2);
        assert!(forward
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            )
            .is_ok());
        let encoded = bincode::options()
            .with_fixint_encoding()
            .serialize(&forward)
            .unwrap();
        let decoded: DirectoryCommitmentBlockV1 = bincode::options()
            .with_fixint_encoding()
            .deserialize(&encoded)
            .unwrap();
        assert_eq!(decoded, forward);
        assert!(decoded.to_string().contains("height=1"));
    }

    #[test]
    fn directory_descriptor_inclusion_proof_roundtrips_odd_tree() {
        let producer = IdentityKeyPair::generate();
        let signed_descriptors = (0u64..5)
            .map(|offset| {
                let identity = IdentityKeyPair::generate();
                let mut descriptor = descriptor_for(&identity);
                descriptor.sequence = descriptor.sequence.saturating_add(offset);
                SignedNodeDescriptor::sign(descriptor, &identity).unwrap()
            })
            .collect::<Vec<_>>();
        let commitments = signed_descriptors
            .iter()
            .map(|descriptor| {
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor).unwrap()
            })
            .collect::<Vec<_>>();
        let observed_at = 1_700_000_100;
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            observed_at,
            [0u8; 32],
            commitments,
            &producer,
        )
        .unwrap();
        let expected_block_hash = block.hash();

        for descriptor in &signed_descriptors {
            let proof =
                DirectoryDescriptorInclusionProofV1::from_block_at(&block, descriptor, observed_at)
                    .unwrap();
            assert_eq!(
                proof.verify_at(
                    &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                    &producer.public_key_bytes(),
                    &expected_block_hash,
                    observed_at,
                ),
                Ok(())
            );
            assert_eq!(proof.block_hash(), expected_block_hash);
            assert!(proof.sibling_hashes.len() <= MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1);
        }
    }

    #[test]
    fn directory_descriptor_inclusion_proof_rejects_wrong_trust_anchors_and_tampering() {
        let producer = IdentityKeyPair::generate();
        let first_identity = IdentityKeyPair::generate();
        let second_identity = IdentityKeyPair::generate();
        let first_signed =
            SignedNodeDescriptor::sign(descriptor_for(&first_identity), &first_identity).unwrap();
        let second_signed =
            SignedNodeDescriptor::sign(descriptor_for(&second_identity), &second_identity).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&first_signed).unwrap(),
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&second_signed).unwrap(),
            ],
            &producer,
        )
        .unwrap();
        let block_hash = block.hash();
        let proof = DirectoryDescriptorInclusionProofV1::from_block_at(
            &block,
            &first_signed,
            1_700_000_100,
        )
        .unwrap();

        assert_eq!(
            proof.verify_at(
                &[0x41; 32],
                &producer.public_key_bytes(),
                &block_hash,
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::WrongChain)
        );
        assert_eq!(
            proof.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &[0x42; 32],
                &block_hash,
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::WrongProducer)
        );
        assert_eq!(
            proof.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &[0x43; 32],
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::WrongBlockHash)
        );

        let mut wrong_descriptor = proof.clone();
        wrong_descriptor.descriptor = second_signed;
        assert_eq!(
            wrong_descriptor.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &block_hash,
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::DescriptorMismatch)
        );

        let mut wrong_path = proof.clone();
        wrong_path.sibling_hashes[0][0] ^= 0x01;
        assert_eq!(
            wrong_path.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &block_hash,
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::InvalidMerkleProof)
        );

        let mut wrong_signature = proof.clone();
        wrong_signature.producer_signature[0] ^= 0x01;
        assert_eq!(
            wrong_signature.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &block_hash,
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::InvalidBlock)
        );

        let mut wrong_length = proof;
        wrong_length.sibling_hashes.clear();
        assert_eq!(
            wrong_length.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &block_hash,
                1_700_000_100,
            ),
            Err(DirectoryDescriptorInclusionProofError::InvalidProofLength)
        );
    }

    #[test]
    fn directory_descriptor_inclusion_proof_is_bounded_and_requires_membership() {
        assert_eq!(
            directory_inclusion_proof_depth(MAX_DIRECTORY_COMMITMENTS_PER_BLOCK),
            MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1
        );

        let producer = IdentityKeyPair::generate();
        let included_identity = IdentityKeyPair::generate();
        let absent_identity = IdentityKeyPair::generate();
        let included =
            SignedNodeDescriptor::sign(descriptor_for(&included_identity), &included_identity)
                .unwrap();
        let absent =
            SignedNodeDescriptor::sign(descriptor_for(&absent_identity), &absent_identity).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(&included).unwrap()],
            &producer,
        )
        .unwrap();

        assert_eq!(
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, &absent, 1_700_000_100,),
            Err(DirectoryDescriptorInclusionProofError::DescriptorMismatch)
        );
    }

    #[test]
    fn test_directory_block_verification_rejects_tampering() {
        let producer = IdentityKeyPair::generate();
        let node = IdentityKeyPair::generate();
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(descriptor_for(&node), &node).unwrap(),
        )
        .unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![commitment],
            &producer,
        )
        .unwrap();

        let mut wrong_chain = block.clone();
        wrong_chain.header.chain_id[0] ^= 0x01;
        assert_eq!(
            wrong_chain.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            ),
            Err(DirectoryCommitmentValidationError::WrongChain)
        );

        let mut wrong_count = block.clone();
        wrong_count.header.commitment_count += 1;
        assert_eq!(
            wrong_count.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            ),
            Err(DirectoryCommitmentValidationError::CommitmentCountMismatch)
        );

        let mut wrong_root = block.clone();
        wrong_root.header.commitment_root[0] ^= 0x01;
        assert_eq!(
            wrong_root.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            ),
            Err(DirectoryCommitmentValidationError::InvalidMerkleRoot)
        );

        let mut wrong_signature = block;
        wrong_signature.producer_signature[0] ^= 0x01;
        assert_eq!(
            wrong_signature.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            ),
            Err(DirectoryCommitmentValidationError::InvalidSignature)
        );
    }

    #[test]
    fn test_directory_block_rejects_invalid_and_unbounded_inputs() {
        let producer = IdentityKeyPair::generate();
        let node = IdentityKeyPair::generate();
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(descriptor_for(&node), &node).unwrap(),
        )
        .unwrap();

        assert_eq!(
            DirectoryCommitmentBlockV1::new_signed(
                1,
                1_700_000_100,
                [0u8; 32],
                Vec::new(),
                &producer,
            ),
            Err(DirectoryCommitmentValidationError::EmptyBlock)
        );
        assert_eq!(
            DirectoryCommitmentBlockV1::new_signed(
                1,
                1_700_000_100,
                [0u8; 32],
                vec![commitment, commitment],
                &producer,
            ),
            Err(DirectoryCommitmentValidationError::DuplicateCommitment)
        );
        assert_eq!(
            DirectoryCommitmentBlockV1::new_signed(
                1,
                1_700_000_100,
                [0u8; 32],
                vec![commitment; MAX_DIRECTORY_COMMITMENTS_PER_BLOCK + 1],
                &producer,
            ),
            Err(DirectoryCommitmentValidationError::TooManyCommitments)
        );
        assert_eq!(
            DirectoryCommitmentBlockV1::new_signed(
                2,
                1_700_000_100,
                [0u8; 32],
                vec![commitment],
                &producer,
            ),
            Err(DirectoryCommitmentValidationError::InvalidPreviousHash)
        );
        assert_eq!(
            DirectoryCommitmentBlockV1::new_signed(1, 0, [0u8; 32], vec![commitment], &producer,),
            Err(DirectoryCommitmentValidationError::InvalidTimestamp)
        );

        let invalid = DirectoryDescriptorCommitmentV1 {
            node_id: [0u8; 32],
            ..commitment
        };
        assert_eq!(
            DirectoryCommitmentBlockV1::new_signed(
                1,
                1_700_000_100,
                [0u8; 32],
                vec![invalid],
                &producer,
            ),
            Err(DirectoryCommitmentValidationError::InvalidCommitment)
        );
    }

    #[test]
    fn test_directory_block_chain_continuity_binds_height_hash_and_time() {
        let producer = IdentityKeyPair::generate();
        let first_node = IdentityKeyPair::generate();
        let second_node = IdentityKeyPair::generate();
        let first_commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(descriptor_for(&first_node), &first_node).unwrap(),
        )
        .unwrap();
        let mut second_descriptor = descriptor_for(&second_node);
        second_descriptor.sequence = 8;
        let second_commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(second_descriptor, &second_node).unwrap(),
        )
        .unwrap();
        let first = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![first_commitment],
            &producer,
        )
        .unwrap();
        let second = DirectoryCommitmentBlockV1::new_signed(
            2,
            1_700_000_101,
            first.hash(),
            vec![second_commitment],
            &producer,
        )
        .unwrap();

        assert!(second
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                2,
                &first.hash(),
                first.header.timestamp,
                1_700_000_101,
            )
            .is_ok());
        assert_eq!(
            second.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                3,
                &first.hash(),
                first.header.timestamp,
                1_700_000_101,
            ),
            Err(DirectoryCommitmentValidationError::InvalidHeight)
        );

        let regressed = DirectoryCommitmentBlockV1::new_signed(
            2,
            1_700_000_099,
            first.hash(),
            vec![second_commitment],
            &producer,
        )
        .unwrap();
        assert_eq!(
            regressed.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                2,
                &first.hash(),
                first.header.timestamp,
                1_700_000_101,
            ),
            Err(DirectoryCommitmentValidationError::InvalidTimestamp)
        );

        let future = DirectoryCommitmentBlockV1::new_signed(
            2,
            1_700_000_222,
            first.hash(),
            vec![second_commitment],
            &producer,
        )
        .unwrap();
        assert_eq!(
            future.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                2,
                &first.hash(),
                first.header.timestamp,
                1_700_000_101,
            ),
            Err(DirectoryCommitmentValidationError::InvalidTimestamp)
        );
    }

    #[test]
    fn test_directory_block_preserves_same_sequence_equivocation_evidence() {
        let producer = IdentityKeyPair::generate();
        let node = IdentityKeyPair::generate();
        let first_descriptor = descriptor_for(&node);
        let mut conflicting_descriptor = first_descriptor.clone();
        conflicting_descriptor.public_endpoint = Some("conflicting.example:443".to_string());
        let first = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(first_descriptor, &node).unwrap(),
        )
        .unwrap();
        let conflicting = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            &SignedNodeDescriptor::sign(conflicting_descriptor, &node).unwrap(),
        )
        .unwrap();

        assert_eq!(first.node_id, conflicting.node_id);
        assert_eq!(first.sequence, conflicting.sequence);
        assert_ne!(first.descriptor_hash, conflicting.descriptor_hash);
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![first, conflicting],
            &producer,
        )
        .unwrap();
        assert_eq!(block.commitments.len(), 2);
        assert!(block
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            )
            .is_ok());
    }

    #[test]
    fn test_directory_observation_checkpoint_is_canonical_and_signed() {
        let observer = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
        let tips = vec![
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 12,
                tip_hash: [0xb2; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 11,
                tip_hash: [0xa1; 32],
            },
        ];
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            2,
            tips.clone(),
            [0x44; 32],
            &observer,
        )
        .unwrap();

        assert!(checkpoint
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                1,
                &[0u8; 32],
                0,
                1_700_000_100,
            )
            .is_ok());
        assert!(checkpoint.producer_tips[0].producer < checkpoint.producer_tips[1].producer);
        let reordered = DirectoryObservationCheckpointV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            2,
            tips.into_iter().rev().collect(),
            [0x44; 32],
            &observer,
        )
        .unwrap();
        assert_eq!(checkpoint.hash(), reordered.hash());
        assert_eq!(checkpoint.observer_signature, reordered.observer_signature);
    }

    #[test]
    fn test_directory_observation_checkpoint_rejects_tamper_and_invalid_history() {
        let observer = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x42; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x43; 32]).unwrap();
        let tips = vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 3,
                tip_hash: [0x51; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 4,
                tip_hash: [0x52; 32],
            },
        ];
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            2,
            1_700_000_200,
            [0x61; 32],
            2,
            tips,
            [0x62; 32],
            &observer,
        )
        .unwrap();

        let mut tampered = checkpoint.clone();
        tampered.observation_root[0] ^= 1;
        assert_eq!(
            tampered.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                2,
                &[0x61; 32],
                1_700_000_100,
                1_700_000_200,
            ),
            Err(DirectoryObservationCheckpointValidationError::InvalidSignature)
        );
        assert_eq!(
            checkpoint.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                3,
                &checkpoint.hash(),
                1_700_000_201,
                1_700_000_200,
            ),
            Err(DirectoryObservationCheckpointValidationError::InvalidPosition)
        );

        let mut noncanonical = checkpoint;
        noncanonical.producer_tips.reverse();
        assert_eq!(
            noncanonical.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                2,
                &[0x61; 32],
                1_700_000_100,
                1_700_000_200,
            ),
            Err(DirectoryObservationCheckpointValidationError::NonCanonicalProducerOrder)
        );
    }

    fn accepted_observation_receipt(
        checkpoint: &DirectoryObservationCheckpointV1,
        witness: &IdentityKeyPair,
        request_id: [u8; 16],
        response_timestamp: u64,
    ) -> DirectoryObservationWitnessReceiptV1 {
        let checkpoint_hash = checkpoint.hash();
        let digest = directory_observation_witness_response_signing_bytes(
            &checkpoint.chain_id,
            &request_id,
            &checkpoint.observer,
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            response_timestamp,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        DirectoryObservationWitnessReceiptV1 {
            chain_id: checkpoint.chain_id,
            request_id,
            observer: checkpoint.observer,
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&digest),
        }
    }

    #[test]
    fn portable_observation_certificate_is_canonical_bounded_and_offline_verifiable() {
        // [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] This fixture
        // proves exact observer/witness signatures without introducing an
        // aggregator signature, vote, quorum, consensus, or finality claim.
        let observer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
        let witness_a = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0x85; 32]).unwrap();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            4,
            1_700_000_400,
            [0x86; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: 18,
                    tip_hash: [0x87; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 19,
                    tip_hash: [0x88; 32],
                },
            ],
            [0x89; 32],
            &observer,
        )
        .unwrap();
        let receipt_a =
            accepted_observation_receipt(&checkpoint, &witness_a, [0x8a; 16], 1_700_000_401);
        let receipt_b =
            accepted_observation_receipt(&checkpoint, &witness_b, [0x8b; 16], 1_700_000_402);
        let certificate = DirectoryObservationCertificateV1::new_verified(
            checkpoint,
            2,
            vec![receipt_b, receipt_a],
            1_700_000_402,
        )
        .unwrap();

        assert!(certificate.receipts[0].responder < certificate.receipts[1].responder);
        assert!(certificate
            .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_402)
            .is_ok());
        let encoded = encode_directory_observation_certificate(&certificate).unwrap();
        assert_eq!(
            encoded.first().copied(),
            Some(DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC)
        );
        assert!(encoded.len() < MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES as usize);
        let decoded = decode_directory_observation_certificate(&encoded).unwrap();
        assert_eq!(decoded, certificate);
        assert_eq!(decoded.hash(), certificate.hash());
        assert_eq!(
            DirectoryObservationWitnessReceiptV1::from_sync_message(
                &decoded.receipts[0].to_sync_message()
            )
            .unwrap(),
            decoded.receipts[0]
        );

        let mut trailing = encoded;
        trailing.push(0);
        assert!(decode_directory_observation_certificate(&trailing).is_err());

        let mut oversized = vec![0u8; MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES as usize + 2];
        oversized[0] = DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC;
        assert!(decode_directory_observation_certificate(&oversized).is_err());
    }

    #[test]
    fn portable_observation_certificate_rejects_partial_duplicate_and_tampered_receipts() {
        let observer = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            2,
            1_700_000_500,
            [0x95; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: 2,
                    tip_hash: [0x96; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 3,
                    tip_hash: [0x97; 32],
                },
            ],
            [0x98; 32],
            &observer,
        )
        .unwrap();
        let receipt =
            accepted_observation_receipt(&checkpoint, &witness, [0x99; 16], 1_700_000_501);

        assert_eq!(
            DirectoryObservationCertificateV1::new_verified(
                checkpoint.clone(),
                2,
                vec![receipt],
                1_700_000_501,
            ),
            Err(DirectoryObservationCertificateValidationError::InvalidReceiptCount)
        );
        let duplicate = DirectoryObservationCertificateV1 {
            protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
            chain_id: checkpoint.chain_id,
            checkpoint: checkpoint.clone(),
            minimum_witnesses: 1,
            receipts: vec![receipt, receipt],
        };
        assert_eq!(
            duplicate.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_501),
            Err(DirectoryObservationCertificateValidationError::DuplicateWitness)
        );

        let mut tampered = receipt;
        tampered.signature[0] ^= 1;
        let tampered = DirectoryObservationCertificateV1 {
            protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
            chain_id: checkpoint.chain_id,
            checkpoint: checkpoint.clone(),
            minimum_witnesses: 1,
            receipts: vec![tampered],
        };
        assert_eq!(
            tampered.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_501),
            Err(DirectoryObservationCertificateValidationError::InvalidReceiptSignature)
        );

        let mut invalid_checkpoint = checkpoint;
        invalid_checkpoint.observer_signature[0] ^= 1;
        let invalid_checkpoint = DirectoryObservationCertificateV1 {
            protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
            chain_id: invalid_checkpoint.chain_id,
            checkpoint: invalid_checkpoint,
            minimum_witnesses: 1,
            receipts: vec![receipt],
        };
        assert_eq!(
            invalid_checkpoint.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_501),
            Err(DirectoryObservationCertificateValidationError::InvalidCheckpoint)
        );
    }

    #[test]
    fn route_domain_certificate_is_canonical_bounded_and_policy_verified() {
        // [ROUTE-DOMAIN-ATTESTATION 2026-08-03 by Codex] The fixture proves
        // exact pinned signatures over an opaque token. It intentionally makes
        // no ASN, operator-independence, consensus, or Sybil-resistance claim.
        let subject = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
        let attestor_a = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
        let attestor_b = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
        let route_domain = [0xa4; 16];
        let issued_at = 1_700_001_000;
        let expires_at = issued_at + 3_600;
        let statement_a = RouteDomainAttestationV1::new_signed(
            subject.public_key_bytes(),
            route_domain,
            issued_at,
            expires_at,
            &attestor_a,
        )
        .unwrap();
        let statement_b = RouteDomainAttestationV1::new_signed(
            subject.public_key_bytes(),
            route_domain,
            issued_at + 1,
            expires_at,
            &attestor_b,
        )
        .unwrap();
        let certificate = RouteDomainAttestationCertificateV1::new_verified(
            subject.public_key_bytes(),
            route_domain,
            vec![statement_b, statement_a],
            issued_at + 2,
        )
        .unwrap();

        assert!(
            certificate.attestations[0].attestor_node_id
                < certificate.attestations[1].attestor_node_id
        );
        assert_eq!(
            certificate
                .verify_with_policy_at(
                    &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                    &[attestor_b.public_key_bytes(), attestor_a.public_key_bytes(),],
                    2,
                    issued_at + 2,
                )
                .unwrap(),
            2
        );
        let encoded = encode_route_domain_attestation_certificate(&certificate).unwrap();
        assert_eq!(
            encoded.first().copied(),
            Some(ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC)
        );
        assert!(encoded.len() <= MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES);
        let decoded = decode_route_domain_attestation_certificate(&encoded).unwrap();
        assert_eq!(decoded, certificate);
        assert_eq!(decoded.hash(), certificate.hash());

        let mut trailing = encoded;
        trailing.push(0);
        assert!(decode_route_domain_attestation_certificate(&trailing).is_err());
        let mut oversized = vec![0u8; MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES + 1];
        oversized[0] = ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC;
        assert!(decode_route_domain_attestation_certificate(&oversized).is_err());
    }

    #[test]
    fn route_domain_certificate_rejects_expiry_tamper_duplicates_and_untrusted_quorum() {
        let subject = IdentityKeyPair::from_bytes(&[0xb1; 32]).unwrap();
        let attestor_a = IdentityKeyPair::from_bytes(&[0xb2; 32]).unwrap();
        let attestor_b = IdentityKeyPair::from_bytes(&[0xb3; 32]).unwrap();
        let untrusted = IdentityKeyPair::from_bytes(&[0xb4; 32]).unwrap();
        let route_domain = [0xb5; 16];
        let issued_at = 1_700_002_000;
        let expires_at = issued_at + 600;
        let statement_a = RouteDomainAttestationV1::new_signed(
            subject.public_key_bytes(),
            route_domain,
            issued_at,
            expires_at,
            &attestor_a,
        )
        .unwrap();
        let statement_b = RouteDomainAttestationV1::new_signed(
            subject.public_key_bytes(),
            route_domain,
            issued_at,
            expires_at,
            &attestor_b,
        )
        .unwrap();
        let certificate = RouteDomainAttestationCertificateV1::new_verified(
            subject.public_key_bytes(),
            route_domain,
            vec![statement_a, statement_b],
            issued_at + 1,
        )
        .unwrap();

        assert_eq!(
            statement_a.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, expires_at),
            Err(RouteDomainAttestationValidationError::Expired)
        );
        assert_eq!(
            RouteDomainAttestationV1::new_signed(
                subject.public_key_bytes(),
                route_domain,
                issued_at,
                issued_at + MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1 + 1,
                &attestor_a,
            ),
            Err(RouteDomainAttestationValidationError::InvalidTimestamp)
        );
        assert_eq!(
            RouteDomainAttestationV1::new_signed(
                subject.public_key_bytes(),
                route_domain,
                issued_at,
                expires_at,
                &subject,
            ),
            Err(RouteDomainAttestationValidationError::InvalidAttestor)
        );

        let duplicate = RouteDomainAttestationCertificateV1 {
            protocol_version: ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            subject_node_id: subject.public_key_bytes(),
            route_domain,
            attestations: vec![statement_a, statement_a],
        };
        assert_eq!(
            duplicate.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, issued_at + 1),
            Err(RouteDomainAttestationCertificateValidationError::DuplicateAttestor)
        );

        let mut tampered = certificate.clone();
        tampered.attestations[0].signature[0] ^= 1;
        assert_eq!(
            tampered.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, issued_at + 1),
            Err(RouteDomainAttestationCertificateValidationError::InvalidAttestation)
        );
        let mut rebound = certificate.clone();
        rebound.route_domain = [0xb6; 16];
        assert_eq!(
            rebound.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, issued_at + 1),
            Err(RouteDomainAttestationCertificateValidationError::InvalidAttestationContract)
        );

        assert_eq!(
            certificate.verify_with_policy_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &[attestor_a.public_key_bytes(), untrusted.public_key_bytes(),],
                2,
                issued_at + 1,
            ),
            Err(RouteDomainAttestationCertificateValidationError::InsufficientTrustedAttestations)
        );
        assert_eq!(
            certificate.verify_with_policy_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &[attestor_a.public_key_bytes(), attestor_a.public_key_bytes(),],
                1,
                issued_at + 1,
            ),
            Err(RouteDomainAttestationCertificateValidationError::InvalidPolicy)
        );
    }

    #[test]
    fn test_directory_observation_witness_frames_are_canonical_and_bound() {
        let observer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let carrier = IdentityKeyPair::from_bytes(&[0x70; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            3,
            1_700_000_300,
            [0x75; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: 8,
                    tip_hash: [0x76; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 9,
                    tip_hash: [0x77; 32],
                },
            ],
            [0x78; 32],
            &observer,
        )
        .unwrap();
        assert!(checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_300)
            .is_ok());

        let request_id = [0x79; 16];
        let checkpoint_hash = checkpoint.hash();
        let request_digest = directory_observation_witness_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            1_700_000_301,
            &checkpoint_hash,
        );
        let request = DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: observer.public_key_bytes(),
            request_timestamp: 1_700_000_301,
            checkpoint: checkpoint.clone(),
            signature: observer.sign(&request_digest),
        };
        let request_frame = encode_directory_sync_message(&request).unwrap();
        let decoded = decode_directory_sync_message(&request_frame).unwrap();
        assert_eq!(decoded, request);

        let response_digest = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            1_700_000_302,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp: 1_700_000_302,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&response_digest),
        };
        let response_frame = encode_directory_sync_message(&response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&response_frame).unwrap(),
            response
        );
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            responder,
            signature,
            ..
        } = response
        else {
            unreachable!();
        };
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&response_digest, &signature)
            .unwrap();

        let carrier_request_id = [0x6f; 16];
        let witness_request_sha256: [u8; 32] = Sha256::digest(&request_frame).into();
        let carrier_request_digest = directory_observation_witness_carrier_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &carrier_request_id,
            &observer.public_key_bytes(),
            1_700_000_303,
            &witness.public_key_bytes(),
            &witness_request_sha256,
            u64::try_from(request_frame.len()).unwrap(),
        );
        let carrier_request = DirectorySyncMessage::ObservationCheckpointWitnessCarrierRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id: carrier_request_id,
            requester: observer.public_key_bytes(),
            request_timestamp: 1_700_000_303,
            witness: witness.public_key_bytes(),
            witness_request_sha256,
            witness_request_frame: request_frame.clone(),
            signature: observer.sign(&carrier_request_digest),
        };
        let carrier_request_frame = encode_directory_sync_message(&carrier_request).unwrap();
        assert_eq!(
            decode_directory_sync_message(&carrier_request_frame).unwrap(),
            carrier_request
        );

        let witness_response_sha256: [u8; 32] = Sha256::digest(&response_frame).into();
        let carrier_response_digest = directory_observation_witness_carrier_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &carrier_request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            &carrier.public_key_bytes(),
            1_700_000_304,
            &witness_request_sha256,
            &witness_response_sha256,
            u64::try_from(response_frame.len()).unwrap(),
        );
        let carrier_response =
            DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id: carrier_request_id,
                requester: observer.public_key_bytes(),
                witness: witness.public_key_bytes(),
                carrier: carrier.public_key_bytes(),
                response_timestamp: 1_700_000_304,
                witness_request_sha256,
                witness_response_sha256,
                witness_response_frame: response_frame,
                signature: carrier.sign(&carrier_response_digest),
            };
        let carrier_response_frame = encode_directory_sync_message(&carrier_response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&carrier_response_frame).unwrap(),
            carrier_response
        );
        assert_ne!(
            carrier_request_digest,
            directory_observation_witness_carrier_request_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &carrier_request_id,
                &observer.public_key_bytes(),
                1_700_000_303,
                &witness.public_key_bytes(),
                &witness_request_sha256,
                u64::try_from(request_frame.len())
                    .unwrap()
                    .saturating_add(1),
            )
        );

        let altered = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            1_700_000_302,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
        );
        assert_ne!(response_digest, altered);

        let policy_request_id = [0x7a; 16];
        let policy_digest = [0x7b; 32];
        let policy_request_digest = directory_policy_anchor_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &policy_request_id,
            &observer.public_key_bytes(),
            1_700_000_303,
            1,
            &[0u8; 32],
            &policy_digest,
        );
        let policy_request = DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id: policy_request_id,
            requester: observer.public_key_bytes(),
            request_timestamp: 1_700_000_303,
            policy_epoch: 1,
            previous_policy_digest: [0u8; 32],
            policy_digest,
            signature: observer.sign(&policy_request_digest),
        };
        let encoded = encode_directory_sync_message(&policy_request).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            policy_request
        );

        let policy_response_digest = directory_policy_anchor_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &policy_request_id,
            &observer.public_key_bytes(),
            1,
            &policy_digest,
            &witness.public_key_bytes(),
            1_700_000_304,
            DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
        );
        let policy_response = DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id: policy_request_id,
            observer: observer.public_key_bytes(),
            policy_epoch: 1,
            policy_digest,
            responder: witness.public_key_bytes(),
            response_timestamp: 1_700_000_304,
            outcome: DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            signature: witness.sign(&policy_response_digest),
        };
        let encoded = encode_directory_sync_message(&policy_response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            policy_response
        );
        assert_ne!(
            policy_response_digest,
            directory_policy_anchor_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &policy_request_id,
                &observer.public_key_bytes(),
                1,
                &policy_digest,
                &witness.public_key_bytes(),
                1_700_000_304,
                DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
            )
        );
    }

    #[test]
    fn test_directory_observation_certificate_exchange_frames_are_canonical_and_bound() {
        let requester = IdentityKeyPair::from_bytes(&[0x7c; 32]).unwrap();
        let responder = IdentityKeyPair::from_bytes(&[0x7d; 32]).unwrap();
        let request_id = [0x7e; 16];
        let request_timestamp = 1_700_000_305;
        let request_digest = directory_observation_certificate_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester.public_key_bytes(),
            request_timestamp,
        );
        let request = DirectorySyncMessage::ObservationCertificateRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp,
            signature: requester.sign(&request_digest),
        };
        let encoded = encode_directory_sync_message(&request).unwrap();
        assert_eq!(decode_directory_sync_message(&encoded).unwrap(), request);

        let certificate_frame = vec![0xa5; 96];
        let certificate_sha256: [u8; 32] = Sha256::digest(&certificate_frame).into();
        let response_timestamp = 1_700_000_306;
        let frame_bytes = u64::try_from(certificate_frame.len()).unwrap();
        let response_digest = directory_observation_certificate_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester.public_key_bytes(),
            &responder.public_key_bytes(),
            response_timestamp,
            &certificate_sha256,
            frame_bytes,
        );
        let response_signature = responder.sign(&response_digest);
        let response = DirectorySyncMessage::ObservationCertificateResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: requester.public_key_bytes(),
            responder: responder.public_key_bytes(),
            response_timestamp,
            certificate_sha256,
            certificate_frame: certificate_frame.clone(),
            signature: response_signature,
        };
        let encoded = encode_directory_sync_message(&response).unwrap();
        assert_eq!(decode_directory_sync_message(&encoded).unwrap(), response);
        IdentityPublicKey::from_bytes(&responder.public_key_bytes())
            .unwrap()
            .verify(&response_digest, &response_signature)
            .unwrap();

        assert_ne!(
            response_digest,
            directory_observation_certificate_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &requester.public_key_bytes(),
                &responder.public_key_bytes(),
                response_timestamp,
                &certificate_sha256,
                frame_bytes + 1,
            )
        );
    }

    #[test]
    fn test_directory_replica_carrier_frames_are_canonical_and_fully_bound() {
        let requester = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let carrier = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
        let descriptor = SignedNodeDescriptor::sign(descriptor_for(&subject), &subject).unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_400,
            [0u8; 32],
            vec![commitment],
            &producer,
        )
        .unwrap();
        let request_id = [0x85; 16];

        let range_request_digest = directory_replica_block_range_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            1,
            1,
            &request_id,
            &requester.public_key_bytes(),
            1_700_000_401,
        );
        let range_request = DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: producer.public_key_bytes(),
            from_height: 1,
            limit: 1,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp: 1_700_000_401,
            signature: requester.sign(&range_request_digest),
        };
        let encoded = encode_directory_sync_message(&range_request).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            range_request
        );

        let block_hash = block.hash();
        let blocks = vec![block];
        let range_response_digest = directory_replica_block_range_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            1_700_000_402,
            &blocks,
            false,
            1,
            &block_hash,
        );
        let range_response = DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            producer: producer.public_key_bytes(),
            carrier: carrier.public_key_bytes(),
            response_timestamp: 1_700_000_402,
            blocks: blocks.clone(),
            has_more: false,
            tip_height: 1,
            tip_hash: block_hash,
            signature: carrier.sign(&range_response_digest),
        };
        let encoded = encode_directory_sync_message(&range_response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            range_response
        );
        let altered_producer_digest = directory_replica_block_range_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &[0x86; 32],
            &carrier.public_key_bytes(),
            1_700_000_402,
            &blocks,
            false,
            1,
            &block_hash,
        );
        assert_ne!(range_response_digest, altered_producer_digest);

        let hashes = vec![commitment.descriptor_hash];
        let object_request_digest = directory_replica_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &hashes,
            &request_id,
            &requester.public_key_bytes(),
            1_700_000_403,
        );
        let object_request = DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: producer.public_key_bytes(),
            descriptor_hashes: hashes.clone(),
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp: 1_700_000_403,
            signature: requester.sign(&object_request_digest),
        };
        let encoded = encode_directory_sync_message(&object_request).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            object_request
        );

        let object_response_digest = directory_replica_descriptor_objects_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            1_700_000_404,
            &hashes,
        );
        let object_response = DirectorySyncMessage::ReplicaDescriptorObjectsResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            producer: producer.public_key_bytes(),
            carrier: carrier.public_key_bytes(),
            response_timestamp: 1_700_000_404,
            descriptor_hashes: hashes,
            objects: vec![descriptor.clone()],
            signature: carrier.sign(&object_response_digest),
        };
        let encoded = encode_directory_sync_message(&object_response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            object_response
        );

        let proof = DirectoryDescriptorInclusionProofV1::from_block_at(
            &blocks[0],
            &descriptor,
            1_700_000_405,
        )
        .unwrap();
        let proof_request_digest =
            directory_replica_descriptor_inclusion_proof_request_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &block_hash,
                &commitment.descriptor_hash,
                &request_id,
                &requester.public_key_bytes(),
                1_700_000_405,
            );
        let proof_request = DirectorySyncMessage::ReplicaDescriptorInclusionProofRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: producer.public_key_bytes(),
            block_hash,
            descriptor_hash: commitment.descriptor_hash,
            request_id,
            requester: requester.public_key_bytes(),
            request_timestamp: 1_700_000_405,
            signature: requester.sign(&proof_request_digest),
        };
        let encoded = encode_directory_sync_message(&proof_request).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            proof_request
        );

        let proof_response_digest =
            directory_replica_descriptor_inclusion_proof_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &producer.public_key_bytes(),
                &carrier.public_key_bytes(),
                1_700_000_406,
                &block_hash,
                &commitment.descriptor_hash,
                &proof,
            );
        let proof_response = DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            producer: producer.public_key_bytes(),
            carrier: carrier.public_key_bytes(),
            response_timestamp: 1_700_000_406,
            block_hash,
            descriptor_hash: commitment.descriptor_hash,
            proof: proof.clone(),
            signature: carrier.sign(&proof_response_digest),
        };
        let encoded = encode_directory_sync_message(&proof_response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            proof_response
        );
        assert_ne!(
            proof_response_digest,
            directory_replica_descriptor_inclusion_proof_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &producer.public_key_bytes(),
                &[0x87; 32],
                1_700_000_406,
                &block_hash,
                &commitment.descriptor_hash,
                &proof,
            )
        );
    }

    #[test]
    fn test_directory_block_v1_canonical_test_vector() {
        let producer = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
        let node = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
        let descriptor = SignedNodeDescriptor::sign(descriptor_for(&node), &node).unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![commitment],
            &producer,
        )
        .unwrap();

        assert_eq!(
            hex::encode(commitment.descriptor_hash),
            "72d814f3d31e2a08d6f2003009cfa548be8e5fd05bc3ba38bb2285cea4432222"
        );
        assert_eq!(
            hex::encode(commitment.hash()),
            "fab10c677239ab88f615137654a4096aaa614b23b8eaea80bb898d1bf736d474"
        );
        assert_eq!(
            hex::encode(block.header.commitment_root),
            "fab10c677239ab88f615137654a4096aaa614b23b8eaea80bb898d1bf736d474"
        );
        assert_eq!(
            hex::encode(block.hash()),
            "51fc47f962be975d17e1f10e2ae9cc38201eea0e072f1bdb9bf3837ff2ad12c2"
        );
        assert_eq!(
            hex::encode(block.producer_signature),
            "8a5963474d6c0a6d94340593cbce67756b99e6a01919bde764c96d50fc57b092f479423b866b2c65036da8f2d2668c56d8c9b90782889e17a7ea2c34b4411e05"
        );
    }
}
