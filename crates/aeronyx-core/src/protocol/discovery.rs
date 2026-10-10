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
//! - Signed, descriptor-bound Anonymous Mailbox admission-work policy metadata
//!   for exact-target clients
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.
//! Every public item is re-exported from this module root, so all existing
//! `protocol::discovery::*` paths are unchanged.
//! - `discovery/protocol_feature.rs`: signed protocol-feature negotiation
//! - `discovery/descriptor.rs`: signed node descriptors
//! - `discovery/mailbox_work_policy.rs`: Anonymous Mailbox admission-work policy
//! - `discovery/directory_block.rs`: Directory Chain V1 commitment blocks
//! - `discovery/inclusion_proof.rs`: Directory descriptor inclusion proofs
//! - `discovery/observation_checkpoint.rs`: Directory observation checkpoints
//! - `discovery/observation_certificate.rs`: Portable Directory observation certificates
//! - `discovery/route_domain.rs`: Portable route-domain attestations
//! - `discovery/sync_message.rs`: Directory Sync V1 wire messages
//! - `discovery/sync_signing.rs`: Directory Sync signing digests
//! - `discovery/gossip.rs`: Discovery bootstrap snapshots and gossip
//! - `discovery/test_support.rs`: shared test fixtures (test builds only)
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
//! v0.36.0-ArchSplit - Split into focused child modules; bodies unchanged
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

mod descriptor;
mod directory_block;
mod gossip;
mod inclusion_proof;
mod mailbox_work_policy;
mod observation_certificate;
mod observation_checkpoint;
mod protocol_feature;
mod route_domain;
mod sync_message;
mod sync_signing;
#[cfg(test)]
mod test_support;

pub use descriptor::{
    NodeCapability, NodeCapacity, NodeDescriptor, NodePolicy, SignedNodeDescriptor,
    MAX_SIGNED_NODE_DESCRIPTOR_BYTES, NODE_DESCRIPTOR_SCHEMA_VERSION,
};
pub use directory_block::{
    DirectoryCommitmentBlockV1, DirectoryCommitmentHeaderV1, DirectoryCommitmentValidationError,
    DirectoryDescriptorCommitmentV1, DIRECTORY_COMMITMENT_BLOCK_VERSION_V1,
    MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
};
pub use gossip::{
    decode_discovery_message, encode_discovery_message, NodeBootstrapSnapshot,
    NodeDiscoveryMessage, NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION,
};
pub use inclusion_proof::{
    DirectoryDescriptorInclusionProofError, DirectoryDescriptorInclusionProofV1,
    DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1,
    MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1,
};
pub use mailbox_work_policy::{
    AnonymousMailboxWorkPolicyError, SignedAnonymousMailboxWorkPolicyV1,
    ANONYMOUS_MAILBOX_WORK_POLICY_VERSION_V1, ANONYMOUS_MAILBOX_WORK_POLICY_WIRE_BYTES_V1,
    MAX_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1, MIN_ANONYMOUS_MAILBOX_TICKET_WORK_BITS_V1,
};
pub use observation_certificate::{
    decode_directory_observation_certificate, encode_directory_observation_certificate,
    DirectoryObservationCertificateV1, DirectoryObservationCertificateValidationError,
    DirectoryObservationWitnessReceiptV1, DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC,
    DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES,
};
pub use observation_checkpoint::{
    DirectoryObservationCheckpointV1, DirectoryObservationCheckpointValidationError,
    DirectoryObservationTipV1, DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1,
    MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1,
};
pub use protocol_feature::NodeProtocolFeature;
pub use route_domain::{
    decode_route_domain_attestation_certificate, encode_route_domain_attestation_certificate,
    route_domain_attestation_signing_bytes, RouteDomainAttestationCertificateV1,
    RouteDomainAttestationCertificateValidationError, RouteDomainAttestationV1,
    RouteDomainAttestationValidationError, MAX_ROUTE_DOMAIN_ATTESTATIONS_V1,
    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES,
    MAX_ROUTE_DOMAIN_ATTESTATION_LIFETIME_SECS_V1, ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_MAGIC,
    ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_VERSION_V1, ROUTE_DOMAIN_ATTESTATION_VERSION_V1,
};
pub use sync_message::{
    decode_directory_sync_message, encode_directory_sync_message, DirectorySyncMessage,
    DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1, DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1, DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
    DIRECTORY_POLICY_ANCHOR_CONFLICT_V1, DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
    DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1, DIRECTORY_SYNC_MAGIC, MAX_DIRECTORY_SYNC_BLOCKS_V1,
    MAX_DIRECTORY_SYNC_OBJECTS_V1,
};
pub use sync_signing::{
    directory_block_range_request_signing_bytes, directory_block_range_response_signing_bytes,
    directory_descriptor_inclusion_proof_request_signing_bytes,
    directory_descriptor_inclusion_proof_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_descriptor_objects_response_signing_bytes,
    directory_observation_certificate_request_signing_bytes,
    directory_observation_certificate_response_signing_bytes,
    directory_observation_witness_carrier_request_signing_bytes,
    directory_observation_witness_carrier_response_signing_bytes,
    directory_observation_witness_request_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_policy_anchor_request_signing_bytes, directory_policy_anchor_response_signing_bytes,
    directory_replica_block_range_request_signing_bytes,
    directory_replica_block_range_response_signing_bytes,
    directory_replica_descriptor_inclusion_proof_request_signing_bytes,
    directory_replica_descriptor_inclusion_proof_response_signing_bytes,
    directory_replica_descriptor_objects_request_signing_bytes,
    directory_replica_descriptor_objects_response_signing_bytes,
    directory_tip_request_signing_bytes, directory_tip_response_signing_bytes,
};

// ============================================
// Serialization constants
// ============================================

/// Stable production chain identifier for public node-directory commitments.
///
/// This is `SHA-256("AeroNyx-Directory-Mainnet-v1")`. Changing it creates a
/// different directory chain and requires an explicit protocol migration.
pub const AERONYX_DIRECTORY_MAINNET_CHAIN_ID: [u8; 32] = [
    0xa0, 0x4a, 0x2f, 0xdf, 0xc8, 0x32, 0x07, 0x08, 0x30, 0x66, 0x2d, 0x43, 0x5a, 0xfc, 0x9e, 0x1e,
    0x78, 0x32, 0xda, 0xde, 0x2f, 0xd5, 0x95, 0x6b, 0xe7, 0x78, 0x28, 0x36, 0xca, 0x61, 0xd2, 0x2f,
];

/// Maximum producer clock lead accepted by a directory verifier.
///
/// Without this bound, a malicious producer could timestamp one validly signed
/// block far in the future and force every later block to follow that clock.
pub const MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS: u64 = 120;

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
