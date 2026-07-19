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
//! - `NodePolicy`: public relay policy hints, including no-exit default
//! - `NodeCapacity`: coarse capacity hints for peer selection
//! - `NodeBootstrapSnapshot`: JSON-friendly bootstrap list of signed descriptors
//! - `NodeDiscoveryMessage`: bounded gossip message envelope for peer sync
//! - Signature-only descriptor verification for local peer-cache retention
//! - `DirectoryCommitmentBlockV1`: signed, hash-linked commitments to public
//!   node descriptor events without embedding endpoint or operator metadata
//! - `DirectoryObservationCheckpointV1`: observer-signed, hash-linked evidence
//!   binding exact producer tips to a recomputable multi-source overlap root
//! - `DirectorySyncMessage`: authenticated, bounded node-to-node transport for
//!   serving one producer's tip, block ranges, descriptor objects, and
//!   independently recomputed observation-checkpoint witness receipts
//! - Opaque policy-head anchor frames that let independent pinned witnesses
//!   retain rollback evidence without receiving policy members or endpoints
//! - Replica-carrier frames that transport already audited producer evidence
//!   without allowing the carrier to replace the producer's signatures
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
//! 10. A pinned carrier may serve an audited producer replica when direct
//!     producer admission is unavailable; receivers still verify both layers
//! 11. A pinned witness may retain one monotonic opaque policy head per
//!     observer and return a signed receipt without learning policy members
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
//! - Observation checkpoints are signed local evidence, not votes, fork choice,
//!   quorum certificates, global consensus, or finality.
//! - A checkpoint witness receipt proves one external node independently
//!   recomputed one exact checkpoint. It is not a vote, quorum, or finality.
//! - A policy-head anchor proves only that one witness retained an opaque
//!   observer-signed epoch/digest at a time. It is not policy approval, a vote,
//!   validator membership, consensus, governance, or finality.
//! - A replica carrier proves transport of its audited copy. It cannot author,
//!   rewrite, finalize, or select the producer's signed chain.
//!
//! ## Last Modified
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

use bincode::Options;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::ledger::merkle_root;

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

/// Maximum producer clock lead accepted by a directory verifier.
///
/// Without this bound, a malicious producer could timestamp one validly signed
/// block far in the future and force every later block to follow that clock.
pub const MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS: u64 = 120;

/// First stable Directory observation-checkpoint hashing contract.
pub const DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1: u16 = 1;

/// Maximum producer tips bound into one observation checkpoint.
pub const MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1: usize = 16;

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

    bincode::options()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(MAX_DESCRIPTOR_BYTES)
        .serialize(&legacy)
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

        bincode::options()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .with_limit(MAX_DESCRIPTOR_BYTES)
            .serialize(self)
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

/// Encodes a canonical bounded Directory Sync frame including its magic byte.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization fails.
pub fn encode_directory_sync_message(message: &DirectorySyncMessage) -> Result<Vec<u8>, CoreError> {
    let payload = bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_SYNC_MESSAGE_BYTES)
        .serialize(message)
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
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_SYNC_MESSAGE_BYTES)
        .reject_trailing_bytes()
        .deserialize(&bytes[1..])
        .map_err(|error| CoreError::malformed(format!("directory sync decode: {error}")))
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
}

/// Encodes a discovery gossip message using bounded bincode.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization fails.
pub fn encode_discovery_message(message: &NodeDiscoveryMessage) -> Result<Vec<u8>, CoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DISCOVERY_MESSAGE_BYTES)
        .serialize(message)
        .map_err(|err| CoreError::malformed(format!("discovery message encode: {err}")))
}

/// Decodes a discovery gossip message using bounded bincode.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when decoding fails.
pub fn decode_discovery_message(bytes: &[u8]) -> Result<NodeDiscoveryMessage, CoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DISCOVERY_MESSAGE_BYTES)
        .deserialize(bytes)
        .map_err(|err| CoreError::malformed(format!("discovery message decode: {err}")))
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

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
            requested_at: 1_700_000_100,
            limit: Some(128),
        };

        let bytes = encode_discovery_message(&message).unwrap();
        let decoded = decode_discovery_message(&bytes).unwrap();

        assert_eq!(decoded, message);
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

    #[test]
    fn test_directory_observation_witness_frames_are_canonical_and_bound() {
        let observer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
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
        let encoded = encode_directory_sync_message(&request).unwrap();
        let decoded = decode_directory_sync_message(&encoded).unwrap();
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
        let encoded = encode_directory_sync_message(&response).unwrap();
        assert_eq!(decode_directory_sync_message(&encoded).unwrap(), response);
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
            objects: vec![descriptor],
            signature: carrier.sign(&object_response_digest),
        };
        let encoded = encode_directory_sync_message(&object_response).unwrap();
        assert_eq!(
            decode_directory_sync_message(&encoded).unwrap(),
            object_response
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
