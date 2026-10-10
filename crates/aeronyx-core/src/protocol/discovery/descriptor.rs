// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/descriptor.rs
// ============================================
//! # Signed node descriptors
//!
//! Owns the descriptor schema/size constants, the `NodeCapability`,
//! `NodePolicy` and `NodeCapacity` hints, the `NodeDescriptor` body
//! (including legacy schema-v1 signing bytes), and the Ed25519-signed
//! `SignedNodeDescriptor` envelope with its canonical bounded codec,
//! verification, and Anonymous Mailbox work-policy accessors.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

use super::directory_block::DirectoryDescriptorCommitmentV1;
use super::mailbox_work_policy::{
    anonymous_mailbox_work_policy_tokens, append_semver_build_identifier,
    AnonymousMailboxWorkPolicyError, SignedAnonymousMailboxWorkPolicyV1,
};
use super::protocol_feature::NodeProtocolFeature;
use super::serde_bytes64;

/// Maximum accepted serialized descriptor size.
///
/// Descriptors are intended to be small control-plane objects. Keeping a
/// strict cap prevents unbounded memory allocation when reading bootstrap
/// snapshots or future gossip payloads.
/// Maximum canonical encoded length of one signed node descriptor.
pub const MAX_SIGNED_NODE_DESCRIPTOR_BYTES: usize = 16 * 1024;
const MAX_DESCRIPTOR_BYTES: u64 = 16 * 1024;

/// Current signed descriptor schema version.
pub const NODE_DESCRIPTOR_SCHEMA_VERSION: u16 = 2;

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

    /// Adds one canonical, target-signed Anonymous Mailbox work-policy token.
    ///
    /// This is additive inside the existing `software_version` metadata and
    /// does not change descriptor schema or old descriptors' signing bytes.
    ///
    /// # Errors
    /// Fails closed for invalid claims, work bounds, identity mismatch, an
    /// existing reserved policy-family token, or a descriptor exceeding its
    /// existing canonical size limit after insertion.
    pub fn with_anonymous_mailbox_work_policy(
        mut self,
        work_bits: u8,
        identity: &IdentityKeyPair,
    ) -> Result<Self, AnonymousMailboxWorkPolicyError> {
        // [ANONYMOUS-MAILBOX-WORK-POLICY 2026-09-07 by Codex] Reserve the
        // complete family case-insensitively. A malformed or future token must
        // not be silently shadowed by a newly generated v1 policy.
        if anonymous_mailbox_work_policy_tokens(&self.software_version)
            .next()
            .is_some()
        {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        let policy = SignedAnonymousMailboxWorkPolicyV1::issue(&self, work_bits, identity)?;
        let token = policy.semver_build_token();
        self.software_version = append_semver_build_identifier(&self.software_version, &token);
        self.signing_bytes()
            .map_err(|_| AnonymousMailboxWorkPolicyError::MalformedPolicy)?;
        Ok(self)
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

    /// Encodes this signed descriptor with the canonical bounded discovery codec.
    ///
    /// # Errors
    /// Returns a malformed-message error when the descriptor exceeds the
    /// existing descriptor bound or cannot be serialized canonically.
    pub fn encode_canonical(&self) -> Result<Vec<u8>, CoreError> {
        encode_bincode_bounded(self, MAX_DESCRIPTOR_BYTES)
            .map_err(|error| CoreError::malformed(format!("signed descriptor encode: {error}")))
    }

    /// Decodes one canonical bounded signed descriptor without trusting it.
    ///
    /// Call [`Self::verify_at`] before using the descriptor as current routing
    /// authority. Trailing bytes and non-canonical encodings fail closed.
    ///
    /// # Errors
    /// Returns a malformed-message error for oversized, trailing, malformed,
    /// or non-canonical bytes.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, CoreError> {
        let descriptor: Self =
            decode_bincode_bounded(bytes, MAX_DESCRIPTOR_BYTES, TrailingBytesPolicy::Reject)
                .map_err(|error| {
                    CoreError::malformed(format!("signed descriptor decode: {error}"))
                })?;
        if descriptor.encode_canonical()?.as_slice() != bytes {
            return Err(CoreError::malformed(
                "signed descriptor encoding is non-canonical",
            ));
        }
        Ok(descriptor)
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

    /// Validates and returns this descriptor's single signed Anonymous Mailbox
    /// work-policy token.
    ///
    /// Outer descriptor authentication is checked before nested policy parsing.
    /// Missing, malformed, duplicate, unknown-family-version, expired, and
    /// mismatched claims all fail closed with coarse typed errors.
    pub fn anonymous_mailbox_work_policy_at(
        &self,
        now: u64,
    ) -> Result<SignedAnonymousMailboxWorkPolicyV1, AnonymousMailboxWorkPolicyError> {
        self.verify_signature()
            .map_err(|_| AnonymousMailboxWorkPolicyError::SignatureRejected)?;
        if !self.descriptor.is_valid_at(now) {
            return Err(AnonymousMailboxWorkPolicyError::NotCurrentlyValid);
        }
        let mut tokens = anonymous_mailbox_work_policy_tokens(&self.descriptor.software_version);
        let token = tokens
            .next()
            .ok_or(AnonymousMailboxWorkPolicyError::MissingPolicy)?;
        if tokens.next().is_some() {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        let policy = SignedAnonymousMailboxWorkPolicyV1::decode_semver_build_token(token)?;
        policy.verify_for_descriptor(&self.descriptor, now)?;
        Ok(policy)
    }

    /// Validates an exact Directory commitment pin before returning the work
    /// policy. This prohibits target substitution or silent descriptor rotation.
    pub fn anonymous_mailbox_work_policy_for_pin_at(
        &self,
        pin: &DirectoryDescriptorCommitmentV1,
        now: u64,
    ) -> Result<SignedAnonymousMailboxWorkPolicyV1, AnonymousMailboxWorkPolicyError> {
        if !pin
            .matches_signed_descriptor(self)
            .map_err(|_| AnonymousMailboxWorkPolicyError::SignatureRejected)?
        {
            return Err(AnonymousMailboxWorkPolicyError::ClaimsConflict);
        }
        self.anonymous_mailbox_work_policy_at(now)
    }
}

#[cfg(test)]
mod tests;
