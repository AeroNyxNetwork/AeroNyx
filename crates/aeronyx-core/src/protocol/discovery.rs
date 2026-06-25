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
//!
//! ## Important Note for Next Developer
//! - Do not put private keys, client IPs, destination metadata, DNS contents,
//!   packet payloads, browsing history, voucher secrets, or wallet-level
//!   traffic in this descriptor.
//! - `bincode` field order is part of the signing contract. Add new fields
//!   only at the end and keep backward compatibility in mind.
//! - Default public policy is no-exit. Future onion routing must opt into any
//!   exit behavior through a separate reviewed policy.
//!
//! ## Last Modified
//! v0.4.0-DiscoverySignatureOnlyVerify - Added signature-only verification for expired peer-cache retention
//! v0.1.0-DiscoveryPhase1 - Initial signed descriptor primitives
//! v0.2.0-DiscoveryPhase2 - Added bounded bootstrap snapshot type
//! v0.3.0-DiscoveryPhase4 - Added bounded discovery gossip messages
// ============================================================================

use bincode::Options;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;

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
pub const NODE_DESCRIPTOR_SCHEMA_VERSION: u16 = 1;

/// Current bootstrap snapshot schema version.
pub const NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION: u16 = 1;

/// Maximum accepted JSON bootstrap snapshot size.
const MAX_BOOTSTRAP_SNAPSHOT_BYTES: usize = 512 * 1024;

/// Maximum accepted binary discovery gossip message size.
const MAX_DISCOVERY_MESSAGE_BYTES: u64 = 512 * 1024;

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
        if self.descriptor.schema_version != NODE_DESCRIPTOR_SCHEMA_VERSION {
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
        if self.descriptor.schema_version != NODE_DESCRIPTOR_SCHEMA_VERSION {
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
}
