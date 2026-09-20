// ============================================
// File: crates/aeronyx-core/src/protocol/version.rs
// ============================================
//! # Protocol Versioning
//!
//! ## Creation Reason
//! Manages protocol versions to ensure compatibility between
//! clients and servers of different versions.
//!
//! ## Main Functionality
//! - `ProtocolVersion`: Version identifier type
//! - Version compatibility checking
//! - Version negotiation support
//!
//! ## Versioning Strategy
//! - Single byte version number (0-255)
//! - Increment for ANY breaking change
//! - Backward compatibility maintained where possible
//! - Server may support multiple versions simultaneously
//!
//! ## Version History
//! | Version | Description |
//! |---------|-------------|
//! | 0x01    | Initial protocol (MVP) |
//! | 0x02    | Directional session keys and transcript-bound handshake |
//!
//! ## ⚠️ Important Note for Next Developer
//! - ALWAYS increment version for wire format changes
//! - Document all versions in version history
//! - Consider supporting version negotiation for upgrades
//!
//! ## Last Modified
//! v0.1.0 - Initial version definitions

use std::fmt;

use serde::{Deserialize, Serialize};

// ============================================
// Constants
// ============================================

/// Current protocol version.
///
/// # Version 0x02
/// - `ClientHello`: 138 bytes
/// - `ServerHello`: 150 bytes
/// - Ed25519 signatures
/// - X25519 key exchange
/// - ChaCha20-Poly1305 transport
/// - Directional session keys and transcript-bound hellos
///
/// [PROTOCOL-V2-DEFAULT-ACTIVATION 2026-09-20 by Codex] New handshakes default
/// to v2 while the supported range remains additive for deployed v1 clients.
pub const CURRENT_PROTOCOL_VERSION: u8 = 0x02;

/// Minimum supported protocol version.
///
/// Servers will reject clients with versions below this.
pub const MIN_SUPPORTED_VERSION: u8 = 0x01;

/// Maximum supported protocol version.
///
/// Servers will reject clients with versions above this.
pub const MAX_SUPPORTED_VERSION: u8 = 0x02;

// ============================================
// ProtocolVersion
// ============================================

/// Protocol version identifier.
///
/// # Purpose
/// Encapsulates version checking logic and provides a type-safe
/// way to handle protocol versions throughout the codebase.
///
/// # Example
/// ```
/// use aeronyx_core::protocol::ProtocolVersion;
///
/// let version = ProtocolVersion::current();
/// assert!(version.is_supported());
///
/// let old_version = ProtocolVersion::new(0);
/// assert!(!old_version.is_supported());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProtocolVersion(u8);

impl ProtocolVersion {
    /// Creates a new protocol version.
    #[must_use]
    pub const fn new(version: u8) -> Self {
        Self(version)
    }

    /// Returns the current protocol version.
    #[must_use]
    pub const fn current() -> Self {
        Self(CURRENT_PROTOCOL_VERSION)
    }

    /// Returns the raw version number.
    #[must_use]
    pub const fn as_u8(&self) -> u8 {
        self.0
    }

    /// Checks if this version is supported by the current implementation.
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        self.0 >= MIN_SUPPORTED_VERSION && self.0 <= MAX_SUPPORTED_VERSION
    }

    /// Checks if this version is compatible with another version.
    ///
    /// For now, versions are compatible only if they're identical.
    /// Future versions may implement more sophisticated negotiation.
    #[must_use]
    pub const fn is_compatible_with(&self, other: &Self) -> bool {
        // For MVP, require exact version match
        self.0 == other.0
    }

    /// Returns the minimum of two versions (for negotiation).
    #[must_use]
    pub const fn negotiate(&self, other: &Self) -> Self {
        if self.0 < other.0 {
            *self
        } else {
            *other
        }
    }
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        Self::current()
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}", self.0 >> 4, self.0 & 0x0F)
    }
}

impl From<u8> for ProtocolVersion {
    fn from(version: u8) -> Self {
        Self(version)
    }
}

impl From<ProtocolVersion> for u8 {
    fn from(version: ProtocolVersion) -> Self {
        version.0
    }
}

// ============================================
// Version Features
// ============================================

/// Features available in each protocol version.
///
/// This allows code to check for feature availability without
/// hardcoding version numbers throughout the codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionFeature {
    /// Basic handshake protocol
    Handshake,
    /// ChaCha20-Poly1305 transport encryption
    ChaCha20Transport,
    /// Ed25519 signatures
    Ed25519Signatures,
    /// X25519 key exchange
    X25519KeyExchange,
}

impl ProtocolVersion {
    /// Checks if a feature is available in this version.
    #[must_use]
    pub const fn has_feature(&self, feature: VersionFeature) -> bool {
        match feature {
            // All features available in v1
            VersionFeature::Handshake
            | VersionFeature::ChaCha20Transport
            | VersionFeature::Ed25519Signatures
            | VersionFeature::X25519KeyExchange => self.0 >= 0x01,
        }
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crypto::{handshake::DefaultHandshakeCrypto, keys::IdentityKeyPair},
        protocol::{
            is_supported_hello_version, ClientHello, MessageType, PROTOCOL_VERSION_V1,
            PROTOCOL_VERSION_V2,
        },
    };

    const SIGNED_HELLO_TIMESTAMP: i64 = 1_700_000_000;
    const V2_EXTENSION: &[u8] = b"deterministic-v2-extension";

    fn deterministic_identity() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[0x42; 32])
            .unwrap_or_else(|_| unreachable!("fixed Ed25519 seed must be valid"))
    }

    fn client_hello_sign_data_v1(hello: &ClientHello) -> Vec<u8> {
        let mut transcript = Vec::with_capacity(74);
        transcript.push(hello.message_type);
        transcript.push(hello.version);
        transcript.extend_from_slice(&hello.client_public_key);
        transcript.extend_from_slice(&hello.client_ephemeral_key);
        transcript.extend_from_slice(&hello.timestamp.to_le_bytes());
        transcript
    }

    fn signed_hello(version: u8, v2_transcript: bool) -> ClientHello {
        let identity = deterministic_identity();
        let mut hello = ClientHello {
            message_type: MessageType::ClientHello as u8,
            version,
            client_public_key: identity.public_key_bytes(),
            client_ephemeral_key: [0x24; 32],
            timestamp: SIGNED_HELLO_TIMESTAMP,
            signature: [0; 64],
        };
        let transcript = if v2_transcript {
            DefaultHandshakeCrypto::client_hello_sign_data_v2(&hello, V2_EXTENSION)
        } else {
            client_hello_sign_data_v1(&hello)
        };
        hello.signature = identity.sign(&transcript);
        hello
    }

    fn signature_matches(hello: &ClientHello, v2_transcript: bool) -> bool {
        let transcript = if v2_transcript {
            DefaultHandshakeCrypto::client_hello_sign_data_v2(hello, V2_EXTENSION)
        } else {
            client_hello_sign_data_v1(hello)
        };
        deterministic_identity()
            .public_key()
            .verify(&transcript, &hello.signature)
            .is_ok()
    }

    #[test]
    fn test_current_version() {
        let version = ProtocolVersion::current();
        assert_eq!(version.as_u8(), CURRENT_PROTOCOL_VERSION);
        assert_eq!(version.as_u8(), PROTOCOL_VERSION_V2);
        assert_eq!(ProtocolVersion::default(), version);
        assert!(version.is_supported());
    }

    #[test]
    fn test_version_support_check() {
        assert!(ProtocolVersion::new(PROTOCOL_VERSION_V1).is_supported());
        assert!(ProtocolVersion::new(PROTOCOL_VERSION_V2).is_supported());
        assert!(!ProtocolVersion::new(0).is_supported());
        assert!(!ProtocolVersion::new(3).is_supported());
        assert!(!ProtocolVersion::new(0xFF).is_supported());
    }

    #[test]
    fn signed_hello_admission_accepts_v1_and_v2_but_rejects_adjacent_versions() {
        // [PROTOCOL-V2-DEFAULT-ACTIVATION 2026-09-20 by Codex] A valid
        // signature never expands the frozen supported-version range.
        let signed_v1 = signed_hello(PROTOCOL_VERSION_V1, false);
        assert!(signature_matches(&signed_v1, false));
        assert!(is_supported_hello_version(signed_v1.version));
        assert!(ProtocolVersion::new(signed_v1.version).is_supported());

        let signed_v2 = signed_hello(PROTOCOL_VERSION_V2, true);
        assert!(signature_matches(&signed_v2, true));
        assert!(is_supported_hello_version(signed_v2.version));
        assert!(ProtocolVersion::new(signed_v2.version).is_supported());

        for rejected in [0, 3] {
            let signed = signed_hello(rejected, false);
            assert!(signature_matches(&signed, false));
            assert!(!is_supported_hello_version(signed.version));
            assert!(!ProtocolVersion::new(signed.version).is_supported());
        }
    }

    #[test]
    fn signed_transcript_rejects_post_signature_version_substitution() {
        let signed_v1 = signed_hello(PROTOCOL_VERSION_V1, false);
        let mut substituted_v1 = signed_v1;
        substituted_v1.version = PROTOCOL_VERSION_V2;
        assert!(!signature_matches(&substituted_v1, false));

        let signed_v2 = signed_hello(PROTOCOL_VERSION_V2, true);
        let mut substituted_v2 = signed_v2;
        substituted_v2.version = PROTOCOL_VERSION_V1;
        assert!(!signature_matches(&substituted_v2, true));
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = ProtocolVersion::new(1);
        let v1_copy = ProtocolVersion::new(1);
        let v2 = ProtocolVersion::new(2);

        assert!(v1.is_compatible_with(&v1_copy));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_version_negotiation() {
        let v1 = ProtocolVersion::new(1);
        let v2 = ProtocolVersion::new(2);

        let negotiated = v1.negotiate(&v2);
        assert_eq!(negotiated, v1);

        let negotiated = v2.negotiate(&v1);
        assert_eq!(negotiated, v1);
    }

    #[test]
    fn test_version_display() {
        let v1 = ProtocolVersion::new(0x01);
        assert_eq!(v1.to_string(), "v0.1");

        let v16 = ProtocolVersion::new(0x10);
        assert_eq!(v16.to_string(), "v1.0");
    }

    #[test]
    fn test_version_features() {
        let v1 = ProtocolVersion::new(1);

        assert!(v1.has_feature(VersionFeature::Handshake));
        assert!(v1.has_feature(VersionFeature::ChaCha20Transport));
        assert!(v1.has_feature(VersionFeature::Ed25519Signatures));
        assert!(v1.has_feature(VersionFeature::X25519KeyExchange));

        let v0 = ProtocolVersion::new(0);
        assert!(!v0.has_feature(VersionFeature::Handshake));
    }
}
