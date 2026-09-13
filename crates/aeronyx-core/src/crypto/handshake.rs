// ============================================
// File: crates/aeronyx-core/src/crypto/handshake.rs
// ============================================
//! # Handshake Cryptography
//!
//! ## Creation Reason
//! Provides cryptographic operations for the handshake protocol,
//! including signature creation/verification and key exchange.
//!
//! ## Main Functionality
//! - `HandshakeCrypto`: Trait for handshake cryptographic operations
//! - `DefaultHandshakeCrypto`: Production implementation
//! - Signature verification for ClientHello messages
//! - Signature creation for ServerHello messages
//!
//! ## Handshake Flow
//! ```text
//! Client                                          Server
//!   │                                               │
//!   │  ClientHello                                  │
//!   │  ├─ client_public_key (Ed25519)              │
//!   │  ├─ client_ephemeral_key (X25519)            │
//!   │  ├─ timestamp                                 │
//!   │  └─ signature ─────────────────────────────►  │
//!   │                                               │
//!   │                           Verify signature    │
//!   │                           Generate ephemeral  │
//!   │                           Derive session key  │
//!   │                                               │
//!   │                                  ServerHello  │
//!   │  ◄───────────────────────────────── signature │
//!   │                    server_ephemeral_key (X25519)
//!   │                    assigned_ip                │
//!   │                    session_id                 │
//!   │                                               │
//!   │  Verify signature                             │
//!   │  Derive session key                           │
//!   │                                               │
//!   │ ═══════════ Encrypted Tunnel ═══════════════ │
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Signature data must be constructed in exact order
//! - Timestamp validation prevents replay attacks
//! - All signature operations should be constant-time where possible
//! - Never log shared secrets, session keys, signatures, signed transcript
//!   bytes, or raw public keys in production node logs.
//!
//! ## Last Modified
//! v0.1.2 - Downgraded successful handshake diagnostic logs to debug level for production nodes
//! v0.1.1 - Redacted sensitive handshake material from crypto logs
//! v0.1.0 - Initial handshake crypto implementation

use crate::crypto::kdf::{
    derive_session_key, derive_session_keys_v2, transcript_hash_v2, SessionKeys,
};
use crate::crypto::keys::{EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey, SessionKey};
use crate::error::{CoreError, Result};
use crate::protocol::messages::PROTOCOL_VERSION_V2;
use crate::protocol::{ClientHello, ServerHello};
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

use aeronyx_common::time::Timestamp;
use tracing::{debug, trace, warn};

// ============================================
// Helper function for privacy-safe diagnostics
// ============================================

/// Returns a length-only placeholder for sensitive bytes.
///
/// The relay/node process may run under third-party operators and centralized
/// log collection. Length-only diagnostics preserve flow visibility without
/// leaking key material or stable identity bytes into logs.
fn redacted_len(bytes: &[u8]) -> String {
    format!("<redacted:{} bytes>", bytes.len())
}

// ============================================
// HandshakeCrypto Trait
// ============================================

/// Trait for handshake cryptographic operations.
///
/// # Purpose
/// Abstracts handshake crypto operations to allow:
/// - Testing with mock implementations
/// - Alternative crypto backends
/// - Hardware security module integration
///
/// # Example
/// ```ignore
/// let crypto = DefaultHandshakeCrypto::new(identity);
///
/// // Process ClientHello
/// crypto.verify_client_hello(&client_hello)?;
///
/// // Create ServerHello
/// let (server_hello, session_key) = crypto.process_handshake(
///     &client_hello,
///     assigned_ip,
///     session_id,
/// )?;
/// ```
pub trait HandshakeCrypto: Send + Sync {
    /// Returns the server's identity public key.
    fn public_key(&self) -> IdentityPublicKey;

    /// Verifies the signature on a ClientHello message.
    ///
    /// # Errors
    /// - `SignatureVerification`: If signature is invalid
    /// - `InvalidTimestamp`: If timestamp is out of acceptable range
    fn verify_client_hello(&self, msg: &ClientHello) -> Result<()>;

    /// Processes a ClientHello and produces a ServerHello with session key.
    ///
    /// # Arguments
    /// * `client_hello` - Validated ClientHello message
    /// * `assigned_ip` - Virtual IP to assign to client
    /// * `session_id` - Unique session identifier
    ///
    /// # Returns
    /// Tuple of (signed ServerHello, derived SessionKey)
    fn process_handshake(
        &self,
        client_hello: &ClientHello,
        assigned_ip: [u8; 4],
        session_id: [u8; 16],
    ) -> Result<(ServerHello, SessionKey)>;

    /// v0x02: like [`verify_client_hello`], but the signature also covers
    /// `SHA-256(extension)` — the voucher bytes that trail the fixed hello —
    /// so a voucher observed on the wire cannot be re-attached to another
    /// client's hello.
    ///
    /// [`verify_client_hello`]: HandshakeCrypto::verify_client_hello
    fn verify_client_hello_v2(&self, msg: &ClientHello, extension: &[u8]) -> Result<()> {
        let _ = (msg, extension);
        Err(CoreError::UnsupportedVersion {
            got: PROTOCOL_VERSION_V2,
            expected: 0x01,
        })
    }

    /// v0x02: answer a verified hello with a transcript-bound ServerHello and
    /// derive the two per-direction session keys.
    fn process_handshake_v2(
        &self,
        client_hello: &ClientHello,
        extension: &[u8],
        assigned_ip: [u8; 4],
        session_id: [u8; 16],
    ) -> Result<(ServerHello, SessionKeys)> {
        let _ = (client_hello, extension, assigned_ip, session_id);
        Err(CoreError::UnsupportedVersion {
            got: PROTOCOL_VERSION_V2,
            expected: 0x01,
        })
    }
}

// ============================================
// DefaultHandshakeCrypto
// ============================================

/// Default production implementation of handshake cryptography.
pub struct DefaultHandshakeCrypto {
    /// Server's long-term identity key pair
    identity: IdentityKeyPair,
    /// Maximum allowed timestamp skew in seconds
    max_timestamp_skew: u64,
}

impl DefaultHandshakeCrypto {
    /// Creates a new handshake crypto instance.
    ///
    /// # Arguments
    /// * `identity` - Server's Ed25519 identity key pair
    #[must_use]
    pub fn new(identity: IdentityKeyPair) -> Self {
        debug!("[CRYPTO-DEBUG] DefaultHandshakeCrypto::new() - Creating handshake crypto instance");
        debug!(
            "[CRYPTO-DEBUG] Server Identity Public Key: {}",
            redacted_len(&identity.public_key_bytes())
        );
        Self {
            identity,
            max_timestamp_skew: 30,
        }
    }

    /// Sets the maximum allowed timestamp skew.
    ///
    /// # Arguments
    /// * `seconds` - Maximum clock difference in seconds
    #[must_use]
    pub fn with_timestamp_skew(mut self, seconds: u64) -> Self {
        debug!(
            "[CRYPTO-DEBUG] Setting max_timestamp_skew to {} seconds",
            seconds
        );
        self.max_timestamp_skew = seconds;
        self
    }

    /// Constructs the data to be signed for ClientHello.
    ///
    /// # Wire Format
    /// ```text
    /// message_type (1 byte) ||
    /// version (1 byte) ||
    /// client_public_key (32 bytes) ||
    /// client_ephemeral_key (32 bytes) ||
    /// timestamp (8 bytes)
    /// ```
    fn client_hello_sign_data(msg: &ClientHello) -> Vec<u8> {
        let mut data = Vec::with_capacity(74);
        data.push(msg.message_type);
        data.push(msg.version);
        data.extend_from_slice(&msg.client_public_key);
        data.extend_from_slice(&msg.client_ephemeral_key);
        data.extend_from_slice(&msg.timestamp.to_le_bytes());

        trace!(
            "[CRYPTO-DEBUG] client_hello_sign_data constructed: {} bytes",
            data.len()
        );

        data
    }

    /// Constructs the data to be signed for ServerHello.
    ///
    /// # Wire Format
    /// ```text
    /// message_type (1 byte) ||
    /// version (1 byte) ||
    /// server_public_key (32 bytes) ||
    /// server_ephemeral_key (32 bytes) ||
    /// assigned_ip (4 bytes) ||
    /// session_id (16 bytes) ||
    /// client_public_key (32 bytes)
    /// ```
    /// v0x02 client signing data: the v0x01 fields followed by the SHA-256 of
    /// the trailing extension (empty extension → hash of the empty string).
    #[must_use]
    pub fn client_hello_sign_data_v2(msg: &ClientHello, extension: &[u8]) -> Vec<u8> {
        let mut data = Self::client_hello_sign_data(msg);
        data.extend_from_slice(&Sha256::digest(extension));
        data
    }

    /// v0x02 server signing data: the v0x01 fields followed by the client's
    /// ephemeral key and timestamp, so this ServerHello answers exactly one
    /// ClientHello and cannot be replayed to a later one.
    #[must_use]
    pub fn server_hello_sign_data_v2(msg: &ServerHello, client: &ClientHello) -> Vec<u8> {
        let mut data = Self::server_hello_sign_data(msg, &client.client_public_key);
        data.extend_from_slice(&client.client_ephemeral_key);
        data.extend_from_slice(&client.timestamp.to_le_bytes());
        data
    }

    fn server_hello_sign_data(msg: &ServerHello, client_public: &[u8; 32]) -> Vec<u8> {
        let mut data = Vec::with_capacity(118);
        data.push(msg.message_type);
        data.push(msg.version);
        data.extend_from_slice(&msg.server_public_key);
        data.extend_from_slice(&msg.server_ephemeral_key);
        data.extend_from_slice(&msg.assigned_ip);
        data.extend_from_slice(&msg.session_id);
        data.extend_from_slice(client_public);

        trace!(
            "[CRYPTO-DEBUG] server_hello_sign_data constructed: {} bytes",
            data.len()
        );

        data
    }
}

impl HandshakeCrypto for DefaultHandshakeCrypto {
    fn public_key(&self) -> IdentityPublicKey {
        self.identity.public_key()
    }

    fn verify_client_hello(&self, msg: &ClientHello) -> Result<()> {
        debug!("[CRYPTO-DEBUG] ========== verify_client_hello START ==========");

        debug!("[CRYPTO-DEBUG] ClientHello contents:");
        debug!("[CRYPTO-DEBUG]   message_type: {}", msg.message_type);
        debug!("[CRYPTO-DEBUG]   version: {}", msg.version);
        debug!(
            "[CRYPTO-DEBUG]   client_public_key: {}",
            redacted_len(&msg.client_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG]   client_ephemeral_key: {}",
            redacted_len(&msg.client_ephemeral_key)
        );
        debug!("[CRYPTO-DEBUG]   timestamp: {}", msg.timestamp);
        debug!(
            "[CRYPTO-DEBUG]   signature: {}",
            redacted_len(&msg.signature)
        );

        // 1. Validate timestamp
        let timestamp = Timestamp::from_secs(msg.timestamp);
        let current_time = Timestamp::now().as_secs();
        debug!(
            "[CRYPTO-DEBUG] Timestamp validation: client={}, current={}, diff={}, max_skew={}",
            msg.timestamp,
            current_time,
            current_time.abs_diff(msg.timestamp),
            self.max_timestamp_skew
        );

        if !timestamp.is_recent(self.max_timestamp_skew) {
            warn!("[CRYPTO-DEBUG] FAILED: Timestamp validation failed");
            return Err(CoreError::invalid_timestamp(format!(
                "Timestamp {} is not recent (max skew: {}s)",
                msg.timestamp, self.max_timestamp_skew
            )));
        }
        debug!("[CRYPTO-DEBUG] Timestamp validation: PASSED");

        // 2. Verify signature
        let sign_data = Self::client_hello_sign_data(msg);
        debug!(
            "[CRYPTO-DEBUG] Sign data for verification: {} bytes",
            sign_data.len()
        );
        debug!(
            "[CRYPTO-DEBUG] Sign data redacted: {}",
            redacted_len(&sign_data)
        );

        let client_public = IdentityPublicKey::from_bytes(&msg.client_public_key)?;
        debug!("[CRYPTO-DEBUG] Verifying signature with client public key...");

        match client_public.verify(&sign_data, &msg.signature) {
            Ok(()) => {
                debug!("[CRYPTO-DEBUG] Signature verification: PASSED");
                debug!("[CRYPTO-DEBUG] ========== verify_client_hello SUCCESS ==========");
                Ok(())
            }
            Err(e) => {
                warn!("[CRYPTO-DEBUG] Signature verification: FAILED - {:?}", e);
                warn!("[CRYPTO-DEBUG] ========== verify_client_hello FAILED ==========");
                Err(e)
            }
        }
    }

    fn process_handshake(
        &self,
        client_hello: &ClientHello,
        assigned_ip: [u8; 4],
        session_id: [u8; 16],
    ) -> Result<(ServerHello, SessionKey)> {
        debug!("[CRYPTO-DEBUG] ========== process_handshake START ==========");

        debug!(
            "[CRYPTO-DEBUG] Input - assigned_ip: {}.{}.{}.{}",
            assigned_ip[0], assigned_ip[1], assigned_ip[2], assigned_ip[3]
        );
        debug!(
            "[CRYPTO-DEBUG] Input - session_id: {}",
            redacted_len(&session_id)
        );
        debug!(
            "[CRYPTO-DEBUG] Input - client_public_key: {}",
            redacted_len(&client_hello.client_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG] Input - client_ephemeral_key: {}",
            redacted_len(&client_hello.client_ephemeral_key)
        );

        // 1. Generate ephemeral key pair for this session
        debug!("[CRYPTO-DEBUG] Step 1: Generating server ephemeral key pair...");
        let ephemeral = EphemeralKeyPair::generate();
        let server_ephemeral_public = ephemeral.public_key_bytes();
        debug!(
            "[CRYPTO-DEBUG] Server Ephemeral Public Key: {}",
            redacted_len(&server_ephemeral_public)
        );

        // 2. Perform key exchange
        debug!("[CRYPTO-DEBUG] Step 2: Performing X25519 key exchange...");
        let shared_secret = ephemeral.exchange(&client_hello.client_ephemeral_key);
        debug!(
            "[CRYPTO-DEBUG] *** Shared Secret: {} ***",
            redacted_len(&shared_secret)
        );

        // 3. Derive session key
        debug!("[CRYPTO-DEBUG] Step 3: Deriving session key...");
        debug!(
            "[CRYPTO-DEBUG] *** Client Identity Public Key (for KDF): {} ***",
            redacted_len(&client_hello.client_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG] *** Server Identity Public Key (for KDF): {} ***",
            redacted_len(&self.identity.public_key_bytes())
        );

        let session_key = derive_session_key(
            &shared_secret,
            &client_hello.client_public_key,
            &self.identity.public_key_bytes(),
        )?;

        debug!(
            "[CRYPTO-DEBUG] *** Derived Session Key: {} ***",
            redacted_len(session_key.as_bytes())
        );

        // 4. Build ServerHello (unsigned)
        debug!("[CRYPTO-DEBUG] Step 4: Building ServerHello message...");
        let mut server_hello = ServerHello {
            message_type: crate::protocol::MessageType::ServerHello as u8,
            version: client_hello.version,
            server_public_key: self.identity.public_key_bytes(),
            server_ephemeral_key: server_ephemeral_public,
            assigned_ip,
            session_id,
            signature: [0u8; 64],
        };

        debug!("[CRYPTO-DEBUG] ServerHello (before signing):");
        debug!(
            "[CRYPTO-DEBUG]   message_type: {}",
            server_hello.message_type
        );
        debug!("[CRYPTO-DEBUG]   version: {}", server_hello.version);
        debug!(
            "[CRYPTO-DEBUG]   server_public_key: {}",
            redacted_len(&server_hello.server_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG]   server_ephemeral_key: {}",
            redacted_len(&server_hello.server_ephemeral_key)
        );
        debug!(
            "[CRYPTO-DEBUG]   assigned_ip: {}.{}.{}.{}",
            server_hello.assigned_ip[0],
            server_hello.assigned_ip[1],
            server_hello.assigned_ip[2],
            server_hello.assigned_ip[3]
        );
        debug!(
            "[CRYPTO-DEBUG]   session_id: {}",
            redacted_len(&server_hello.session_id)
        );

        // 5. Sign ServerHello
        debug!("[CRYPTO-DEBUG] Step 5: Signing ServerHello...");
        let sign_data =
            Self::server_hello_sign_data(&server_hello, &client_hello.client_public_key);
        debug!(
            "[CRYPTO-DEBUG] ServerHello sign_data: {} bytes",
            sign_data.len()
        );
        debug!(
            "[CRYPTO-DEBUG] ServerHello sign_data redacted: {}",
            redacted_len(&sign_data)
        );

        server_hello.signature = self.identity.sign(&sign_data);
        debug!(
            "[CRYPTO-DEBUG] ServerHello signature: {}",
            redacted_len(&server_hello.signature)
        );

        debug!("[CRYPTO-DEBUG] ========== process_handshake SUCCESS ==========");
        debug!("[CRYPTO-DEBUG] Summary:");
        debug!(
            "[CRYPTO-DEBUG]   Shared Secret: {}",
            redacted_len(&shared_secret)
        );
        debug!(
            "[CRYPTO-DEBUG]   Client Identity Public: {}",
            redacted_len(&client_hello.client_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG]   Server Identity Public: {}",
            redacted_len(&self.identity.public_key_bytes())
        );
        debug!(
            "[CRYPTO-DEBUG]   Derived Session Key: {}",
            redacted_len(session_key.as_bytes())
        );

        Ok((server_hello, session_key))
    }

    fn verify_client_hello_v2(&self, msg: &ClientHello, extension: &[u8]) -> Result<()> {
        if msg.version != PROTOCOL_VERSION_V2 {
            return Err(CoreError::UnsupportedVersion {
                got: msg.version,
                expected: PROTOCOL_VERSION_V2,
            });
        }
        let timestamp = Timestamp::from_secs(msg.timestamp);
        if !timestamp.is_recent(self.max_timestamp_skew) {
            return Err(CoreError::invalid_timestamp(format!(
                "Timestamp {} is not recent (max skew: {}s)",
                msg.timestamp, self.max_timestamp_skew
            )));
        }
        let sign_data = Self::client_hello_sign_data_v2(msg, extension);
        let client_public = IdentityPublicKey::from_bytes(&msg.client_public_key)?;
        client_public.verify(&sign_data, &msg.signature)
    }

    fn process_handshake_v2(
        &self,
        client_hello: &ClientHello,
        extension: &[u8],
        assigned_ip: [u8; 4],
        session_id: [u8; 16],
    ) -> Result<(ServerHello, SessionKeys)> {
        if client_hello.version != PROTOCOL_VERSION_V2 {
            return Err(CoreError::UnsupportedVersion {
                got: client_hello.version,
                expected: PROTOCOL_VERSION_V2,
            });
        }
        let ephemeral = EphemeralKeyPair::generate();
        let server_ephemeral_public = ephemeral.public_key_bytes();
        let mut shared_secret = ephemeral.exchange(&client_hello.client_ephemeral_key);

        let mut server_hello = ServerHello {
            message_type: crate::protocol::MessageType::ServerHello as u8,
            version: PROTOCOL_VERSION_V2,
            server_public_key: self.identity.public_key_bytes(),
            server_ephemeral_key: server_ephemeral_public,
            assigned_ip,
            session_id,
            signature: [0u8; 64],
        };
        let server_sign_data = Self::server_hello_sign_data_v2(&server_hello, client_hello);
        server_hello.signature = self.identity.sign(&server_sign_data);

        let client_sign_data = Self::client_hello_sign_data_v2(client_hello, extension);
        let transcript = transcript_hash_v2(&client_sign_data, &server_sign_data);
        let keys = derive_session_keys_v2(&shared_secret, &transcript);
        shared_secret.zeroize();
        Ok((server_hello, keys?))
    }
}

// ============================================
// Client-side Handshake Helpers
// ============================================

/// Verifies a ServerHello signature from the client's perspective.
///
/// # Arguments
/// * `server_hello` - The ServerHello message to verify
/// * `client_public` - The client's public key (for signature binding)
///
/// # Errors
/// Returns `SignatureVerification` error if signature is invalid.
pub fn verify_server_hello(server_hello: &ServerHello, client_public: &[u8; 32]) -> Result<()> {
    debug!("[CRYPTO-DEBUG] ========== verify_server_hello START ==========");

    debug!("[CRYPTO-DEBUG] ServerHello contents:");
    debug!(
        "[CRYPTO-DEBUG]   message_type: {}",
        server_hello.message_type
    );
    debug!("[CRYPTO-DEBUG]   version: {}", server_hello.version);
    debug!(
        "[CRYPTO-DEBUG]   server_public_key: {}",
        redacted_len(&server_hello.server_public_key)
    );
    debug!(
        "[CRYPTO-DEBUG]   server_ephemeral_key: {}",
        redacted_len(&server_hello.server_ephemeral_key)
    );
    debug!(
        "[CRYPTO-DEBUG]   assigned_ip: {}.{}.{}.{}",
        server_hello.assigned_ip[0],
        server_hello.assigned_ip[1],
        server_hello.assigned_ip[2],
        server_hello.assigned_ip[3]
    );
    debug!(
        "[CRYPTO-DEBUG]   session_id: {}",
        redacted_len(&server_hello.session_id)
    );
    debug!(
        "[CRYPTO-DEBUG]   signature: {}",
        redacted_len(&server_hello.signature)
    );
    debug!(
        "[CRYPTO-DEBUG] Client public key (for binding): {}",
        redacted_len(client_public)
    );

    let sign_data = DefaultHandshakeCrypto::server_hello_sign_data(server_hello, client_public);

    debug!(
        "[CRYPTO-DEBUG] Sign data for verification: {} bytes",
        sign_data.len()
    );
    debug!(
        "[CRYPTO-DEBUG] Sign data redacted: {}",
        redacted_len(&sign_data)
    );

    let server_public = IdentityPublicKey::from_bytes(&server_hello.server_public_key)?;

    match server_public.verify(&sign_data, &server_hello.signature) {
        Ok(()) => {
            debug!("[CRYPTO-DEBUG] ServerHello signature verification: PASSED");
            debug!("[CRYPTO-DEBUG] ========== verify_server_hello SUCCESS ==========");
            Ok(())
        }
        Err(e) => {
            warn!(
                "[CRYPTO-DEBUG] ServerHello signature verification: FAILED - {:?}",
                e
            );
            warn!("[CRYPTO-DEBUG] ========== verify_server_hello FAILED ==========");
            Err(e)
        }
    }
}

/// Creates a signed ClientHello message.
///
/// # Arguments
/// * `identity` - Client's identity key pair
/// * `ephemeral_public` - Client's ephemeral X25519 public key
/// * `version` - Protocol version to use
///
/// # Returns
/// A signed ClientHello ready for transmission.
pub fn create_client_hello(
    identity: &IdentityKeyPair,
    ephemeral_public: [u8; 32],
    version: u8,
) -> ClientHello {
    debug!("[CRYPTO-DEBUG] ========== create_client_hello START ==========");

    let timestamp = Timestamp::now().as_secs();

    debug!(
        "[CRYPTO-DEBUG] Client Identity Public Key: {}",
        redacted_len(&identity.public_key_bytes())
    );
    debug!(
        "[CRYPTO-DEBUG] Client Ephemeral Public Key: {}",
        redacted_len(&ephemeral_public)
    );
    debug!("[CRYPTO-DEBUG] Timestamp: {}", timestamp);
    debug!("[CRYPTO-DEBUG] Version: {}", version);

    let mut msg = ClientHello {
        message_type: crate::protocol::MessageType::ClientHello as u8,
        version,
        client_public_key: identity.public_key_bytes(),
        client_ephemeral_key: ephemeral_public,
        timestamp,
        signature: [0u8; 64],
    };

    // Sign the message
    let sign_data = DefaultHandshakeCrypto::client_hello_sign_data(&msg);
    debug!(
        "[CRYPTO-DEBUG] ClientHello sign_data: {} bytes",
        sign_data.len()
    );
    debug!(
        "[CRYPTO-DEBUG] ClientHello sign_data redacted: {}",
        redacted_len(&sign_data)
    );

    msg.signature = identity.sign(&sign_data);
    debug!(
        "[CRYPTO-DEBUG] ClientHello signature: {}",
        redacted_len(&msg.signature)
    );

    debug!("[CRYPTO-DEBUG] ========== create_client_hello SUCCESS ==========");

    msg
}

// ============================================
// Tests
// ============================================

// ============================================
// Client-side v0x02 helpers (used by tests, smoke clients, and the app core's
// reference implementation)
// ============================================

/// Build and sign a v0x02 ClientHello. `extension` is the exact trailing byte
/// string that will follow the 138-byte hello on the wire (voucher block or
/// empty); it is covered by the signature.
#[must_use]
pub fn create_client_hello_v2(
    identity: &IdentityKeyPair,
    ephemeral_public: [u8; 32],
    extension: &[u8],
) -> ClientHello {
    let mut msg = ClientHello {
        message_type: crate::protocol::MessageType::ClientHello as u8,
        version: PROTOCOL_VERSION_V2,
        client_public_key: identity.public_key_bytes(),
        client_ephemeral_key: ephemeral_public,
        timestamp: Timestamp::now().as_secs(),
        signature: [0u8; 64],
    };
    let sign_data = DefaultHandshakeCrypto::client_hello_sign_data_v2(&msg, extension);
    msg.signature = identity.sign(&sign_data);
    msg
}

/// Verify a v0x02 ServerHello against the ClientHello it must answer.
/// `expected_server_public` pins the node identity when the caller knows it
/// (from the signed node directory); a mismatch fails before the signature
/// is even checked.
pub fn verify_server_hello_v2(
    server_hello: &ServerHello,
    client_hello: &ClientHello,
    expected_server_public: Option<&[u8; 32]>,
) -> Result<()> {
    if server_hello.version != PROTOCOL_VERSION_V2 {
        return Err(CoreError::UnsupportedVersion {
            got: server_hello.version,
            expected: PROTOCOL_VERSION_V2,
        });
    }
    if let Some(expected) = expected_server_public {
        if expected != &server_hello.server_public_key {
            return Err(CoreError::SignatureVerification);
        }
    }
    let sign_data = DefaultHandshakeCrypto::server_hello_sign_data_v2(server_hello, client_hello);
    let server_public = IdentityPublicKey::from_bytes(&server_hello.server_public_key)?;
    server_public.verify(&sign_data, &server_hello.signature)
}

/// The client's side of the v0x02 key schedule.
pub fn derive_client_session_keys_v2(
    shared_secret: &[u8; 32],
    client_hello: &ClientHello,
    extension: &[u8],
    server_hello: &ServerHello,
) -> Result<SessionKeys> {
    let client_sign_data = DefaultHandshakeCrypto::client_hello_sign_data_v2(client_hello, extension);
    let server_sign_data = DefaultHandshakeCrypto::server_hello_sign_data_v2(server_hello, client_hello);
    let transcript = transcript_hash_v2(&client_sign_data, &server_sign_data);
    derive_session_keys_v2(shared_secret, &transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::CURRENT_PROTOCOL_VERSION;

    // ── v0x02 ────────────────────────────────────────────────────────────

    fn v2_pair() -> (DefaultHandshakeCrypto, IdentityKeyPair) {
        (
            DefaultHandshakeCrypto::new(IdentityKeyPair::generate()),
            IdentityKeyPair::generate(),
        )
    }

    #[test]
    fn v2_handshake_agrees_on_two_distinct_direction_keys() {
        let (server, client) = v2_pair();
        let client_eph = EphemeralKeyPair::generate();
        let extension = b"AVCH\x03\x00abc";
        let hello = create_client_hello_v2(&client, client_eph.public_key_bytes(), extension);
        server.verify_client_hello_v2(&hello, extension).expect("client hello verifies");
        let (server_hello, server_keys) = server
            .process_handshake_v2(&hello, extension, [10, 7, 0, 2], [0x33u8; 16])
            .expect("server side");

        verify_server_hello_v2(&server_hello, &hello, Some(&server.public_key().to_bytes()))
            .expect("server hello verifies and pins");
        let shared = client_eph.exchange(&server_hello.server_ephemeral_key);
        let client_keys =
            derive_client_session_keys_v2(&shared, &hello, extension, &server_hello).unwrap();

        assert_eq!(client_keys.c2s, server_keys.c2s, "c2s agrees");
        assert_eq!(client_keys.s2c, server_keys.s2c, "s2c agrees");
        assert_ne!(server_keys.c2s, server_keys.s2c, "the two directions never share a key");
        assert_eq!(server_hello.version, PROTOCOL_VERSION_V2);
    }

    #[test]
    fn v2_voucher_extension_is_covered_by_the_client_signature() {
        let (server, client) = v2_pair();
        let eph = EphemeralKeyPair::generate();
        let hello = create_client_hello_v2(&client, eph.public_key_bytes(), b"AVCH\x01\x00x");
        assert!(server.verify_client_hello_v2(&hello, b"AVCH\x01\x00y").is_err(),
            "a swapped voucher must not verify under the original signature");
        assert!(server.verify_client_hello_v2(&hello, b"").is_err(),
            "a stripped voucher must not verify either");
    }

    #[test]
    fn v2_server_hello_answers_exactly_one_client_hello() {
        let (server, client) = v2_pair();
        let eph = EphemeralKeyPair::generate();
        let hello = create_client_hello_v2(&client, eph.public_key_bytes(), b"");
        let (server_hello, _) = server
            .process_handshake_v2(&hello, b"", [10, 7, 0, 3], [0x44u8; 16])
            .unwrap();
        // The same client, a fresh hello: the old ServerHello no longer verifies.
        let later = create_client_hello_v2(&client, EphemeralKeyPair::generate().public_key_bytes(), b"");
        assert!(verify_server_hello_v2(&server_hello, &later, None).is_err());
        // Pinning: a different expected node key fails before the signature.
        assert!(verify_server_hello_v2(&server_hello, &hello, Some(&[0u8; 32])).is_err());
    }

    #[test]
    fn v2_rejects_a_v1_hello_and_v1_path_rejects_nothing_it_used_to_accept() {
        let (server, client) = v2_pair();
        let eph = EphemeralKeyPair::generate();
        let v1 = create_client_hello(&client, eph.public_key_bytes(), 0x01);
        assert!(server.verify_client_hello_v2(&v1, b"").is_err());
        assert!(server.verify_client_hello(&v1).is_ok(), "v0x01 clients keep working");
    }

    /// Interop vector shared with the app core (rust/src/udp_client.rs tests):
    /// fixed shared secret and fixed signing bodies must yield these keys.
    #[test]
    fn v2_key_schedule_vector() {
        let shared = [0x11u8; 32];
        let client_sign_data: Vec<u8> = (0u8..74).collect();
        let server_sign_data: Vec<u8> = (100u8..218).collect();
        let th = transcript_hash_v2(&client_sign_data, &server_sign_data);
        let keys = derive_session_keys_v2(&shared, &th).unwrap();
        assert_eq!(hex::encode(th), "743d6e151ea92bb3965f9dd98d843730b64a5e4036090b62537fb5f923a4426f");
        assert_eq!(hex::encode(keys.c2s.as_bytes()), "f8569af8dd0f13954c07be5b80a4b09efda38100c358b821c653aaecdb9e3b1a");
        assert_eq!(hex::encode(keys.s2c.as_bytes()), "ee00bd70f541a1e05e04e6960737e4d4d41f522a7ed7bf0e2191a42b6ae047d0");
    }

    #[test]
    fn test_full_handshake() {
        // Server setup
        let server_identity = IdentityKeyPair::generate();
        let server_crypto = DefaultHandshakeCrypto::new(server_identity);

        // Client setup
        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_ephemeral_public = client_ephemeral.public_key_bytes();

        // Client creates ClientHello
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral_public,
            CURRENT_PROTOCOL_VERSION,
        );

        // Server verifies ClientHello
        assert!(server_crypto.verify_client_hello(&client_hello).is_ok());

        // Server processes handshake
        let assigned_ip = [100, 64, 0, 2];
        let session_id = [0x42u8; 16];

        let (server_hello, server_session_key) = server_crypto
            .process_handshake(&client_hello, assigned_ip, session_id)
            .unwrap();

        // Client verifies ServerHello
        assert!(verify_server_hello(&server_hello, &client_identity.public_key_bytes(),).is_ok());

        // Client derives session key
        let shared_secret = client_ephemeral.exchange(&server_hello.server_ephemeral_key);
        let client_session_key = derive_session_key(
            &shared_secret,
            &client_identity.public_key_bytes(),
            &server_hello.server_public_key,
        )
        .unwrap();

        // Both sides should have the same session key
        assert_eq!(server_session_key.as_bytes(), client_session_key.as_bytes());
    }

    #[test]
    fn test_invalid_signature_rejected() {
        let server_identity = IdentityKeyPair::generate();
        let server_crypto = DefaultHandshakeCrypto::new(server_identity);

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();

        let mut client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );

        // Corrupt the signature
        client_hello.signature[0] ^= 0xFF;

        // Verification should fail
        assert!(server_crypto.verify_client_hello(&client_hello).is_err());
    }

    #[test]
    fn test_old_timestamp_rejected() {
        let server_identity = IdentityKeyPair::generate();
        let server_crypto = DefaultHandshakeCrypto::new(server_identity).with_timestamp_skew(30);

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();

        // Create a ClientHello with old timestamp
        let old_timestamp = Timestamp::now().as_secs() - 60; // 60 seconds old

        let mut client_hello = ClientHello {
            message_type: crate::protocol::MessageType::ClientHello as u8,
            version: CURRENT_PROTOCOL_VERSION,
            client_public_key: client_identity.public_key_bytes(),
            client_ephemeral_key: client_ephemeral.public_key_bytes(),
            timestamp: old_timestamp,
            signature: [0u8; 64],
        };

        // Sign with correct key
        let sign_data = DefaultHandshakeCrypto::client_hello_sign_data(&client_hello);
        client_hello.signature = client_identity.sign(&sign_data);

        // Verification should fail due to timestamp
        let result = server_crypto.verify_client_hello(&client_hello);
        assert!(matches!(result, Err(CoreError::InvalidTimestamp { .. })));
    }
}
