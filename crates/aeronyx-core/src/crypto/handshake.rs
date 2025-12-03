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
//!
//! ## Last Modified
//! v0.1.0 - Initial handshake crypto implementation

use crate::crypto::kdf::derive_session_key;
use crate::crypto::keys::{
    EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey, SessionKey,
};
use crate::error::{CoreError, Result};
use crate::protocol::{ClientHello, ServerHello};

use aeronyx_common::time::Timestamp;
use tracing::{debug, info, trace};

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
        info!("[CRYPTO-DEBUG] DefaultHandshakeCrypto::new() - Creating handshake crypto instance");
        debug!(
            "[CRYPTO-DEBUG] Server Identity Public Key: {}",
            hex::encode(identity.public_key_bytes())
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
        debug!("[CRYPTO-DEBUG] Setting max_timestamp_skew to {} seconds", seconds);
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
            "[CRYPTO-DEBUG] client_hello_sign_data constructed: {} bytes, hash: {}",
            data.len(),
            hex::encode(&sha256_hash(&data)[..8])
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
            "[CRYPTO-DEBUG] server_hello_sign_data constructed: {} bytes, hash: {}",
            data.len(),
            hex::encode(&sha256_hash(&data)[..8])
        );
        
        data
    }
}

impl HandshakeCrypto for DefaultHandshakeCrypto {
    fn public_key(&self) -> IdentityPublicKey {
        self.identity.public_key()
    }

    fn verify_client_hello(&self, msg: &ClientHello) -> Result<()> {
        info!("[CRYPTO-DEBUG] ========== verify_client_hello START ==========");
        
        debug!("[CRYPTO-DEBUG] ClientHello contents:");
        debug!("[CRYPTO-DEBUG]   message_type: {}", msg.message_type);
        debug!("[CRYPTO-DEBUG]   version: {}", msg.version);
        debug!(
            "[CRYPTO-DEBUG]   client_public_key: {}",
            hex::encode(&msg.client_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG]   client_ephemeral_key: {}",
            hex::encode(&msg.client_ephemeral_key)
        );
        debug!("[CRYPTO-DEBUG]   timestamp: {}", msg.timestamp);
        debug!(
            "[CRYPTO-DEBUG]   signature: {}",
            hex::encode(&msg.signature)
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
            info!("[CRYPTO-DEBUG] FAILED: Timestamp validation failed");
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
            "[CRYPTO-DEBUG] Sign data hex: {}",
            hex::encode(&sign_data)
        );
        
        let client_public = IdentityPublicKey::from_bytes(&msg.client_public_key)?;
        debug!("[CRYPTO-DEBUG] Verifying signature with client public key...");
        
        match client_public.verify(&sign_data, &msg.signature) {
            Ok(()) => {
                info!("[CRYPTO-DEBUG] Signature verification: PASSED");
                info!("[CRYPTO-DEBUG] ========== verify_client_hello SUCCESS ==========");
                Ok(())
            }
            Err(e) => {
                info!("[CRYPTO-DEBUG] Signature verification: FAILED - {:?}", e);
                info!("[CRYPTO-DEBUG] ========== verify_client_hello FAILED ==========");
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
        info!("[CRYPTO-DEBUG] ========== process_handshake START ==========");
        
        debug!(
            "[CRYPTO-DEBUG] Input - assigned_ip: {}.{}.{}.{}",
            assigned_ip[0], assigned_ip[1], assigned_ip[2], assigned_ip[3]
        );
        debug!(
            "[CRYPTO-DEBUG] Input - session_id: {}",
            hex::encode(&session_id)
        );
        debug!(
            "[CRYPTO-DEBUG] Input - client_public_key: {}",
            hex::encode(&client_hello.client_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG] Input - client_ephemeral_key: {}",
            hex::encode(&client_hello.client_ephemeral_key)
        );

        // 1. Generate ephemeral key pair for this session
        info!("[CRYPTO-DEBUG] Step 1: Generating server ephemeral key pair...");
        let ephemeral = EphemeralKeyPair::generate();
        let server_ephemeral_public = ephemeral.public_key_bytes();
        debug!(
            "[CRYPTO-DEBUG] Server Ephemeral Public Key: {}",
            hex::encode(&server_ephemeral_public)
        );

        // 2. Perform key exchange
        info!("[CRYPTO-DEBUG] Step 2: Performing X25519 key exchange...");
        let shared_secret = ephemeral.exchange(&client_hello.client_ephemeral_key);
        info!(
            "[CRYPTO-DEBUG] *** Shared Secret: {} ***",
            hex::encode(&shared_secret)
        );

        // 3. Derive session key
        info!("[CRYPTO-DEBUG] Step 3: Deriving session key...");
        info!(
            "[CRYPTO-DEBUG] *** Client Identity Public Key (for KDF): {} ***",
            hex::encode(&client_hello.client_public_key)
        );
        info!(
            "[CRYPTO-DEBUG] *** Server Identity Public Key (for KDF): {} ***",
            hex::encode(&self.identity.public_key_bytes())
        );
        
        let session_key = derive_session_key(
            &shared_secret,
            &client_hello.client_public_key,
            &self.identity.public_key_bytes(),
        )?;
        
        info!(
            "[CRYPTO-DEBUG] *** Derived Session Key: {} ***",
            hex::encode(session_key.as_bytes())
        );

        // 4. Build ServerHello (unsigned)
        info!("[CRYPTO-DEBUG] Step 4: Building ServerHello message...");
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
        debug!("[CRYPTO-DEBUG]   message_type: {}", server_hello.message_type);
        debug!("[CRYPTO-DEBUG]   version: {}", server_hello.version);
        debug!(
            "[CRYPTO-DEBUG]   server_public_key: {}",
            hex::encode(&server_hello.server_public_key)
        );
        debug!(
            "[CRYPTO-DEBUG]   server_ephemeral_key: {}",
            hex::encode(&server_hello.server_ephemeral_key)
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
            hex::encode(&server_hello.session_id)
        );

        // 5. Sign ServerHello
        info!("[CRYPTO-DEBUG] Step 5: Signing ServerHello...");
        let sign_data = Self::server_hello_sign_data(
            &server_hello,
            &client_hello.client_public_key,
        );
        debug!(
            "[CRYPTO-DEBUG] ServerHello sign_data: {} bytes",
            sign_data.len()
        );
        debug!(
            "[CRYPTO-DEBUG] ServerHello sign_data hex: {}",
            hex::encode(&sign_data)
        );
        
        server_hello.signature = self.identity.sign(&sign_data);
        debug!(
            "[CRYPTO-DEBUG] ServerHello signature: {}",
            hex::encode(&server_hello.signature)
        );

        info!("[CRYPTO-DEBUG] ========== process_handshake SUCCESS ==========");
        info!("[CRYPTO-DEBUG] Summary:");
        info!(
            "[CRYPTO-DEBUG]   Shared Secret: {}",
            hex::encode(&shared_secret)
        );
        info!(
            "[CRYPTO-DEBUG]   Client Identity Public: {}",
            hex::encode(&client_hello.client_public_key)
        );
        info!(
            "[CRYPTO-DEBUG]   Server Identity Public: {}",
            hex::encode(&self.identity.public_key_bytes())
        );
        info!(
            "[CRYPTO-DEBUG]   Derived Session Key: {}",
            hex::encode(session_key.as_bytes())
        );

        Ok((server_hello, session_key))
    }
}

// ============================================
// Helper function for debug logging
// ============================================

/// Compute SHA256 hash for debug logging purposes
fn sha256_hash(data: &[u8]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
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
pub fn verify_server_hello(
    server_hello: &ServerHello,
    client_public: &[u8; 32],
) -> Result<()> {
    info!("[CRYPTO-DEBUG] ========== verify_server_hello START ==========");
    
    debug!("[CRYPTO-DEBUG] ServerHello contents:");
    debug!("[CRYPTO-DEBUG]   message_type: {}", server_hello.message_type);
    debug!("[CRYPTO-DEBUG]   version: {}", server_hello.version);
    debug!(
        "[CRYPTO-DEBUG]   server_public_key: {}",
        hex::encode(&server_hello.server_public_key)
    );
    debug!(
        "[CRYPTO-DEBUG]   server_ephemeral_key: {}",
        hex::encode(&server_hello.server_ephemeral_key)
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
        hex::encode(&server_hello.session_id)
    );
    debug!(
        "[CRYPTO-DEBUG]   signature: {}",
        hex::encode(&server_hello.signature)
    );
    debug!(
        "[CRYPTO-DEBUG] Client public key (for binding): {}",
        hex::encode(client_public)
    );

    let sign_data = DefaultHandshakeCrypto::server_hello_sign_data(
        server_hello,
        client_public,
    );
    
    debug!(
        "[CRYPTO-DEBUG] Sign data for verification: {} bytes",
        sign_data.len()
    );
    debug!(
        "[CRYPTO-DEBUG] Sign data hex: {}",
        hex::encode(&sign_data)
    );
    
    let server_public = IdentityPublicKey::from_bytes(&server_hello.server_public_key)?;
    
    match server_public.verify(&sign_data, &server_hello.signature) {
        Ok(()) => {
            info!("[CRYPTO-DEBUG] ServerHello signature verification: PASSED");
            info!("[CRYPTO-DEBUG] ========== verify_server_hello SUCCESS ==========");
            Ok(())
        }
        Err(e) => {
            info!("[CRYPTO-DEBUG] ServerHello signature verification: FAILED - {:?}", e);
            info!("[CRYPTO-DEBUG] ========== verify_server_hello FAILED ==========");
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
    info!("[CRYPTO-DEBUG] ========== create_client_hello START ==========");
    
    let timestamp = Timestamp::now().as_secs();
    
    debug!(
        "[CRYPTO-DEBUG] Client Identity Public Key: {}",
        hex::encode(&identity.public_key_bytes())
    );
    debug!(
        "[CRYPTO-DEBUG] Client Ephemeral Public Key: {}",
        hex::encode(&ephemeral_public)
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
        "[CRYPTO-DEBUG] ClientHello sign_data hex: {}",
        hex::encode(&sign_data)
    );
    
    msg.signature = identity.sign(&sign_data);
    debug!(
        "[CRYPTO-DEBUG] ClientHello signature: {}",
        hex::encode(&msg.signature)
    );

    info!("[CRYPTO-DEBUG] ========== create_client_hello SUCCESS ==========");
    
    msg
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::CURRENT_PROTOCOL_VERSION;

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
        assert!(verify_server_hello(
            &server_hello,
            &client_identity.public_key_bytes(),
        ).is_ok());

        // Client derives session key
        let shared_secret = client_ephemeral.exchange(&server_hello.server_ephemeral_key);
        let client_session_key = derive_session_key(
            &shared_secret,
            &client_identity.public_key_bytes(),
            &server_hello.server_public_key,
        ).unwrap();

        // Both sides should have the same session key
        assert_eq!(
            server_session_key.as_bytes(),
            client_session_key.as_bytes()
        );
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
        let server_crypto = DefaultHandshakeCrypto::new(server_identity)
            .with_timestamp_skew(30);

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
