// ============================================
// File: crates/aeronyx-core/src/crypto/handshake.rs
// ============================================
//! # Handshake Cryptography (Enhanced Debug Version)
//!
//! ## Modification Reason
//! Added extensive debugging to diagnose session key derivation issues
//!
//! ## Main Functionality
//! - `HandshakeCrypto`: Trait for handshake cryptographic operations
//! - `DefaultHandshakeCrypto`: Production implementation
//! - Signature verification for ClientHello messages
//! - Signature creation for ServerHello messages
//! - **ENHANCED**: Detailed handshake debugging
//!
//! ## Last Modified
//! v0.1.1 - Enhanced handshake debugging for troubleshooting

use crate::crypto::kdf::derive_session_key;
use crate::crypto::keys::{
    EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey, SessionKey,
};
use crate::error::{CoreError, Result};
use crate::protocol::{ClientHello, ServerHello};

use aeronyx_common::time::Timestamp;
use tracing::info;

// ============================================
// HandshakeCrypto Trait
// ============================================

/// Trait for handshake cryptographic operations.
pub trait HandshakeCrypto: Send + Sync {
    /// Returns the server's identity public key.
    fn public_key(&self) -> IdentityPublicKey;

    /// Verifies the signature on a ClientHello message.
    fn verify_client_hello(&self, msg: &ClientHello) -> Result<()>;

    /// Processes a ClientHello and produces a ServerHello with session key.
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
    #[must_use]
    pub fn new(identity: IdentityKeyPair) -> Self {
        Self {
            identity,
            max_timestamp_skew: 30,
        }
    }

    /// Sets the maximum allowed timestamp skew.
    #[must_use]
    pub fn with_timestamp_skew(mut self, seconds: u64) -> Self {
        self.max_timestamp_skew = seconds;
        self
    }

    /// Constructs the data to be signed for ClientHello.
    fn client_hello_sign_data(msg: &ClientHello) -> Vec<u8> {
        let mut data = Vec::with_capacity(74);
        data.push(msg.message_type);
        data.push(msg.version);
        data.extend_from_slice(&msg.client_public_key);
        data.extend_from_slice(&msg.client_ephemeral_key);
        data.extend_from_slice(&msg.timestamp.to_le_bytes());
        data
    }

    /// Constructs the data to be signed for ServerHello.
    fn server_hello_sign_data(msg: &ServerHello, client_public: &[u8; 32]) -> Vec<u8> {
        let mut data = Vec::with_capacity(118);
        data.push(msg.message_type);
        data.push(msg.version);
        data.extend_from_slice(&msg.server_public_key);
        data.extend_from_slice(&msg.server_ephemeral_key);
        data.extend_from_slice(&msg.assigned_ip);
        data.extend_from_slice(&msg.session_id);
        data.extend_from_slice(client_public);
        data
    }
}

impl HandshakeCrypto for DefaultHandshakeCrypto {
    fn public_key(&self) -> IdentityPublicKey {
        self.identity.public_key()
    }

    fn verify_client_hello(&self, msg: &ClientHello) -> Result<()> {
        info!("[HANDSHAKE-SERVER] 📨 Verifying ClientHello");
        
        // 1. Validate timestamp
        let timestamp = Timestamp::from_secs(msg.timestamp);
        if !timestamp.is_recent(self.max_timestamp_skew) {
            info!("[HANDSHAKE-SERVER] ❌ Timestamp validation failed");
            return Err(CoreError::invalid_timestamp(format!(
                "Timestamp {} is not recent (max skew: {}s)",
                msg.timestamp, self.max_timestamp_skew
            )));
        }
        info!("[HANDSHAKE-SERVER] ✅ Timestamp valid");

        // 2. Verify signature
        let sign_data = Self::client_hello_sign_data(msg);
        let client_public = IdentityPublicKey::from_bytes(&msg.client_public_key)?;
        client_public.verify(&sign_data, &msg.signature)?;
        
        info!("[HANDSHAKE-SERVER] ✅ ClientHello signature verified");
        Ok(())
    }

    fn process_handshake(
        &self,
        client_hello: &ClientHello,
        assigned_ip: [u8; 4],
        session_id: [u8; 16],
    ) -> Result<(ServerHello, SessionKey)> {
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("[HANDSHAKE-SERVER] 🤝 PROCESSING HANDSHAKE");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 1. Generate ephemeral key pair for this session
        info!("[HANDSHAKE-SERVER] 🔑 Generating server ephemeral key pair...");
        let ephemeral = EphemeralKeyPair::generate();
        let server_ephemeral_public = ephemeral.public_key_bytes();
        info!("[HANDSHAKE-SERVER] ✅ Server ephemeral public: {}", hex::encode(&server_ephemeral_public));

        // 2. Perform key exchange
        info!("[HANDSHAKE-SERVER] 🔄 Performing X25519 key exchange...");
        info!("[HANDSHAKE-SERVER]   Client ephemeral: {}", hex::encode(&client_hello.client_ephemeral_key));
        let shared_secret = ephemeral.exchange(&client_hello.client_ephemeral_key);
        info!("[HANDSHAKE-SERVER] ✅ Shared secret derived: {}", hex::encode(&shared_secret));

        // 3. Derive session key
        info!("[HANDSHAKE-SERVER] 🔐 Deriving session key...");
        info!("[HANDSHAKE-SERVER]   Parameters for KDF:");
        info!("[HANDSHAKE-SERVER]     - Shared secret: {}", hex::encode(&shared_secret));
        info!("[HANDSHAKE-SERVER]     - Client public (arg 2): {}", hex::encode(&client_hello.client_public_key));
        info!("[HANDSHAKE-SERVER]     - Server public (arg 3): {}", hex::encode(&self.identity.public_key_bytes()));
        
        let session_key = derive_session_key(
            &shared_secret,
            &client_hello.client_public_key,
            &self.identity.public_key_bytes(),
        )?;
        
        info!("[HANDSHAKE-SERVER] ✅ Session key derived!");
        info!("[HANDSHAKE-SERVER]   Session Key (hex): {}", hex::encode(session_key.as_bytes()));
        info!("[HANDSHAKE-SERVER]   Session Key (base64): {}", base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD, 
            session_key.as_bytes()
        ));

        // 4. Build ServerHello (unsigned)
        info!("[HANDSHAKE-SERVER] 📝 Building ServerHello message...");
        let mut server_hello = ServerHello {
            message_type: crate::protocol::MessageType::ServerHello as u8,
            version: client_hello.version,
            server_public_key: self.identity.public_key_bytes(),
            server_ephemeral_key: server_ephemeral_public,
            assigned_ip,
            session_id,
            signature: [0u8; 64],
        };

        // 5. Sign ServerHello
        info!("[HANDSHAKE-SERVER] ✍️ Signing ServerHello...");
        let sign_data = Self::server_hello_sign_data(
            &server_hello,
            &client_hello.client_public_key,
        );
        server_hello.signature = self.identity.sign(&sign_data);
        info!("[HANDSHAKE-SERVER] ✅ ServerHello signed");

        info!("[HANDSHAKE-SERVER] ✅ HANDSHAKE COMPLETE");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        Ok((server_hello, session_key))
    }
}

// ============================================
// Client-side Handshake Helpers
// ============================================

/// Verifies a ServerHello signature from the client's perspective.
pub fn verify_server_hello(
    server_hello: &ServerHello,
    client_public: &[u8; 32],
) -> Result<()> {
    info!("[HANDSHAKE-CLIENT] 📨 Verifying ServerHello signature");
    
    let sign_data = DefaultHandshakeCrypto::server_hello_sign_data(
        server_hello,
        client_public,
    );
    
    let server_public = IdentityPublicKey::from_bytes(&server_hello.server_public_key)?;
    server_public.verify(&sign_data, &server_hello.signature)?;
    
    info!("[HANDSHAKE-CLIENT] ✅ ServerHello signature verified");
    Ok(())
}

/// Creates a signed ClientHello message.
pub fn create_client_hello(
    identity: &IdentityKeyPair,
    ephemeral_public: [u8; 32],
    version: u8,
) -> ClientHello {
    info!("[HANDSHAKE-CLIENT] 📝 Creating ClientHello");
    info!("[HANDSHAKE-CLIENT]   Client public: {}", hex::encode(&identity.public_key_bytes()));
    info!("[HANDSHAKE-CLIENT]   Client ephemeral: {}", hex::encode(&ephemeral_public));
    
    let timestamp = Timestamp::now().as_secs();
    
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
    msg.signature = identity.sign(&sign_data);
    
    info!("[HANDSHAKE-CLIENT] ✅ ClientHello created and signed");

    msg
}

/// Client-side session key derivation (with debugging).
pub fn client_derive_session_key(
    ephemeral: &EphemeralKeyPair,
    server_hello: &ServerHello,
    client_identity: &IdentityKeyPair,
) -> Result<SessionKey> {
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("[HANDSHAKE-CLIENT] 🤝 CLIENT SESSION KEY DERIVATION");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Perform key exchange
    info!("[HANDSHAKE-CLIENT] 🔄 Performing X25519 key exchange...");
    info!("[HANDSHAKE-CLIENT]   Server ephemeral: {}", hex::encode(&server_hello.server_ephemeral_key));
    let shared_secret = ephemeral.exchange(&server_hello.server_ephemeral_key);
    info!("[HANDSHAKE-CLIENT] ✅ Shared secret: {}", hex::encode(&shared_secret));

    // Derive session key
    info!("[HANDSHAKE-CLIENT] 🔐 Deriving session key...");
    info!("[HANDSHAKE-CLIENT]   Parameters for KDF:");
    info!("[HANDSHAKE-CLIENT]     - Shared secret: {}", hex::encode(&shared_secret));
    info!("[HANDSHAKE-CLIENT]     - Client public (arg 2): {}", hex::encode(&client_identity.public_key_bytes()));
    info!("[HANDSHAKE-CLIENT]     - Server public (arg 3): {}", hex::encode(&server_hello.server_public_key));

    let session_key = derive_session_key(
        &shared_secret,
        &client_identity.public_key_bytes(),
        &server_hello.server_public_key,
    )?;

    info!("[HANDSHAKE-CLIENT] ✅ Session key derived!");
    info!("[HANDSHAKE-CLIENT]   Session Key (hex): {}", hex::encode(session_key.as_bytes()));
    info!("[HANDSHAKE-CLIENT]   Session Key (base64): {}", base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        session_key.as_bytes()
    ));
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(session_key)
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
        let client_session_key = client_derive_session_key(
            &client_ephemeral,
            &server_hello,
            &client_identity,
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
