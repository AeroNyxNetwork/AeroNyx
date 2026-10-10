// ============================================
// File: crates/aeronyx-node-tls/src/lib.rs
// ============================================
//! # Identity-bound TLS for AeroNyx node endpoints
//!
//! [NODE-TLS-BINDING 2026-10-10 by Claude] A node proves its HTTPS endpoint
//! with its own Ed25519 identity instead of a certificate authority or a
//! domain name.
//!
//! The node generates an ordinary TLS key (ECDSA P-256, supported by rustls and
//! BoringSSL) and a self-signed certificate. One Subject Alternative Name URI
//! carries the binding:
//!
//! ```text
//! aeronyx-tls-binding:v1:<node_id hex>:<not_after unix secs>:<signature hex>
//! ```
//!
//! where the signature is the node identity's Ed25519 signature over
//! [`binding_message`]: a domain-separated digest of the certificate's own
//! SubjectPublicKeyInfo, the node id and the expiry.
//!
//! A party that already knows which node it wants to reach (from a signed
//! descriptor) accepts the certificate only when [`verify_certificate`]
//! succeeds for that node id. The TLS handshake itself proves possession of the
//! certificate's key; the binding proves the node identity endorsed that key.
//! CA chains, host names and IP SANs play no part.
//!
//! The identity key never signs TLS handshakes: it signs exactly one binding
//! message per certificate, which no other AeroNyx signature can be confused
//! with (its own prefix, and it is not a 32-byte digest).

#![forbid(unsafe_code)]

use der::{Decode, Encode};
use ed25519_dalek::{Signature, VerifyingKey};
#[cfg(feature = "generate")]
use rcgen::PublicKeyData;
use sha2::{Digest, Sha256};
use x509_cert::ext::pkix::name::GeneralName;
use x509_cert::ext::pkix::SubjectAltName;
use x509_cert::Certificate;

/// URI scheme and version prefix of the binding SAN entry.
pub const BINDING_URI_PREFIX: &str = "aeronyx-tls-binding:v1:";

/// Domain separation for the identity signature.
const BINDING_SIGNING_DOMAIN: &[u8] = b"AeroNyx-Node-TLS-Binding-v1\0";

/// Longest accepted remaining lifetime of a binding. A node rotates its TLS
/// key well before this; a verifier rejects bindings claiming more, so a
/// leaked TLS key cannot stay endorsed for long.
pub const MAX_BINDING_LIFETIME_SECS: u64 = 31 * 24 * 60 * 60;

/// Why a certificate is not accepted for a node.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NodeTlsError {
    /// The certificate is not valid DER X.509.
    #[error("certificate is not valid X.509")]
    Certificate,
    /// No binding SAN entry.
    #[error("certificate carries no AeroNyx node binding")]
    MissingBinding,
    /// More than one binding SAN entry.
    #[error("certificate carries more than one AeroNyx node binding")]
    MultipleBindings,
    /// The binding entry is malformed.
    #[error("AeroNyx node binding is malformed")]
    MalformedBinding,
    /// The binding names a different node.
    #[error("certificate is bound to a different node")]
    NodeMismatch,
    /// The binding has expired.
    #[error("AeroNyx node binding has expired")]
    Expired,
    /// The binding claims a longer lifetime than any node issues.
    #[error("AeroNyx node binding lifetime exceeds the limit")]
    LifetimeTooLong,
    /// The identity signature does not verify over this certificate's key.
    #[error("AeroNyx node binding signature is invalid")]
    BadSignature,
    /// Certificate generation failed (node only).
    #[error("certificate generation failed: {0}")]
    Generation(String),
}

/// The exact bytes the node identity signs for one certificate key.
#[must_use]
pub fn binding_message(spki_der: &[u8], node_id: &[u8; 32], not_after: u64) -> Vec<u8> {
    let mut message = Vec::with_capacity(BINDING_SIGNING_DOMAIN.len() + 32 + 32 + 8);
    message.extend_from_slice(BINDING_SIGNING_DOMAIN);
    message.extend_from_slice(&Sha256::digest(spki_der));
    message.extend_from_slice(node_id);
    message.extend_from_slice(&not_after.to_le_bytes());
    message
}

/// The SAN URI that carries a binding.
#[must_use]
pub fn binding_uri(node_id: &[u8; 32], not_after: u64, signature: &[u8; 64]) -> String {
    format!(
        "{BINDING_URI_PREFIX}{}:{not_after}:{}",
        hex::encode(node_id),
        hex::encode(signature)
    )
}

struct Binding {
    node_id: [u8; 32],
    not_after: u64,
    signature: [u8; 64],
}

fn parse_binding_uri(uri: &str) -> Result<Binding, NodeTlsError> {
    let rest = uri
        .strip_prefix(BINDING_URI_PREFIX)
        .ok_or(NodeTlsError::MalformedBinding)?;
    let mut fields = rest.split(':');
    let (Some(node_id), Some(not_after), Some(signature), None) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return Err(NodeTlsError::MalformedBinding);
    };
    let lower_hex = |value: &str| {
        value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    if node_id.len() != 64
        || signature.len() != 128
        || !lower_hex(node_id)
        || !lower_hex(signature)
        || not_after.is_empty()
        || !not_after.bytes().all(|b| b.is_ascii_digit())
        || (not_after.len() > 1 && not_after.starts_with('0'))
    {
        return Err(NodeTlsError::MalformedBinding);
    }
    let mut node = [0u8; 32];
    let mut sig = [0u8; 64];
    hex::decode_to_slice(node_id, &mut node).map_err(|_| NodeTlsError::MalformedBinding)?;
    hex::decode_to_slice(signature, &mut sig).map_err(|_| NodeTlsError::MalformedBinding)?;
    Ok(Binding {
        node_id: node,
        not_after: not_after
            .parse()
            .map_err(|_| NodeTlsError::MalformedBinding)?,
        signature: sig,
    })
}

/// Accepts `cert_der` as the TLS certificate of `expected_node_id` at `now`.
///
/// Checks: valid X.509; exactly one binding SAN URI; it names
/// `expected_node_id`; it has not expired and claims no more than
/// [`MAX_BINDING_LIFETIME_SECS`]; and the node's Ed25519 signature (strict
/// verification) covers this certificate's own SubjectPublicKeyInfo.
///
/// The caller's TLS stack must separately verify the handshake signature with
/// the certificate's key (rustls and BoringSSL do this as part of TLS 1.3).
///
/// # Errors
/// Returns the first [`NodeTlsError`] that applies.
pub fn verify_certificate(
    cert_der: &[u8],
    expected_node_id: &[u8; 32],
    now: u64,
) -> Result<(), NodeTlsError> {
    let certificate = Certificate::from_der(cert_der).map_err(|_| NodeTlsError::Certificate)?;
    let tbs = certificate.tbs_certificate();
    let spki_der = tbs
        .subject_public_key_info()
        .to_der()
        .map_err(|_| NodeTlsError::Certificate)?;

    let mut binding = None;
    for extension in tbs.extensions().into_iter().flatten() {
        if extension.extn_id != <SubjectAltName as der::oid::AssociatedOid>::OID {
            continue;
        }
        let names = SubjectAltName::from_der(extension.extn_value.as_bytes())
            .map_err(|_| NodeTlsError::Certificate)?;
        for name in names.0 {
            if let GeneralName::UniformResourceIdentifier(uri) = name {
                if uri.as_str().starts_with(BINDING_URI_PREFIX) {
                    if binding.is_some() {
                        return Err(NodeTlsError::MultipleBindings);
                    }
                    binding = Some(parse_binding_uri(uri.as_str())?);
                }
            }
        }
    }
    let binding = binding.ok_or(NodeTlsError::MissingBinding)?;
    if &binding.node_id != expected_node_id {
        return Err(NodeTlsError::NodeMismatch);
    }
    if binding.not_after <= now {
        return Err(NodeTlsError::Expired);
    }
    if binding.not_after - now > MAX_BINDING_LIFETIME_SECS {
        return Err(NodeTlsError::LifetimeTooLong);
    }
    let key = VerifyingKey::from_bytes(expected_node_id).map_err(|_| NodeTlsError::BadSignature)?;
    let signature = Signature::from_bytes(&binding.signature);
    key.verify_strict(
        &binding_message(&spki_der, expected_node_id, binding.not_after),
        &signature,
    )
    .map_err(|_| NodeTlsError::BadSignature)
}

/// A freshly generated node TLS certificate and its private key.
#[cfg(feature = "generate")]
#[derive(Clone)]
pub struct GeneratedCertificate {
    /// DER X.509 certificate.
    pub cert_der: Vec<u8>,
    /// PKCS#8 DER private key (ECDSA P-256).
    pub key_pkcs8_der: Vec<u8>,
    /// Expiry of the binding (Unix seconds).
    pub not_after: u64,
}

/// Generates a TLS key and a self-signed certificate bound to `node_id`.
///
/// `sign` must return the node identity's Ed25519 signature over the given
/// message; it is called exactly once, with [`binding_message`] for the new key.
///
/// # Errors
/// Returns [`NodeTlsError::Generation`] when rcgen cannot build the certificate.
#[cfg(feature = "generate")]
pub fn generate_certificate(
    node_id: [u8; 32],
    now: u64,
    lifetime_secs: u64,
    sign: impl FnOnce(&[u8]) -> [u8; 64],
) -> Result<GeneratedCertificate, NodeTlsError> {
    let generation = |error: rcgen::Error| NodeTlsError::Generation(error.to_string());
    if lifetime_secs == 0 || lifetime_secs > MAX_BINDING_LIFETIME_SECS {
        return Err(NodeTlsError::Generation(
            "lifetime out of range".to_string(),
        ));
    }
    let not_after = now
        .checked_add(lifetime_secs)
        .ok_or_else(|| NodeTlsError::Generation("expiry overflows".to_string()))?;
    let key_pair =
        rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(generation)?;
    let signature = sign(&binding_message(
        &key_pair.subject_public_key_info(),
        &node_id,
        not_after,
    ));
    let cert_der = certificate_with_binding_uri(
        &key_pair,
        &binding_uri(&node_id, not_after, &signature),
        now,
        not_after,
    )?;
    Ok(GeneratedCertificate {
        cert_der,
        key_pkcs8_der: key_pair.serialize_der(),
        not_after,
    })
}

#[cfg(feature = "generate")]
fn certificate_with_binding_uri(
    key_pair: &rcgen::KeyPair,
    uri: &str,
    now: u64,
    not_after: u64,
) -> Result<Vec<u8>, NodeTlsError> {
    let generation = |error: rcgen::Error| NodeTlsError::Generation(error.to_string());
    // 9999-12-31T23:59:59Z, the last instant X.509 GeneralizedTime can hold.
    const LAST_X509_SECOND: u64 = 253_402_300_799;
    let to_time = |secs: u64| {
        if secs > LAST_X509_SECOND {
            return Err(NodeTlsError::Generation("time out of range".to_string()));
        }
        Ok(rcgen::date_time_ymd(1970, 1, 1) + std::time::Duration::from_secs(secs))
    };
    let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).map_err(generation)?;
    params.subject_alt_names = vec![rcgen::SanType::URI(
        rcgen::string::Ia5String::try_from(uri.to_string()).map_err(generation)?,
    )];
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "aeronyx-node");
    // A little slack for verifier clock skew; the binding expiry is what counts.
    params.not_before = to_time(now.saturating_sub(300))?;
    params.not_after = to_time(not_after)?;
    let certificate = params.self_signed(key_pair).map_err(generation)?;
    Ok(certificate.der().to_vec())
}

#[cfg(all(test, feature = "generate"))]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const NOW: u64 = 1_790_000_000;

    fn identity(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn generate(identity: &SigningKey, lifetime: u64) -> GeneratedCertificate {
        generate_certificate(
            identity.verifying_key().to_bytes(),
            NOW,
            lifetime,
            |message| identity.sign(message).to_bytes(),
        )
        .unwrap()
    }

    #[test]
    fn a_generated_certificate_verifies_for_its_node() {
        let node = identity(0x11);
        let cert = generate(&node, 7 * 24 * 3600);
        assert_eq!(
            verify_certificate(&cert.cert_der, &node.verifying_key().to_bytes(), NOW + 60),
            Ok(())
        );
        assert!(!cert.key_pkcs8_der.is_empty());
    }

    #[test]
    fn another_node_id_is_rejected() {
        let node = identity(0x12);
        let other = identity(0x13);
        let cert = generate(&node, 3600);
        assert_eq!(
            verify_certificate(&cert.cert_der, &other.verifying_key().to_bytes(), NOW),
            Err(NodeTlsError::NodeMismatch)
        );
    }

    #[test]
    fn expiry_and_lifetime_bounds_are_enforced() {
        let node = identity(0x14);
        let id = node.verifying_key().to_bytes();
        let cert = generate(&node, 3600);
        assert_eq!(
            verify_certificate(&cert.cert_der, &id, NOW + 3600),
            Err(NodeTlsError::Expired)
        );
        // A binding claiming more than the limit (as seen from an earlier
        // verifier clock) is refused.
        let long = generate(&node, MAX_BINDING_LIFETIME_SECS);
        assert_eq!(
            verify_certificate(&long.cert_der, &id, NOW - 10),
            Err(NodeTlsError::LifetimeTooLong)
        );
        assert!(
            generate_certificate(id, NOW, MAX_BINDING_LIFETIME_SECS + 1, |m| {
                node.sign(m).to_bytes()
            })
            .is_err()
        );
    }

    #[test]
    fn a_binding_signed_by_someone_else_is_rejected() {
        let node = identity(0x15);
        let impostor = identity(0x16);
        let cert = generate_certificate(node.verifying_key().to_bytes(), NOW, 3600, |message| {
            impostor.sign(message).to_bytes()
        })
        .unwrap();
        assert_eq!(
            verify_certificate(&cert.cert_der, &node.verifying_key().to_bytes(), NOW),
            Err(NodeTlsError::BadSignature)
        );
    }

    #[test]
    fn a_valid_binding_moved_onto_another_key_is_rejected() {
        // The node signs a binding for key A; an attacker puts that exact URI
        // into a certificate for its own key B.
        let node = identity(0x17);
        let id = node.verifying_key().to_bytes();
        let key_a = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let key_b = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let not_after = NOW + 3600;
        let signature = node
            .sign(&binding_message(
                &key_a.subject_public_key_info(),
                &id,
                not_after,
            ))
            .to_bytes();
        let uri = binding_uri(&id, not_after, &signature);
        let genuine = certificate_with_binding_uri(&key_a, &uri, NOW, not_after).unwrap();
        let stolen = certificate_with_binding_uri(&key_b, &uri, NOW, not_after).unwrap();
        assert_eq!(verify_certificate(&genuine, &id, NOW), Ok(()));
        assert_eq!(
            verify_certificate(&stolen, &id, NOW),
            Err(NodeTlsError::BadSignature)
        );
    }

    #[test]
    fn missing_duplicate_and_malformed_bindings_are_rejected() {
        let node = identity(0x18);
        let id = node.verifying_key().to_bytes();
        let key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let plain = rcgen::CertificateParams::new(vec!["example.com".to_string()])
            .unwrap()
            .self_signed(&key)
            .unwrap();
        assert_eq!(
            verify_certificate(plain.der(), &id, NOW),
            Err(NodeTlsError::MissingBinding)
        );

        let not_after = NOW + 3600;
        let signature = node
            .sign(&binding_message(
                &key.subject_public_key_info(),
                &id,
                not_after,
            ))
            .to_bytes();
        let uri = binding_uri(&id, not_after, &signature);
        let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
        params.subject_alt_names = vec![
            rcgen::SanType::URI(rcgen::string::Ia5String::try_from(uri.clone()).unwrap()),
            rcgen::SanType::URI(rcgen::string::Ia5String::try_from(uri.clone()).unwrap()),
        ];
        let twice = params.self_signed(&key).unwrap();
        assert_eq!(
            verify_certificate(twice.der(), &id, NOW),
            Err(NodeTlsError::MultipleBindings)
        );

        for broken in [
            uri.replace(BINDING_URI_PREFIX, &format!("{BINDING_URI_PREFIX}x")),
            uri.to_uppercase()
                .replace("AERONYX-TLS-BINDING:V1:", BINDING_URI_PREFIX),
            format!("{uri}:extra"),
            uri.replace(&format!(":{not_after}:"), &format!(":0{not_after}:")),
        ] {
            let cert = certificate_with_binding_uri(&key, &broken, NOW, not_after).unwrap();
            assert_eq!(
                verify_certificate(&cert, &id, NOW),
                Err(NodeTlsError::MalformedBinding),
                "{broken}"
            );
        }
        assert_eq!(
            verify_certificate(b"not a certificate", &id, NOW),
            Err(NodeTlsError::Certificate)
        );
    }
}
