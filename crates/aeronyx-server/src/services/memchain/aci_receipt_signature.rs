// ============================================
// File: crates/aeronyx-server/src/services/memchain/aci_receipt_signature.rs
// ============================================
//! Strict ACI/1 receipt-signature verification for the pinned profile.
//! Phala evidence JSON is checked for duplicate members before schema parsing.

use k256::ecdsa::{
    signature::hazmat::PrehashVerifier, RecoveryId, Signature as Secp256k1Signature, VerifyingKey,
};
use sha2::Digest;

const ED25519_SIGNATURE_HEX_BYTES: usize = 128;
const ED25519_PUBLIC_KEY_HEX_BYTES: usize = 64;
const SECP256K1_SIGNATURE_HEX_BYTES: usize = 130;

// [PHALA-133171-PROFILE 2026-10-08 by Codex] Identity endorsements use
// non-recoverable r||s; receipts keep their distinct r||s||v contract.
pub(super) fn verify_aci_identity_endorsement(
    algorithm: &str, public_key: &str, signature: &str, payload: &[u8],
) -> bool {
    if algorithm == "ed25519" {
        return verify_aci_receipt_signature(algorithm, signature, public_key, payload);
    }
    if algorithm != "ecdsa-secp256k1" || !is_lower_hex(signature, 128)
        || !matches!(public_key.len(), 66 | 130)
        || !is_lower_hex(public_key, public_key.len())
    { return false; }
    let Ok(key_bytes) = hex::decode(public_key) else { return false; };
    let Ok(key) = VerifyingKey::from_sec1_bytes(&key_bytes) else { return false; };
    let Ok(signature_bytes) = hex::decode(signature) else { return false; };
    let Ok(signature) = Secp256k1Signature::from_slice(&signature_bytes) else { return false; };
    key.verify_prehash(&sha2::Sha256::digest(payload), &signature).is_ok()
}

// [PHALA-133171-PROFILE 2026-10-08 by Codex] The 133171 dstack profile
// proves the secp256k1 IDENTITY key, not a991's receipt-key derivation link.
// The other operational roles still depend on accepted measured code. This
// deliberately fixes the purpose that upstream's generic helper only reads.
pub(super) fn recover_aci_identity_kms_root(
    evidence: &serde_json::Value, identity_public_key: &str, app_id: &[u8],
) -> Option<String> {
    if app_id.is_empty() || app_id.len() > 64
        || !matches!(identity_public_key.len(), 66 | 130)
        || !is_lower_hex(identity_public_key, identity_public_key.len())
    { return None; }
    let custody = evidence.get("key_custody")?;
    if custody.get("provider")?.as_str()? != "dstack-kms" { return None; }
    let keys = custody.get("keys")?.as_array()?;
    if keys.len() > 64 { return None; }
    let mut identities = keys.iter().filter(|key| key.get("role").and_then(|v| v.as_str()) == Some("identity"));
    let identity = identities.next()?;
    if identities.next().is_some()
        || identity.get("purpose")?.as_str()? != "aci.identity.v1"
        || identity.get("public_key")?.as_str()? != identity_public_key
    { return None; }
    let chain = identity.get("signature_chain")?.as_array()?;
    if chain.len() != 2 { return None; }
    let public_key = hex::decode(identity_public_key).ok()?;
    let public_key = VerifyingKey::from_sec1_bytes(&public_key).ok()?;
    let purpose_message = format!("aci.identity.v1:{}", hex::encode(public_key.to_encoded_point(true).as_bytes()));
    let recover = |message: &[u8], raw: &serde_json::Value| -> Option<VerifyingKey> {
        let raw = raw.as_str()?;
        let raw = raw.strip_prefix("0x").unwrap_or(raw);
        if !is_lower_hex(raw, 130) { return None; }
        let raw = hex::decode(raw).ok()?;
        let recovery = match raw[64] {
            v @ 0..=3 => v,
            v @ 27..=30 => v - 27,
            _ => return None,
        };
        let signature = Secp256k1Signature::from_slice(&raw[..64]).ok()?;
        let digest = sha3::Keccak256::digest(message);
        VerifyingKey::recover_from_prehash(&digest, &signature, RecoveryId::from_byte(recovery)?).ok()
    };
    let app_key = recover(purpose_message.as_bytes(), &chain[0])?;
    let mut root_message = b"dstack-kms-issued:".to_vec();
    root_message.extend_from_slice(app_id);
    // Dstack's recovered keys use compressed SEC1, including this chain link.
    root_message.extend_from_slice(app_key.to_encoded_point(true).as_bytes());
    let root = recover(&root_message, &chain[1])?;
    Some(format!("0x{}", hex::encode(root.to_encoded_point(true).as_bytes())))
}

// [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Value deserialization
// discards duplicate members before JCS/signature verification can see them.
// Use serde's JSON parser and recursion limit, retaining only each active
// object's decoded names. This validates input; it never rewrites signed bytes.
pub(crate) fn validate_phala_json(body: &[u8], max_bytes: usize) -> Result<(), &'static str> {
    if body.len() > max_bytes { return Err("phala_json_oversized"); }
    struct UniqueJson;
    impl<'de> serde::Deserialize<'de> for UniqueJson {
        fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
            struct UniqueVisitor;
            impl<'de> serde::de::Visitor<'de> for UniqueVisitor {
                type Value = UniqueJson;
                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("JSON with unique object member names")
                }
                fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<UniqueJson, E> { Ok(UniqueJson) }
                fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<UniqueJson, E> { Ok(UniqueJson) }
                fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<UniqueJson, E> { Ok(UniqueJson) }
                fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<UniqueJson, E> { Ok(UniqueJson) }
                fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<UniqueJson, E> { Ok(UniqueJson) }
                fn visit_unit<E: serde::de::Error>(self) -> Result<UniqueJson, E> { Ok(UniqueJson) }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<UniqueJson, A::Error> {
                    while sequence.next_element::<UniqueJson>()?.is_some() {}
                    Ok(UniqueJson)
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<UniqueJson, A::Error> {
                    let mut names = std::collections::HashSet::new();
                    while let Some(name) = map.next_key::<String>()? {
                        if !names.insert(name) {
                            return Err(serde::de::Error::custom("duplicate JSON member"));
                        }
                        map.next_value::<UniqueJson>()?;
                    }
                    Ok(UniqueJson)
                }
            }
            decoder.deserialize_any(UniqueVisitor)
        }
    }
    serde_json::from_slice::<UniqueJson>(body)
        .map(|_| ()).map_err(|_| "phala_json_malformed")
}

// [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Local restrictive
// profile for Intel tcbInfo/enclaveIdentity ONLY, not ACI receipts/JCS.
// dcap-qvl 0.3.12 verifies the supplied bytes; Intel C++ QVL d12717e3's
// RapidJSON Writer reserializes them. This subset avoids differing escapes
// and number spellings without trying a second signature representation.
pub(crate) fn validate_phala_intel_signed_json(body: &[u8], max_bytes: usize) -> Result<(), &'static str> {
    if body.len() > max_bytes { return Err("peer_collateral_oversized"); }
    validate_phala_json(body, max_bytes).map_err(|_| "peer_collateral_malformed")?;
    if body.iter().copied().find(|byte| !matches!(*byte, b' ' | b'\t' | b'\r' | b'\n')) != Some(b'{') {
        return Err("peer_collateral_malformed");
    }
    const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
    let mut index = 0;
    let mut quoted = false;
    let mut escaped = false;
    while index < body.len() {
        let byte = body[index];
        if quoted {
            if escaped {
                if matches!(byte, b'u' | b'/') {
                    return Err("peer_collateral_unsupported_form");
                }
                escaped = false;
            } else if byte == b'\\' { escaped = true; }
            else if byte == b'"' { quoted = false; }
        } else if byte == b'"' {
            quoted = true;
        } else if byte == b'-' {
            return Err("peer_collateral_unsupported_form");
        } else if byte.is_ascii_digit() {
            // Inspect original digits, never a parsed f64 or rounded integer.
            let mut number = 0u64;
            while index < body.len() && body[index].is_ascii_digit() {
                number = number.checked_mul(10)
                    .and_then(|value| value.checked_add(u64::from(body[index] - b'0')))
                    .filter(|value| *value <= MAX_SAFE_INTEGER)
                    .ok_or("peer_collateral_unsupported_form")?;
                index += 1;
            }
            if index < body.len() && matches!(body[index], b'.' | b'e' | b'E') {
                return Err("peer_collateral_unsupported_form");
            }
            continue;
        }
        index += 1;
    }
    Ok(())
}

// [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Receipt crypto
// shares the appraisal permit pool, and keeps ownership through cancellation.
pub(super) async fn verify_bounded_aci_receipt_signature(
    algorithm: String,
    signature_hex: String,
    public_key_hex: String,
    signing_input: Vec<u8>,
) -> Result<bool, super::llm_provider::LlmError> {
    use super::llm_provider::LlmError;
    if signing_input.len() > 1024 * 1024 {
        return Err(LlmError::AciResponseContractViolation);
    }
    crate::api::discovery::with_phala_appraisal_budget(|permit| async move {
        crate::api::discovery::run_phala_appraisal_crypto(permit, move || {
            verify_aci_receipt_signature(&algorithm, &signature_hex, &public_key_hex, &signing_input)
        }).await
    }).await.map_err(|_| LlmError::AciVerifierUnavailable)?
        .map_err(|_| LlmError::AciVerifierUnavailable)
}

// [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex] ACI/1 defines
// Ed25519 signatures over JCS bytes and recoverable secp256k1 r||s||v over
// SHA-256(JCS bytes). The algorithm and public key come from the attested keyset.
pub(super) fn verify_aci_receipt_signature(
    algorithm: &str,
    signature_hex: &str,
    public_key_hex: &str,
    signing_input: &[u8],
) -> bool {
    match algorithm {
        "ed25519" => {
            if !is_lower_hex(signature_hex, ED25519_SIGNATURE_HEX_BYTES)
                || !is_lower_hex(public_key_hex, ED25519_PUBLIC_KEY_HEX_BYTES)
            {
                return false;
            }
            let Ok(signature_bytes) = hex::decode(signature_hex) else { return false; };
            let Ok(public_key_bytes) = hex::decode(public_key_hex) else { return false; };
            let Ok(public_key_bytes) = <[u8; 32]>::try_from(public_key_bytes.as_slice()) else {
                return false;
            };
            let Ok(signature_bytes) = <[u8; 64]>::try_from(signature_bytes.as_slice()) else {
                return false;
            };
            let Ok(public_key) = ed25519_dalek::PublicKey::from_bytes(&public_key_bytes) else {
                return false;
            };
            let Ok(signature) = ed25519_dalek::Signature::from_bytes(&signature_bytes) else {
                return false;
            };
            public_key.verify_strict(signing_input, &signature).is_ok()
        }
        "ecdsa-secp256k1" => {
            if !is_lower_hex(signature_hex, SECP256K1_SIGNATURE_HEX_BYTES) {
                return false;
            }
            let Ok(signature_bytes) = hex::decode(signature_hex) else { return false; };
            // ACI uses recoverable r||s||v, with v encoded as 0..3 or 27..30.
            let recovery_id = match signature_bytes[64] {
                value @ 0..=3 => value,
                value @ 27..=30 => value - 27,
                _ => return false,
            };
            let Some(recovery_id) = RecoveryId::from_byte(recovery_id) else {
                return false;
            };
            let Ok(signature) = Secp256k1Signature::from_slice(&signature_bytes[..64]) else {
                return false;
            };
            let public_key = public_key_hex.strip_prefix("0x").unwrap_or(public_key_hex);
            if !is_lower_hex(public_key, public_key.len())
                || !matches!(public_key.len(), 66 | 130)
            {
                return false;
            }
            let Ok(public_key_bytes) = hex::decode(public_key) else { return false; };
            let Ok(verifying_key) = VerifyingKey::from_sec1_bytes(&public_key_bytes) else {
                return false;
            };
            let digest = sha2::Sha256::digest(signing_input);
            verifying_key.verify_prehash(&digest, &signature).is_ok()
                && VerifyingKey::recover_from_prehash(&digest, &signature, recovery_id)
                    .is_ok_and(|recovered| recovered == verifying_key)
        }
        _ => false,
    }
}

fn is_lower_hex(value: &str, exact_chars: usize) -> bool {
    value.len() == exact_chars
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::{ecdsa::SigningKey, elliptic_curve::sec1::ToEncodedPoint};

    // [PHALA-133171-PROFILE 2026-10-08 by Codex] Synthetic mechanism
    // fixtures, authored only. They do not establish TDX or production trust.
    #[test]
    fn identity_endorsements_keep_receipt_and_endorsement_encodings_separate() {
        use ed25519_dalek::Signer;
        let payload = b"synthetic canonical keyset endorsement";
        let secret = ed25519_dalek::SecretKey::from_bytes(&[23u8; 32]).unwrap();
        let public = ed25519_dalek::PublicKey::from(&secret);
        let keypair = ed25519_dalek::Keypair { secret, public };
        let ed_public = hex::encode(keypair.public.as_bytes());
        let ed_signature = hex::encode(keypair.sign(payload).to_bytes());
        assert!(verify_aci_identity_endorsement("ed25519", &ed_public, &ed_signature, payload));
        assert!(!verify_aci_identity_endorsement("ed25519", &ed_public, &ed_signature, b"changed"));

        let key = SigningKey::from_slice(&[24u8; 32]).unwrap();
        let public = hex::encode(key.verifying_key().to_encoded_point(false).as_bytes());
        let (signature, recovery) = key.sign_prehash_recoverable(&sha2::Sha256::digest(payload)).unwrap();
        let endorsement = hex::encode(signature.to_bytes());
        assert!(verify_aci_identity_endorsement("ecdsa-secp256k1", &public, &endorsement, payload));
        assert!(!verify_aci_receipt_signature("ecdsa-secp256k1", &endorsement, &public, payload));
        let mut receipt = signature.to_bytes().to_vec();
        receipt.push(recovery.to_byte());
        let receipt = hex::encode(receipt);
        assert!(verify_aci_receipt_signature("ecdsa-secp256k1", &receipt, &public, payload));
        assert!(!verify_aci_identity_endorsement("ecdsa-secp256k1", &public, &receipt, payload));
        assert!(!verify_aci_identity_endorsement("ecdsa-secp256k1", &public, &endorsement, b"changed"));
        assert!(!verify_aci_identity_endorsement("ecdsa", &public, &endorsement, payload));
    }

    #[test]
    fn identity_kms_chain_fixes_purpose_and_requires_unique_identity_and_accepted_root() {
        let identity = SigningKey::from_slice(&[25u8; 32]).unwrap();
        let app = SigningKey::from_slice(&[26u8; 32]).unwrap();
        let root = SigningKey::from_slice(&[27u8; 32]).unwrap();
        let identity_public = hex::encode(identity.verifying_key().to_encoded_point(false).as_bytes());
        let identity_compressed = hex::encode(identity.verifying_key().to_encoded_point(true).as_bytes());
        let expected_root = format!("0x{}", hex::encode(root.verifying_key().to_encoded_point(true).as_bytes()));
        let app_id = [28u8; 20];
        let sign = |key: &SigningKey, message: &[u8]| {
            let (signature, recovery) = key.sign_prehash_recoverable(&sha3::Keccak256::digest(message)).unwrap();
            let mut bytes = signature.to_bytes().to_vec();
            bytes.push(recovery.to_byte());
            hex::encode(bytes)
        };
        let first = sign(&app, format!("aci.identity.v1:{identity_compressed}").as_bytes());
        let mut root_message = b"dstack-kms-issued:".to_vec();
        root_message.extend_from_slice(&app_id);
        root_message.extend_from_slice(app.verifying_key().to_encoded_point(true).as_bytes());
        let evidence = serde_json::json!({"key_custody": {"provider": "dstack-kms", "keys": [{
            "role": "identity", "purpose": "aci.identity.v1", "public_key": identity_public,
            "signature_chain": [first, sign(&root, &root_message)]
        }]}});
        assert_eq!(recover_aci_identity_kms_root(&evidence, &identity_public, &app_id), Some(expected_root.clone()));
        let mut legacy_v = evidence.clone();
        for entry in legacy_v["key_custody"]["keys"][0]["signature_chain"].as_array_mut().unwrap() {
            let mut bytes = hex::decode(entry.as_str().unwrap()).unwrap();
            bytes[64] += 27;
            *entry = serde_json::json!(format!("0x{}", hex::encode(bytes)));
        }
        assert_eq!(recover_aci_identity_kms_root(&legacy_v, &identity_public, &app_id), Some(expected_root.clone()));
        for (pointer, value) in [
            ("/key_custody/provider", serde_json::json!("other")),
            ("/key_custody/keys/0/purpose", serde_json::json!("aci.receipt.v1")),
            ("/key_custody/keys/0/role", serde_json::json!("receipt")),
            ("/key_custody/keys/0/public_key", serde_json::json!(identity_compressed)),
            ("/key_custody/keys/0/signature_chain/0", serde_json::json!("00".repeat(64))),
            ("/key_custody/keys/0/signature_chain/1", serde_json::json!(format!("{}ff", "00".repeat(64)))),
        ] {
            let mut invalid = evidence.clone();
            *invalid.pointer_mut(pointer).unwrap() = value;
            assert_eq!(recover_aci_identity_kms_root(&invalid, &identity_public, &app_id), None, "{pointer}");
        }
        let mut duplicate = evidence.clone();
        let entry = duplicate["key_custody"]["keys"][0].clone();
        duplicate["key_custody"]["keys"].as_array_mut().unwrap().push(entry);
        assert_eq!(recover_aci_identity_kms_root(&duplicate, &identity_public, &app_id), None);
        assert_eq!(recover_aci_identity_kms_root(&evidence, &identity_public, b""), None);
        // Recovery alone is not authorization: changed signed messages may
        // recover another key. The production caller MUST check root policy.
        assert_ne!(recover_aci_identity_kms_root(&evidence, &identity_public, &[29u8; 20]), Some(expected_root));
    }

    // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Authored only;
    // these establish a lexical boundary, not valid Intel signatures/quotes.
    #[test]
    fn intel_signed_profile_is_token_aware_and_scoped() {
        for body in [
            r#"{ "literal":"\\u0008\\/", "short":"\b\f\n\r\t\"\\", "space":" a b " }"#.to_owned(),
            r#"{"nested":[true,null,{"n":9007199254740991,"zero":0}]}"#.to_owned(),
            format!("{{\"utf8\":\"{}{}\"}}", '\u{e9}', '\u{7f}'),
        ] {
            let original = body.clone();
            assert_eq!(validate_phala_intel_signed_json(body.as_bytes(), 4096), Ok(()));
            assert_eq!(body, original);
        }
        for body in [
            r#"{"x":"\u0008"}"#, r#"{"x":"\u0009"}"#, r#"{"x":"\u000A"}"#,
            r#"{"x":"\u000C"}"#, r#"{"x":"\u000D"}"#, r#"{"x":"\u007F"}"#,
            r#"{"x":"\u001F"}"#, r#"{"x":"\u00e9"}"#, r#"{"x":"a\/b"}"#,
            r#"{"\u0078":1}"#, r#"{"a":[{"x":"\u0061"}]}"#,
            r#"{"x":-0}"#, r#"{"x":-1}"#, r#"{"x":1.0}"#, r#"{"x":1e2}"#,
            r#"{"x":1E+2}"#, r#"{"x":9007199254740992}"#,
            r#"{"x":18446744073709551616}"#,
        ] {
            assert_eq!(validate_phala_json(body.as_bytes(), 4096), Ok(()), "general guard stays broad");
            assert_eq!(validate_phala_intel_signed_json(body.as_bytes(), 4096), Err("peer_collateral_unsupported_form"));
        }
        for body in [r#"{"x":01}"#, r#"{"x":1,"x":2}"#, r#"{"x":1,"\u0078":2}"#, "{} {}", "[]", "null"] {
            assert_eq!(validate_phala_intel_signed_json(body.as_bytes(), 4096), Err("peer_collateral_malformed"));
        }
        assert_eq!(validate_phala_intel_signed_json(b"{}", 1), Err("peer_collateral_oversized"));
    }

    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Authored only.
    // The old Value parser accepts these fixtures and silently chooses a value.
    #[test]
    fn evidence_json_rejects_duplicate_decoded_names_at_every_depth() {
        for body in [
            r#"{"receipt_id":"first","receipt_id":"last"}"#,
            r#"{"receipt_id":"first","\u0072eceipt_id":"last"}"#,
            r#"{"workload_keyset":{"e2ee_public_keys":[],"e2ee_public_keys":[{}]}}"#,
            r#"{"events":[{"claims":{"verified":true,"verified":false}}]}"#,
            r#"{"unknown":{"x":1,"x":2}}"#,
        ] {
            assert!(serde_json::from_str::<serde_json::Value>(body).is_ok(), "calibrate old parser");
            assert_eq!(validate_phala_json(body.as_bytes(), 4096), Err("phala_json_malformed"));
        }
    }

    #[test]
    fn evidence_json_keeps_bytes_and_allows_names_in_distinct_objects() {
        let body = br#"{ "a": {"x":1.00e+02}, "b":[{"x":true},{"x":null}], "s":"\u0061\/" }"#;
        let original = body.to_vec();
        assert!(validate_phala_json(body, body.len()).is_ok());
        assert_eq!(body.as_slice(), original.as_slice());
        assert_eq!(validate_phala_json(body, body.len() - 1), Err("phala_json_oversized"));
        for malformed in [b"{} {}".as_slice(), b"{\"a\":}".as_slice(), b"\xff".as_slice(), b"".as_slice()] {
            assert_eq!(validate_phala_json(malformed, 4096), Err("phala_json_malformed"));
        }
        let nested = format!("{}null{}", "[".repeat(256), "]".repeat(256));
        assert_eq!(validate_phala_json(nested.as_bytes(), 4096), Err("phala_json_malformed"));
    }

    // [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Authored,
    // not run: use the real async ownership wrapper with synthetic signatures.
    #[tokio::test]
    async fn bounded_receipt_verifier_preserves_signature_semantics() {
        use ed25519_dalek::Signer;
        let secret = ed25519_dalek::SecretKey::from_bytes(&[17u8; 32]).unwrap();
        let public = ed25519_dalek::PublicKey::from(&secret);
        let keypair = ed25519_dalek::Keypair { secret, public };
        let message = b"synthetic canonical ACI receipt".to_vec();
        let signature = hex::encode(keypair.sign(&message).to_bytes());
        let public = hex::encode(keypair.public.as_bytes());
        assert!(verify_bounded_aci_receipt_signature(
            "ed25519".into(), signature.clone(), public.clone(), message,
        ).await.unwrap());
        assert!(!verify_bounded_aci_receipt_signature(
            "ed25519".into(), signature, public, b"modified".to_vec(),
        ).await.unwrap());
        assert!(matches!(verify_bounded_aci_receipt_signature(
            "ed25519".into(), String::new(), String::new(), vec![0; 1024 * 1024 + 1],
        ).await, Err(super::super::llm_provider::LlmError::AciResponseContractViolation)));
    }

    // [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex]
    // Authored for the ACI/1 signature encodings; not executed in this phase.
    #[test]
    fn unsupported_algorithms_and_malformed_shapes_fail_closed() {
        assert!(!verify_aci_receipt_signature(
            "ecdsa-secp256k1", &"0".repeat(128), &"0".repeat(66), b"receipt",
        ));
        assert!(!verify_aci_receipt_signature(
            "ecdsa-secp256k1", &"0".repeat(130), &"0".repeat(66), b"receipt",
        ));
        assert!(!verify_aci_receipt_signature(
            "ed25519",
            &"0".repeat(126),
            &"0".repeat(64),
            b"receipt",
        ));
        assert!(!verify_aci_receipt_signature(
            "ed25519",
            &"0".repeat(128),
            &"0".repeat(62),
            b"receipt",
        ));
    }

    // [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex]
    // Deterministic test-only key; authored but not executed in this phase.
    #[test]
    fn ed25519_verifies_jcs_bytes_and_rejects_modified_input() {
        use ed25519_dalek::Signer;

        let secret = ed25519_dalek::SecretKey::from_bytes(&[7u8; 32]).unwrap();
        let public = ed25519_dalek::PublicKey::from(&secret);
        let keypair = ed25519_dalek::Keypair { secret, public };
        let signing_input = b"canonical ACI receipt bytes";
        let signature = keypair.sign(signing_input);

        assert!(verify_aci_receipt_signature(
            "ed25519",
            &hex::encode(signature.to_bytes()),
            &hex::encode(keypair.public.as_bytes()),
            signing_input,
        ));
        assert!(!verify_aci_receipt_signature(
            "ed25519",
            &hex::encode(signature.to_bytes()),
            &hex::encode(keypair.public.as_bytes()),
            b"modified receipt bytes",
        ));
    }

    // [MEMCHAIN-PHALA-ACI-RECEIPT-COMPAT 2026-10-06 by Codex]
    // Test-only key material; authored but not executed in this development phase.
    #[test]
    fn secp256k1_recovery_id_must_resolve_to_the_attested_key() {
        let signing_key = SigningKey::from_slice(&[1u8; 32]).unwrap();
        let signing_input = b"synthetic ACI receipt";
        let digest = sha2::Sha256::digest(signing_input);
        let (signature, recovery_id) = signing_key.sign_prehash_recoverable(&digest).unwrap();
        let mut signature_bytes = signature.to_bytes().to_vec();
        signature_bytes.push(recovery_id.to_byte());
        let public_key = hex::encode(signing_key.verifying_key().to_encoded_point(false).as_bytes());

        assert!(verify_aci_receipt_signature(
            "ecdsa-secp256k1",
            &hex::encode(&signature_bytes),
            &public_key,
            signing_input,
        ));

        signature_bytes[64] = 27 + (recovery_id.to_byte() ^ 1);
        assert!(!verify_aci_receipt_signature(
            "ecdsa-secp256k1",
            &hex::encode(signature_bytes),
            &public_key,
            signing_input,
        ));
    }
}
