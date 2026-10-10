// ============================================
// File: crates/aeronyx-node-tls/tests/golden.rs
// ============================================
//! [NODE-TLS-BINDING 2026-10-10 by Claude] Frozen certificates with their
//! expected verdicts. Every verifier of this format (this crate, the Flutter
//! client's vendored copy, its Dart tests) checks the same file, so a change
//! to the format or the verification order shows up as a vector failure.
//!
//! The file is generated once (P-256 keys and serials are random) and then
//! never rewritten; `AERONYX_NODE_TLS_WRITE_VECTORS=1 cargo test -p
//! aeronyx-node-tls --features generate -- --ignored` regenerates it only on a
//! deliberate format change.

use aeronyx_node_tls::{verify_certificate, NodeTlsError};

const VECTORS: &str = include_str!("vectors/node_tls_v1.json");

fn verdict(result: Result<(), NodeTlsError>) -> &'static str {
    match result {
        Ok(()) => "ok",
        Err(NodeTlsError::Certificate) => "certificate",
        Err(NodeTlsError::MissingBinding) => "missing_binding",
        Err(NodeTlsError::MultipleBindings) => "multiple_bindings",
        Err(NodeTlsError::MalformedBinding) => "malformed_binding",
        Err(NodeTlsError::NodeMismatch) => "node_mismatch",
        Err(NodeTlsError::Expired) => "expired",
        Err(NodeTlsError::LifetimeTooLong) => "lifetime_too_long",
        Err(NodeTlsError::BadSignature) => "bad_signature",
        Err(NodeTlsError::Generation(_)) => "generation",
    }
}

fn hex32(value: &str) -> [u8; 32] {
    hex::decode(value).unwrap().try_into().unwrap()
}

#[test]
fn every_frozen_vector_gets_its_expected_verdict() {
    let vectors: serde_json::Value = serde_json::from_str(VECTORS).unwrap();
    assert_eq!(vectors["format"], "aeronyx-tls-binding:v1");
    let cases = vectors["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let cert = hex::decode(case["cert_der_hex"].as_str().unwrap()).unwrap();
        let node = hex32(case["expected_node_id_hex"].as_str().unwrap());
        let now = case["now"].as_u64().unwrap();
        assert_eq!(
            verdict(verify_certificate(&cert, &node, now)),
            case["verdict"].as_str().unwrap(),
            "{name}"
        );
    }
}

#[cfg(feature = "generate")]
#[test]
#[ignore = "writes the frozen vector file; run only on a deliberate format change"]
fn regenerate_golden_vectors() {
    use aeronyx_node_tls::{binding_uri, generate_certificate};
    use ed25519_dalek::{Signer, SigningKey};

    if std::env::var("AERONYX_NODE_TLS_WRITE_VECTORS").as_deref() != Ok("1") {
        return;
    }
    const NOW: u64 = 1_791_590_400; // 2026-10-10T00:00:00Z
    const LIFETIME: u64 = 7 * 24 * 60 * 60;
    let node = SigningKey::from_bytes(&[0x41; 32]);
    let other = SigningKey::from_bytes(&[0x42; 32]);
    let node_id = node.verifying_key().to_bytes();
    let other_id = other.verifying_key().to_bytes();
    let sign_node = |message: &[u8]| node.sign(message).to_bytes();

    let genuine = generate_certificate(node_id, NOW, LIFETIME, sign_node).unwrap();
    let forged = generate_certificate(node_id, NOW, LIFETIME, |message| {
        other.sign(message).to_bytes()
    })
    .unwrap();
    let long = generate_certificate(node_id, NOW, 31 * 24 * 60 * 60, sign_node).unwrap();

    // The genuine binding URI copied into a certificate for a different key.
    let key_b = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
    let parsed = genuine_binding_signature(&genuine.cert_der);
    let mut params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
    params.subject_alt_names = vec![rcgen::SanType::URI(
        rcgen::string::Ia5String::try_from(binding_uri(&node_id, genuine.not_after, &parsed))
            .unwrap(),
    )];
    let moved = params.self_signed(&key_b).unwrap().der().to_vec();

    let case = |name: &str, cert: &[u8], id: &[u8; 32], now: u64, verdict: &str| {
        serde_json::json!({
            "name": name,
            "cert_der_hex": hex::encode(cert),
            "expected_node_id_hex": hex::encode(id),
            "now": now,
            "verdict": verdict,
        })
    };
    let vectors = serde_json::json!({
        "format": "aeronyx-tls-binding:v1",
        "note": "Generated once by aeronyx-node-tls tests/golden.rs; never edit by hand.",
        "node_identity_seed_hex": hex::encode([0x41u8; 32]),
        "cases": [
            case("valid", &genuine.cert_der, &node_id, NOW + 60, "ok"),
            case("valid_just_before_expiry", &genuine.cert_der, &node_id, genuine.not_after - 1, "ok"),
            case("wrong_node", &genuine.cert_der, &other_id, NOW + 60, "node_mismatch"),
            case("expired", &genuine.cert_der, &node_id, genuine.not_after, "expired"),
            case("forged_signature", &forged.cert_der, &node_id, NOW + 60, "bad_signature"),
            case("binding_moved_to_another_key", &moved, &node_id, NOW + 60, "bad_signature"),
            case("lifetime_too_long_for_verifier_clock", &long.cert_der, &node_id, NOW - 60, "lifetime_too_long"),
            case("not_a_certificate", b"not a certificate", &node_id, NOW, "certificate"),
        ],
    });
    std::fs::write(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/vectors/node_tls_v1.json"
        ),
        serde_json::to_string_pretty(&vectors).unwrap() + "\n",
    )
    .unwrap();
}

#[cfg(feature = "generate")]
fn genuine_binding_signature(cert_der: &[u8]) -> [u8; 64] {
    // The URI is ASCII inside the DER; find it and take the signature field.
    let text = String::from_utf8_lossy(cert_der);
    let start = text.find(aeronyx_node_tls::BINDING_URI_PREFIX).unwrap();
    let uri: String = text[start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == ':' || *c == '-')
        .collect();
    let signature = uri.rsplit(':').next().unwrap();
    hex::decode(&signature[..128]).unwrap().try_into().unwrap()
}
