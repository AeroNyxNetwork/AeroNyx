// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn memchain_startup_integrity_accepts_valid_sighted_and_blind_records() {
    let sighted = MemoryRecord::new(
        [0x11; 32],
        1_700_000_000,
        MemoryLayer::Knowledge,
        vec!["compatibility".into()],
        "legacy".into(),
        b"node-visible-content".to_vec(),
        vec![0.1, 0.2],
    );
    assert_eq!(memchain_index_rejection_reason(&sighted), None);

    let owner = IdentityKeyPair::generate();
    let mut blind = MemoryRecord::new(
        owner.public_key_bytes(),
        1_700_000_001,
        MemoryLayer::Episode,
        vec!["sealed".into()],
        "client".into(),
        b"opaque-client-ciphertext".to_vec(),
        vec![0.3, 0.4],
    );
    blind.blind = true;
    blind.signature = owner.sign(&blind.record_id);
    assert_eq!(memchain_index_rejection_reason(&blind), None);
}

#[test]
fn memchain_startup_integrity_rejects_tampering_and_bad_blind_signature() {
    let owner = IdentityKeyPair::generate();
    let mut blind = MemoryRecord::new(
        owner.public_key_bytes(),
        1_700_000_002,
        MemoryLayer::Episode,
        vec![],
        "client".into(),
        b"opaque-client-ciphertext".to_vec(),
        vec![0.5, 0.6],
    );
    blind.blind = true;
    assert_eq!(
        memchain_index_rejection_reason(&blind),
        Some("owner_signature_invalid")
    );

    blind.signature = owner.sign(&blind.record_id);
    blind.encrypted_content.push(0xFF);
    assert_eq!(
        memchain_index_rejection_reason(&blind),
        Some("record_id_mismatch")
    );
}
