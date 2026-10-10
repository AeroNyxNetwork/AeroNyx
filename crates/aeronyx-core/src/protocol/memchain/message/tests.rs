// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/message/tests.rs
// ============================================
//! # Tests: wire message enum
//!
//! Unit tests for per-variant `MemChainMessage` round trips, discriminant stability, and
//! backward-compatible decoding, moved from the former
//! `protocol::memchain::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use super::*;

use crate::crypto::IdentityKeyPair;
#[allow(deprecated)]
use crate::ledger::{
    MemoryLayer, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, BLOCK_TYPE_NORMAL, GENESIS_PREV_HASH,
};
use crate::protocol::chat::ChatContentType;
use crate::protocol::memchain::{
    decode_memchain, encode_memchain, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1, MEMCHAIN_MAGIC,
};

#[test]
fn test_broadcast_fact_roundtrip() {
    let fact = Fact::new(1_700_000_000, "s".into(), "p".into(), "o".into());
    let msg = MemChainMessage::BroadcastFact(fact.clone());
    let encoded = encode_memchain(&msg).expect("encode");
    assert_eq!(encoded[0], MEMCHAIN_MAGIC);
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::BroadcastFact(f) => assert_eq!(f, fact),
        other => panic!("Expected BroadcastFact, got {:?}", other),
    }
}

#[test]
fn test_sync_request_roundtrip() {
    let msg = MemChainMessage::SyncRequest {
        last_known_hash: [0xAB; 32],
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::SyncRequest { last_known_hash } => {
            assert_eq!(last_known_hash, [0xAB; 32]);
        }
        other => panic!("Expected SyncRequest, got {:?}", other),
    }
}

#[test]
fn test_ping_pong_roundtrip() {
    let msg = MemChainMessage::Ping { nonce: 42 };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::Ping { nonce } => assert_eq!(nonce, 42),
        other => panic!("Expected Ping, got {:?}", other),
    }
}

#[test]
fn test_block_announce_roundtrip() {
    let header = BlockHeader {
        height: 42,
        timestamp: 1_700_000_000,
        prev_block_hash: GENESIS_PREV_HASH,
        merkle_root: [0xBB; 32],
        block_type: BLOCK_TYPE_NORMAL,
    };
    let msg = MemChainMessage::BlockAnnounce(header.clone());
    let encoded = encode_memchain(&msg).expect("encode");
    assert!(
        encoded.len() < 200,
        "BlockAnnounce must be <200 bytes, got {}",
        encoded.len()
    );
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::BlockAnnounce(h) => {
            assert_eq!(h.height, 42);
            assert_eq!(h, header);
        }
        other => panic!("Expected BlockAnnounce, got {:?}", other),
    }
}

#[test]
fn test_broadcast_record_roundtrip() {
    let record = MemoryRecord::new(
        [0xAA; 32],
        1_700_000_000,
        MemoryLayer::Episode,
        vec!["test".into(), "memory".into()],
        "aeronyx-memory-v1".into(),
        b"encrypted_content".to_vec(),
        vec![0.1, 0.2, 0.3],
    );
    let msg = MemChainMessage::BroadcastRecord(record.clone());
    let encoded = encode_memchain(&msg).expect("encode");
    assert_eq!(encoded[0], MEMCHAIN_MAGIC);
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::BroadcastRecord(r) => {
            assert_eq!(r.record_id, record.record_id);
            assert_eq!(r.layer, MemoryLayer::Episode);
            assert_eq!(r.source_ai, "aeronyx-memory-v1");
            assert_eq!(r.topic_tags, vec!["test", "memory"]);
        }
        other => panic!("Expected BroadcastRecord, got {:?}", other),
    }
}

#[test]
fn test_sync_record_request_roundtrip() {
    let msg = MemChainMessage::SyncRecordRequest {
        owner: [0xCC; 32],
        after_timestamp: 1_700_000_000,
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::SyncRecordRequest {
            owner,
            after_timestamp,
        } => {
            assert_eq!(owner, [0xCC; 32]);
            assert_eq!(after_timestamp, 1_700_000_000);
        }
        other => panic!("Expected SyncRecordRequest, got {:?}", other),
    }
}

#[test]
fn test_sync_record_response_roundtrip() {
    let record = MemoryRecord::new(
        [0xAA; 32],
        1_700_000_000,
        MemoryLayer::Knowledge,
        vec![],
        "test".into(),
        b"data".to_vec(),
        vec![],
    );
    let msg = MemChainMessage::SyncRecordResponse {
        records: vec![record.clone()],
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::SyncRecordResponse { records } => {
            assert_eq!(records.len(), 1);
            assert_eq!(records[0], record);
        }
        other => panic!("Expected SyncRecordResponse, got {:?}", other),
    }
}

// ── Discriminant stability ───────────────────────────────────────────

/// Verify that all discriminant indices are preserved.
/// Extended in v1.3.0-Sovereign to include WalletPresence = 17.
/// This test catches accidental reordering of variants.
#[test]
fn test_discriminant_stability() {
    fn disc(bytes: &[u8]) -> u32 {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    // BroadcastFact = 0
    let b = bincode::serialize(&MemChainMessage::BroadcastFact(Fact::new(
        0,
        "s".into(),
        "p".into(),
        "o".into(),
    )))
    .unwrap();
    assert_eq!(disc(&b), 0, "BroadcastFact must be discriminant 0");

    // BlockAnnounce = 7
    let b = bincode::serialize(&MemChainMessage::BlockAnnounce(BlockHeader {
        height: 0,
        timestamp: 0,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        block_type: 0x01,
    }))
    .unwrap();
    assert_eq!(disc(&b), 7, "BlockAnnounce must be discriminant 7");

    // BroadcastRecord = 8
    let b = bincode::serialize(&MemChainMessage::BroadcastRecord(MemoryRecord::new(
        [0; 32],
        0,
        MemoryLayer::Episode,
        vec![],
        "".into(),
        vec![],
        vec![],
    )))
    .unwrap();
    assert_eq!(disc(&b), 8, "BroadcastRecord must be discriminant 8");

    // ChatRelay = 11
    let kp = IdentityKeyPair::generate();
    let mut env = crate::protocol::chat::ChatEnvelope {
        message_id: [0x01; 16],
        sender: kp.public_key_bytes(),
        receiver: [0xBB; 32],
        timestamp: 1_700_000_000,
        ciphertext: b"test".to_vec(),
        nonce: [0x02; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    let data = env.sign_data();
    env.signature = kp.sign(&data);
    let b = bincode::serialize(&MemChainMessage::ChatRelay(env)).unwrap();
    assert_eq!(disc(&b), 11, "ChatRelay must be discriminant 11");

    // ChatPull = 12
    let b = bincode::serialize(&MemChainMessage::ChatPull {
        wallet: [0xAA; 32],
        after_timestamp: 0,
        cursor: [0u8; 16],
        limit: 50,
        request_timestamp: 1_700_000_000,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(disc(&b), 12, "ChatPull must be discriminant 12");

    // ChatPullResponse = 13
    let b = bincode::serialize(&MemChainMessage::ChatPullResponse {
        envelopes: vec![],
        has_more: false,
    })
    .unwrap();
    assert_eq!(disc(&b), 13, "ChatPullResponse must be discriminant 13");

    // ChatAck = 14
    let b = bincode::serialize(&MemChainMessage::ChatAck {
        message_ids: vec![[0u8; 16]],
        wallet: [0xAA; 32],
        ack_timestamp: 1_700_000_000,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(disc(&b), 14, "ChatAck must be discriminant 14");

    // ChatExpired = 15
    let b = bincode::serialize(&MemChainMessage::ChatExpired {
        message_ids: vec![[0u8; 16]],
        receiver: [0xCC; 32],
    })
    .unwrap();
    assert_eq!(disc(&b), 15, "ChatExpired must be discriminant 15");

    // DeviceRegister = 16
    let b = bincode::serialize(&MemChainMessage::DeviceRegister {
        device_id: [0x01u8; 16],
        device_name: "test-device".to_string(),
        wallet_pubkey: [0xAA; 32],
        timestamp: 1_700_000_000,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(disc(&b), 16, "DeviceRegister must be discriminant 16");

    // WalletPresence = 17
    let b = bincode::serialize(&MemChainMessage::WalletPresence {
        wallet_pubkey: [0xBB; 32],
        timestamp: 1_700_000_000,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(disc(&b), 17, "WalletPresence must be discriminant 17");

    let identity = IdentityKeyPair::generate();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        1_700_000_001,
        GENESIS_PREV_HASH,
        vec![[0x11; 32]],
        &identity,
    );
    let b = bincode::serialize(&MemChainMessage::RecordBlockAnnounceV1 {
        header: block.header.clone(),
        proposer_signature: block.proposer_signature,
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        18,
        "RecordBlockAnnounceV1 must be discriminant 18"
    );

    let b = bincode::serialize(&MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height: 1,
        limit: 16,
        request_id: [0x22; 16],
        requester: identity.public_key_bytes(),
        request_timestamp: 1_700_000_002,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        19,
        "RecordBlockRangeRequestV1 must be discriminant 19"
    );

    let b = bincode::serialize(&MemChainMessage::RecordBlockRangeResponseV1 {
        request_id: [0x22; 16],
        responder: identity.public_key_bytes(),
        response_timestamp: 1_700_000_003,
        blocks: vec![block],
        has_more: false,
        tip_height: 1,
        tip_hash: [0x33; 32],
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        20,
        "RecordBlockRangeResponseV1 must be discriminant 20"
    );

    let b = bincode::serialize(&MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height: 1,
        known_tip_hash: [0x33; 32],
        request_id: [0x44; 16],
        requester: identity.public_key_bytes(),
        request_timestamp: 1_700_000_004,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        21,
        "RecordChainCheckpointRequestV1 must be discriminant 21"
    );

    let b = bincode::serialize(&MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        request_id: [0x44; 16],
        responder: identity.public_key_bytes(),
        response_timestamp: 1_700_000_005,
        checkpoint_height: 1,
        checkpoint_hash: [0x33; 32],
        tip_height: 1,
        tip_hash: [0x33; 32],
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        22,
        "RecordChainCheckpointResponseV1 must be discriminant 22"
    );

    let b = bincode::serialize(&MemChainMessage::ChatPullV2 {
        wallet: [0x55; 32],
        after_timestamp: 0,
        cursor: vec![],
        limit: 100,
        request_timestamp: 1_700_000_006,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(disc(&b), 23, "ChatPullV2 must be discriminant 23");

    let b = bincode::serialize(&MemChainMessage::ChatPullResponseV2 {
        envelopes: vec![],
        next_cursor: vec![0x77; 57],
        has_more: true,
    })
    .unwrap();
    assert_eq!(disc(&b), 24, "ChatPullResponseV2 must be discriminant 24");

    let b = bincode::serialize(&MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height: 1,
        known_tip_hash: [0x33; 32],
        request_id: [0x88; 16],
        requester: identity.public_key_bytes(),
        request_timestamp: 1_700_000_007,
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        25,
        "RecordCheckpointCertificateRequestV1 must be discriminant 25"
    );

    let member = RecordCheckpointCertificateMemberV1 {
        request_id: [0x44; 16],
        responder: identity.public_key_bytes(),
        response_timestamp: 1_700_000_005,
        checkpoint_height: 1,
        checkpoint_hash: [0x33; 32],
        tip_height: 1,
        tip_hash: [0x33; 32],
        signature: [0u8; 64],
    };
    let b = bincode::serialize(&MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        request_id: [0x88; 16],
        responder: identity.public_key_bytes(),
        response_timestamp: 1_700_000_008,
        checkpoint_height: 1,
        checkpoint_hash: [0x33; 32],
        certificate_digest: [0x99; 32],
        required_signers: 2,
        members: [Some(member), Some(member), None],
        signature: [0u8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        26,
        "RecordCheckpointCertificateResponseV1 must be discriminant 26"
    );

    // [SESSION-TERMINATION 2026-08-15 by Codex] New client control frames
    // append after every deployed node-peer frame; moving this value would
    // silently reinterpret existing bincode traffic.
    let b = bincode::serialize(&MemChainMessage::SessionCloseV1 {
        session_id: [0xA5; 16],
        close_timestamp: 1_700_000_009,
        signature: [0x5A; 64],
    })
    .unwrap();
    assert_eq!(disc(&b), 35, "SessionCloseV1 must be discriminant 35");

    let anchor = CustodyAuditAnchorV1::signed(1, 8, 4_096, [0xC6; 32], &identity)
        .expect("sign custody anchor");
    let b = bincode::serialize(&MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        request_id: [0xC7; 16],
        requester: identity.public_key_bytes(),
        request_timestamp: 1_700_000_010,
        anchor: anchor.clone(),
        signature: [0xC8; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        36,
        "CustodyAuditAnchorWitnessRequestV1 must be discriminant 36"
    );

    let receipt = CustodyAuditWitnessReceiptV1::signed(
        identity.public_key_bytes(),
        1,
        [0xC9; 32],
        1_700_000_011,
        1,
        [0xC9; 32],
        crate::protocol::chat::CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
        &IdentityKeyPair::from_bytes(&[0xCA; 32]).expect("witness identity"),
    )
    .expect("sign custody receipt");
    let b = bincode::serialize(&MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id: [0xC7; 16],
        requester: identity.public_key_bytes(),
        witness: receipt.witness_node_id,
        response_timestamp: receipt.observed_at,
        receipt,
        signature: [0xCB; 64],
    })
    .unwrap();
    assert_eq!(
        disc(&b),
        37,
        "CustodyAuditAnchorWitnessResponseV1 must be discriminant 37"
    );

    // [CHAT-VERIFIED-SUBMIT 2026-08-22 by Codex] New opt-in client frames
    // append after every deployed control frame. Legacy ChatRelay remains
    // index 11 and receives no response unless the client selects v1 here.
    let mut envelope = ChatEnvelope {
        message_id: [0xD1; 16],
        sender: identity.public_key_bytes(),
        receiver: [0xD2; 32],
        timestamp: 1_700_000_012,
        ciphertext: vec![0xD3; 32],
        nonce: [0xD4; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    envelope.signature = identity.sign(&envelope.sign_data());
    let b = bincode::serialize(&MemChainMessage::ChatRelayVerifiedSubmitV1(
        ChatRelayVerifiedSubmitRequestV1 {
            request_id: [0xD5; 16],
            envelope: envelope.clone(),
            request_timestamp: 1_700_000_013,
            signature: [0xD6; 64],
        },
    ))
    .unwrap();
    assert_eq!(
        disc(&b),
        38,
        "ChatRelayVerifiedSubmitV1 must be discriminant 38"
    );

    let b = bincode::serialize(&MemChainMessage::ChatRelayVerifiedSubmitResponseV1(
        ChatRelayVerifiedSubmitResponseV1 {
            request_id: [0xD5; 16],
            message_id: envelope.message_id,
            result: CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1,
            terminal_receipt: None,
        },
    ))
    .unwrap();
    assert_eq!(
        disc(&b),
        39,
        "ChatRelayVerifiedSubmitResponseV1 must be discriminant 39"
    );

    // [ANONYMOUS-MAILBOX-V1 2026-09-02 by Codex] Append-only wire ids;
    // no legacy enum position or ChatEnvelope byte layout changes.
    let b = bincode::serialize(&MemChainMessage::AnonymousMailboxRouteV1(
        AnonymousMailboxRouteRequestV1 {
            version: 1,
            request_id: [0xe1; 16],
            target_node_id: [0xe2; 32],
            sealed_terminal_frame: vec![0xe3; 32],
            requested_at: 1_800_000_000,
            signature: [0xe4; 64],
        },
    ))
    .unwrap();
    assert_eq!(disc(&b), 40, "AnonymousMailboxRouteV1 must be 40");

    let b = bincode::serialize(&MemChainMessage::AnonymousMailboxRouteResponseV1(
        AnonymousMailboxRouteResponseV1 {
            version: 1,
            request_id: [0xe1; 16],
            request_commitment: [0xe5; 32],
            outcome: crate::protocol::anonymous_mailbox::AnonymousMailboxOutcomeV1::Accepted,
            sealed_terminal_response: vec![0xe6; 32],
            responded_at: 1_800_000_001,
            responder_node_id: [0xe2; 32],
            signature: [0xe7; 64],
        },
    ))
    .unwrap();
    assert_eq!(disc(&b), 41, "AnonymousMailboxRouteResponseV1 must be 41");
}

#[test]
fn test_session_close_v1_roundtrip() {
    let request = MemChainMessage::SessionCloseV1 {
        session_id: [0xC1; 16],
        close_timestamp: 1_700_100_100,
        signature: [0x3C; 64],
    };
    let encoded = encode_memchain(&request).expect("encode SessionCloseV1");
    match decode_memchain(&encoded[1..]).expect("decode SessionCloseV1") {
        MemChainMessage::SessionCloseV1 {
            session_id,
            close_timestamp,
            signature,
        } => {
            assert_eq!(session_id, [0xC1; 16]);
            assert_eq!(close_timestamp, 1_700_100_100);
            assert_eq!(signature, [0x3C; 64]);
        }
        other => panic!("Expected SessionCloseV1, got {other:?}"),
    }
}

#[test]
fn test_chat_pull_v2_roundtrip_and_cursor_bound() {
    let request = MemChainMessage::ChatPullV2 {
        wallet: [0x51; 32],
        after_timestamp: 1_700_100_000,
        cursor: vec![0xA5; 57],
        limit: 25,
        request_timestamp: 1_700_100_001,
        signature: [0x7C; 64],
    };
    let encoded = encode_memchain(&request).expect("encode bounded ChatPullV2");
    let decoded = decode_memchain(&encoded[1..]).expect("decode bounded ChatPullV2");
    match decoded {
        MemChainMessage::ChatPullV2 {
            wallet,
            after_timestamp,
            cursor,
            limit,
            request_timestamp,
            signature,
        } => {
            assert_eq!(wallet, [0x51; 32]);
            assert_eq!(after_timestamp, 1_700_100_000);
            assert_eq!(cursor, vec![0xA5; 57]);
            assert_eq!(limit, 25);
            assert_eq!(request_timestamp, 1_700_100_001);
            assert_eq!(signature, [0x7C; 64]);
        }
        other => panic!("Expected ChatPullV2, got {:?}", other),
    }

    let oversized = MemChainMessage::ChatPullV2 {
        wallet: [0x51; 32],
        after_timestamp: 0,
        cursor: vec![0xA5; MAX_CHAT_PULL_CURSOR_V2_BYTES + 1],
        limit: 25,
        request_timestamp: 1_700_100_001,
        signature: [0x7C; 64],
    };
    let encoded = encode_memchain(&oversized).expect("serialization remains infallible");
    assert!(
        decode_memchain(&encoded[1..]).is_err(),
        "oversized ChatPullV2 cursor must be rejected during decode"
    );

    let response = MemChainMessage::ChatPullResponseV2 {
        envelopes: vec![],
        next_cursor: vec![0x42; 57],
        has_more: false,
    };
    let encoded = encode_memchain(&response).expect("encode ChatPullResponseV2");
    match decode_memchain(&encoded[1..]).expect("decode ChatPullResponseV2") {
        MemChainMessage::ChatPullResponseV2 {
            envelopes,
            next_cursor,
            has_more,
        } => {
            assert!(envelopes.is_empty());
            assert_eq!(next_cursor, vec![0x42; 57]);
            assert!(!has_more);
        }
        other => panic!("Expected ChatPullResponseV2, got {:?}", other),
    }
}

// ── Chat Relay variant roundtrip tests ───────────────────────────────

#[test]
fn test_chat_relay_roundtrip() {
    let kp = IdentityKeyPair::generate();
    let mut env = crate::protocol::chat::ChatEnvelope {
        message_id: [0xDE; 16],
        sender: kp.public_key_bytes(),
        receiver: [0xBE; 32],
        timestamp: 1_700_000_001,
        ciphertext: b"hello encrypted world".to_vec(),
        nonce: [0x05; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    let data = env.sign_data();
    env.signature = kp.sign(&data);

    let msg = MemChainMessage::ChatRelay(env.clone());
    let encoded = encode_memchain(&msg).expect("encode");
    assert_eq!(encoded[0], MEMCHAIN_MAGIC);

    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::ChatRelay(e) => {
            assert_eq!(e.message_id, env.message_id);
            assert_eq!(e.sender, env.sender);
            assert_eq!(e.receiver, env.receiver);
            assert_eq!(e.ciphertext, env.ciphertext);
            assert!(
                e.verify_signature().is_ok(),
                "Signature must survive roundtrip"
            );
        }
        other => panic!("Expected ChatRelay, got {:?}", other),
    }
}

#[test]
fn test_chat_pull_roundtrip() {
    let msg = MemChainMessage::ChatPull {
        wallet: [0xAA; 32],
        after_timestamp: 1_700_000_000,
        cursor: [0x01; 16],
        limit: 50,
        request_timestamp: 1_700_000_001,
        signature: [0xBB; 64],
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::ChatPull {
            wallet,
            after_timestamp,
            cursor,
            limit,
            request_timestamp,
            signature,
        } => {
            assert_eq!(wallet, [0xAA; 32]);
            assert_eq!(after_timestamp, 1_700_000_000);
            assert_eq!(cursor, [0x01; 16]);
            assert_eq!(limit, 50);
            assert_eq!(request_timestamp, 1_700_000_001);
            assert_eq!(signature, [0xBB; 64]);
        }
        other => panic!("Expected ChatPull, got {:?}", other),
    }
}

#[test]
fn test_chat_pull_response_roundtrip() {
    let msg = MemChainMessage::ChatPullResponse {
        envelopes: vec![],
        has_more: true,
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::ChatPullResponse {
            envelopes,
            has_more,
        } => {
            assert!(envelopes.is_empty());
            assert!(has_more);
        }
        other => panic!("Expected ChatPullResponse, got {:?}", other),
    }
}

#[test]
fn test_chat_ack_roundtrip() {
    let ids: Vec<[u8; 16]> = vec![[0xAA; 16], [0xBB; 16]];
    let msg = MemChainMessage::ChatAck {
        message_ids: ids.clone(),
        wallet: [0xCC; 32],
        ack_timestamp: 1_700_000_000,
        signature: [0xDD; 64],
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::ChatAck {
            message_ids,
            wallet,
            ack_timestamp,
            signature,
        } => {
            assert_eq!(message_ids, ids);
            assert_eq!(wallet, [0xCC; 32]);
            assert_eq!(ack_timestamp, 1_700_000_000);
            assert_eq!(signature, [0xDD; 64]);
        }
        other => panic!("Expected ChatAck, got {:?}", other),
    }
}

#[test]
fn test_chat_expired_roundtrip() {
    let ids: Vec<[u8; 16]> = vec![[0xCC; 16]];
    let receiver = [0xDD; 32];
    let msg = MemChainMessage::ChatExpired {
        message_ids: ids.clone(),
        receiver,
    };
    let encoded = encode_memchain(&msg).expect("encode");
    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::ChatExpired {
            message_ids,
            receiver: r,
        } => {
            assert_eq!(message_ids, ids);
            assert_eq!(r, receiver);
        }
        other => panic!("Expected ChatExpired, got {:?}", other),
    }
}

// ── v1.3.0-Sovereign: new variant roundtrip tests ────────────────────

#[test]
fn test_device_register_roundtrip_v130() {
    let kp = IdentityKeyPair::generate();
    let msg = MemChainMessage::DeviceRegister {
        device_id: [0xABu8; 16],
        device_name: "iPhone 14 Pro".to_string(),
        wallet_pubkey: kp.public_key_bytes(),
        timestamp: 1_700_000_000,
        signature: kp.sign(b"dummy_sign_data_for_roundtrip"),
    };
    let encoded = encode_memchain(&msg).expect("encode");
    assert_eq!(encoded[0], MEMCHAIN_MAGIC);

    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::DeviceRegister {
            device_id,
            device_name,
            wallet_pubkey,
            timestamp,
            signature,
        } => {
            assert_eq!(device_id, [0xABu8; 16]);
            assert_eq!(device_name, "iPhone 14 Pro");
            assert_eq!(wallet_pubkey, kp.public_key_bytes());
            assert_eq!(timestamp, 1_700_000_000);
            // Signature bytes must survive roundtrip intact
            assert_eq!(signature, kp.sign(b"dummy_sign_data_for_roundtrip"));
        }
        other => panic!("Expected DeviceRegister, got {:?}", other),
    }
}

#[test]
fn test_wallet_presence_roundtrip() {
    let kp = IdentityKeyPair::generate();
    let ts = 1_700_000_042u64;
    let sig = kp.sign(b"dummy_presence_sign_data");

    let msg = MemChainMessage::WalletPresence {
        wallet_pubkey: kp.public_key_bytes(),
        timestamp: ts,
        signature: sig,
    };
    let encoded = encode_memchain(&msg).expect("encode");
    assert_eq!(encoded[0], MEMCHAIN_MAGIC);

    let decoded = decode_memchain(&encoded[1..]).expect("decode");
    match decoded {
        MemChainMessage::WalletPresence {
            wallet_pubkey,
            timestamp,
            signature,
        } => {
            assert_eq!(wallet_pubkey, kp.public_key_bytes());
            assert_eq!(timestamp, ts);
            assert_eq!(signature, sig);
        }
        other => panic!("Expected WalletPresence, got {:?}", other),
    }
}

#[test]
fn test_wallet_presence_size_reasonable() {
    let kp = IdentityKeyPair::generate();
    let msg = MemChainMessage::WalletPresence {
        wallet_pubkey: kp.public_key_bytes(),
        timestamp: 1_700_000_000,
        signature: [0u8; 64],
    };
    let encoded = encode_memchain(&msg).expect("encode");
    // 1 (magic) + 4 (discriminant) + 32 (pubkey) + 8 (ts) + 64 (sig) = 109 bytes
    assert!(
        encoded.len() < 200,
        "WalletPresence must be <200 bytes for UDP friendliness, got {}",
        encoded.len()
    );
}

/// Regression: v1.0.0 messages still decode correctly after v1.3.0 additions.
#[test]
fn test_backward_compat_v100_messages_still_decode() {
    let msg = MemChainMessage::SyncRecordRequest {
        owner: [0xEE; 32],
        after_timestamp: 9_999_999,
    };
    let bytes = bincode::serialize(&msg).expect("serialize");
    let disc = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(disc, 9, "SyncRecordRequest discriminant must remain 9");

    let decoded: MemChainMessage = bincode::deserialize(&bytes).expect("deserialize");
    match decoded {
        MemChainMessage::SyncRecordRequest {
            owner,
            after_timestamp,
        } => {
            assert_eq!(owner, [0xEE; 32]);
            assert_eq!(after_timestamp, 9_999_999);
        }
        other => panic!("Backward compat failed: got {:?}", other),
    }
}
