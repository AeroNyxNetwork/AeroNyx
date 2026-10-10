// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/peer_control/tests.rs
// ============================================
//! # Tests: node-peer control frames
//!
//! Unit tests for node-peer control-frame round trips and their canonical signing bytes, moved from the former
//! `protocol::memchain::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use super::*;

use crate::crypto::IdentityKeyPair;
use crate::ledger::{AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH};
use crate::protocol::chat::{CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1};
use crate::protocol::memchain::{
    decode_memchain, encode_memchain, MemChainMessage, MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1,
};

#[test]
fn test_record_block_range_messages_roundtrip_and_signatures() {
    let requester = IdentityKeyPair::generate();
    let request_id = [0xA1; 16];
    let request_timestamp = 1_700_100_000;
    let request_bytes = record_block_range_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        1,
        8,
        &request_id,
        &requester.public_key_bytes(),
        request_timestamp,
    );
    let request = MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height: 1,
        limit: 8,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp,
        signature: requester.sign(&request_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode block range request");
    let decoded = decode_memchain(&encoded[1..]).expect("decode block range request");
    match decoded {
        MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id,
            from_height,
            limit,
            request_id,
            requester: requester_key,
            request_timestamp,
            signature,
        } => {
            let signed = record_block_range_request_signing_bytes(
                &chain_id,
                from_height,
                limit,
                &request_id,
                &requester_key,
                request_timestamp,
            );
            requester
                .verify(&signed, &signature)
                .expect("request signature");
        }
        other => panic!("Expected RecordBlockRangeRequestV1, got {other:?}"),
    }

    let responder = IdentityKeyPair::generate();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        request_timestamp,
        GENESIS_PREV_HASH,
        vec![[0x42; 32]],
        &responder,
    );
    let response_timestamp = request_timestamp + 1;
    let response_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder.public_key_bytes(),
        response_timestamp,
        std::slice::from_ref(&block),
        false,
        1,
        &block.hash(),
    );
    let response = MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder: responder.public_key_bytes(),
        response_timestamp,
        blocks: vec![block.clone()],
        has_more: false,
        tip_height: 1,
        tip_hash: block.hash(),
        signature: responder.sign(&response_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode block range response");
    let decoded = decode_memchain(&encoded[1..]).expect("decode block range response");
    match decoded {
        MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder: responder_key,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            let signed = record_block_range_response_signing_bytes(
                &request_id,
                &responder_key,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            responder
                .verify(&signed, &signature)
                .expect("response signature");
            assert_eq!(blocks, vec![block]);
        }
        other => panic!("Expected RecordBlockRangeResponseV1, got {other:?}"),
    }
}

#[test]
fn test_record_chain_checkpoint_messages_roundtrip_and_signatures() {
    let requester = IdentityKeyPair::generate();
    let responder = IdentityKeyPair::generate();
    let request_id = [0xB1; 16];
    let known_tip_hash = [0xB2; 32];
    let request_timestamp = 1_700_200_000;
    let request_bytes = record_chain_checkpoint_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        7,
        &known_tip_hash,
        &request_id,
        &requester.public_key_bytes(),
        request_timestamp,
    );
    let request = MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height: 7,
        known_tip_hash,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp,
        signature: requester.sign(&request_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode checkpoint request");
    let decoded = decode_memchain(&encoded[1..]).expect("decode checkpoint request");
    let MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester: requester_key,
        request_timestamp,
        signature,
    } = decoded
    else {
        panic!("expected checkpoint request");
    };
    let signed = record_chain_checkpoint_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester_key,
        request_timestamp,
    );
    requester
        .verify(&signed, &signature)
        .expect("checkpoint request signature");

    let response_timestamp = request_timestamp + 1;
    let checkpoint_hash = [0xC1; 32];
    let tip_hash = [0xC2; 32];
    let response_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder.public_key_bytes(),
        response_timestamp,
        7,
        &checkpoint_hash,
        9,
        &tip_hash,
    );
    let response = MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder: responder.public_key_bytes(),
        response_timestamp,
        checkpoint_height: 7,
        checkpoint_hash,
        tip_height: 9,
        tip_hash,
        signature: responder.sign(&response_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode checkpoint response");
    let decoded = decode_memchain(&encoded[1..]).expect("decode checkpoint response");
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder: responder_key,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = decoded
    else {
        panic!("expected checkpoint response");
    };
    let signed = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder_key,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    responder
        .verify(&signed, &signature)
        .expect("checkpoint response signature");
}

#[test]
fn test_checkpoint_certificate_messages_roundtrip_and_digest() {
    use sha2::{Digest, Sha256};

    let requester = IdentityKeyPair::generate();
    let serving_node = IdentityKeyPair::generate();
    let witnesses = [IdentityKeyPair::generate(), IdentityKeyPair::generate()];
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let height = 7;
    let hash = [0xD1; 32];
    let request_id = [0xD2; 16];
    let request_timestamp = 1_700_300_000;
    let request_bytes = record_checkpoint_certificate_request_signing_bytes(
        &chain_id,
        height,
        &hash,
        &request_id,
        &requester.public_key_bytes(),
        request_timestamp,
    );
    let request = MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id,
        known_tip_height: height,
        known_tip_hash: hash,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp,
        signature: requester.sign(&request_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode certificate request");
    let decoded = decode_memchain(&encoded[1..]).expect("decode certificate request");
    let MemChainMessage::RecordCheckpointCertificateRequestV1 { signature, .. } = decoded else {
        panic!("expected certificate request");
    };
    requester
        .verify(&request_bytes, &signature)
        .expect("certificate request signature");

    let mut member_slots = [None; MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1];
    let mut digest_members = Vec::new();
    for (index, witness) in witnesses.iter().enumerate() {
        let member_request_id = [0xE0 + index as u8; 16];
        let member_timestamp = request_timestamp.saturating_sub(60 - index as u64);
        let signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &chain_id,
            &member_request_id,
            &witness.public_key_bytes(),
            member_timestamp,
            height,
            &hash,
            height,
            &hash,
        );
        let member = RecordCheckpointCertificateMemberV1 {
            request_id: member_request_id,
            responder: witness.public_key_bytes(),
            response_timestamp: member_timestamp,
            checkpoint_height: height,
            checkpoint_hash: hash,
            tip_height: height,
            tip_hash: hash,
            signature: witness.sign(&signing_bytes),
        };
        let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id,
            request_id: member.request_id,
            responder: member.responder,
            response_timestamp: member.response_timestamp,
            checkpoint_height: member.checkpoint_height,
            checkpoint_hash: member.checkpoint_hash,
            tip_height: member.tip_height,
            tip_hash: member.tip_hash,
            signature: member.signature,
        })
        .expect("encode certificate member");
        digest_members.push((member.responder, Sha256::digest(frame).into()));
        member_slots[index] = Some(member);
    }
    member_slots
        .sort_unstable_by_key(|member| member.map_or([0xFF; 32], |present| present.responder));
    digest_members.sort_unstable_by_key(|member| member.0);
    let certificate_digest =
        record_checkpoint_certificate_digest_v1(&chain_id, height, &hash, 2, &digest_members);
    let response_timestamp = request_timestamp + 1;
    let response_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &serving_node.public_key_bytes(),
        response_timestamp,
        height,
        &hash,
        &certificate_digest,
        2,
        2,
    );
    let response = MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder: serving_node.public_key_bytes(),
        response_timestamp,
        checkpoint_height: height,
        checkpoint_hash: hash,
        certificate_digest,
        required_signers: 2,
        members: member_slots,
        signature: serving_node.sign(&response_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode certificate response");
    let decoded = decode_memchain(&encoded[1..]).expect("decode certificate response");
    let MemChainMessage::RecordCheckpointCertificateResponseV1 {
        members, signature, ..
    } = decoded
    else {
        panic!("expected certificate response");
    };
    assert_eq!(members.iter().flatten().count(), 2);
    serving_node
        .verify(&response_bytes, &signature)
        .expect("certificate response signature");

    let tampered =
        record_checkpoint_certificate_digest_v1(&chain_id, height, &[0xD3; 32], 2, &digest_members);
    assert_ne!(tampered, certificate_digest);
}

#[test]
fn test_coordinator_lease_messages_roundtrip_and_signatures() {
    let coordinator = IdentityKeyPair::generate();
    let witness = IdentityKeyPair::generate();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let instance_id = [0xA1; 32];
    let tip_height = 41;
    let tip_hash = [0xA2; 32];
    let request_id = [0xA3; 16];
    let request_timestamp = 1_700_400_000;
    let requested_ttl_secs = 120;
    let request_bytes = record_coordinator_lease_request_signing_bytes(
        &chain_id,
        &coordinator.public_key_bytes(),
        &instance_id,
        tip_height,
        &tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id,
        coordinator: coordinator.public_key_bytes(),
        instance_id,
        known_tip_height: tip_height,
        known_tip_hash: tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature: coordinator.sign(&request_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode lease request");
    let decoded = decode_memchain(&encoded[1..]).expect("decode lease request");
    let MemChainMessage::RecordCoordinatorLeaseRequestV1 { signature, .. } = decoded else {
        panic!("expected coordinator lease request");
    };
    coordinator
        .verify(&request_bytes, &signature)
        .expect("lease request signature");
    let tampered_request = record_coordinator_lease_request_signing_bytes(
        &chain_id,
        &coordinator.public_key_bytes(),
        &[0xFF; 32],
        tip_height,
        &tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    assert!(coordinator.verify(&tampered_request, &signature).is_err());

    let response_timestamp = request_timestamp + 1;
    let lease_epoch = 3;
    let lease_expires_at = response_timestamp + u64::from(requested_ttl_secs);
    let response_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator.public_key_bytes(),
        &instance_id,
        &witness.public_key_bytes(),
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator: coordinator.public_key_bytes(),
        instance_id,
        witness: witness.public_key_bytes(),
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height: tip_height,
        witness_tip_hash: tip_hash,
        signature: witness.sign(&response_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode lease response");
    let decoded = decode_memchain(&encoded[1..]).expect("decode lease response");
    let MemChainMessage::RecordCoordinatorLeaseResponseV1 { signature, .. } = decoded else {
        panic!("expected coordinator lease response");
    };
    witness
        .verify(&response_bytes, &signature)
        .expect("lease response signature");

    let release_request_id = [0xA4; 16];
    let release_timestamp = response_timestamp + 2;
    let release_request_bytes = record_coordinator_lease_release_request_signing_bytes(
        &chain_id,
        &coordinator.public_key_bytes(),
        &instance_id,
        &release_request_id,
        release_timestamp,
    );
    let release_request = MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id,
        coordinator: coordinator.public_key_bytes(),
        instance_id,
        request_id: release_request_id,
        request_timestamp: release_timestamp,
        signature: coordinator.sign(&release_request_bytes),
    };
    let encoded = encode_memchain(&release_request).expect("encode lease release request");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 29);
    let decoded = decode_memchain(&encoded[1..]).expect("decode lease release request");
    let MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 { signature, .. } = decoded else {
        panic!("expected coordinator lease release request");
    };
    coordinator
        .verify(&release_request_bytes, &signature)
        .expect("lease release request signature");

    let release_response_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &release_request_id,
        &coordinator.public_key_bytes(),
        &instance_id,
        &witness.public_key_bytes(),
        release_timestamp,
        lease_epoch,
    );
    let release_response = MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id: release_request_id,
        coordinator: coordinator.public_key_bytes(),
        instance_id,
        witness: witness.public_key_bytes(),
        released_at: release_timestamp,
        lease_epoch,
        signature: witness.sign(&release_response_bytes),
    };
    let encoded = encode_memchain(&release_response).expect("encode lease release response");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 30);
    let decoded = decode_memchain(&encoded[1..]).expect("decode lease release response");
    let MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 { signature, .. } = decoded else {
        panic!("expected coordinator lease release response");
    };
    witness
        .verify(&release_response_bytes, &signature)
        .expect("lease release response signature");
}

#[test]
fn test_verified_delivery_anchor_witness_messages_roundtrip_and_signatures() {
    let requester = IdentityKeyPair::generate();
    let witness = IdentityKeyPair::generate();
    let generation = 42;
    let anchor_digest = [0xA5; 32];
    let request_id = [0xB6; 16];
    let request_timestamp = 1_700_900_000;
    let request_signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
        &requester.public_key_bytes(),
        generation,
        &anchor_digest,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
        requester: requester.public_key_bytes(),
        generation,
        anchor_digest,
        request_id,
        request_timestamp,
        signature: requester.sign(&request_signing_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode delivery witness request");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 31);
    let decoded = decode_memchain(&encoded[1..]).expect("decode delivery witness request");
    let MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
        requester: requester_key,
        generation: decoded_generation,
        anchor_digest: decoded_digest,
        request_id: decoded_request_id,
        request_timestamp: decoded_timestamp,
        signature,
    } = decoded
    else {
        panic!("expected delivery witness request");
    };
    assert_eq!(decoded_generation, generation);
    assert_eq!(decoded_digest, anchor_digest);
    let decoded_signing_bytes = verified_delivery_anchor_witness_request_signing_bytes(
        &requester_key,
        decoded_generation,
        &decoded_digest,
        &decoded_request_id,
        decoded_timestamp,
    );
    requester
        .verify(&decoded_signing_bytes, &signature)
        .expect("delivery witness request signature");

    let response_timestamp = request_timestamp + 1;
    let response_signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &request_id,
        &requester.public_key_bytes(),
        generation,
        &anchor_digest,
        &witness.public_key_bytes(),
        response_timestamp,
        generation,
        &anchor_digest,
        VERIFIED_DELIVERY_WITNESS_ADVANCED_V1,
    );
    let response = MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id,
        requester: requester.public_key_bytes(),
        requested_generation: generation,
        requested_anchor_digest: anchor_digest,
        witness: witness.public_key_bytes(),
        response_timestamp,
        witness_generation: generation,
        witness_anchor_digest: anchor_digest,
        outcome: VERIFIED_DELIVERY_WITNESS_ADVANCED_V1,
        signature: witness.sign(&response_signing_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode delivery witness response");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 32);
    let decoded = decode_memchain(&encoded[1..]).expect("decode delivery witness response");
    let MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id: decoded_request_id,
        requester: requester_key,
        requested_generation,
        requested_anchor_digest,
        witness: witness_key,
        response_timestamp: decoded_timestamp,
        witness_generation,
        witness_anchor_digest,
        outcome,
        signature,
    } = decoded
    else {
        panic!("expected delivery witness response");
    };
    let decoded_signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &decoded_request_id,
        &requester_key,
        requested_generation,
        &requested_anchor_digest,
        &witness_key,
        decoded_timestamp,
        witness_generation,
        &witness_anchor_digest,
        outcome,
    );
    witness
        .verify(&decoded_signing_bytes, &signature)
        .expect("delivery witness response signature");

    assert_ne!(
        verified_delivery_anchor_witness_request_signing_bytes(
            &requester.public_key_bytes(),
            generation + 1,
            &anchor_digest,
            &request_id,
            request_timestamp,
        ),
        request_signing_bytes,
        "generation must be covered by the request signature"
    );
}

#[test]
fn test_custody_audit_anchor_witness_messages_roundtrip_and_signatures() {
    // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Lock both nested
    // portable signatures and outer request correlation before any HTTP
    // endpoint consumes these append-only wire variants.
    let requester = IdentityKeyPair::from_bytes(&[0xD1; 32]).expect("requester identity");
    let witness = IdentityKeyPair::from_bytes(&[0xD2; 32]).expect("witness identity");
    let anchor = CustodyAuditAnchorV1::signed(7, 65_600, 512 * 1024 * 1024, [0xD3; 32], &requester)
        .expect("sign custody anchor");
    let anchor_sha256 = crate::protocol::chat::custody_audit_anchor_frame_sha256(&anchor)
        .expect("hash custody anchor");
    let request_id = [0xD4; 16];
    let request_timestamp = 1_787_300_000;
    let request_signing_bytes = custody_audit_anchor_witness_request_signing_bytes(
        &request_id,
        &requester.public_key_bytes(),
        request_timestamp,
        &anchor_sha256,
    );
    let request = MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp,
        anchor: anchor.clone(),
        signature: requester.sign(&request_signing_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode custody witness request");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 36);
    let decoded = decode_memchain(&encoded[1..]).expect("decode custody witness request");
    let MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        request_id: decoded_request_id,
        requester: decoded_requester,
        request_timestamp: decoded_timestamp,
        anchor: decoded_anchor,
        signature,
    } = decoded
    else {
        panic!("expected custody witness request");
    };
    assert_eq!(decoded_anchor, anchor);
    requester
        .verify(
            &custody_audit_anchor_witness_request_signing_bytes(
                &decoded_request_id,
                &decoded_requester,
                decoded_timestamp,
                &crate::protocol::chat::custody_audit_anchor_frame_sha256(&decoded_anchor)
                    .expect("hash decoded anchor"),
            ),
            &signature,
        )
        .expect("verify custody witness request");

    let response_timestamp = request_timestamp + 1;
    let receipt = CustodyAuditWitnessReceiptV1::signed(
        requester.public_key_bytes(),
        anchor.checkpoint_generation,
        anchor_sha256,
        response_timestamp,
        anchor.checkpoint_generation,
        anchor_sha256,
        crate::protocol::chat::CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
        &witness,
    )
    .expect("sign custody witness receipt");
    let receipt_sha256 =
        crate::protocol::chat::custody_audit_witness_receipt_frame_sha256(&receipt)
            .expect("hash custody witness receipt");
    let response_signing_bytes = custody_audit_anchor_witness_response_signing_bytes(
        &request_id,
        &requester.public_key_bytes(),
        &witness.public_key_bytes(),
        response_timestamp,
        &receipt_sha256,
    );
    let response = MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id,
        requester: requester.public_key_bytes(),
        witness: witness.public_key_bytes(),
        response_timestamp,
        receipt: receipt.clone(),
        signature: witness.sign(&response_signing_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode custody witness response");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 37);
    let decoded = decode_memchain(&encoded[1..]).expect("decode custody witness response");
    let MemChainMessage::CustodyAuditAnchorWitnessResponseV1 {
        request_id: decoded_request_id,
        requester: decoded_requester,
        witness: decoded_witness,
        response_timestamp: decoded_timestamp,
        receipt: decoded_receipt,
        signature,
    } = decoded
    else {
        panic!("expected custody witness response");
    };
    assert_eq!(decoded_receipt, receipt);
    witness
        .verify(
            &custody_audit_anchor_witness_response_signing_bytes(
                &decoded_request_id,
                &decoded_requester,
                &decoded_witness,
                decoded_timestamp,
                &crate::protocol::chat::custody_audit_witness_receipt_frame_sha256(
                    &decoded_receipt,
                )
                .expect("hash decoded receipt"),
            ),
            &signature,
        )
        .expect("verify custody witness response");
    decoded_receipt
        .verify_accepted_for_anchor(
            &anchor,
            &anchor_sha256,
            &requester.public_key_bytes(),
            &witness.public_key_bytes(),
            anchor.checkpoint_generation,
        )
        .expect("verify portable custody witness receipt");

    assert_ne!(
        custody_audit_anchor_witness_request_signing_bytes(
            &[0xFF; 16],
            &requester.public_key_bytes(),
            request_timestamp,
            &anchor_sha256,
        ),
        request_signing_bytes,
        "request id must be covered by the outer signature"
    );
}

#[test]
fn test_coordinator_handover_exchange_roundtrip_and_signatures() {
    // [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] Appending these
    // variants must preserve every earlier bincode discriminant while the
    // response signature binds the complete dual-signed proof envelope.
    let requester = IdentityKeyPair::generate();
    let responder = IdentityKeyPair::generate();
    let previous = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let proof = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        [0x91; 32],
        [0x92; 16],
        1_701_000_000,
        &previous,
        &next,
    );
    let request_id = [0x93; 16];
    let request_timestamp = 1_701_000_001;
    let request_signing_bytes = record_coordinator_handover_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        0,
        &request_id,
        &requester.public_key_bytes(),
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        after_authority_epoch: 0,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp,
        signature: requester.sign(&request_signing_bytes),
    };
    let encoded = encode_memchain(&request).expect("encode handover request");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 33);
    let decoded = decode_memchain(&encoded[1..]).expect("decode handover request");
    let MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id,
        after_authority_epoch,
        request_id: decoded_request_id,
        requester: decoded_requester,
        request_timestamp: decoded_timestamp,
        signature,
    } = decoded
    else {
        panic!("expected coordinator handover request");
    };
    let decoded_request_signing_bytes = record_coordinator_handover_request_signing_bytes(
        &chain_id,
        after_authority_epoch,
        &decoded_request_id,
        &decoded_requester,
        decoded_timestamp,
    );
    requester
        .verify(&decoded_request_signing_bytes, &signature)
        .expect("handover request signature");

    let response_timestamp = request_timestamp + 1;
    let response_signing_bytes = record_coordinator_handover_response_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &request_id,
        &responder.public_key_bytes(),
        response_timestamp,
        Some(&proof),
        1,
    );
    let response = MemChainMessage::RecordCoordinatorHandoverResponseV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        request_id,
        responder: responder.public_key_bytes(),
        response_timestamp,
        handover: Some(proof.clone()),
        latest_authority_epoch: 1,
        signature: responder.sign(&response_signing_bytes),
    };
    let encoded = encode_memchain(&response).expect("encode handover response");
    assert_eq!(u32::from_le_bytes(encoded[1..5].try_into().unwrap()), 34);
    let decoded = decode_memchain(&encoded[1..]).expect("decode handover response");
    let MemChainMessage::RecordCoordinatorHandoverResponseV1 {
        chain_id,
        request_id: decoded_request_id,
        responder: decoded_responder,
        response_timestamp: decoded_timestamp,
        handover,
        latest_authority_epoch,
        signature,
    } = decoded
    else {
        panic!("expected coordinator handover response");
    };
    assert_eq!(handover, Some(proof.clone()));
    let decoded_response_signing_bytes = record_coordinator_handover_response_signing_bytes(
        &chain_id,
        &decoded_request_id,
        &decoded_responder,
        decoded_timestamp,
        handover.as_ref(),
        latest_authority_epoch,
    );
    responder
        .verify(&decoded_response_signing_bytes, &signature)
        .expect("handover response signature");

    assert_ne!(
        record_coordinator_handover_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder.public_key_bytes(),
            response_timestamp,
            None,
            1,
        ),
        response_signing_bytes,
        "proof presence must be covered by the response signature"
    );
}
