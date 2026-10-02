// File: crates/aeronyx-server/src/server/session_ingress.rs
// Purpose: Session MemChain dispatch, expiry notifications and encrypted writes.
// Dependencies: parent server services and the existing core/transport codecs.
// Flow: authenticate -> read custody -> coalesce V1 prefix -> encrypted write.
// Boundary: V1 coalescing never ACKs/deletes records or changes V2 cursors.
// [ARCH-SPLIT 2026-10-02] Private items remain pub(super) for parent composition.
// [CHAT-V1-COALESCING 2026-10-03 by Codex] Bound multi-envelope coalescing,
// preserving the existing single-envelope send path above the target.
// [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Budget V2 before cursor creation;
// retain single-item compatibility and observe local write failure without ACK.
// Last Modified: 2026-10-03.
use super::*;

impl Server {
    // [CHAT-V1-COALESCING 2026-10-03 by Codex] Full UDP payload target,
    // NOT a hard admission ceiling, socket capacity, or Internet PMTU promise.
    pub(super) const LEGACY_CHAT_PULL_COALESCING_TARGET: usize = 1200;

    // [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] Full UDP coalescing target,
    // not an admission limit or PMTU assertion; oversized first items stay whole.
    pub(super) const CHAT_PULL_V2_COALESCING_TARGET: usize = 1200;

    // Keep an ordered, whole-envelope prefix. A first envelope above the target
    // is sent alone for single-message compatibility, even if the old transport
    // may fail to carry it. No repository, cursor or ACK mutation happens here.
    pub(super) fn coalesce_legacy_chat_pull(
        mut envelopes: Vec<ChatEnvelope>,
        has_more: bool,
    ) -> std::result::Result<MemChainMessage, &'static str> {
        use aeronyx_core::protocol::messages::DATA_PACKET_HEADER_SIZE;
        // Frozen fixed-int V1 layout: magic + enum + Vec length + bool.
        let overhead = DATA_PACKET_HEADER_SIZE
            .checked_add(ENCRYPTION_OVERHEAD)
            .ok_or("size_overflow")?;
        let mut expected_bytes = overhead.checked_add(1 + 4 + 8 + 1).ok_or("size_overflow")?;
        let mut selected = 0;
        for envelope in &envelopes {
            // id16 + sender32 + receiver32 + timestamp8 + Vec length8 +
            // nonce24 + content enum4 + raw signature64 = 188 fixed bytes.
            let item_bytes = 188usize
                .checked_add(envelope.ciphertext.len())
                .ok_or("size_overflow")?;
            let next_bytes = expected_bytes
                .checked_add(item_bytes)
                .ok_or("size_overflow")?;
            if selected != 0 && next_bytes > Self::LEGACY_CHAT_PULL_COALESCING_TARGET {
                break;
            }
            expected_bytes = next_bytes;
            selected += 1;
            if expected_bytes > Self::LEGACY_CHAT_PULL_COALESCING_TARGET {
                break;
            }
        }
        let has_more = has_more || selected < envelopes.len();
        envelopes.truncate(selected);
        let response = MemChainMessage::ChatPullResponse {
            envelopes,
            has_more,
        };
        // The actual bounded production encoder is the final authority. Fail
        // closed if its layout drifts from the checked size calculation.
        let actual_bytes = encode_memchain(&response)
            .map_err(|_| "encode_failed")?
            .len()
            .checked_add(overhead)
            .ok_or("size_overflow")?;
        if actual_bytes != expected_bytes
            || (actual_bytes > Self::LEGACY_CHAT_PULL_COALESCING_TARGET && selected != 1)
        {
            return Err("encoded_size_mismatch");
        }
        Ok(response)
    }

    // ============================================
    // MemChain Message Handler
    // ============================================

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_memchain_message(
        msg: MemChainMessage,
        mempool: Option<&Arc<MemPool>>,
        aof_writer: Option<&Arc<TokioMutex<AofWriter>>>,
        storage: &Option<Arc<MemoryStorage>>,
        vector_index: &Option<Arc<VectorIndex>>,
        config: &MemChainConfig,
        server_pubkey_hex: &str,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
        sessions: &Arc<SessionManager>,
        chat_relay: &Option<Arc<ChatRelayService>>,
        peer_store: &Arc<PeerStore>,
        self_node_id: &[u8; 32],
        node_identity: &IdentityKeyPair,
        chat_peer_client: Option<&reqwest::Client>,
    ) {
        let storage_access = match MemChainStorageRequirement::for_message(&msg)
            .authorize(mempool, aof_writer, storage)
        {
            Ok(access) => access,
            Err(error) => {
                // [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex]
                // Missing optional persistence is routine in chat-only mode;
                // retain only fixed aggregate buckets without per-frame warning
                // amplification. Internal access mismatches remain warnings/errors.
                debug!(
                    reason = error.reason_bucket(),
                    family = error.family_bucket(),
                    "[MEMCHAIN] Storage-owned message rejected"
                );
                return;
            }
        };
        match msg {
            MemChainMessage::BroadcastFact(fact) => {
                let MemChainStorageAccess::FactAof {
                    mempool,
                    aof_writer,
                } = storage_access
                else {
                    error!(
                        reason = "dispatch_gate_invariant",
                        "[MEMCHAIN] Message rejected"
                    );
                    return;
                };
                let origin_hex = hex::encode(fact.origin);
                let sig_ok = match IdentityPublicKey::from_bytes(&fact.origin) {
                    Ok(pk) => pk.verify(&fact.fact_id, &fact.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok {
                    warn!("[MEMCHAIN] BroadcastFact sig failed");
                    return;
                }
                if !config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
                    warn!("[MEMCHAIN] BroadcastFact untrusted origin");
                    return;
                }
                if mempool.add_fact(fact.clone()) {
                    let mut w = aof_writer.lock().await;
                    let _ = w.append_fact(&fact).await;
                }
            }
            MemChainMessage::BroadcastRecord(record) => {
                let MemChainStorageAccess::RecordStore { storage } = storage_access else {
                    error!(
                        reason = "dispatch_gate_invariant",
                        "[MEMCHAIN] Message rejected"
                    );
                    return;
                };
                let owner_hex = record.owner_hex();
                let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                    Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                    Err(_) => false,
                };
                if !sig_ok {
                    warn!(owner = %owner_hex, "[MEMCHAIN] BroadcastRecord sig failed");
                    return;
                }
                if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                    warn!(owner = %owner_hex, "[MEMCHAIN] BroadcastRecord untrusted");
                    return;
                }
                if !record.verify_id() {
                    warn!(owner = %owner_hex, id = hex::encode(record.record_id), "[MEMCHAIN] record_id hash mismatch");
                    return;
                }
                if storage.insert(&record, "p2p-remote").await {
                    info!(
                        id = hex::encode(record.record_id),
                        "[MEMCHAIN] BroadcastRecord stored"
                    );
                    if record.has_embedding() {
                        if let Some(ref vi) = vector_index {
                            vi.upsert(
                                record.record_id,
                                record.embedding.clone(),
                                record.layer,
                                record.timestamp,
                                &record.owner,
                                "p2p-remote",
                            );
                        }
                    }
                }
            }
            MemChainMessage::SyncRequest { last_known_hash } => {
                let MemChainStorageAccess::FactAof { mempool, .. } = storage_access else {
                    error!(
                        reason = "dispatch_gate_invariant",
                        "[MEMCHAIN] Message rejected"
                    );
                    return;
                };
                let facts = mempool.get_facts_after(last_known_hash);
                let resp = MemChainMessage::SyncResponse { facts };
                Self::send_to_session(&resp, session, udp, crypto).await;
            }
            MemChainMessage::SyncResponse { facts } => {
                let MemChainStorageAccess::FactAof {
                    mempool,
                    aof_writer,
                } = storage_access
                else {
                    error!(
                        reason = "dispatch_gate_invariant",
                        "[MEMCHAIN] Message rejected"
                    );
                    return;
                };
                for fact in facts {
                    let origin_hex = hex::encode(fact.origin);
                    let sig_ok = match IdentityPublicKey::from_bytes(&fact.origin) {
                        Ok(pk) => pk.verify(&fact.fact_id, &fact.signature).is_ok(),
                        Err(_) => false,
                    };
                    if !sig_ok || !config.is_origin_trusted(&origin_hex, server_pubkey_hex) {
                        continue;
                    }
                    if mempool.add_fact(fact.clone()) {
                        let mut w = aof_writer.lock().await;
                        let _ = w.append_fact(&fact).await;
                    }
                }
            }
            MemChainMessage::SyncRecordRequest {
                owner,
                after_timestamp,
            } => {
                let MemChainStorageAccess::RecordStore { storage } = storage_access else {
                    error!(
                        reason = "dispatch_gate_invariant",
                        "[MEMCHAIN] Message rejected"
                    );
                    return;
                };
                let records = storage.query_by_owner_after(&owner, after_timestamp).await;
                let resp = MemChainMessage::SyncRecordResponse { records };
                Self::send_to_session(&resp, session, udp, crypto).await;
            }
            MemChainMessage::SyncRecordResponse { records } => {
                let MemChainStorageAccess::RecordStore { storage } = storage_access else {
                    error!(
                        reason = "dispatch_gate_invariant",
                        "[MEMCHAIN] Message rejected"
                    );
                    return;
                };
                for record in records {
                    let owner_hex = record.owner_hex();
                    let sig_ok = match IdentityPublicKey::from_bytes(&record.owner) {
                        Ok(pk) => pk.verify(&record.record_id, &record.signature).is_ok(),
                        Err(_) => false,
                    };
                    if !sig_ok {
                        warn!(owner = %owner_hex, "[MEMCHAIN] SyncRecordResponse sig failed");
                        continue;
                    }
                    if !config.is_origin_trusted(&owner_hex, server_pubkey_hex) {
                        continue;
                    }
                    if !record.verify_id() {
                        warn!(owner = %owner_hex, id = hex::encode(record.record_id), "[MEMCHAIN] SyncRecordResponse hash mismatch");
                        continue;
                    }
                    let _ = storage.insert(&record, "p2p-sync").await;
                }
            }
            MemChainMessage::BlockAnnounce(header) => {
                info!(
                    height = header.height,
                    hash = hex::encode(header.hash()),
                    "[MEMCHAIN] BlockAnnounce received"
                );
            }
            MemChainMessage::RecordBlockAnnounceV1 {
                header,
                proposer_signature,
            } => {
                let now = unix_now_secs();
                let session_key = session.client_public_key.to_bytes();
                let signature_valid = IdentityPublicKey::from_bytes(&header.proposer)
                    .and_then(|key| key.verify(&header.hash(), &proposer_signature))
                    .is_ok();
                let known_peer = peer_store.get_valid(&header.proposer, now).is_some();
                if !signature_valid || !known_peer || session_key != header.proposer {
                    warn!(
                        signature_valid,
                        known_peer,
                        session_binding_valid = session_key == header.proposer,
                        "[MEMCHAIN_BLOCK] Rejected announcement outside authenticated node peer boundary"
                    );
                    return;
                }
                info!(
                    height = header.height,
                    hash = %header.hash_hex(),
                    "[MEMCHAIN_BLOCK] Authenticated peer tip announcement received"
                );
            }
            MemChainMessage::RecordBlockRangeRequestV1 { .. }
            | MemChainMessage::RecordBlockRangeResponseV1 { .. }
            | MemChainMessage::RecordChainCheckpointRequestV1 { .. }
            | MemChainMessage::RecordChainCheckpointResponseV1 { .. } => {
                // Ledger sync and checkpoint proofs are intentionally
                // unavailable on the VPN/client DataPacket path. They belong
                // to the signed node-to-node peer API so ordinary clients
                // cannot enumerate commitments or probe peer chain tips.
                warn!(
                    "[MEMCHAIN_BLOCK] Rejected ledger sync on client tunnel; node peer API required"
                );
            }
            MemChainMessage::ChatRelay(envelope) => {
                if envelope.verify_signature().is_err() {
                    warn!(
                        reason = "invalid_signature",
                        "[CHAT_RELAY] Envelope dropped"
                    );
                    return;
                }
                let authenticated_sender = session.client_public_key.to_bytes();
                if !envelope.sender_matches_authenticated_identity(&authenticated_sender) {
                    warn!(
                        reason = "session_sender_mismatch",
                        "[CHAT_RELAY] Envelope dropped"
                    );
                    return;
                }
                let Some(ref relay) = chat_relay else {
                    warn!(
                        reason = "relay_unavailable",
                        "[CHAT_RELAY] Envelope dropped"
                    );
                    return;
                };
                if relay.is_online_duplicate(&envelope.message_id) {
                    debug!(reason = "duplicate", "[CHAT_RELAY] Envelope dropped");
                    return;
                }
                relay.wallet_routes.announce(
                    &authenticated_sender,
                    session.id.clone(),
                    session.client_endpoint,
                );
                let receiver = envelope.receiver;
                let target_routes = relay.wallet_routes.lookup(&receiver);

                if !target_routes.is_empty() {
                    let mut all_failed = true;
                    let device_count = target_routes.len();
                    for (target_sid, _endpoint) in &target_routes {
                        if let Some(target_session) = sessions.get(target_sid) {
                            if Self::send_to_session(
                                &MemChainMessage::ChatRelay(envelope.clone()),
                                &target_session,
                                udp,
                                crypto,
                            )
                            .await
                            {
                                all_failed = false;
                            } else {
                                warn!(
                                    reason = "transport_write_failed",
                                    "[CHAT_RELAY] Online delivery failed"
                                );
                            }
                        } else {
                            relay.wallet_routes.remove_session(target_sid);
                            debug!("[CHAT_RELAY] Pruned stale route during delivery");
                        }
                    }
                    if all_failed {
                        let onion_outcome = Self::relay_authenticated_chat_over_onion_paths(
                            chat_peer_client,
                            Some(relay.as_ref()),
                            peer_store,
                            node_identity,
                            self_node_id,
                            &envelope,
                            None,
                        )
                        .await;
                        if !onion_outcome.delivered()
                            && onion_outcome.compatibility_direct_fallback_allowed()
                        {
                            Self::relay_chat_envelope_to_discovered_peers(
                                chat_peer_client,
                                Some(relay.as_ref()),
                                peer_store,
                                node_identity,
                                &envelope,
                            )
                            .await;
                        }
                        if let Err(e) = relay.store_pending(&envelope) {
                            warn!(
                                reason = e.reason_bucket(),
                                "[CHAT_RELAY] Fallback store failed"
                            );
                        } else {
                            debug!("[CHAT_RELAY] All routes stale; stored for offline delivery");
                        }
                    } else {
                        debug!(
                            devices = device_count,
                            "[CHAT_RELAY] Online delivery complete"
                        );
                    }
                } else {
                    let onion_outcome = Self::relay_authenticated_chat_over_onion_paths(
                        chat_peer_client,
                        Some(relay.as_ref()),
                        peer_store,
                        node_identity,
                        self_node_id,
                        &envelope,
                        None,
                    )
                    .await;
                    if !onion_outcome.delivered()
                        && onion_outcome.compatibility_direct_fallback_allowed()
                    {
                        Self::relay_chat_envelope_to_discovered_peers(
                            chat_peer_client,
                            Some(relay.as_ref()),
                            peer_store,
                            node_identity,
                            &envelope,
                        )
                        .await;
                    }
                    if let Err(e) = relay.store_pending(&envelope) {
                        warn!(
                            reason = e.reason_bucket(),
                            "[CHAT_RELAY] Pending store failed"
                        );
                    } else {
                        debug!("[CHAT_RELAY] Stored for offline delivery");
                    }
                }
            }
            MemChainMessage::ChatRelayVerifiedSubmitV1(request) => {
                let response = Self::handle_verified_chat_submit(
                    request,
                    session,
                    chat_relay,
                    peer_store.as_ref(),
                    self_node_id,
                    node_identity,
                    chat_peer_client,
                )
                .await;
                if !Self::send_to_session(
                    &MemChainMessage::ChatRelayVerifiedSubmitResponseV1(response),
                    session,
                    udp,
                    crypto,
                )
                .await
                {
                    warn!(
                        reason = "verified_submit_response_write_failed",
                        "[CHAT_RELAY] Verified submit response failed"
                    );
                }
            }
            MemChainMessage::ChatRelayVerifiedSubmitResponseV1(_) => {
                // Server-to-client only. Accepting it from a client would let
                // arbitrary sessions manufacture local delivery UI state.
                warn!(
                    reason = "client_sent_server_response",
                    "[CHAT_RELAY] Verified submit response rejected"
                );
            }
            MemChainMessage::ChatPull {
                wallet,
                after_timestamp,
                cursor,
                limit,
                request_timestamp,
                signature,
            } => {
                let Some(ref relay) = chat_relay else {
                    return;
                };
                let at_bytes = after_timestamp.to_le_bytes();
                let limit_bytes = limit.to_le_bytes();
                let rts_bytes = request_timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_CHAT_PULL,
                    &[
                        wallet.as_ref(),
                        at_bytes.as_ref(),
                        cursor.as_ref(),
                        limit_bytes.as_ref(),
                        rts_bytes.as_ref(),
                    ],
                    &wallet,
                    &signature,
                    request_timestamp,
                );
                if verify_result.is_err() {
                    return;
                }
                // [CHAT-PULL-ROUTE-AUTHORITY 2026-10-01 by Codex] Pull signs
                // query claims, not this session. Preserve delegated retrieval;
                // delegated route changes require session-bound register/presence.
                if wallet == session.client_public_key.to_bytes() {
                    relay
                        .wallet_routes
                        .announce(&wallet, session.id.clone(), session.endpoint());
                }
                match relay.pull_pending(&wallet, after_timestamp, &cursor, limit) {
                    Ok((messages, message_has_more)) => {
                        let envelopes: Vec<_> = messages.into_iter().map(|m| m.envelope).collect();
                        let mut has_more = message_has_more;
                        match relay.pull_pending_notifications(&wallet) {
                            Ok((notifications, notification_has_more)) => {
                                let delivery_complete = Self::push_expired_notifications(
                                    relay,
                                    notifications,
                                    session,
                                    udp,
                                    crypto,
                                )
                                .await;
                                has_more |= notification_has_more || !delivery_complete;
                            }
                            Err(e) => {
                                warn!(
                                    reason = e.reason_bucket(),
                                    "[CHAT_RELAY] Expiry notification pull failed"
                                );
                            }
                        }
                        // [CHAT-V1-COALESCING 2026-10-03 by Codex] A byte-
                        // shortened prefix remains pending until owner ACK.
                        let resp = match Self::coalesce_legacy_chat_pull(envelopes, has_more) {
                            Ok(response) => response,
                            Err(reason) => {
                                debug!(reason, "[CHAT_RELAY] Legacy pull assembly failed");
                                return;
                            }
                        };
                        if !Self::send_to_session(&resp, session, udp, crypto).await {
                            // No retry amplification, fake success or deletion.
                            // Single-envelope socket failures remain possible.
                            debug!(
                                reason = "response_write_failed",
                                "[CHAT_RELAY] Legacy pull not written"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            reason = e.reason_bucket(),
                            "[CHAT_RELAY] pull_pending failed"
                        );
                    }
                }
            }
            MemChainMessage::ChatPullV2 {
                wallet,
                after_timestamp,
                cursor,
                limit,
                request_timestamp,
                signature,
            } => {
                let Some(ref relay) = chat_relay else {
                    return;
                };
                if cursor.len() > MAX_CHAT_PULL_CURSOR_V2_BYTES {
                    warn!(
                        reason = "pull_cursor_too_large",
                        "[CHAT_RELAY] ChatPullV2 rejected"
                    );
                    return;
                }
                let Ok(cursor_len) = u16::try_from(cursor.len()) else {
                    return;
                };
                let at_bytes = after_timestamp.to_le_bytes();
                let cursor_len_bytes = cursor_len.to_le_bytes();
                let limit_bytes = limit.to_le_bytes();
                let rts_bytes = request_timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_CHAT_PULL_V2,
                    &[
                        wallet.as_ref(),
                        at_bytes.as_ref(),
                        cursor_len_bytes.as_ref(),
                        cursor.as_slice(),
                        limit_bytes.as_ref(),
                        rts_bytes.as_ref(),
                    ],
                    &wallet,
                    &signature,
                    request_timestamp,
                );
                if verify_result.is_err() {
                    debug!(
                        reason = "invalid_signature",
                        "[CHAT_RELAY] ChatPullV2 rejected"
                    );
                    return;
                }
                // [CHAT-PULL-ROUTE-AUTHORITY 2026-10-01 by Codex] A bounded
                // cursor does not bind the signed query to its carrying session.
                // Only the matching transport identity may refresh this route.
                if wallet == session.client_public_key.to_bytes() {
                    relay
                        .wallet_routes
                        .announce(&wallet, session.id.clone(), session.endpoint());
                }
                // [CHAT-V2-BYTE-PAGING 2026-10-03 by Codex] The domain must
                // shorten the page before sealing its continuation cursor.
                match relay.pull_pending_v2_coalesced(
                    &wallet,
                    after_timestamp,
                    &cursor,
                    limit,
                    Self::CHAT_PULL_V2_COALESCING_TARGET,
                ) {
                    Ok(page) => {
                        let envelopes: Vec<_> = page
                            .messages
                            .into_iter()
                            .map(|message| message.envelope)
                            .collect();
                        let mut has_more = page.has_more;
                        match relay.pull_pending_notifications(&wallet) {
                            Ok((notifications, notification_has_more)) => {
                                let delivery_complete = Self::push_expired_notifications(
                                    relay,
                                    notifications,
                                    session,
                                    udp,
                                    crypto,
                                )
                                .await;
                                has_more |= notification_has_more || !delivery_complete;
                            }
                            Err(e) => {
                                warn!(
                                    reason = e.reason_bucket(),
                                    "[CHAT_RELAY] Expiry notification pull failed"
                                );
                            }
                        }
                        let response = MemChainMessage::ChatPullResponseV2 {
                            envelopes,
                            next_cursor: page.next_cursor,
                            has_more,
                        };
                        if !Self::send_to_session(&response, session, udp, crypto).await {
                            debug!(
                                reason = "response_write_failed",
                                "[CHAT_RELAY] Snapshot pull not written"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            reason = e.reason_bucket(),
                            "[CHAT_RELAY] pull_pending_v2 failed"
                        );
                    }
                }
            }
            MemChainMessage::ChatAck {
                message_ids,
                wallet,
                ack_timestamp,
                signature,
            } => {
                let Some(ref relay) = chat_relay else {
                    return;
                };
                if message_ids.is_empty() {
                    return;
                }
                if message_ids.len() > MAX_CHAT_ACK_MESSAGE_IDS {
                    warn!(
                        reason = "ack_batch_too_large",
                        "[CHAT_RELAY] ChatAck rejected"
                    );
                    return;
                }
                let mut id_hasher = Sha256::new();
                for mid in &message_ids {
                    id_hasher.update(mid.as_ref());
                }
                let ids_hash: [u8; 32] = id_hasher.finalize().into();
                let ack_ts_bytes = ack_timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_CHAT_ACK,
                    &[wallet.as_ref(), ack_ts_bytes.as_ref(), ids_hash.as_ref()],
                    &wallet,
                    &signature,
                    ack_timestamp,
                );
                if verify_result.is_err() {
                    warn!(
                        reason = "invalid_signature",
                        "[CHAT_RELAY] ChatAck rejected"
                    );
                    return;
                }
                match relay.ack_messages(&message_ids, &wallet) {
                    Ok(deleted) => {
                        debug!(deleted, "[CHAT_RELAY] ChatAck processed");
                    }
                    Err(e) => {
                        warn!(
                            reason = e.reason_bucket(),
                            "[CHAT_RELAY] ack_messages failed"
                        );
                    }
                }
            }
            MemChainMessage::DeviceRegister {
                device_id,
                device_name: _,
                wallet_pubkey,
                timestamp,
                signature,
            } => {
                let ts_bytes = timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_DEVICE_REGISTER,
                    &[
                        session.id.as_bytes().as_ref(),
                        device_id.as_ref(),
                        wallet_pubkey.as_ref(),
                        ts_bytes.as_ref(),
                    ],
                    &wallet_pubkey,
                    &signature,
                    timestamp,
                );
                if verify_result.is_err() {
                    warn!(
                        reason = "invalid_signature",
                        "[CHAT_RELAY] DeviceRegister rejected"
                    );
                    return;
                }
                let Some(ref relay) = chat_relay else {
                    return;
                };
                // [SESSION-WALLET-BOUND 2026-07-29 by Codex] Keep the
                // independent online-route and device indexes transactional
                // from the handler's perspective. A rejected side must not
                // leave the other side advertising a half-registered wallet.
                if !relay.wallet_routes.announce(
                    &wallet_pubkey,
                    session.id.clone(),
                    session.client_endpoint,
                ) {
                    warn!(
                        reason = "session_wallet_limit",
                        "[CHAT_RELAY] DeviceRegister rejected"
                    );
                    return;
                }
                if !sessions.register_device(&wallet_pubkey, device_id, session.id.clone()) {
                    relay
                        .wallet_routes
                        .remove_route(&wallet_pubkey, &session.id);
                    warn!(
                        reason = "session_device_index_rejected",
                        "[CHAT_RELAY] DeviceRegister rejected"
                    );
                    return;
                }
                info!("[CHAT_RELAY] Device registered");
                match relay.pull_pending(&wallet_pubkey, 0, &[0u8; 16], 100) {
                    Ok((messages, _has_more)) if !messages.is_empty() => {
                        let count = messages.len();
                        for pm in messages {
                            Self::send_to_session(
                                &MemChainMessage::ChatRelay(pm.envelope),
                                session,
                                udp,
                                crypto,
                            )
                            .await;
                        }
                        info!(count, "[CHAT_RELAY] Delivered pending messages on register");
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!(
                            reason = e.reason_bucket(),
                            "[CHAT_RELAY] pull_pending on register failed"
                        );
                    }
                }
            }
            MemChainMessage::WalletPresence {
                wallet_pubkey,
                timestamp,
                signature,
            } => {
                let Some(ref relay) = chat_relay else {
                    return;
                };
                let ts_bytes = timestamp.to_le_bytes();
                let verify_result = verify_signed_message(
                    DOMAIN_WALLET_PRESENCE,
                    &[
                        session.id.as_bytes().as_ref(),
                        wallet_pubkey.as_ref(),
                        ts_bytes.as_ref(),
                    ],
                    &wallet_pubkey,
                    &signature,
                    timestamp,
                );
                if verify_result.is_err() {
                    debug!(
                        reason = "invalid_signature",
                        "[CHAT_RELAY] WalletPresence rejected"
                    );
                    return;
                }
                relay.wallet_routes.announce(
                    &wallet_pubkey,
                    session.id.clone(),
                    session.client_endpoint,
                );
                debug!("[CHAT_RELAY] WalletPresence route refreshed");
            }
            _ => {
                debug!("[MEMCHAIN] Unhandled message variant");
            }
        }
    }

    /// Sends one bounded page of durable `ChatExpired` control events.
    ///
    /// Successfully written rows are marked in one atomic database batch.
    /// Unsent or unmarkable rows remain pending and are retried on a later
    /// authenticated pull. Payloads and routing identifiers never enter logs.
    pub(super) async fn push_expired_notifications(
        relay: &ChatRelayService,
        notifications: Vec<ExpiredNotification>,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) -> bool {
        let offered = notifications.len();
        if offered == 0 {
            return true;
        }

        let mut pushed_ids = Vec::with_capacity(offered);
        for notification in notifications {
            let message_ids = match notification.message_ids() {
                Ok(message_ids) => message_ids,
                Err(e) => {
                    warn!(
                        reason = e.reason_bucket(),
                        "[CHAT_RELAY] Expiry notification decode failed"
                    );
                    break;
                }
            };
            let frame = MemChainMessage::ChatExpired {
                message_ids,
                receiver: notification.receiver,
            };
            if !Self::send_to_session(&frame, session, udp, crypto).await {
                break;
            }
            pushed_ids.push(notification.id);
        }

        let pushed = pushed_ids.len();
        if pushed > 0 {
            if let Err(e) = relay.mark_notifications_pushed(&pushed_ids) {
                warn!(
                    reason = e.reason_bucket(),
                    "[CHAT_RELAY] Expiry notification mark failed"
                );
                return false;
            }
            debug!(offered, pushed, "[CHAT_RELAY] Expiry notifications written");
        }

        pushed == offered
    }

    /// Writes one encrypted MemChain frame to a client session.
    ///
    /// The boolean reports only local encode/encrypt/socket-write success; it
    /// is not an application-level delivery receipt.
    pub(super) async fn send_to_session(
        msg: &MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) -> bool {
        let plaintext = match encode_memchain(msg) {
            Ok(p) => p,
            Err(_) => {
                error!(reason = "encode_failed", "[MEMCHAIN_TX] Frame write failed");
                return false;
            }
        };
        let counter = session.next_tx_counter();
        let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
        let len = match crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            &plaintext,
            &mut encrypted,
        ) {
            Ok(l) => l,
            Err(_) => {
                error!(
                    reason = "encrypt_failed",
                    "[MEMCHAIN_TX] Frame write failed"
                );
                return false;
            }
        };
        encrypted.truncate(len);
        let pkt = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
        let bytes = encode_data_packet(&pkt).to_vec();
        match udp.send(&bytes, &session.endpoint()).await {
            Ok(_) => true,
            Err(_) => {
                warn!(
                    reason = "socket_write_failed",
                    "[MEMCHAIN_TX] Frame write failed"
                );
                false
            }
        }
    }
}
