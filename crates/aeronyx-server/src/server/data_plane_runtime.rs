// ============================================
// File: crates/aeronyx-server/src/server/data_plane_runtime.rs
// ============================================
// [DATA-PLANE-RUNTIME 2026-09-25 by Codex] Keep UDP/TUN session ingress,
// authenticated teardown, traffic snapshots, and keepalive task lifecycles
// together. Parent Server still owns startup order and MemChain/chat handlers.
use super::*;

const KEEPALIVE_PROBE_INTERVAL_SECS: u64 = 60;
const KEEPALIVE_ACK_TIMEOUT_SECS: u64 = 90;
/// Closed, privacy-safe reason vocabulary for handshake admission failures.
///
/// [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] The UDP dispatcher must not
/// re-expand the service's typed failure into endpoint or attacker-controlled
/// error text. Unknown/new errors deliberately collapse to `internal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HandshakeRejectionClass {
    Policy,
    Capacity,
    Authentication,
    SessionAdmission,
    Internal,
}

impl HandshakeRejectionClass {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Policy => "policy",
            Self::Capacity => "capacity",
            Self::Authentication => "authentication",
            Self::SessionAdmission => "session_admission",
            Self::Internal => "internal",
        }
    }
}

pub(super) const fn handshake_rejection_class(error: &ServerError) -> HandshakeRejectionClass {
    match error {
        ServerError::WalletDenied { .. } | ServerError::NodePolicyRejected { .. } => {
            HandshakeRejectionClass::Policy
        }
        ServerError::IpPoolExhausted
        | ServerError::IpAlreadyAssigned(_)
        | ServerError::SessionLimitReached { .. } => HandshakeRejectionClass::Capacity,
        ServerError::Core(_) => HandshakeRejectionClass::Authentication,
        ServerError::SessionCreationFailed { .. } | ServerError::SessionExists => {
            HandshakeRejectionClass::SessionAdmission
        }
        _ => HandshakeRejectionClass::Internal,
    }
}

pub(super) fn log_handshake_rejection_class(
    reason: HandshakeRejectionClass,
    active_sessions: usize,
) {
    warn!(
        event = "handshake_rejected",
        reason = reason.as_str(),
        active_sessions,
        "Handshake rejected"
    );
}

pub(super) fn log_handshake_rejection(error: &ServerError, active_sessions: usize) {
    log_handshake_rejection_class(handshake_rejection_class(error), active_sessions);
}

/// Admit only the frozen handshake versions before any stateful ingress gate.
///
/// [HANDSHAKE-PRE-ADMISSION-VERSION 2026-09-21 by Codex] Datagram shape has
/// already established a fixed ClientHello header, so byte 1 is bounded and
/// sufficient. This deliberately performs no logging or mutation: unknown
/// versions cannot consume global/per-IP tokens or voucher observations.
pub(super) fn client_hello_wire_version_is_supported(datagram: &[u8]) -> bool {
    matches!(
        datagram.get(1).copied(),
        Some(PROTOCOL_VERSION_V1) | Some(PROTOCOL_VERSION_V2)
    )
}

impl Server {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spawn_udp_task(
        &self,
        udp: Arc<UdpTransport>,
        #[cfg(target_os = "linux")] tun: Arc<LinuxTun>,
        handshake: Arc<HandshakeService>,
        packet_handler: Arc<PacketHandler>,
        voucher_verifier: Arc<VoucherVerifier>,
        sessions: Arc<SessionManager>,
        session_events: SessionEventSender,
        mempool: Option<Arc<MemPool>>,
        aof_writer: Option<Arc<TokioMutex<AofWriter>>>,
        storage: Option<Arc<MemoryStorage>>,
        vector_index: Option<Arc<VectorIndex>>,
        memchain_config: MemChainConfig,
        server_pubkey_hex: String,
        chat_relay: Option<Arc<ChatRelayService>>,
        routing: Arc<RoutingService>,
        peer_store: Arc<PeerStore>,
        control_http_client: Arc<reqwest::Client>,
        traffic_tracker: Arc<TrafficTracker>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let udp_reply = Arc::clone(&udp);
        let self_node_id = self.identity.public_key_bytes();
        let node_identity = self.identity.clone();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            let crypto = DefaultTransportCrypto::new();
            let handshake_limiter = crate::services::HandshakeLimiter::production();
            let mut consecutive_receive_failures = 0u32;

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    result = udp.recv(&mut buf) => {
                        match result {
                            Ok((len, source)) => {
                                consecutive_receive_failures = 0;
                                if shutdown.load(Ordering::SeqCst) { break; }
                                let data = &buf[..len];

                                match ProtocolCodec::classify_datagram(data) {
                                    MessageType::ClientHello => {
                                        if !client_hello_wire_version_is_supported(data) {
                                            continue;
                                        }
                                        // [2026-09-12] Two token buckets before any crypto.
                                        if !handshake_limiter.allow(source.addr.ip()) {
                                            continue;
                                        }
                                        let extension = if data.len() > CLIENT_HELLO_SIZE {
                                            data[CLIENT_HELLO_SIZE..].to_vec()
                                        } else {
                                            Vec::new()
                                        };
                                        let verifier = Arc::clone(&voucher_verifier);
                                        if !verifier.accept_client_hello_extension(extension.clone()).await {
                                            // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex]
                                            // Voucher failure is part of the same v1/v2 admission
                                            // boundary and must not expose the source endpoint.
                                            log_handshake_rejection_class(
                                                HandshakeRejectionClass::Authentication,
                                                sessions.count(),
                                            );
                                            continue;
                                        }

                                        if let Ok(hello) = decode_client_hello(data) {
                                            match handshake.process(&hello, &extension, source.addr) {
                                                Ok(result) => {
                                                    // [PROTOCOL-V2-ADMISSION-HOTFIX 2026-09-13 by Codex]
                                                    // The handshake service returns evictions only
                                                    // after authenticating the claimed identity and
                                                    // atomically admitting its replacement session.
                                                    for termination in
                                                        result.session.take_admission_evictions()
                                                    {
                                                        Self::finalize_session_termination(
                                                            termination,
                                                            &routing,
                                                            &session_events,
                                                            chat_relay.as_deref(),
                                                            &traffic_tracker,
                                                        );
                                                    }
                                                    let sid        = BASE64.encode(&result.response.session_id);
                                                    let wallet_hex = hex::encode(result.session.client_public_key.to_bytes());
                                                    session_events.session_created(
                                                        &sid,
                                                        Some(wallet_hex),
                                                        Some(result.session.virtual_ip.to_string()),
                                                    );
                                                    let resp = encode_server_hello(&result.response);
                                                    let _ = udp.send(&resp, &source.addr).await;
                                                }
                                                Err(error) => {
                                                    // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex]
                                                    // Do not reintroduce the peer endpoint or raw
                                                    // typed error after the service has rejected a
                                                    // v1/v2 handshake.
                                                    log_handshake_rejection(
                                                        &error,
                                                        sessions.count(),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                    MessageType::Keepalive => {
                                        if len >= KEEPALIVE_PACKET_SIZE {
                                            let mut sid = [0u8; 16];
                                            sid.copy_from_slice(&data[1..17]);
                                            if let Some(id) = SessionId::from_bytes(&sid) {
                                                if let Some(s) = sessions.get(&id) { s.touch(); }
                                            }
                                        }
                                    }
                                    MessageType::Data | MessageType::ServerHello => {
                                        match packet_handler.handle_udp_packet(data, source.addr) {
                                            Ok((_sess, DecryptedPayload::Vpn(pkt))) => {
                                                #[cfg(target_os = "linux")]
                                                { let _ = tun.write(&pkt).await; }
                                            }
                                            // [PROTOCOL-V2] Authenticated liveness and close.
                                            Ok((session, DecryptedPayload::ControlPing { id })) => {
                                                if let Ok(bytes) = packet_handler
                                                    .seal_control(&session, &[0x00, 0x02, id[0], id[1]])
                                                {
                                                    let _ = udp_reply.send(&bytes, &session.endpoint()).await;
                                                }
                                            }
                                            Ok((_session, DecryptedPayload::ControlPong)) => {}
                                            Ok((session, DecryptedPayload::ControlDisconnect { reason })) => {
                                                info!(session_id = %session.id, reason, "[PROTOCOL-V2] Client closed the session");
                                                if let Some(termination) =
                                                    sessions.terminate_with_cooldown(&session.id)
                                                {
                                                    Self::finalize_session_termination(
                                                        termination,
                                                        &routing,
                                                        &session_events,
                                                        chat_relay.as_deref(),
                                                        &traffic_tracker,
                                                    );
                                                }
                                            }
                                            Ok((session, DecryptedPayload::KeepaliveAck { rtt_ms })) => {
                                                trace!(
                                                    session_id = %session.id,
                                                    rtt_ms,
                                                    "[KEEPALIVE] ACK consumed"
                                                );
                                            }
                                            Err(ref e) if e.is_session_not_found() => {
                                                let reset = [0xFFu8];
                                                let _ = udp_reply.send(&reset, &source.addr).await;
                                                debug!(src = %source.addr, "[SESSION] Sent RESET to stale client");
                                            }
                                            Err(_) => {}
                                            Ok((_session, DecryptedPayload::VoiceSignal { discriminant, target_wallet, payload })) => {
                                                let signal_name = match discriminant { 31 => "Offer", 32 => "Answer", 33 => "End", _ => "Unknown" };
                                                match target_wallet {
                                                    None => { warn!(reason = "missing_target", signal = signal_name, "[VOICE_SIGNAL] Dropped"); }
                                                    Some(ref wallet_hex) => {
                                                        let target_bytes = hex::decode(wallet_hex).ok().and_then(|b| {
                                                            if b.len() == 32 { let mut arr = [0u8; 32]; arr.copy_from_slice(&b); Some(arr) } else { None }
                                                        });
                                                        match target_bytes {
                                                            None => { warn!(reason = "invalid_target", signal = signal_name, "[VOICE_SIGNAL] Dropped"); }
                                                            Some(pk) => {
                                                                let target = sessions.get_by_wallet(&pk).or_else(|| {
                                                                    sessions.all_sessions().into_iter().find(|s| s.client_public_key.to_bytes() == pk)
                                                                });
                                                                match target {
                                                                    None => { debug!(reason = "target_offline", signal = signal_name, "[VOICE_SIGNAL] Dropped"); }
                                                                    Some(target_session) => {
                                                                        let counter = target_session.next_tx_counter();
                                                                        let mut encrypted = vec![0u8; payload.len() + ENCRYPTION_OVERHEAD];
                                                                        match crypto.encrypt(&target_session.session_key, counter, target_session.id.as_bytes(), &payload, &mut encrypted) {
                                                                            Ok(len) => {
                                                                                encrypted.truncate(len);
                                                                                let pkt   = aeronyx_core::protocol::DataPacket::new(*target_session.id.as_bytes(), counter, encrypted);
                                                                                let bytes = aeronyx_core::protocol::codec::encode_data_packet(&pkt).to_vec();
                                                                                let _ = udp_reply.send(&bytes, &target_session.endpoint()).await;
                                                                                debug!(signal = signal_name, "[VOICE_SIGNAL] Forwarded");
                                                                            }
                                                                            Err(_) => { warn!(reason = "encryption_failed", signal = signal_name, "[VOICE_SIGNAL] Forward failed"); }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            Ok((_session, DecryptedPayload::Voice { dst_ip, payload })) => {
                                                if let Some(target_sid) = routing.lookup(dst_ip) {
                                                    if let Some(target) = sessions.get(&target_sid) {
                                                        let counter = target.next_tx_counter();
                                                        let mut encrypted = vec![0u8; payload.len() + ENCRYPTION_OVERHEAD];
                                                        match crypto.encrypt(&target.session_key, counter, target.id.as_bytes(), &payload, &mut encrypted) {
                                                            Ok(len) => {
                                                                encrypted.truncate(len);
                                                                let pkt   = aeronyx_core::protocol::DataPacket::new(*target.id.as_bytes(), counter, encrypted);
                                                                let bytes = aeronyx_core::protocol::codec::encode_data_packet(&pkt).to_vec();
                                                                let _ = udp_reply.send(&bytes, &target.endpoint()).await;
                                                                trace!("[VOICE] Relayed voice packet");
                                                            }
                                                            Err(_) => { warn!(reason = "encryption_failed", "[VOICE] Relay failed"); }
                                                        }
                                                    } else {
                                                        debug!(reason = "target_disconnected", "[VOICE] Relay dropped");
                                                    }
                                                } else {
                                                    debug!(reason = "no_route", "[VOICE] Relay dropped");
                                                }
                                            }
                                            Ok((session, DecryptedPayload::MemChain(msg))) => {
                                                // [SESSION-TERMINATION 2026-08-15 by Codex]
                                                // Graceful close is transport lifecycle, not
                                                // MemChain storage. Dispatch it even when the
                                                // optional memory subsystem is disabled.
                                                if let MemChainMessage::SessionCloseV1 {
                                                    session_id,
                                                    close_timestamp,
                                                    signature,
                                                } = &msg
                                                {
                                                    if Self::session_close_v1_is_authenticated(
                                                        &session,
                                                        session_id,
                                                        *close_timestamp,
                                                        signature,
                                                    ) {
                                                        if let Some(termination) = sessions
                                                            .terminate_with_cooldown(&session.id)
                                                        {
                                                            Self::finalize_session_termination(
                                                                termination,
                                                                &routing,
                                                                &session_events,
                                                                chat_relay.as_deref(),
                                                                &traffic_tracker,
                                                            );
                                                        }
                                                    } else {
                                                        warn!(
                                                            reason = "authentication_failed",
                                                            "[SESSION] Graceful close rejected"
                                                        );
                                                    }
                                                    continue;
                                                }
                                                // [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex]
                                                // Chat runtime admission is independent from the
                                                // optional Fact/AOF and record stores. The handler's
                                                // typed gate rejects only messages whose own effect
                                                // requires unavailable persistence.
                                                Self::handle_memchain_message(
                                                    msg, mempool.as_ref(), aof_writer.as_ref(),
                                                    &storage, &vector_index, &memchain_config,
                                                    &server_pubkey_hex, &session, &udp_reply,
                                                    &crypto, &sessions, &chat_relay, &peer_store,
                                                    &self_node_id, &node_identity,
                                                    Some(control_http_client.as_ref()),
                                                ).await;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                if !retry_required_data_plane_receive(
                                    "udp",
                                    &e,
                                    &mut consecutive_receive_failures,
                                    shutdown.as_ref(),
                                    &mut shutdown_rx,
                                )
                                .await
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    /// Verifies one graceful-close frame against its encrypted outer session.
    pub(super) fn session_close_v1_is_authenticated(
        session: &crate::services::Session,
        claimed_session_id: &[u8; 16],
        close_timestamp: u64,
        signature: &[u8; 64],
    ) -> bool {
        // [SESSION-TERMINATION 2026-08-15 by Codex] Binding both the wire field
        // and signer to the already-decrypted session prevents a valid close
        // request from being replayed into another tunnel during its 60-second
        // timestamp window.
        if claimed_session_id != session.id.as_bytes() {
            return false;
        }
        let timestamp_bytes = close_timestamp.to_le_bytes();
        verify_signed_message(
            DOMAIN_SESSION_CLOSE_V1,
            &[claimed_session_id.as_ref(), timestamp_bytes.as_ref()],
            &session.client_public_key.to_bytes(),
            signature,
            close_timestamp,
        )
        .is_ok()
    }

    /// Finalizes resources owned outside [`SessionManager`] exactly once.
    pub(super) fn finalize_session_termination(
        termination: SessionTermination,
        routing: &RoutingService,
        events: &SessionEventSender,
        chat_relay: Option<&ChatRelayService>,
        traffic_tracker: &TrafficTracker,
    ) {
        // [SESSION-TERMINATION 2026-08-15 by Codex] Conditional route removal
        // protects a replacement session if stale cleanup races with IP reuse.
        routing.remove_route_for_session(termination.virtual_ip, &termination.session_id);
        events.session_ended(
            &termination.session_id.to_string(),
            Some(termination.wallet_hex.clone()),
            Some(termination.virtual_ip.to_string()),
            termination.stats.bytes_rx,
            termination.stats.bytes_tx,
            quality_from_stats(termination.stats),
        );
        if let Some(relay) = chat_relay {
            relay.wallet_routes.remove_session(&termination.session_id);
        }
        traffic_tracker.remove_wallet(&termination.wallet_hex);
    }

    // ============================================
    // TUN Task
    // ============================================

    #[cfg(target_os = "linux")]
    pub(super) fn spawn_tun_task(
        &self,
        tun: Arc<LinuxTun>,
        udp: Arc<UdpTransport>,
        handler: Arc<PacketHandler>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            let mut consecutive_receive_failures = 0u32;
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    result = tun.read(&mut buf) => {
                        match result {
                            Ok(len) => {
                                consecutive_receive_failures = 0;
                                if shutdown.load(Ordering::SeqCst) { break; }
                                if let Ok((enc, ep)) = handler.handle_tun_packet(&buf[..len]) {
                                    let _ = udp.send(&enc, &ep).await;
                                }
                            }
                            Err(e) => {
                                if !retry_required_data_plane_receive(
                                    "tun",
                                    &e,
                                    &mut consecutive_receive_failures,
                                    shutdown.as_ref(),
                                    &mut rx,
                                )
                                .await
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // Traffic Snapshot Task
    // ============================================

    pub(super) fn spawn_traffic_snapshot_task(
        &self,
        sessions: Arc<SessionManager>,
        events: SessionEventSender,
        interval_secs: u64,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        let interval_secs = interval_secs.clamp(10, 300);
        tokio::spawn(async move {
            // [BACKGROUND-SHUTDOWN-COOPERATION 2026-08-12 by Codex] A plain
            // startup sleep kept this task alive for up to five minutes after
            // SIGTERM. Treat any receiver completion as shutdown; a closed
            // broadcast means the process owner has disappeared as well.
            if Self::runtime_delay_interrupted_by_shutdown(
                &mut rx,
                Duration::from_secs(interval_secs),
            )
            .await
            {
                return;
            }
            let mut timer = tokio::time::interval(Duration::from_secs(interval_secs));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        let all      = sessions.all_sessions();
                        let mut reported = 0usize;
                        for session in all {
                            if !session.is_established() { continue; }
                            let snap = session.stats_snapshot();
                            let wallet_hex = session.wallet_hex.clone();
                            let sid        = BASE64.encode(session.id.as_bytes());
                            events.session_traffic_snapshot(
                                &sid,
                                Some(wallet_hex),
                                Some(session.virtual_ip.to_string()),
                                snap.bytes_rx,
                                snap.bytes_tx,
                                quality_from_stats(snap),
                            );
                            reported += 1;
                        }
                        if reported > 0 {
                            debug!(
                                sessions = reported,
                                interval_secs,
                                "[TRAFFIC_SNAPSHOT] Sent quality snapshots for {} active session(s)",
                                reported
                            );
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // VPN Keepalive / RTT Probe Task
    // ============================================

    pub(super) fn spawn_keepalive_probe_task(
        &self,
        sessions: Arc<SessionManager>,
        udp: Arc<UdpTransport>,
        packet_handler: Arc<PacketHandler>,
        gateway_ip: Ipv4Addr,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            // [KEEPALIVE-SHUTDOWN-COOPERATION 2026-08-12 by Codex] Keepalive
            // probes need a warm-up, but service shutdown must not wait for a
            // fixed sleep that has no session or network work to preserve.
            if Self::runtime_delay_interrupted_by_shutdown(&mut rx, Duration::from_secs(30)).await {
                return;
            }
            let mut timer =
                tokio::time::interval(Duration::from_secs(KEEPALIVE_PROBE_INTERVAL_SECS));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }
                        let mut sent = 0usize;
                        for session in sessions.all_sessions() {
                            if !session.is_established() { continue; }
                            match packet_handler.build_keepalive_probe(
                                &session,
                                gateway_ip,
                                Duration::from_secs(KEEPALIVE_ACK_TIMEOUT_SECS),
                            ) {
                                Ok((bytes, endpoint)) => {
                                    if udp.send(&bytes, &endpoint).await.is_ok() {
                                        sent += 1;
                                    }
                                }
                                Err(e) => {
                                    debug!(
                                        session_id = %session.id,
                                        error = %e,
                                        "[KEEPALIVE] Probe build failed"
                                    );
                                }
                            }
                        }
                        if sent > 0 {
                            trace!(sessions = sent, "[KEEPALIVE] ICMP probes sent");
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_close_authentication_binds_transport_identity_and_id() {
        let identity = IdentityKeyPair::generate();
        let session_id = aeronyx_common::types::SessionId::generate();
        let session = crate::services::Session::new(
            session_id.clone(),
            identity.public_key(),
            aeronyx_core::crypto::SessionKey::from_bytes([0x41; 32]),
            Ipv4Addr::new(100, 64, 0, 41),
            "127.0.0.1:1041".parse().unwrap(),
        );
        let close_timestamp = unix_now_secs();
        let timestamp_bytes = close_timestamp.to_le_bytes();
        let mut hasher = Sha256::new();
        hasher.update(aeronyx_core::protocol::DOMAIN_SESSION_CLOSE_V1.as_bytes());
        hasher.update(session_id.as_bytes());
        hasher.update(timestamp_bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        let signature = identity.sign(&digest);

        assert!(Server::session_close_v1_is_authenticated(
            &session,
            session_id.as_bytes(),
            close_timestamp,
            &signature,
        ));
        // [SESSION-TERMINATION 2026-08-15 by Codex] Neither a different outer
        // session nor a different ClientHello identity may reuse this frame.
        assert!(!Server::session_close_v1_is_authenticated(
            &session,
            &[0x42; 16],
            close_timestamp,
            &signature,
        ));
        let wrong_identity = IdentityKeyPair::generate();
        let wrong_session = crate::services::Session::new(
            session_id,
            wrong_identity.public_key(),
            aeronyx_core::crypto::SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 42),
            "127.0.0.1:1042".parse().unwrap(),
        );
        assert!(!Server::session_close_v1_is_authenticated(
            &wrong_session,
            wrong_session.id.as_bytes(),
            close_timestamp,
            &signature,
        ));
    }
}
