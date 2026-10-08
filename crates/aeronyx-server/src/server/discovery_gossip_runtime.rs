// ============================================
// File: crates/aeronyx-server/src/server/discovery_gossip_runtime.rs
// ============================================
// [DISCOVERY-GOSSIP-RUNTIME 2026-09-25 by Codex] Owns bounded peer sampling,
// optional proof negotiation, and synthetic route probes. Shared client
// delivery primitives and the Server startup/shutdown composition remain in
// the parent module; no route, receipt, quarantine, or backpressure contract
// changes are introduced by this extraction.
use super::*;
// [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] Keep the same
// cfg-gated loopback policy as the outbound transport owner.
use super::chat_outbound_runtime::is_test_loopback_peer_endpoint;
use crate::services::discovery_peer_sampling::sample_public_gossip_peers_with_policy;

// [PHALA-APPRAISAL-TASK-OWNERSHIP 2026-10-07 by Codex] The gossip
// future owns one child even when startup unwinds or its registry aborts it.
// Blocking QVL/publication work may outlive async cancellation; the shared
// stop bit fences publication without releasing QVL's retained permit early.
pub(super) struct PhalaAppraisalTaskOwner {
    task: Option<JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
}

impl PhalaAppraisalTaskOwner {
    pub(super) fn new() -> Self {
        Self { task: None, stopped: Arc::new(AtomicBool::new(false)) }
    }

    pub(super) fn can_start(&self) -> bool {
        !self.stopped.load(Ordering::Acquire)
            && self.task.as_ref().is_none_or(|task| task.is_finished())
    }

    pub(super) fn stop_signal(&self) -> Arc<AtomicBool> { Arc::clone(&self.stopped) }

    pub(super) fn start(&mut self, work: impl std::future::Future<Output = ()> + Send + 'static) -> bool {
        if !self.can_start() { return false; }
        self.task = Some(tokio::spawn(work));
        true
    }

    fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Some(task) = &self.task { task.abort(); }
    }

    pub(super) async fn stop_and_join(&mut self) {
        self.request_stop();
        // Keep the handle owned if this join itself is cancelled.
        if let Some(task) = self.task.as_mut() { let _ = task.await; }
        self.task = None;
    }
}

impl Drop for PhalaAppraisalTaskOwner {
    fn drop(&mut self) { self.request_stop(); }
}

/// Minimal additive fields read from a peer's public discovery summary.
///
/// [DIRECTORY-GOSSIP-NEGOTIATION 2026-07-27 by Codex] This unsigned document
/// is only a transport optimization hint. A false positive can at most cause
/// one optional proof request before mandatory legacy fallback; a false
/// negative can only suppress that optional request. It never grants trust.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct DiscoveryNegotiationSummary {
    #[serde(default)]
    pub(super) protocol_features: DiscoveryNegotiationFeatures,
}

#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct DiscoveryNegotiationFeatures {
    #[serde(default)]
    pub(super) directory_descriptor_proof_gossip_v1: bool,
    /// Whether this relay accepts a terminal receipt propagated through more
    /// than one middle hop. Missing means legacy/unsupported.
    #[serde(default)]
    pub(super) multihop_delivery_receipt_v1: bool,
    /// Whether current terminal ACKs commit to both payload and canonical
    /// route purpose. Missing means v1-only/unsupported.
    #[serde(default)]
    pub(super) purpose_bound_delivery_receipt_v2: bool,
}

/// Terminal result for optional Directory-authenticated proof transmission.
///
/// [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] Buckets are stable and
/// privacy-safe. They intentionally omit peer, endpoint, producer, descriptor,
/// block, proof, route, message, payload, client, and traffic dimensions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum DirectoryProofGossipResult {
    #[default]
    NotAttempted,
    Accepted,
    EvidenceRejected,
    ReplicaUnavailable,
    RateLimited,
    ProtocolRejected,
    TransportFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiscoveryGossipTargetPolicy {
    round_peer_limit: usize,
    include_operator_seeds: bool,
    include_cached_peers: bool,
}

impl DiscoveryGossipTargetPolicy {
    fn for_config(config: &ServerConfig, backpressure_active: bool) -> Self {
        // [PHALA-PRIVATE-RECIPIENT-RELAY-ONLY 2026-10-06 by Codex] A private
        // worker may refresh only its operator-pinned relay; neither configured
        // seeds nor cached public peers are outbound authority for this role.
        if config.reverse_onion.recipient.enabled {
            return Self {
                round_peer_limit: 1,
                include_operator_seeds: false,
                include_cached_peers: false,
            };
        }

        let seed_limit = config.discovery.seed_endpoints.len().max(1);
        Self {
            round_peer_limit: if backpressure_active {
                usize::from(config.discovery.gossip_peer_limit).min(seed_limit)
            } else {
                usize::from(config.discovery.gossip_peer_limit)
            },
            include_operator_seeds: true,
            include_cached_peers: !backpressure_active || config.discovery.seed_endpoints.is_empty(),
        }
    }
}

impl DirectoryProofGossipResult {
    pub(super) const fn from_http_status(status: reqwest::StatusCode) -> Self {
        match status {
            reqwest::StatusCode::UNPROCESSABLE_ENTITY => Self::EvidenceRejected,
            reqwest::StatusCode::SERVICE_UNAVAILABLE => Self::ReplicaUnavailable,
            reqwest::StatusCode::TOO_MANY_REQUESTS => Self::RateLimited,
            _ => Self::ProtocolRejected,
        }
    }
}

/// Capability-negotiation stage for one optional proof exchange.
///
/// A typed stage prevents impossible boolean combinations such as "attempted
/// but not capable" from entering round telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum DirectoryProofGossipPeerState {
    #[default]
    NotChecked,
    LegacyOnly,
    NegotiationFailed,
    Attempted,
}

/// Per-peer result for the optional Directory-authenticated gossip frames.
///
/// [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] This is process-local,
/// aggregate-only telemetry. Capability hints are non-authoritative, and these
/// fields must never gain peer, endpoint, descriptor, producer, block, proof,
/// route, message, payload, client, user, or traffic dimensions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DirectoryProofGossipOutcome {
    pub(super) state: DirectoryProofGossipPeerState,
    pub(super) frames_attempted: usize,
    pub(super) evidence_rejected: usize,
    pub(super) result: DirectoryProofGossipResult,
}

impl DirectoryProofGossipOutcome {
    pub(super) const fn negotiation_failed(result: DirectoryProofGossipResult) -> Self {
        Self {
            state: DirectoryProofGossipPeerState::NegotiationFailed,
            frames_attempted: 0,
            evidence_rejected: 0,
            result,
        }
    }

    pub(super) const fn capability_checked(self) -> bool {
        !matches!(self.state, DirectoryProofGossipPeerState::NotChecked)
    }

    pub(super) const fn capable(self) -> bool {
        matches!(self.state, DirectoryProofGossipPeerState::Attempted)
    }

    pub(super) const fn attempted(self) -> bool {
        matches!(self.state, DirectoryProofGossipPeerState::Attempted)
    }

    pub(super) const fn accepted(self) -> bool {
        matches!(self.result, DirectoryProofGossipResult::Accepted)
    }
}

/// Fixed phase of a mandatory compatibility gossip failure.
///
/// [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] Typed phases prevent URLs
/// or peer-controlled text from entering status, logs, or audit events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DiscoveryGossipPhase {
    Peer,
    AnnounceRequest,
    AnnounceStatus,
    SnapshotRequest,
    SnapshotStatus,
    SnapshotResponse,
}

impl DiscoveryGossipPhase {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Peer => "peer",
            Self::AnnounceRequest => "announce_request",
            Self::AnnounceStatus => "announce_status",
            Self::SnapshotRequest => "snapshot_request",
            Self::SnapshotStatus => "snapshot_status",
            Self::SnapshotResponse => "snapshot_response",
        }
    }
}

/// Privacy-safe class of a mandatory gossip failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DiscoveryGossipFailureKind {
    Timeout,
    Connect,
    Http(u16),
    HttpStatus,
    Decode,
    Body,
    Request,
    BoundedResponse(BoundedHttpResponseError),
    Unknown,
}

/// Typed internal failure retained by a per-peer report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DiscoveryGossipFailure {
    pub(super) phase: DiscoveryGossipPhase,
    pub(super) kind: DiscoveryGossipFailureKind,
}

impl DiscoveryGossipFailure {
    pub(super) const fn peer_timeout() -> Self {
        Self {
            phase: DiscoveryGossipPhase::Peer,
            kind: DiscoveryGossipFailureKind::Timeout,
        }
    }

    pub(super) fn from_reqwest(phase: DiscoveryGossipPhase, error: &reqwest::Error) -> Self {
        let kind = if error.is_timeout() {
            DiscoveryGossipFailureKind::Timeout
        } else if error.is_connect() {
            DiscoveryGossipFailureKind::Connect
        } else if error.is_status() {
            error
                .status()
                .map_or(DiscoveryGossipFailureKind::HttpStatus, |status| {
                    DiscoveryGossipFailureKind::Http(status.as_u16())
                })
        } else if error.is_decode() {
            DiscoveryGossipFailureKind::Decode
        } else if error.is_body() {
            DiscoveryGossipFailureKind::Body
        } else if error.is_request() {
            DiscoveryGossipFailureKind::Request
        } else {
            DiscoveryGossipFailureKind::Unknown
        };
        Self { phase, kind }
    }

    pub(super) const fn from_bounded_response(
        phase: DiscoveryGossipPhase,
        error: BoundedHttpResponseError,
    ) -> Self {
        Self {
            phase,
            kind: DiscoveryGossipFailureKind::BoundedResponse(error),
        }
    }

    pub(super) fn bucket(self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for DiscoveryGossipFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}_", self.phase.as_str())?;
        match self.kind {
            DiscoveryGossipFailureKind::Timeout => formatter.write_str("timeout"),
            DiscoveryGossipFailureKind::Connect => formatter.write_str("connect"),
            DiscoveryGossipFailureKind::Http(status) => write!(formatter, "http_{status}"),
            DiscoveryGossipFailureKind::HttpStatus => formatter.write_str("http_status"),
            DiscoveryGossipFailureKind::Decode => formatter.write_str("decode"),
            DiscoveryGossipFailureKind::Body => formatter.write_str("body"),
            DiscoveryGossipFailureKind::Request => formatter.write_str("request"),
            DiscoveryGossipFailureKind::BoundedResponse(error) => {
                formatter.write_str(error.as_str())
            }
            DiscoveryGossipFailureKind::Unknown => formatter.write_str("unknown"),
        }
    }
}

/// Unique verified node identities observed for canonical gossip URLs.
///
/// [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] Signed descriptors may
/// independently claim the same endpoint. Such a collision is not enough to
/// choose either identity, so the URL remains ambiguous for this hint set and
/// proof selection uses its bounded identity-agnostic fallback.
#[derive(Debug, Default)]
pub(super) struct DiscoveryPeerIdentityHints {
    pub(super) by_url: HashMap<String, Option<[u8; 32]>>,
    // Keep the route's transport pin separate from gossip identity hints:
    // identity ambiguity must not silently select the generic DNS client.
    private_transport_by_url: HashMap<String, Option<[u8; 32]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] Internal scope also
// admits the existing server-level policy regression fixtures.
pub(super) enum GossipTransportPin {
    Generic,
    Pinned([u8; 32]),
    Reject,
}

impl DiscoveryPeerIdentityHints {
    pub(super) fn observe_verified(&mut self, url: String, node_id: [u8; 32]) {
        match self.by_url.entry(url) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Some(node_id));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if entry.get().as_ref() != Some(&node_id) {
                    entry.insert(None);
                }
            }
        }
    }

    pub(super) fn unique_node_id(&self, url: &str) -> Option<[u8; 32]> {
        self.by_url.get(url).copied().flatten()
    }

    // [REVERSE-ONION-GOSSIP-FAIL-CLOSED 2026-10-06 by Codex]
    pub(super) fn observe_private_transport_target(
        &mut self,
        url: String,
        node_id: [u8; 32],
    ) {
        match self.private_transport_by_url.entry(url) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Some(node_id));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if entry.get().as_ref() != Some(&node_id) {
                    entry.insert(None);
                }
            }
        }
    }

    pub(super) fn transport_pin_for(
        &self,
        url: &str,
        verified_node_id: Option<[u8; 32]>,
    ) -> GossipTransportPin {
        match self.private_transport_by_url.get(url) {
            None => GossipTransportPin::Generic,
            Some(None) => GossipTransportPin::Reject,
            Some(Some(expected)) if verified_node_id.is_some_and(|node| node != *expected) => {
                GossipTransportPin::Reject
            }
            Some(Some(expected)) => GossipTransportPin::Pinned(*expected),
        }
    }
}

/// Immutable dependencies and policy for one outbound gossip round.
///
/// [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] Keeping these values in a
/// borrowed execution context prevents long parameter lists and guarantees
/// every concurrently polled peer receives the same limits and proof set.
#[derive(Clone, Copy)]
pub(super) struct DiscoveryGossipExecution<'a> {
    pub(super) client: &'a reqwest::Client,
    pub(super) peer_store: &'a PeerStore,
    pub(super) local_identity: Option<&'a IdentityKeyPair>,
    /// [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Applies to all
    /// private-recipient gossip, including recovery-only descriptor refresh.
    pub(super) private_recipient: Option<&'a crate::config_reverse_onion::ReverseOnionRecipientConfig>,
    /// The one R identity for which a local P may issue its grant.
    pub(super) private_authorization_target: Option<[u8; 32]>,
    /// [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Exact,
    /// bounded operator-pinned source identities selected for this round.
    pub(super) private_authorization_source_targets: &'a [[u8; 32]],
    /// P identity named by the local queue's operator pin; signatures and
    /// current descriptors still come from authenticated gossip.
    pub(super) private_authorization_recipient: Option<[u8; 32]>,
    pub(super) directory_announcements: &'a [DirectoryReplicaGossipAnnouncement],
    /// Verified `PeerStore` identity hints keyed by canonical gossip URL.
    ///
    /// [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] This remains a local
    /// optimization only. Missing or stale hints cannot grant trust; they only
    /// decide whether a receiver's own producer proof is skipped.
    pub(super) peer_identity_hints: Option<&'a DiscoveryPeerIdentityHints>,
    pub(super) now: u64,
    pub(super) snapshot_limit: u16,
    pub(super) peer_timeout: Duration,
}

/// Immutable selection input for the verified public-peer portion of one
/// gossip round. It carries no receiver, session, or message identity.
// [PERMISSIONLESS-GOSSIP-RUNTIME 2026-09-24 by Codex] Keep the sampling
// boundary typed and independent of mutable seed/URL deduplication state.
pub(super) struct DiscoveryGossipSampleRequest<'a> {
    pub(super) now: u64,
    pub(super) round_nonce: [u8; 32],
    pub(super) round_peer_limit: usize,
    pub(super) self_node_id: &'a [u8; 32],
    pub(super) self_gossip_url: Option<&'a str>,
}

/// Complete result of one outbound peer gossip attempt.
///
/// [GOSSIP-OUTCOME-INTEGRITY 2026-07-28 by Codex] Optional proof telemetry is
/// retained even when the mandatory legacy descriptor/snapshot exchange fails.
/// The error renders as a bounded phase bucket from typed local variants;
/// it must never contain a URL, peer id, descriptor, proof, payload, or client.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DiscoveryPeerGossipReport {
    pub(super) directory_proof: DirectoryProofGossipOutcome,
    pub(super) legacy_error: Option<DiscoveryGossipFailure>,
}

/// Process-local accumulator for one outbound gossip round.
///
/// This owns the counter fan-in so the task loop cannot accidentally count a
/// proof only when the later legacy exchange succeeds. All fields remain
/// aggregate control-plane observations without peer or traffic dimensions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DiscoveryGossipRoundAccumulator {
    pub(super) attempted: usize,
    pub(super) succeeded: usize,
    pub(super) directory_proof: PeerStoreDirectoryProofGossipRound,
    pub(super) last_failure_reason: Option<DiscoveryGossipFailure>,
}

impl DiscoveryGossipRoundAccumulator {
    pub(super) fn observe(&mut self, report: DiscoveryPeerGossipReport) {
        self.attempted = self.attempted.saturating_add(1);

        let proof = report.directory_proof;
        self.directory_proof.capability_checked = self
            .directory_proof
            .capability_checked
            .saturating_add(usize::from(proof.capability_checked()));
        self.directory_proof.capable = self
            .directory_proof
            .capable
            .saturating_add(usize::from(proof.capable()));
        self.directory_proof.peers_attempted = self
            .directory_proof
            .peers_attempted
            .saturating_add(usize::from(proof.attempted()));
        self.directory_proof.frames_attempted = self
            .directory_proof
            .frames_attempted
            .saturating_add(proof.frames_attempted);
        self.directory_proof.accepted = self
            .directory_proof
            .accepted
            .saturating_add(usize::from(proof.accepted()));
        self.directory_proof.evidence_rejected = self
            .directory_proof
            .evidence_rejected
            .saturating_add(proof.evidence_rejected);
        match proof.result {
            DirectoryProofGossipResult::ReplicaUnavailable => {
                self.directory_proof.replica_unavailable =
                    self.directory_proof.replica_unavailable.saturating_add(1);
            }
            DirectoryProofGossipResult::RateLimited => {
                self.directory_proof.rate_limited =
                    self.directory_proof.rate_limited.saturating_add(1);
            }
            DirectoryProofGossipResult::ProtocolRejected => {
                self.directory_proof.protocol_rejected =
                    self.directory_proof.protocol_rejected.saturating_add(1);
            }
            DirectoryProofGossipResult::TransportFailed => {
                self.directory_proof.transport_failed =
                    self.directory_proof.transport_failed.saturating_add(1);
            }
            DirectoryProofGossipResult::NotAttempted
            | DirectoryProofGossipResult::Accepted
            | DirectoryProofGossipResult::EvidenceRejected => {}
        }

        if let Some(reason) = report.legacy_error {
            self.last_failure_reason = Some(reason);
        } else {
            self.succeeded = self.succeeded.saturating_add(1);
        }
    }

    pub(super) const fn directory_proof_acceptance_percent(&self) -> usize {
        if self.directory_proof.capable == 0 {
            0
        } else {
            self.directory_proof.accepted.saturating_mul(100) / self.directory_proof.capable
        }
    }
}

impl Server {
    // [PHALA-APPRAISAL-EGRESS-PIN 2026-10-07 by Codex] Derive outbound
    // appraisal scope from enabled roles, including recovery-only startup.
    // Invalid private configuration disables appraisal, never public fallback.
    pub(super) fn phala_peer_appraisal_egress_for_config(
        config: &ServerConfig,
    ) -> Option<crate::services::peer_store::PhalaPeerAppraisalEgress> {
        use crate::services::peer_store::PhalaPeerAppraisalEgress;
        if config.reverse_onion.recipient.enabled {
            let mut relay = [0u8; 32];
            let pin = &config.reverse_onion.recipient.relay_node_id;
            if pin.len() != 64 || hex::decode_to_slice(pin, &mut relay).is_err() { return None; }
            return PhalaPeerAppraisalEgress::fixed_relay(
                relay, &config.reverse_onion.recipient.relay_endpoint, true,
            );
        }
        if config.reverse_onion.source.enabled {
            let (relay, _, endpoint) = config.reverse_onion.source.identity_pins().ok()?;
            return PhalaPeerAppraisalEgress::fixed_relay(relay, &endpoint, false);
        }
        Some(PhalaPeerAppraisalEgress::discovery())
    }

    pub(super) fn spawn_discovery_gossip_task(
        &self,
        peer_store: Arc<PeerStore>,
        directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
        chat_relay_runtime_ready: bool,
        blind_vault: Option<Arc<BlindVaultService>>,
        anonymous_mailbox_runtime_ready: bool,
        gossip_http_client: Arc<reqwest::Client>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.discovery.enabled || !self.config.discovery.gossip_enabled {
            return None;
        }

        let config = self.config.clone();
        let identity = self.identity.clone();
        let self_node_id = identity.public_key_bytes();
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        let phala_appraisal_egress = Self::phala_peer_appraisal_egress_for_config(&config);

        Some(tokio::spawn(async move {
            let mut run_immediately = true;
            let mut last_blind_relay_probe_at = 0u64;
            let mut last_two_hop_blind_relay_probe_at = 0u64;
            let mut last_three_hop_blind_relay_probe_at = 0u64;
            // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Keep
            // source delivery bounded by the existing concurrency policy and
            // an eight-target cap; ordinary peer selection remains independent.
            let mut private_authorization_source_cursor = 0usize;
            // [PHALA-AUTHORITY-DESCRIPTOR-EPOCH 2026-10-08 by Codex]
            // The owner retains public bytes only, after PeerStore accepted
            // them. A new task/process starts fresh; no persisted KEM secret
            // or cached self descriptor can restore this local generation.
            let mut private_authority_descriptor_epoch = None;
            // [PHALA-PEER-ATTESTATION-SCHEDULER 2026-10-06 by Codex] Appraise
            // at most one advertised Phala peer per discovery round.
            let mut phala_peer_attestation_cursor = 0usize;
            let mut phala_peer_appraisals = PhalaAppraisalTaskOwner::new();
            'gossip: loop {
                if run_immediately {
                    run_immediately = false;
                    peer_store.record_gossip_schedule(unix_now_secs(), false, 0, 0);
                } else {
                    let schedule_now = unix_now_secs();
                    let consecutive_failures = peer_store.consecutive_gossip_failures();
                    let (delay, backpressure_active, delay_secs, jitter_secs) =
                        Self::discovery_gossip_schedule(
                            &config.discovery,
                            &self_node_id,
                            schedule_now,
                            consecutive_failures,
                        );
                    peer_store.record_gossip_schedule(
                        schedule_now,
                        backpressure_active,
                        delay_secs,
                        jitter_secs,
                    );

                    tokio::select! {
                        _ = rx.recv() => break,
                        _ = tokio::time::sleep(delay) => {}
                    }
                }

                if shutdown.load(Ordering::SeqCst) {
                    break;
                }

                let now = unix_now_secs();
                if config.discovery.phala_attested_peers_required
                    && phala_peer_appraisals.can_start()
                {
                    // [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex]
                    // A fixed recovery relay need not publish public discovery.
                    // Only PeerStore can mint that private appraisal scope;
                    // ordinary peers and Stage-A promotion keep their own gate.
                    if let Some(target) = phala_appraisal_egress.as_ref().and_then(|egress| {
                        peer_store.next_phala_peer_appraisal_target(
                            now, config.discovery.max_peers, phala_peer_attestation_cursor, egress,
                        )
                    }) {
                        let descriptor = target.descriptor().clone();
                        phala_peer_attestation_cursor =
                            phala_peer_attestation_cursor.wrapping_add(1);
                        let appraisal_store = Arc::clone(&peer_store);
                        let trusted_app_ids = config.discovery.phala_trusted_app_ids.clone();
                        let trusted_compose_hashes =
                            config.discovery.phala_trusted_compose_hashes.clone();
                        // [PHALA-APPRAISAL-TASK-OWNERSHIP 2026-10-07 by Codex]
                        let appraisal_stopped = phala_peer_appraisals.stop_signal();
                        let appraisal_shutdown = Arc::clone(&shutdown);
                        phala_peer_appraisals.start(async move {
                            if appraisal_stopped.load(Ordering::Acquire)
                                || appraisal_shutdown.load(Ordering::Acquire) { return; }
                            // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex]
                            // The verifier owns the common timeout/permits. Cache
                            // its challenge age, never restamp network evidence.
                            if let Ok(appraisal) = crate::api::discovery::verify_phala_peer_appraisal_target(
                                &target, &trusted_app_ids, &trusted_compose_hashes,
                            ).await {
                                // Route authority may be draining a DB writer;
                                // do not synchronously wait on the async worker.
                                let _ = tokio::task::spawn_blocking(move || {
                                    appraisal_store.record_verified_phala_peer_attestation_if(
                                        &descriptor, &appraisal, || {
                                            !appraisal_stopped.load(Ordering::SeqCst)
                                                && !appraisal_shutdown.load(Ordering::SeqCst)
                                        },
                                    )
                                }).await;
                            }
                        });
                    }
                }
                // Rotate the onion key on the discovery cadence (no-op until the
                // rotation period elapses). Forward secrecy — see onion_keys.
                // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] No new
                // authority is signed for a failed/rollback key observation.
                if crate::services::onion_keys::try_tick_rotation(now).is_err() {
                    peer_store.record_gossip_round(
                        now, 0, 0, 0, Some("onion_key_epoch_rejected".to_string()),
                    );
                    continue;
                }
                // [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex]
                // Refresh physical/logical admission readiness every gossip
                // round. A stale `BlindVaultReplica` claim could otherwise
                // route fresh replicas to a node that is already fail-closed.
                let blind_vault_runtime_ready =
                    Self::observe_blind_vault_admission_readiness(blind_vault.clone(), now).await;
                // [PRIVATE-ONION-PULL-READINESS 2026-10-05 by Codex] Renew
                // the operation feature only while the read-only terminal is
                // actually reachable; public admission policy is independent.
                // [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
                let private_pull_runtime_ready = if config.reverse_onion.recipient.permits_new_claims() {
                    Self::observe_blind_vault_terminal_readiness(blind_vault.clone()).await
                } else {
                    false
                };
                // [REVERSE-ONION-AUTHORITY-RENEWAL 2026-10-05 by Codex] The
                // current R descriptor/KEM is refreshed normally. P's signed
                // authority is checked per new enqueue against this PeerStore.
                // [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex]
                // Choose the counter before signing nested runtime policies.
                let self_sequence = match Self::private_authority_self_descriptor_sequence(
                    &config, &identity, private_authority_descriptor_epoch.as_ref(), &peer_store, now,
                ) {
                    Ok(sequence) => sequence,
                    Err(_) => {
                        peer_store.record_gossip_round(
                            now, 0, 0, 0, Some("self_descriptor_sequence_rejected".to_string()),
                        );
                        continue;
                    }
                };
                let self_descriptor = match Self::pinned_private_experiment_self_descriptor_for(
                    &config,
                    &identity,
                    now,
                    chat_relay_runtime_ready,
                    blind_vault_runtime_ready,
                    anonymous_mailbox_runtime_ready,
                ) {
                    Ok(Some(descriptor)) => descriptor,
                    Ok(None) => match Self::build_self_discovery_descriptor_for_runtime_state_with_private_pull_and_sequence(
                        &config,
                        &identity,
                        now,
                        chat_relay_runtime_ready,
                        blind_vault_runtime_ready,
                        anonymous_mailbox_runtime_ready,
                        private_pull_runtime_ready,
                        self_sequence,
                    ) {
                        Ok(descriptor) => descriptor,
                        Err(_) => {
                            warn!("[DISCOVERY] Skipping outbound gossip; self descriptor build failed");
                            peer_store.record_gossip_round(
                                now,
                                0,
                                0,
                                0,
                                Some("self_descriptor_build_failed".to_string()),
                            );
                            continue;
                        }
                    },
                    Err(_) => {
                        // [REVERSE-ONION-GOSSIP-FALLBACK 2026-10-04 by Codex]
                        // A stale private experiment must stop only its
                        // private advertisement/admission. Keep ordinary
                        // discovery exchange, directory proofs, and route
                        // probes alive with the normal runtime descriptor.
                        warn!("[DISCOVERY] Private reverse-onion authority unavailable; falling back to ordinary descriptor");
                        match Self::build_self_discovery_descriptor_for_runtime_state_with_private_pull_and_sequence(
                            &config,
                            &identity,
                            now,
                            chat_relay_runtime_ready,
                            blind_vault_runtime_ready,
                            anonymous_mailbox_runtime_ready,
                            private_pull_runtime_ready,
                            self_sequence,
                        ) {
                            Ok(descriptor) => descriptor,
                            Err(_) => {
                                warn!("[DISCOVERY] Skipping outbound gossip; self descriptor build failed");
                                peer_store.record_gossip_round(
                                    now,
                                    0,
                                    0,
                                    0,
                                    Some("self_descriptor_build_failed".to_string()),
                                );
                                continue;
                            }
                        }
                    }
                };
                // [PHALA-AUTHORITY-DESCRIPTOR-EPOCH 2026-10-08 by Codex]
                // Build current readiness/KEM first, then retain an unchanged
                // signed epoch without extending its original expiry.
                // [PHALA-SELF-DESCRIPTOR-SEQUENCE 2026-10-08 by Codex]
                // An independently imported newer local descriptor must not
                // cause the owner to re-publish its older readiness surface.
                let cached_self = peer_store.get_signature_verified_cached(&self_node_id);
                let retained_owner = private_authority_descriptor_epoch.as_ref().filter(|owner|
                    cached_self.as_ref().map_or(true, |cached| cached == *owner));
                let self_descriptor = Self::select_private_authority_descriptor_epoch(
                    &config, self_descriptor, retained_owner, now,
                );
                // [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex]
                // Keep local route selection aligned with the descriptor sent
                // to peers. Otherwise this process could retain its startup
                // capability after the physical or logical gate closed.
                if peer_store
                    .upsert_verified_from_source(self_descriptor.clone(), now, "self")
                    .is_err()
                {
                    warn!(
                        "[DISCOVERY] Skipping outbound gossip; refreshed self descriptor rejected"
                    );
                    peer_store.record_gossip_round(
                        now,
                        0,
                        0,
                        0,
                        Some("self_descriptor_refresh_rejected".to_string()),
                    );
                    continue;
                }

                private_authority_descriptor_epoch = Some(self_descriptor.clone());

                let consecutive_failures = peer_store.consecutive_gossip_failures();
                let backpressure_active = Self::discovery_gossip_backpressure_active(
                    &config.discovery,
                    consecutive_failures,
                );
                let target_policy =
                    DiscoveryGossipTargetPolicy::for_config(&config, backpressure_active);
                let round_peer_limit = target_policy.round_peer_limit;
                let include_cached_peers = target_policy.include_cached_peers;
                let self_gossip_url = config
                    .discovery
                    .public_endpoint
                    .as_deref()
                    .or(config.network.public_endpoint.as_deref())
                    .and_then(Self::discovery_gossip_url);
                let mut seen_urls = HashSet::new();
                let mut gossip_urls = Vec::new();
                let mut gossip_peer_identity_hints = DiscoveryPeerIdentityHints::default();

                if target_policy.include_operator_seeds {
                    for endpoint in &config.discovery.seed_endpoints {
                        let Some(url) = Self::discovery_gossip_url(endpoint) else {
                            continue;
                        };
                        if self_gossip_url.as_deref() == Some(url.as_str()) {
                            continue;
                        }
                        if seen_urls.insert(url.clone()) {
                            gossip_urls.push(url);
                        }
                        if gossip_urls.len() >= round_peer_limit {
                            break;
                        }
                    }
                }

                // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
                // The private recipient's pinned relay endpoint bootstraps the
                // first signed descriptor exchange; later rounds use the exact
                // verified descriptor identity hint above.
                // [PHALA-RECIPIENT-DESCRIPTOR-RECOVERY 2026-10-06 by Codex]
                // Both live and recovery-only workers need this pinned
                // transport to refresh the relay descriptor. Grant issuance
                // remains separately gated by permits_new_claims below.
                if config.reverse_onion.recipient.enabled {
                    if let Some(url) = Self::reverse_onion_private_gossip_url(
                        &config.reverse_onion.recipient.relay_endpoint,
                    ) {
                        let mut relay_id = [0u8; 32];
                        let relay_hint = hex::decode_to_slice(
                            &config.reverse_onion.recipient.relay_node_id, &mut relay_id,
                        ).is_ok();
                        if relay_hint {
                            gossip_peer_identity_hints.observe_private_transport_target(
                                url.clone(), relay_id,
                            );
                            gossip_peer_identity_hints.observe_verified(url.clone(), relay_id);
                        }
                        if self_gossip_url.as_deref() != Some(url.as_str())
                            && seen_urls.insert(url.clone())
                        {
                            gossip_urls.push(url);
                        }
                    }
                }

                // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
                // A source always refreshes through its configured relay
                // identity, even when an expired descriptor is not yet a
                // routeable PeerStore candidate.
                if config.reverse_onion.source.enabled {
                    if let Ok((relay_id, _, endpoint)) =
                        config.reverse_onion.source.identity_pins()
                    {
                        if let Some(url) = Self::reverse_onion_private_gossip_url(&endpoint) {
                            if self_gossip_url.as_deref() != Some(url.as_str()) {
                                gossip_peer_identity_hints.observe_private_transport_target(
                                    url.clone(), relay_id,
                                );
                                gossip_peer_identity_hints.observe_verified(url.clone(), relay_id);
                                if seen_urls.insert(url.clone()) {
                                    gossip_urls.push(url);
                                }
                            }
                        }
                    }
                }

                // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] The
                // configured relay target above also serves recovery. Never
                // append a signed descriptor's rotated endpoint: identity/key
                // renewal is not operator permission to disclose P elsewhere.

                // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
                // [PHALA-REVERSE-AUTHORITY-BOOTSTRAP 2026-10-06 by Codex]
                // The relay forwards P's current signed grant only to a
                // configured source. Descriptor/grant delivery is a gossip
                // bootstrap; the queue still enforces the same current signed
                // R/P/grant snapshot at every durable effect boundary.
                // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                // Keep the selected bounded pin batch immutable for the round.
                let (private_authorization_recipient, private_authorization_source_targets) =
                    if config.reverse_onion.queue.enabled
                        && !config.reverse_onion.queue.recovery_only
                    {
                        // [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex]
                        // Never forward a grant using a partially parsed or
                        // role-overlapping operator allowlist.
                        match config.reverse_onion.queue.live_identity_pins(identity.public_key_bytes()) {
                            Ok(pins) => {
                                let selected = Self::private_authorization_source_batch(
                                    pins.sources(), &mut private_authorization_source_cursor,
                                    usize::from(config.discovery.gossip_concurrency_limit),
                                );
                                (Some(pins.recipient()), selected)
                            }
                            Err(_) => (None, Vec::new()),
                        }
                    } else {
                        (None, Vec::new())
                    };
                for source_id in private_authorization_source_targets.iter().copied() {
                    if let Some(source) = peer_store.get_valid(&source_id, now) {
                        if let Some(endpoint) = source.descriptor.public_endpoint.as_deref() {
                            if let Some(url) = Self::reverse_onion_private_gossip_url(endpoint) {
                                if self_gossip_url.as_deref() != Some(url.as_str()) {
                                    gossip_peer_identity_hints.observe_private_transport_target(
                                        url.clone(), source_id,
                                    );
                                    gossip_peer_identity_hints.observe_verified(url.clone(), source_id);
                                    if seen_urls.insert(url.clone()) {
                                        gossip_urls.push(url);
                                    }
                                }
                            }
                        }
                    }
                }

                let mut gossip_round = DiscoveryGossipRoundAccumulator::default();

                let seed_attempted = gossip_urls.len();

                if include_cached_peers {
                    // [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] Build
                    // identity hints from every valid public descriptor, not
                    // the round-limited selection snapshot. This ensures a
                    // colliding descriptor cannot hide just beyond fan-out.
                    for (peer_node_id, endpoint) in peer_store.valid_public_endpoint_identities(now)
                    {
                        let Some(descriptor) = peer_store.get_valid(&peer_node_id, now) else {
                            continue;
                        };
                        let Some(url) = Self::attested_discovered_peer_gossip_url(
                            &peer_store,
                            &descriptor,
                            now,
                        ) else {
                            continue;
                        };
                        if self_gossip_url.as_deref() == Some(url.as_str()) {
                            continue;
                        }
                        gossip_peer_identity_hints.observe_verified(url, peer_node_id);
                    }

                    // [PERMISSIONLESS-GOSSIP-RUNTIME 2026-09-24 by Codex]
                    // The bootstrap export is sorted and truncated before
                    // transport selection, so low node IDs can monopolize
                    // every round. Sample the complete verified public view
                    // with fresh private entropy, preserving seed priority and
                    // the complete identity-hint collision check above.
                    let mut round_nonce = [0u8; 32];
                    if rand::rngs::OsRng.try_fill_bytes(&mut round_nonce).is_ok() {
                        Self::append_sampled_discovered_peer_gossip_urls(
                            &peer_store,
                            &DiscoveryGossipSampleRequest {
                                now,
                                round_nonce,
                                round_peer_limit,
                                self_node_id: &self_node_id,
                                self_gossip_url: self_gossip_url.as_deref(),
                            },
                            &mut seen_urls,
                            &mut gossip_urls,
                        );
                    } else {
                        // [PERMISSIONLESS-GOSSIP-RUNTIME 2026-09-24 by Codex]
                        // Failed entropy cannot become a predictable peer
                        // rank. Operator seeds still retain this round.
                        warn!("[DISCOVERY] Skipping sampled peers; entropy unavailable");
                    }
                }

                // [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] Keep optional
                // proof publication behind replica convergence while legacy
                // descriptor and snapshot gossip remain immediate.
                let directory_proof_min_age_secs = config
                    .discovery
                    .effective_directory_gossip_proof_min_age_secs();
                let directory_gossip_announcements = if gossip_urls.is_empty() {
                    Vec::new()
                } else if let Some(store) = directory_replica_store.as_ref() {
                    let store = Arc::clone(store);
                    let proof_shutdown = Arc::clone(&shutdown);
                    // [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] Divide by
                    // the cadence so common intervals (for example, 60
                    // seconds) do not preserve the same modulo forever.
                    let selection_seed = now / config.discovery.gossip_interval_secs.max(1);
                    match tokio::task::spawn_blocking(move || {
                        let mut announcements =
                            Vec::with_capacity(DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT);
                        for offset in 0..DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT {
                            // [BACKGROUND-SHUTDOWN-COOPERATION 2026-08-12 by Codex]
                            // A running blocking audit cannot be preempted, but
                            // shutdown must prevent the next complete producer
                            // audit from starting.
                            if proof_shutdown.load(Ordering::Acquire) {
                                break;
                            }
                            let candidate = store.audited_live_descriptor_gossip_announcement(
                                now,
                                directory_proof_min_age_secs,
                                selection_seed.saturating_add(offset as u64),
                            )?;
                            let Some(candidate) = candidate else {
                                continue;
                            };
                            if announcements.iter().any(
                                |existing: &DirectoryReplicaGossipAnnouncement| {
                                    existing.producer == candidate.producer
                                },
                            ) {
                                continue;
                            }
                            announcements.push(candidate);
                        }
                        Ok::<_, crate::services::DirectoryReplicaStoreError>(announcements)
                    })
                    .await
                    {
                        Ok(Ok(announcement)) => announcement,
                        Ok(Err(_)) => {
                            // The error text may contain persistence details.
                            // Keep the network log coarse and identity-free.
                            debug!("[DISCOVERY] Directory gossip proof selection failed closed");
                            Vec::new()
                        }
                        Err(error) => {
                            warn!(
                                task_cancelled = error.is_cancelled(),
                                task_panicked = error.is_panic(),
                                "[DISCOVERY] Directory gossip proof selection worker failed"
                            );
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };
                if shutdown.load(Ordering::Acquire) {
                    break 'gossip;
                }

                let gossip_started_at = tokio::time::Instant::now();
                let concurrency_limit = usize::from(config.discovery.gossip_concurrency_limit);
                let peer_timeout = Duration::from_secs(config.discovery.fetch_timeout_secs);
                let execution = DiscoveryGossipExecution {
                    client: gossip_http_client.as_ref(),
                    peer_store: &peer_store,
                    local_identity: Some(&identity),
                    private_recipient: config.reverse_onion.recipient.enabled
                        .then_some(&config.reverse_onion.recipient),
                    // [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
                    private_authorization_target: if config.reverse_onion.recipient.permits_new_claims() {
                        let mut target = [0u8; 32];
                        hex::decode_to_slice(&config.reverse_onion.recipient.relay_node_id, &mut target)
                            .ok().map(|()| target)
                    } else { None },
                    // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                    private_authorization_source_targets: &private_authorization_source_targets,
                    private_authorization_recipient,
                    directory_announcements: &directory_gossip_announcements,
                    peer_identity_hints: Some(&gossip_peer_identity_hints),
                    now,
                    snapshot_limit: config.discovery.gossip_peer_limit,
                    peer_timeout,
                };
                let reports = Self::gossip_with_peers_bounded(
                    execution,
                    gossip_urls,
                    &self_descriptor,
                    concurrency_limit,
                )
                .await;
                let gossip_elapsed_ms = gossip_started_at.elapsed().as_millis();

                for report in reports {
                    if let Some(reason) = report.legacy_error.as_ref() {
                        debug!(
                            reason = %reason,
                            backpressure_active, "[DISCOVERY] Outbound gossip peer sync failed"
                        );
                    }
                    gossip_round.observe(report);
                }

                if gossip_round.attempted > 0 {
                    let proof = gossip_round.directory_proof;
                    let directory_proof_acceptance_percent =
                        gossip_round.directory_proof_acceptance_percent();
                    info!(
                        attempted = gossip_round.attempted,
                        succeeded = gossip_round.succeeded,
                        directory_proof_capability_checked = proof.capability_checked,
                        directory_proof_capable = proof.capable,
                        directory_proof_attempted = proof.peers_attempted,
                        directory_proof_frames_attempted = proof.frames_attempted,
                        directory_proof_accepted = proof.accepted,
                        directory_proof_acceptance_percent,
                        directory_proof_evidence_rejected = proof.evidence_rejected,
                        directory_proof_replica_unavailable = proof.replica_unavailable,
                        directory_proof_rate_limited = proof.rate_limited,
                        directory_proof_protocol_rejected = proof.protocol_rejected,
                        directory_proof_transport_failed = proof.transport_failed,
                        directory_proof_min_age_secs,
                        concurrency_limit,
                        peer_timeout_secs = config.discovery.fetch_timeout_secs,
                        gossip_elapsed_ms,
                        backpressure_active,
                        "[DISCOVERY] Outbound gossip round complete"
                    );
                }
                peer_store.record_directory_proof_gossip_round(now, gossip_round.directory_proof);
                peer_store.record_gossip_round(
                    now,
                    gossip_round.attempted,
                    gossip_round.succeeded,
                    seed_attempted,
                    gossip_round
                        .last_failure_reason
                        .map(DiscoveryGossipFailure::bucket),
                );
                if shutdown.load(Ordering::Acquire) {
                    break 'gossip;
                }
                if chat_relay_runtime_ready
                    && !config.reverse_onion.recipient.enabled
                    && gossip_round.succeeded > 0
                {
                    let probe_now = unix_now_secs();
                    let probe_cooldown_secs = Self::blind_relay_probe_cooldown_secs_for_status(
                        &config.discovery,
                        &peer_store,
                        probe_now,
                    );
                    let probe_due = last_blind_relay_probe_at == 0
                        || probe_now.saturating_sub(last_blind_relay_probe_at)
                            >= probe_cooldown_secs;

                    if probe_due {
                        let max_candidates = if last_blind_relay_probe_at == 0 {
                            BLIND_RELAY_STARTUP_WARMUP_MAX_CANDIDATES
                        } else {
                            1
                        };
                        let probes_started = Self::probe_blind_relay_candidates(
                            gossip_http_client.as_ref(),
                            &peer_store,
                            &identity,
                            &self_node_id,
                            probe_now,
                            max_candidates,
                        )
                        .await;
                        if shutdown.load(Ordering::Acquire) {
                            break 'gossip;
                        }
                        if probes_started > 0 {
                            last_blind_relay_probe_at = probe_now;
                        }
                    } else {
                        trace!(
                            cooldown_secs = probe_cooldown_secs,
                            last_probe_age_secs =
                                probe_now.saturating_sub(last_blind_relay_probe_at),
                            "[DISCOVERY] Blind relay synthetic probe skipped during cooldown"
                        );
                    }

                    let two_hop_probe_due = last_two_hop_blind_relay_probe_at == 0
                        || probe_now.saturating_sub(last_two_hop_blind_relay_probe_at)
                            >= probe_cooldown_secs;

                    if two_hop_probe_due {
                        let outcome = Self::probe_two_hop_blind_relay_path(
                            gossip_http_client.as_ref(),
                            &peer_store,
                            &identity,
                            &self_node_id,
                            probe_now,
                        )
                        .await;
                        if shutdown.load(Ordering::Acquire) {
                            break 'gossip;
                        }
                        if outcome.attempted {
                            last_two_hop_blind_relay_probe_at = probe_now;
                        }

                        // [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex]
                        // Three-hop traffic is attempted only after this same
                        // round has cryptographically verified a two-hop
                        // terminal receipt. This bounds cold-start fan-out and
                        // avoids using an unhealthy mesh as a deeper probe.
                        let three_hop_cooldown_secs = probe_cooldown_secs.saturating_mul(2);
                        let three_hop_probe_due = last_three_hop_blind_relay_probe_at == 0
                            || probe_now.saturating_sub(last_three_hop_blind_relay_probe_at)
                                >= three_hop_cooldown_secs;
                        if outcome.terminal_delivery_verified && three_hop_probe_due {
                            let three_hop_outcome = Self::probe_three_hop_blind_relay_path(
                                gossip_http_client.as_ref(),
                                &peer_store,
                                &identity,
                                &self_node_id,
                                probe_now,
                            )
                            .await;
                            if shutdown.load(Ordering::Acquire) {
                                break 'gossip;
                            }
                            if three_hop_outcome.attempted {
                                last_three_hop_blind_relay_probe_at = probe_now;
                            }
                        }
                    } else {
                        trace!(
                            cooldown_secs = probe_cooldown_secs,
                            last_probe_age_secs =
                                probe_now.saturating_sub(last_two_hop_blind_relay_probe_at),
                            "[DISCOVERY] Two-hop blind relay synthetic proof skipped during cooldown"
                        );
                    }
                }
            }
            phala_peer_appraisals.stop_and_join().await;
        }))
    }

    pub(super) fn blind_relay_probe_cooldown_secs(discovery: &DiscoveryConfig) -> u64 {
        discovery
            .gossip_interval_secs
            .saturating_mul(3)
            .max(BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS)
    }

    pub(super) fn blind_relay_probe_recovery_cooldown_secs(discovery: &DiscoveryConfig) -> u64 {
        discovery.gossip_interval_secs.clamp(
            BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS,
            BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS,
        )
    }

    pub(super) fn blind_relay_probe_cooldown_secs_for_status(
        discovery: &DiscoveryConfig,
        peer_store: &PeerStore,
        now: u64,
    ) -> u64 {
        let status = peer_store.status(now);
        let proof = &status.two_hop_path_proof_history;
        let two_hop_delivery_ready =
            proof.recent_message_delivery_ready && !proof.failure_streak_active;
        let two_hop_stability_ready =
            proof.stability_ready && !proof.failure_circuit_breaker_active;

        if status.blind_relay_quality.quality_ready
            && two_hop_delivery_ready
            && two_hop_stability_ready
        {
            Self::blind_relay_probe_cooldown_secs(discovery)
        } else {
            Self::blind_relay_probe_recovery_cooldown_secs(discovery)
        }
    }

    pub(super) fn prioritize_probe_candidates(
        peer_store: &PeerStore,
        now: u64,
        candidates: &mut [SignedNodeDescriptor],
    ) {
        candidates.sort_by_key(|candidate| {
            let node_id = candidate.node_id();
            let route_quarantined = peer_store.is_route_quarantined_now(&node_id, now);
            let routeable = peer_store.is_routeable_now(&node_id, now);
            // Low-frequency probes should expand coverage first: fresh signed
            // peers with unknown/stale routeability are tried before peers that
            // are already proven. Quarantined peers stay last so local failure
            // isolation remains stronger than coverage convergence.
            (route_quarantined, routeable)
        });
    }

    pub(super) fn discovery_gossip_backpressure_active(
        discovery: &DiscoveryConfig,
        consecutive_failures: u64,
    ) -> bool {
        consecutive_failures >= discovery.gossip_backpressure_failure_threshold
    }

    pub(super) fn discovery_gossip_schedule(
        discovery: &DiscoveryConfig,
        self_node_id: &[u8],
        now: u64,
        consecutive_failures: u64,
    ) -> (Duration, bool, u64, i64) {
        let base_secs = discovery.gossip_interval_secs.max(1);
        let backpressure_active =
            Self::discovery_gossip_backpressure_active(discovery, consecutive_failures);
        let backoff_multiplier = if backpressure_active {
            let backoff_steps = consecutive_failures
                .saturating_sub(discovery.gossip_backpressure_failure_threshold)
                .min(5);
            1u64 << backoff_steps
        } else {
            1
        };
        let raw_delay_secs = base_secs
            .saturating_mul(backoff_multiplier)
            .min(discovery.gossip_failure_backoff_max_secs.max(base_secs));
        let jitter_secs = Self::discovery_gossip_jitter_seconds(
            raw_delay_secs,
            discovery.gossip_jitter_percent,
            now,
            self_node_id,
            consecutive_failures,
        );
        let delayed = i128::from(raw_delay_secs) + i128::from(jitter_secs);
        let delay_secs = delayed.clamp(
            1,
            i128::from(discovery.gossip_failure_backoff_max_secs.max(base_secs)),
        ) as u64;

        (
            Duration::from_secs(delay_secs),
            backpressure_active,
            delay_secs,
            jitter_secs,
        )
    }

    pub(super) fn discovery_gossip_jitter_seconds(
        base_secs: u64,
        jitter_percent: u8,
        now: u64,
        self_node_id: &[u8],
        consecutive_failures: u64,
    ) -> i64 {
        let window = base_secs.saturating_mul(u64::from(jitter_percent)) / 100;
        if window == 0 {
            return 0;
        }

        let period = now / base_secs.max(1);
        let mut mixed = period ^ consecutive_failures.rotate_left(13) ^ base_secs.rotate_left(7);
        for byte in self_node_id.iter().take(16) {
            mixed = mixed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(u64::from(*byte));
        }
        let span = window.saturating_mul(2).saturating_add(1);
        (mixed % span) as i64 - window as i64
    }

    pub(super) fn discovery_summary_url_from_gossip_url(gossip_url: &str) -> Option<String> {
        const GOSSIP_SUFFIX: &str = "/api/discovery/gossip";
        let base = gossip_url.strip_suffix(GOSSIP_SUFFIX)?;
        if base.is_empty() {
            return None;
        }
        Some(format!("{base}/api/discovery/summary"))
    }

    /// Returns whether a peer explicitly advertises Directory proof gossip V1.
    ///
    /// [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] A valid feature value
    /// remains only an optimization hint. Transport and malformed-response
    /// failures are now typed separately from a valid legacy-only response so
    /// fleet convergence telemetry cannot mislabel an unavailable peer.
    pub(super) async fn peer_supports_directory_proof_gossip(
        client: &reqwest::Client,
        gossip_url: &str,
    ) -> std::result::Result<bool, DirectoryProofGossipResult> {
        let Some(summary_url) = Self::discovery_summary_url_from_gossip_url(gossip_url) else {
            return Err(DirectoryProofGossipResult::ProtocolRejected);
        };
        let features = Self::peer_discovery_negotiation_features(client, &summary_url).await?;
        Ok(features.directory_descriptor_proof_gossip_v1)
    }

    /// Reads one bounded unsigned feature document from a public peer.
    ///
    /// [THREE-HOP-FEATURE-NEGOTIATION 2026-08-02 by Codex] Sharing this reader
    /// keeps optional Directory proof and multihop receipt negotiation under
    /// the same response ceiling and fail-closed decoding rules. The returned
    /// hints never grant identity, route, witness, or delivery authority.
    pub(super) async fn peer_discovery_negotiation_features(
        client: &reqwest::Client,
        summary_url: &str,
    ) -> std::result::Result<DiscoveryNegotiationFeatures, DirectoryProofGossipResult> {
        let response = client
            .get(summary_url)
            .send()
            .await
            .map_err(|_| DirectoryProofGossipResult::TransportFailed)?;
        if !response.status().is_success() {
            return Err(DirectoryProofGossipResult::from_http_status(
                response.status(),
            ));
        }
        let summary = decode_bounded_json_response::<DiscoveryNegotiationSummary>(
            response,
            DISCOVERY_NEGOTIATION_SUMMARY_MAX_BYTES,
        )
        .await
        .map_err(|_| DirectoryProofGossipResult::ProtocolRejected)?;
        Ok(summary.protocol_features)
    }

    /// Returns true only when a signed descriptor's public endpoint advertises
    /// purpose-bound version-2 terminal receipts.
    ///
    /// [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] The summary is
    /// an unsigned bootstrap hint. It can suppress an optional v2 probe, but it
    /// never populates route-authority evidence; only a verified v2 receipt can.
    pub(super) async fn peer_advertises_purpose_bound_delivery_receipt(
        client: &reqwest::Client,
        endpoint: &str,
    ) -> bool {
        let Some(summary_url) =
            Self::permissionless_peer_transport_url(endpoint, "/api/discovery/summary")
        else {
            return false;
        };
        Self::peer_discovery_negotiation_features(client, &summary_url)
            .await
            .map(|features| {
                features.multihop_delivery_receipt_v1 && features.purpose_bound_delivery_receipt_v2
            })
            .unwrap_or(false)
    }

    /// Returns candidates with fresh verified v2 evidence or an explicit v2
    /// bootstrap claim. Cryptographic process-local evidence and the exact
    /// signed descriptor both avoid an unnecessary unsigned network request.
    /// The legacy summary remains a compatibility hint only. Input health order
    /// is preserved by `buffered`; negotiation cannot become a hidden
    /// route-ranking signal.
    pub(super) async fn purpose_bound_delivery_receipt_advertisers(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        candidates: &[SignedNodeDescriptor],
        now: u64,
    ) -> HashSet<[u8; 32]> {
        let mut supported = candidates
            .iter()
            .filter(|candidate| {
                // [SIGNED-RECEIPT-NEGOTIATION 2026-08-11 by Codex] A signed
                // claim grants only permission to send the optional v2 probe.
                // The returned receipt remains the sole route-authority proof.
                candidate
                    .descriptor
                    .advertises_protocol_feature(NodeProtocolFeature::PurposeBoundDeliveryReceiptV2)
                    || peer_store.has_fresh_purpose_bound_delivery_receipt_capability(
                        &candidate.node_id(),
                        now,
                    )
            })
            .map(SignedNodeDescriptor::node_id)
            .collect::<HashSet<_>>();
        let unknown = candidates
            .iter()
            .filter(|candidate| !supported.contains(&candidate.node_id()))
            .cloned()
            .collect::<Vec<_>>();
        let advertised = futures::stream::iter(unknown)
            .map(|candidate| async move {
                let advertised = match candidate.descriptor.public_endpoint.as_deref() {
                    Some(endpoint) => tokio::time::timeout(
                        DISCOVERY_NEGOTIATION_HINT_TIMEOUT,
                        Self::peer_advertises_purpose_bound_delivery_receipt(client, endpoint),
                    )
                    .await
                    .unwrap_or(false),
                    None => false,
                };
                advertised.then(|| candidate.node_id())
            })
            .buffered(ONION_ROUTE_SELECTION_CANDIDATE_LIMIT)
            .filter_map(|node_id| async move { node_id })
            .collect::<HashSet<_>>()
            .await;
        supported.extend(advertised);
        supported
    }

    /// Runs one peer exchange under a total lifetime budget.
    ///
    /// Optional proof work receives at most one third of the budget. Any unused
    /// proof time is carried into mandatory legacy synchronization, while a
    /// stalled proof can never consume the compatibility reserve.
    pub(super) async fn gossip_with_peer(
        execution: DiscoveryGossipExecution<'_>,
        url: &str,
        self_descriptor: SignedNodeDescriptor,
    ) -> DiscoveryPeerGossipReport {
        // [GOSSIP-OUTCOME-INTEGRITY 2026-07-28 by Codex] These exchanges have
        // independent outcome domains. Proof delivery is optional evidence;
        // descriptor/snapshot exchange remains the compatibility-critical path.
        let started_at = tokio::time::Instant::now();
        let proof_budget = (execution.peer_timeout / 3).max(Duration::from_millis(1));
        let peer_node_id = execution
            .peer_identity_hints
            .and_then(|hints| hints.unique_node_id(url));
        let transport_pin = execution.peer_identity_hints
            .map(|hints| hints.transport_pin_for(url, peer_node_id))
            .unwrap_or(GossipTransportPin::Generic);
        // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Fence the actual
        // request boundary too, before DNS, proofs, or private announcement.
        // This is independent of live grant issuance, so recovery cannot
        // accidentally fall back to generic gossip transport.
        let private_egress_rejected = execution.private_recipient.is_some_and(|recipient| {
            !Self::private_recipient_gossip_target_allowed(recipient, peer_node_id, url)
        });
        if transport_pin == GossipTransportPin::Reject || private_egress_rejected {
            return DiscoveryPeerGossipReport {
                directory_proof: DirectoryProofGossipOutcome::negotiation_failed(
                    DirectoryProofGossipResult::TransportFailed,
                ),
                legacy_error: Some(DiscoveryGossipFailure::peer_timeout()),
            };
        }
        // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Only
        // selected pinned sources get the private transport path this round.
        let private_target = peer_node_id.is_some_and(|peer_id| {
            execution.private_authorization_target == Some(peer_id)
                || execution.private_authorization_source_targets.contains(&peer_id)
        });
        // [PHALA-DNS-SEED-PIN 2026-10-07 by Codex] Operator seeds have no
        // descriptor identity hint, but HTTPS DNS must still resolve to a
        // wholly public answer set before the generic gossip client connects.
        let dns_target_requires_pin = reqwest::Url::parse(url)
            .is_ok_and(|parsed| crate::api::peer_http_target_requires_dns_pin(&parsed));
        let pinned_target = if matches!(transport_pin, GossipTransportPin::Pinned(_))
            || private_target
            || dns_target_requires_pin
        {
            let parsed = reqwest::Url::parse(url).ok();
            match parsed {
                Some(url) => match crate::api::resolve_pinned_peer_http_target(
                    url, execution.peer_timeout,
                ).await {
                    Ok(target) => Some(target),
                    Err(_) => return DiscoveryPeerGossipReport {
                        directory_proof: DirectoryProofGossipOutcome::negotiation_failed(
                            DirectoryProofGossipResult::TransportFailed,
                        ),
                        legacy_error: Some(DiscoveryGossipFailure::peer_timeout()),
                    },
                },
                None => return DiscoveryPeerGossipReport {
                    directory_proof: DirectoryProofGossipOutcome::negotiation_failed(
                        DirectoryProofGossipResult::TransportFailed,
                    ),
                    legacy_error: Some(DiscoveryGossipFailure::peer_timeout()),
                },
            }
        } else {
            None
        };
        let client = pinned_target.as_ref()
            .map(|target| &target.client)
            .unwrap_or(execution.client);
        let peer_url = pinned_target.as_ref()
            .map(|target| target.url.as_str())
            .unwrap_or(url);
        let remaining_proof_budget = proof_budget.saturating_sub(started_at.elapsed());
        let directory_proof = if execution.directory_announcements.is_empty()
            || remaining_proof_budget.is_zero()
        {
            DirectoryProofGossipOutcome::default()
        } else {
            tokio::time::timeout(
                remaining_proof_budget,
                Self::gossip_directory_proofs_with_peer(
                    client,
                    peer_url,
                    execution.directory_announcements,
                    peer_node_id,
                ),
            )
            .await
            .unwrap_or_else(|_| {
                DirectoryProofGossipOutcome::negotiation_failed(
                    DirectoryProofGossipResult::TransportFailed,
                )
            })
        };
        let legacy_budget = execution.peer_timeout.saturating_sub(started_at.elapsed());
        let legacy_error = tokio::time::timeout(
            legacy_budget,
            Self::gossip_legacy_with_peer(
                client,
                execution.peer_store,
                peer_url,
                self_descriptor,
                execution.now,
                execution.snapshot_limit,
            ),
        )
        .await
        .map_or_else(
            |_| Some(DiscoveryGossipFailure::peer_timeout()),
            |result| result.err(),
        );

        // [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex] P issues the
        // fresh grant to its configured R; R later forwards the exact signed
        // grant only to selected source identities in its configured allowlist.
        // Both directions require the target's signed support feature and a
        // successful legacy descriptor exchange first.
        if legacy_error.is_none() {
            // [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex] Legacy
            // exchange/DNS may consume the round's timestamp. Renewal starts
            // from a new wall-clock sample; rollback never mints a fresh grant.
            let authority_now = unix_now_secs();
            if authority_now == 0 || authority_now < execution.now {
                return DiscoveryPeerGossipReport { directory_proof, legacy_error };
            }
            if let (Some(local_identity), Some(peer_id)) =
                (execution.local_identity, peer_node_id)
            {
                let peer = execution.peer_store.get_valid(&peer_id, authority_now);
                let supports_grant_gossip = peer.as_ref().is_some_and(|descriptor| {
                    descriptor.descriptor.advertises_protocol_feature(
                        NodeProtocolFeature::PrivateOnionAuthorizationGossipV1,
                    )
                });
                if supports_grant_gossip {
                    // [PRIVATE-ONION-AUTHORITY-PURPOSES 2026-10-05 by Codex]
                    // Keep each workload grant independent end-to-end; never
                    // replace a Pull grant with admission authority in cache.
                    let purposes = [
                        OnionRoutePurpose::BlindVaultPull.as_str(),
                        OnionRoutePurpose::BlindVaultLeaseAdmission.as_str(),
                    ];
                    let issued: Vec<_> = if execution.private_authorization_target == Some(peer_id) {
                        // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                        // Reuse an ACK-confirmed current grant verbatim rather
                        // than revoking it with a fresh signature each heartbeat.
                        purposes.into_iter().filter_map(|purpose| {
                            Self::private_recipient_authorization_for_gossip(
                                execution.peer_store, local_identity, peer_id,
                                purpose, authority_now, unix_now_secs,
                            ).filter(|(_, relay, _, _)| {
                                relay.descriptor.public_endpoint.as_deref()
                                    .and_then(Self::reverse_onion_private_gossip_url)
                                    .as_deref() == Some(peer_url)
                            })
                        }).collect()
                    } else if execution.private_authorization_source_targets.contains(&peer_id) {
                        // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                        // Batch membership selects transport, not authority.
                        let selected_source_is_current = execution.peer_store
                            .get_valid(&peer_id, authority_now)
                            .and_then(|source| source.descriptor.public_endpoint)
                            .and_then(|endpoint| Self::reverse_onion_private_gossip_url(&endpoint))
                            .as_deref() == Some(peer_url);
                        execution.private_authorization_recipient
                            .filter(|_| selected_source_is_current)
                            .map(|recipient_id| {
                                // [REVERSE-ONION-AUTHORITY-SNAPSHOT-GOSSIP 2026-10-05 by Codex]
                                // Fetch R, P and the cached P grant under one PeerStore
                                // authority epoch. Never combine independently read
                                // generations into a bundle that the receiver rejects.
                                purposes.into_iter().filter_map(|purpose| {
                                    execution.peer_store
                                        .current_private_onion_authority_snapshot_for_purpose(
                                            &local_identity.public_key_bytes(),
                                            &recipient_id,
                                            purpose,
                                            authority_now,
                                        )
                                        .map(|(relay, recipient, authorization)| {
                                            // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                                            // Retain the fresh lookup's lower clock bound.
                                            (authorization, relay, recipient, authority_now)
                                        })
                                }).collect()
                            }).unwrap_or_default()
                    } else {
                        Vec::new()
                    };
                    // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex]
                    // Cached grant age never replaces this attempt's clock floor.
                    for (authorization, relay_descriptor, recipient_descriptor, selected_at) in issued {
                        let message = NodeDiscoveryMessage::PrivateOnionRecipientAuthorizationV1 {
                            authorization: authorization.clone(),
                            relay_descriptor,
                            recipient_descriptor,
                        };
                        // [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex]
                        // Finish request construction before the final exact
                        // target/R/P/grant check. Never reselect after a wait.
                        let Some(selected_peer) = peer.as_ref() else { break; };
                        let Ok(request) = client.post(peer_url).json(&message).build() else { break; };
                        let Some(sent_at) = Self::private_authorization_gossip_admitted_at(
                            execution.peer_store, local_identity.public_key_bytes(), selected_peer,
                            &message, peer_url, selected_at, unix_now_secs,
                        ) else { break; };
                        let remaining = execution.peer_timeout.saturating_sub(started_at.elapsed());
                        if !remaining.is_zero() {
                            if let Ok(Ok(response)) = tokio::time::timeout(
                                remaining,
                                client.execute(request),
                            ).await {
                                if response.status().is_success()
                                    && authorization.recipient_node_id()
                                        == local_identity.public_key_bytes()
                                {
                                    // [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex]
                                    // ACK does not freeze route validity. The
                                    // cache writer rechecks current R/P under
                                    // its own epoch using this new timestamp.
                                    let _ = execution.peer_store
                                        .try_remember_issued_private_onion_authorization_at(
                                            authorization,
                                            local_identity.public_key_bytes(),
                                            sent_at,
                                            unix_now_secs,
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }

        DiscoveryPeerGossipReport {
            directory_proof,
            legacy_error,
        }
    }

    /// Runs selected peer exchanges concurrently under a strict fan-out cap.
    ///
    /// Completion order is intentionally discarded: reports are restored to
    /// selection order before aggregation so the historical "last selected
    /// failure" status behavior remains deterministic across runtime timing.
    pub(super) async fn gossip_with_peers_bounded(
        execution: DiscoveryGossipExecution<'_>,
        gossip_urls: Vec<String>,
        self_descriptor: &SignedNodeDescriptor,
        concurrency_limit: usize,
    ) -> Vec<DiscoveryPeerGossipReport> {
        let concurrency_limit = concurrency_limit.max(1).min(gossip_urls.len().max(1));
        let mut reports = futures::stream::iter(gossip_urls.into_iter().enumerate())
            .map(|(index, url)| {
                    let self_descriptor = self_descriptor.clone();
                    async move {
                    let report = Self::gossip_with_peer(execution, &url, self_descriptor).await;
                    (index, report)
                }
            })
            .buffer_unordered(concurrency_limit)
            .collect::<Vec<_>>()
            .await;
        reports.sort_unstable_by_key(|(index, _)| *index);
        reports.into_iter().map(|(_, report)| report).collect()
    }

    /// Sends bounded optional Directory proof frames to one capable peer.
    ///
    /// This function never runs or suppresses the mandatory legacy exchange.
    /// Its output is retained independently by `DiscoveryPeerGossipReport`.
    pub(super) async fn gossip_directory_proofs_with_peer(
        client: &reqwest::Client,
        url: &str,
        directory_gossip_announcements: &[DirectoryReplicaGossipAnnouncement],
        peer_node_id: Option<[u8; 32]>,
    ) -> DirectoryProofGossipOutcome {
        let mut outcome = DirectoryProofGossipOutcome::default();
        if !directory_gossip_announcements.is_empty() {
            match Self::peer_supports_directory_proof_gossip(client, url).await {
                Ok(false) => {
                    outcome.state = DirectoryProofGossipPeerState::LegacyOnly;
                }
                Err(result) => {
                    outcome = DirectoryProofGossipOutcome::negotiation_failed(result);
                }
                Ok(true) => {
                    outcome.state = DirectoryProofGossipPeerState::Attempted;
                    // [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] Only
                    // an exact-evidence miss advances to one alternate audited
                    // proof. Other failures stop optional work.
                    for announcement in directory_gossip_announcements
                        .iter()
                        .filter(|announcement| peer_node_id != Some(announcement.producer))
                        .take(DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT)
                    {
                        outcome.frames_attempted += 1;
                        let proof_message = NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
                            producer: announcement.producer,
                            block_hash: announcement.block_hash,
                            descriptor_hash: announcement.descriptor_hash,
                            proof: announcement.proof.clone(),
                        };
                        let result = match client.post(url).json(&proof_message).send().await {
                            Ok(response) if response.status().is_success() => {
                                DirectoryProofGossipResult::Accepted
                            }
                            Ok(response) => {
                                DirectoryProofGossipResult::from_http_status(response.status())
                            }
                            Err(_) => DirectoryProofGossipResult::TransportFailed,
                        };
                        outcome.result = result;
                        match result {
                            DirectoryProofGossipResult::EvidenceRejected => {
                                outcome.evidence_rejected += 1;
                            }
                            DirectoryProofGossipResult::Accepted
                            | DirectoryProofGossipResult::NotAttempted
                            | DirectoryProofGossipResult::ReplicaUnavailable
                            | DirectoryProofGossipResult::RateLimited
                            | DirectoryProofGossipResult::ProtocolRejected
                            | DirectoryProofGossipResult::TransportFailed => break,
                        }
                    }
                }
            }
        }

        outcome
    }

    /// Executes the backward-compatible descriptor and snapshot exchange.
    ///
    /// Successful completion is the only path that marks `last_gossip_at`.
    /// Errors remain bounded phase buckets and contain no endpoint or identity.
    pub(super) async fn gossip_legacy_with_peer(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        url: &str,
        self_descriptor: SignedNodeDescriptor,
        now: u64,
        limit: u16,
    ) -> std::result::Result<(), DiscoveryGossipFailure> {
        let announce_response = client
            .post(url)
            .json(&NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: self_descriptor,
            })
            .send()
            .await
            .map_err(|error| {
                DiscoveryGossipFailure::from_reqwest(DiscoveryGossipPhase::AnnounceRequest, &error)
            })?;

        announce_response.error_for_status().map_err(|error| {
            DiscoveryGossipFailure::from_reqwest(DiscoveryGossipPhase::AnnounceStatus, &error)
        })?;

        let snapshot_response = client
            .post(url)
            .json(&NodeDiscoveryMessage::SnapshotRequest {
                requested_at: now,
                limit: Some(limit),
            })
            .send()
            .await
            .map_err(|error| {
                DiscoveryGossipFailure::from_reqwest(DiscoveryGossipPhase::SnapshotRequest, &error)
            })?;

        let snapshot_response = snapshot_response.error_for_status().map_err(|error| {
            DiscoveryGossipFailure::from_reqwest(DiscoveryGossipPhase::SnapshotStatus, &error)
        })?;
        let response = decode_bounded_json_response::<GossipResponse>(
            snapshot_response,
            DISCOVERY_GOSSIP_RESPONSE_MAX_BYTES,
        )
        .await
        .map_err(|error| {
            DiscoveryGossipFailure::from_bounded_response(
                DiscoveryGossipPhase::SnapshotResponse,
                error,
            )
        })?;

        if let Some(message) = response.response {
            peer_store.apply_discovery_message(&message, now);
        }
        peer_store.mark_gossip_at(now);

        Ok(())
    }

    /// Directly probes a bounded set of signed ChatRelay candidates.
    ///
    /// The first recovery round may cover up to three peers concurrently so a
    /// small node mesh does not spend multiple gossip intervals in a partially
    /// routeable state after restart. The hard clamp prevents peer-count-based
    /// request amplification; steady-state callers pass one.
    pub(super) async fn probe_blind_relay_candidates(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        now: u64,
        max_candidates: usize,
    ) -> usize {
        if max_candidates == 0 {
            return 0;
        }
        let mut candidates = peer_store.route_probe_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            8,
            &[*self_node_id],
        );
        Self::prioritize_probe_candidates(peer_store, now, &mut candidates);
        let candidates = candidates
            .into_iter()
            .take(max_candidates.min(BLIND_RELAY_STARTUP_WARMUP_MAX_CANDIDATES))
            .collect::<Vec<_>>();
        let attempted = candidates.len();
        let probes = candidates.into_iter().map(|candidate| {
            Self::probe_blind_relay_candidate_descriptor(
                client,
                peer_store,
                identity,
                self_node_id,
                candidate,
                now,
                false,
            )
        });
        futures::future::join_all(probes).await;
        attempted
    }

    pub(super) async fn probe_blind_relay_candidate_descriptor(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        candidate: SignedNodeDescriptor,
        now: u64,
        require_signed_terminal_receipt: bool,
    ) {
        let next_hop = candidate.node_id();
        let Some(endpoint) = candidate.descriptor.public_endpoint.as_deref() else {
            peer_store.record_blind_relay_probe_result(now, false, "missing_endpoint");
            let _ = peer_store.record_route_forward_failure_for_descriptor(
                &candidate,
                now,
                "missing_endpoint",
            );
            return;
        };
        // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex] Only
        // signed promotion control probes may enter the appraised DNS lane;
        // legacy warmup/probe URL policy remains public-IP-only.
        let dns_promotion_probe = require_signed_terminal_receipt
            && !peer_endpoint_is_public_ip(endpoint)
            && !is_test_loopback_peer_endpoint(endpoint);
        let probe_url = if dns_promotion_probe {
            Self::promotion_blind_relay_probe_url(peer_store, &candidate, now)
        } else {
            Self::blind_relay_probe_url(endpoint)
        };
        let Some(mut url) = probe_url else {
            // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-08 by Codex]
            // Local quote expiry/lock contention is not a peer failure.
            if dns_promotion_probe {
                peer_store.record_blind_relay_probe_result(now, false, "promotion_unavailable");
                return;
            }
            peer_store.record_blind_relay_probe_result(now, false, "invalid_endpoint");
            let _ = peer_store.record_route_forward_failure_for_descriptor(
                &candidate,
                now,
                "invalid_endpoint",
            );
            return;
        };
        let mut probe_client = client.clone();
        if dns_promotion_probe {
            let Some((pinned_client, pinned_url)) = Self::resolve_onion_middle_blind_relay_target(
                endpoint, client,
            ).await else {
                peer_store.record_blind_relay_probe_result(now, false, "dns_target_unavailable");
                return;
            };
            probe_client = pinned_client;
            url = pinned_url;
        }

        let promotion_route_id = if require_signed_terminal_receipt {
            let mut route_id = [0u8; 16];
            if rand::rngs::OsRng.try_fill_bytes(&mut route_id).is_err() {
                peer_store.record_blind_relay_probe_result(now, false, "nonce_unavailable");
                return;
            }
            Some(route_id)
        } else {
            None
        };
        let preparation_identity = (*identity).clone();
        let preparation_self_node_id = *self_node_id;
        let request = match prepare_peer_blind_relay_http_request_with(move || {
            let envelope = BlindRelayEnvelope {
                route_id: promotion_route_id.unwrap_or_else(|| {
                    Self::blind_relay_probe_route_id(now, &preparation_self_node_id, &next_hop)
                }),
                next_hop,
                ttl: 1,
                encrypted_blob: Self::blind_relay_probe_blob(
                    now,
                    &preparation_self_node_id,
                    &next_hop,
                ),
                timestamp: now,
                signature: [0u8; 64],
            }
            .sign_with(&preparation_identity);
            Ok::<_, std::convert::Infallible>((
                PeerBlindRelayRequest {
                    envelope: envelope.clone(),
                    previous_hop_node_id: preparation_self_node_id,
                    onward_envelope: None,
                    onward_descriptor_hint: None,
                },
                envelope,
            ))
        })
        .await
        {
            Ok((request, envelope)) => (request, envelope),
            Err(BlindRelayRequestPreparationError::Build(never)) => match never {},
            Err(BlindRelayRequestPreparationError::Local(error)) => {
                // [OUTBOUND-BLIND-REQUEST-PREPARATION 2026-08-31 by Codex]
                // No peer observed this request. Keep local saturation visible
                // to aggregate readiness without poisoning route reputation.
                peer_store.record_blind_relay_probe_result(now, false, error.reason_bucket());
                return;
            }
        };
        let (request, probe_envelope) = request;

        // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex] DNS
        // and request preparation can wait. Recheck the exact gate/quote at
        // POST entry, without falling back to the unpinned generic client.
        if dns_promotion_probe
            && Self::promotion_blind_relay_probe_url(peer_store, &candidate, unix_now_secs())
                .as_deref() != Some(url.as_str())
        {
            peer_store.record_blind_relay_probe_result(now, false, "stale_promotion");
            return;
        }

        match probe_client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(request.body())
            .send()
            .await
        {
            Ok(response) if response.status().is_success() && response.url().as_str() == url => {
                match decode_bounded_json_response::<PeerBlindRelayResponse>(
                    response,
                    PEER_ACK_RESPONSE_MAX_BYTES,
                )
                .await
                {
                    Ok(ack)
                        if !require_signed_terminal_receipt && ack.accepted
                            || require_signed_terminal_receipt
                                && Self::permissionless_promotion_probe_ack_valid(
                                    &probe_envelope,
                                    &ack,
                                    &next_hop,
                                    now,
                                    unix_now_secs(),
                                ) =>
                    {
                        if require_signed_terminal_receipt {
                            // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex]
                            // Generic route-health writes never open a promoted
                            // candidate. Only this verified control transition can.
                            // [PHALA-PROMOTION-DNS-CONTROL-PROBE 2026-10-07 by Codex]
                            // Receipt observation cannot reuse pre-DNS wall time.
                            let recorded = peer_store.record_permissionless_promotion_probe_verified(
                                &candidate, unix_now_secs(),
                            );
                            peer_store.record_blind_relay_probe_result(
                                now,
                                recorded,
                                if recorded {
                                    "accepted"
                                } else {
                                    "stale_promotion"
                                },
                            );
                        } else {
                            peer_store.record_blind_relay_probe_result(now, true, "accepted");
                            let _ = peer_store
                                .record_route_forward_success_for_descriptor(&candidate, now);
                        }
                    }
                    Ok(_ack) => {
                        let reason = if require_signed_terminal_receipt {
                            "signed_ack_rejected"
                        } else {
                            "ack_rejected"
                        };
                        peer_store.record_blind_relay_probe_result(now, false, reason);
                        let _ = peer_store
                            .record_route_forward_failure_for_descriptor(&candidate, now, reason);
                    }
                    Err(error) => {
                        debug!(
                            reason = error.as_str(),
                            "[DISCOVERY] Blind relay probe ACK rejected"
                        );
                        let reason = format!("ack_{}", error.as_str());
                        peer_store.record_blind_relay_probe_result(now, false, &reason);
                        let _ = peer_store
                            .record_route_forward_failure_for_descriptor(&candidate, now, &reason);
                    }
                }
            }
            Ok(response) => {
                let reason = format!("http_{}", response.status().as_u16());
                peer_store.record_blind_relay_probe_result(now, false, &reason);
                let _ =
                    peer_store.record_route_forward_failure_for_descriptor(&candidate, now, reason);
            }
            Err(error) => {
                let reason = Self::classify_reqwest_error("blind_relay_probe", &error);
                debug!(
                    reason = %reason,
                    "[DISCOVERY] Blind relay probe failed"
                );
                peer_store.record_blind_relay_probe_result(now, false, &reason);
                let _ =
                    peer_store.record_route_forward_failure_for_descriptor(&candidate, now, reason);
            }
        }
    }

    /// An unsigned JSON boolean is not route authority for a newly promoted
    /// permissionless peer. Require the exact target's signed terminal proof
    /// over this fresh, nonce-bound control envelope and response shape.
    // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Legacy warmup
    // keeps its prior behavior; only this new authority transition is strict.
    pub(super) fn permissionless_promotion_probe_ack_valid(
        envelope: &BlindRelayEnvelope,
        ack: &PeerBlindRelayResponse,
        target_node_id: &[u8; 32],
        sent_at: u64,
        observed_at: u64,
    ) -> bool {
        if !ack.accepted
            || !ack.terminal
            || ack.forwarded
            || ack.ttl_remaining != envelope.ttl
            || ack.reason.as_deref() != Some("terminal_next_hop")
            || ack.delivery_receipt.is_some()
            || ack.failure_receipt.is_some()
            || ack.opaque_terminal_response_b64.is_some()
            || envelope.next_hop != *target_node_id
        {
            return false;
        }
        let Some(receipt) = ack.success_receipt.as_ref() else {
            return false;
        };
        if receipt.accepted_at < sent_at.saturating_sub(30)
            || receipt.accepted_at > observed_at.saturating_add(30)
            || observed_at > receipt.accepted_at.saturating_add(30)
        {
            return false;
        }
        receipt
            .verify_expected(
                envelope,
                true,
                false,
                ack.ttl_remaining,
                ack.reason.as_deref(),
                None,
                None,
                target_node_id,
            )
            .is_ok()
    }

    pub(super) async fn probe_two_hop_blind_relay_path(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        now: u64,
    ) -> TwoHopBlindRelayProbeOutcome {
        let mut middle_candidates = peer_store
            .multi_hop_route_probe_candidates_with_capability_excluding(
                NodeCapability::OnionMiddle,
                now,
                8,
                &[*self_node_id],
            );
        Self::prioritize_probe_candidates(peer_store, now, &mut middle_candidates);
        if middle_candidates.is_empty() {
            return TwoHopBlindRelayProbeOutcome::default();
        }

        // [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] This set is
        // only a bootstrap optimization. Legacy candidates remain in the loop
        // for control-plane reachability; they are excluded only from the v2
        // terminal-delivery attempt that can establish App route authority.
        let purpose_bound_middle_advertisers = Self::purpose_bound_delivery_receipt_advertisers(
            client,
            peer_store,
            &middle_candidates,
            now,
        )
        .await;
        let middle_candidate_count = middle_candidates.len();
        let mut attempted = false;
        let mut network_diversity_blocked = false;
        let mut legacy_delivery_contexts = Vec::with_capacity(TWO_HOP_PROBE_REQUEST_LIMIT);
        let mut request_count = 0usize;
        'candidate_search: for middle in middle_candidates {
            let middle_node_id = middle.node_id();
            let mut terminal_candidates = peer_store
                .multi_hop_route_probe_candidates_with_capability_excluding(
                    NodeCapability::ChatRelay,
                    now,
                    8,
                    &[*self_node_id, middle_node_id],
                );
            Self::prioritize_probe_candidates(peer_store, now, &mut terminal_candidates);
            let pre_diversity_candidate_count = terminal_candidates.len();
            terminal_candidates.retain(|terminal| {
                PeerStore::route_endpoints_are_network_diverse(&middle, terminal)
            });
            network_diversity_blocked |=
                pre_diversity_candidate_count > 0 && terminal_candidates.is_empty();
            let terminal_candidate_count = terminal_candidates.len();
            if terminal_candidates.is_empty() {
                continue;
            }
            let purpose_bound_terminal_advertisers =
                if purpose_bound_middle_advertisers.contains(&middle_node_id) {
                    Self::purpose_bound_delivery_receipt_advertisers(
                        client,
                        peer_store,
                        &terminal_candidates,
                        now,
                    )
                    .await
                } else {
                    HashSet::new()
                };

            'terminal_candidates: for terminal in terminal_candidates {
                if request_count >= TWO_HOP_PROBE_REQUEST_LIMIT {
                    break 'candidate_search;
                }
                attempted = true;
                let terminal_node_id = terminal.node_id();
                let Some(endpoint) = middle.descriptor.public_endpoint.as_deref() else {
                    peer_store.record_blind_relay_two_hop_probe_result_with_context(
                        now,
                        false,
                        "middle_missing_endpoint",
                        middle_candidate_count,
                        terminal_candidate_count,
                        2,
                        1,
                    );
                    let _ = peer_store.record_route_forward_failure_for_descriptor(
                        &middle,
                        now,
                        "missing_endpoint",
                    );
                    continue;
                };
                let Some(url) = Self::blind_relay_probe_url(endpoint) else {
                    peer_store.record_blind_relay_two_hop_probe_result_with_context(
                        now,
                        false,
                        "middle_invalid_endpoint",
                        middle_candidate_count,
                        terminal_candidate_count,
                        2,
                        1,
                    );
                    let _ = peer_store.record_route_forward_failure_for_descriptor(
                        &middle,
                        now,
                        "invalid_endpoint",
                    );
                    continue;
                };

                // Milestone 2 probe: prefer a real onion-wrapped ChatEnvelope
                // delivery over the older onward-envelope control-plane probe.
                // Both participants must advertise v2 before this optional
                // request. The signed receipt remains the sole authority.
                let purpose_bound_probe_allowed =
                    purpose_bound_terminal_advertisers.contains(&terminal_node_id);
                let purpose_bound_probe_request = if purpose_bound_probe_allowed {
                    request_count = request_count.saturating_add(1);
                    let preparation_identity = (*identity).clone();
                    let preparation_self_node_id = *self_node_id;
                    let preparation_middle = middle.clone();
                    let preparation_terminal = terminal.clone();
                    match prepare_peer_blind_relay_http_request_with(move || {
                        Self::build_two_hop_onion_delivery_probe_request(
                            &preparation_identity,
                            &preparation_self_node_id,
                            &preparation_middle,
                            &preparation_terminal,
                            now,
                        )
                        .ok_or(())
                    })
                    .await
                    {
                        Ok(request) => Some(request),
                        Err(BlindRelayRequestPreparationError::Build(())) => {
                            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                now,
                                false,
                                "onion_route_contract_rejected",
                                middle_candidate_count,
                                terminal_candidate_count,
                                2,
                                1,
                            );
                            None
                        }
                        Err(BlindRelayRequestPreparationError::Local(error)) => {
                            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                now,
                                false,
                                error.reason_bucket(),
                                middle_candidate_count,
                                terminal_candidate_count,
                                2,
                                1,
                            );
                            continue 'terminal_candidates;
                        }
                    }
                } else {
                    None
                };
                if let Some((request, payload_commitment)) = purpose_bound_probe_request {
                    let request_started_at = Instant::now();
                    match client
                        .post(&url)
                        .header(reqwest::header::CONTENT_TYPE, "application/json")
                        .body(request.body())
                        .send()
                        .await
                    {
                        Ok(response) if response.status().is_success() => {
                            // [MULTIHOP-PROOF-RESPONSE-TIME 2026-08-11 by Codex]
                            // Bind verification to response observation rather
                            // than the earlier route-selection snapshot. The
                            // elapsed projection preserves injected test clocks.
                            let observed_at =
                                now.saturating_add(request_started_at.elapsed().as_secs());
                            match decode_bounded_json_response::<PeerBlindRelayResponse>(
                                response,
                                PEER_ACK_RESPONSE_MAX_BYTES,
                            )
                            .await
                            {
                                Ok(ack) if ack.accepted && ack.forwarded => {
                                    match verify_blind_relay_delivery_receipt(
                                        ack.delivery_receipt,
                                        *request.route_id(),
                                        payload_commitment,
                                        terminal_node_id,
                                        observed_at,
                                    )
                                    .await
                                    {
                                        Ok(_) => {
                                            if peer_store
                                                .record_verified_two_hop_probe_delivery(
                                                    &middle,
                                                    &terminal,
                                                    observed_at,
                                                    middle_candidate_count,
                                                    terminal_candidate_count,
                                                )
                                            {
                                                return TwoHopBlindRelayProbeOutcome {
                                                    attempted: true,
                                                    route_accepted: true,
                                                    terminal_delivery_verified: true,
                                                };
                                            }
                                            // A real receipt cannot authorize a path
                                            // whose signed surface rotated in flight.
                                            continue 'terminal_candidates;
                                        }
                                        Err(BlindRelayDeliveryReceiptVerificationFailure::Missing) => {
                                            // Only absence can prove a rolling-upgrade
                                            // legacy path. Invalid v2 evidence must never
                                            // downgrade into compatibility success.
                                            legacy_delivery_contexts.push((
                                                middle.clone(),
                                                terminal.clone(),
                                                middle_candidate_count,
                                                terminal_candidate_count,
                                                observed_at,
                                            ));
                                            continue 'terminal_candidates;
                                        }
                                        Err(BlindRelayDeliveryReceiptVerificationFailure::Invalid) => {
                                            peer_store
                                                .record_blind_relay_two_hop_probe_result_with_context(
                                                    observed_at,
                                                    false,
                                                    "onion_receipt_unverified",
                                                    middle_candidate_count,
                                                    terminal_candidate_count,
                                                    2,
                                                    1,
                                                );
                                            let _ = Self::record_onion_route_failure(
                                                peer_store,
                                                &middle,
                                                observed_at,
                                                "delivery_receipt_invalid",
                                                OnionRouteFailureAttribution::EndToEnd,
                                            );
                                            continue 'terminal_candidates;
                                        }
                                        Err(
                                            BlindRelayDeliveryReceiptVerificationFailure::Unavailable,
                                        ) => {
                                            peer_store
                                                .record_blind_relay_two_hop_probe_result_with_context(
                                                    observed_at,
                                                    false,
                                                    "onion_receipt_verifier_unavailable",
                                                    middle_candidate_count,
                                                    terminal_candidate_count,
                                                    2,
                                                    1,
                                            );
                                            continue 'terminal_candidates;
                                        }
                                    }
                                }
                                Ok(_ack) => {
                                    peer_store
                                        .record_blind_relay_two_hop_probe_result_with_context(
                                            observed_at,
                                            false,
                                            "onion_ack_rejected",
                                            middle_candidate_count,
                                            terminal_candidate_count,
                                            2,
                                            1,
                                        );
                                    let _ = peer_store.record_route_forward_failure_for_descriptor(
                                        &middle,
                                        observed_at,
                                        "onion_ack_rejected",
                                    );
                                }
                                Err(error) => {
                                    let reason = format!("onion_ack_{}", error.as_str());
                                    debug!(
                                        reason = %reason,
                                        "[DISCOVERY] Two-hop onion delivery probe ACK rejected"
                                    );
                                    peer_store
                                        .record_blind_relay_two_hop_probe_result_with_context(
                                            observed_at,
                                            false,
                                            &reason,
                                            middle_candidate_count,
                                            terminal_candidate_count,
                                            2,
                                            1,
                                        );
                                    let _ = peer_store.record_route_forward_failure_for_descriptor(
                                        &middle,
                                        observed_at,
                                        &reason,
                                    );
                                }
                            }
                        }
                        Ok(response) => {
                            let observed_at =
                                now.saturating_add(request_started_at.elapsed().as_secs());
                            let reason = format!("onion_http_{}", response.status().as_u16());
                            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                observed_at,
                                false,
                                &reason,
                                middle_candidate_count,
                                terminal_candidate_count,
                                2,
                                1,
                            );
                            let _ = peer_store.record_route_forward_failure_for_descriptor(
                                &middle,
                                observed_at,
                                reason,
                            );
                        }
                        Err(error) => {
                            let observed_at =
                                now.saturating_add(request_started_at.elapsed().as_secs());
                            let reason = Self::classify_reqwest_error(
                                "two_hop_onion_delivery_probe",
                                &error,
                            );
                            debug!(
                                reason = %reason,
                                "[DISCOVERY] Two-hop onion delivery probe failed"
                            );
                            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                observed_at,
                                false,
                                &reason,
                                middle_candidate_count,
                                terminal_candidate_count,
                                2,
                                1,
                            );
                            let _ = peer_store.record_route_forward_failure_for_descriptor(
                                &middle,
                                observed_at,
                                reason,
                            );
                        }
                    }
                }

                if request_count >= TWO_HOP_PROBE_REQUEST_LIMIT {
                    break 'candidate_search;
                }
                request_count = request_count.saturating_add(1);

                let preparation_identity = (*identity).clone();
                let preparation_self_node_id = *self_node_id;
                let preparation_terminal = terminal.clone();
                let request = match prepare_peer_blind_relay_http_request_with(move || {
                    let outer_envelope = BlindRelayEnvelope {
                        route_id: Self::blind_relay_two_hop_probe_route_id(
                            now,
                            &preparation_self_node_id,
                            &middle_node_id,
                            &terminal_node_id,
                            b"outer",
                        ),
                        next_hop: middle_node_id,
                        ttl: 2,
                        encrypted_blob: Self::blind_relay_probe_blob(
                            now,
                            &preparation_self_node_id,
                            &middle_node_id,
                        ),
                        timestamp: now,
                        signature: [0u8; 64],
                    }
                    .sign_with(&preparation_identity);
                    let onward_envelope = BlindRelayEnvelope {
                        route_id: Self::blind_relay_two_hop_probe_route_id(
                            now,
                            &preparation_self_node_id,
                            &middle_node_id,
                            &terminal_node_id,
                            b"onward",
                        ),
                        next_hop: terminal_node_id,
                        ttl: 1,
                        encrypted_blob: Self::blind_relay_probe_blob(
                            now,
                            &preparation_self_node_id,
                            &terminal_node_id,
                        ),
                        timestamp: now,
                        signature: [0u8; 64],
                    }
                    .sign_with(&preparation_identity);
                    Ok::<_, std::convert::Infallible>((
                        PeerBlindRelayRequest {
                            envelope: outer_envelope,
                            previous_hop_node_id: preparation_self_node_id,
                            onward_envelope: Some(onward_envelope),
                            onward_descriptor_hint: Some(preparation_terminal),
                        },
                        (),
                    ))
                })
                .await
                {
                    Ok((request, ())) => request,
                    Err(BlindRelayRequestPreparationError::Build(never)) => match never {},
                    Err(BlindRelayRequestPreparationError::Local(error)) => {
                        peer_store.record_blind_relay_two_hop_probe_result_with_context(
                            now,
                            false,
                            error.reason_bucket(),
                            middle_candidate_count,
                            terminal_candidate_count,
                            2,
                            1,
                        );
                        continue 'terminal_candidates;
                    }
                };

                let request_started_at = Instant::now();
                match client
                    .post(&url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(request.body())
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        let observed_at =
                            now.saturating_add(request_started_at.elapsed().as_secs());
                        match decode_bounded_json_response::<PeerBlindRelayResponse>(
                            response,
                            PEER_ACK_RESPONSE_MAX_BYTES,
                        )
                        .await
                        {
                            Ok(ack) if ack.accepted && ack.forwarded => {
                                // A legacy onward-envelope proof preserves
                                // compatibility, but it is not a terminal-signed
                                // delivery receipt. Keep it as fallback evidence
                                // and continue the bounded candidate search so an
                                // older peer cannot mask a receipt-capable path.
                                legacy_delivery_contexts.push((
                                    middle.clone(),
                                    terminal.clone(),
                                    middle_candidate_count,
                                    terminal_candidate_count,
                                    observed_at,
                                ));
                                continue 'terminal_candidates;
                            }
                            Ok(_ack) => {
                                peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                    observed_at,
                                    false,
                                    "ack_rejected",
                                    middle_candidate_count,
                                    terminal_candidate_count,
                                    2,
                                    1,
                                );
                                let _ = peer_store.record_route_forward_failure_for_descriptor(
                                    &middle,
                                    observed_at,
                                    "ack_rejected",
                                );
                            }
                            Err(error) => {
                                let reason = format!("ack_{}", error.as_str());
                                debug!(
                                    reason = %reason,
                                    "[DISCOVERY] Two-hop blind relay proof ACK rejected"
                                );
                                peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                    observed_at,
                                    false,
                                    &reason,
                                    middle_candidate_count,
                                    terminal_candidate_count,
                                    2,
                                    1,
                                );
                                let _ = peer_store.record_route_forward_failure_for_descriptor(
                                    &middle,
                                    observed_at,
                                    &reason,
                                );
                            }
                        }
                    }
                    Ok(response) => {
                        let observed_at =
                            now.saturating_add(request_started_at.elapsed().as_secs());
                        let reason = format!("http_{}", response.status().as_u16());
                        peer_store.record_blind_relay_two_hop_probe_result_with_context(
                            observed_at,
                            false,
                            &reason,
                            middle_candidate_count,
                            terminal_candidate_count,
                            2,
                            1,
                        );
                        let _ = peer_store.record_route_forward_failure_for_descriptor(
                            &middle,
                            observed_at,
                            reason,
                        );
                    }
                    Err(error) => {
                        let observed_at =
                            now.saturating_add(request_started_at.elapsed().as_secs());
                        let reason =
                            Self::classify_reqwest_error("two_hop_blind_relay_probe", &error);
                        debug!(
                            reason = %reason,
                            "[DISCOVERY] Two-hop blind relay proof failed"
                        );
                        peer_store.record_blind_relay_two_hop_probe_result_with_context(
                            observed_at,
                            false,
                            &reason,
                            middle_candidate_count,
                            terminal_candidate_count,
                            2,
                            1,
                        );
                        let _ = peer_store.record_route_forward_failure_for_descriptor(
                            &middle,
                            observed_at,
                            reason,
                        );
                    }
                }
            }
        }

        // [LEGACY-CONTROL-PROOF-SURFACE-BINDING 2026-08-11 by Codex] Try the
        // newest bounded fallback first. A delayed ACK may prove only the exact
        // descriptors it carried; it cannot authorize a replacement surface.
        for (middle, terminal, middle_candidates, terminal_candidates, observed_at) in
            legacy_delivery_contexts.into_iter().rev()
        {
            if peer_store.record_verified_two_hop_control_probe(
                &middle,
                &terminal,
                observed_at,
                middle_candidates,
                terminal_candidates,
            ) {
                return TwoHopBlindRelayProbeOutcome {
                    attempted: true,
                    route_accepted: true,
                    terminal_delivery_verified: false,
                };
            }
        }

        if !attempted {
            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                now,
                false,
                if network_diversity_blocked {
                    "no_network_diverse_path"
                } else {
                    "no_distinct_path"
                },
                middle_candidate_count,
                0,
                2,
                1,
            );
        }
        TwoHopBlindRelayProbeOutcome {
            attempted,
            route_accepted: false,
            terminal_delivery_verified: false,
        }
    }

    /// Proves entry -> middle -> middle -> terminal delivery with one opaque
    /// synthetic ChatEnvelope and a terminal-signed receipt.
    ///
    /// [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex] Candidate identities and
    /// endpoints stay process-local. Public status receives only bounded
    /// success/failure buckets through `PeerStore`; this function never logs or
    /// persists the selected path, route id, receiver, commitment, or payload.
    pub(super) async fn probe_three_hop_blind_relay_path(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        now: u64,
    ) -> TwoHopBlindRelayProbeOutcome {
        // [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Three-hop
        // proof is layered on top of successful two-hop v2 evidence. Every
        // participant must have recently carried a cryptographically verified,
        // purpose-bound receipt; unsigned summary hints cannot authorize this
        // higher-hop probe or cause a legacy peer to receive failure penalties.
        let mut first_middle_candidates = peer_store
            .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
                NodeCapability::OnionMiddle,
                now,
                ONION_ROUTE_SELECTION_CANDIDATE_LIMIT,
                &[*self_node_id],
            );
        Self::prioritize_probe_candidates(peer_store, now, &mut first_middle_candidates);
        if first_middle_candidates.is_empty() {
            return TwoHopBlindRelayProbeOutcome::default();
        }

        let first_middle_candidate_count = first_middle_candidates.len();
        let mut attempted = false;
        let mut network_diversity_blocked = false;
        let mut request_count = 0usize;

        'first_middle_search: for first_middle in first_middle_candidates {
            let first_middle_node_id = first_middle.node_id();
            let mut second_middle_candidates = peer_store
                .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
                    NodeCapability::OnionMiddle,
                    now,
                    ONION_ROUTE_SELECTION_CANDIDATE_LIMIT,
                    &[*self_node_id, first_middle_node_id],
                );
            Self::prioritize_probe_candidates(peer_store, now, &mut second_middle_candidates);
            let second_before_diversity = second_middle_candidates.len();
            second_middle_candidates.retain(|second_middle| {
                PeerStore::route_endpoints_are_network_diverse(&first_middle, second_middle)
            });
            network_diversity_blocked |=
                second_before_diversity > 0 && second_middle_candidates.is_empty();

            for second_middle in second_middle_candidates {
                let second_middle_node_id = second_middle.node_id();
                let mut terminal_candidates = peer_store
                    .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
                        NodeCapability::ChatRelay,
                        now,
                        ONION_ROUTE_SELECTION_CANDIDATE_LIMIT,
                        &[*self_node_id, first_middle_node_id, second_middle_node_id],
                    );
                Self::prioritize_probe_candidates(peer_store, now, &mut terminal_candidates);
                let terminal_before_diversity = terminal_candidates.len();
                terminal_candidates.retain(|terminal| {
                    PeerStore::route_endpoints_are_network_diverse(&first_middle, terminal)
                        && PeerStore::route_endpoints_are_network_diverse(&second_middle, terminal)
                });
                network_diversity_blocked |=
                    terminal_before_diversity > 0 && terminal_candidates.is_empty();
                if terminal_candidates.is_empty() {
                    continue;
                }

                let effective_middle_candidate_count =
                    first_middle_candidate_count.min(second_before_diversity.max(1));
                let terminal_candidate_count = terminal_candidates.len();
                let Some(endpoint) = first_middle.descriptor.public_endpoint.as_deref() else {
                    attempted = true;
                    peer_store.record_blind_relay_three_hop_probe_result_with_context(
                        now,
                        false,
                        "middle_missing_endpoint",
                        effective_middle_candidate_count,
                        terminal_candidate_count,
                        3,
                        2,
                    );
                    let _ = peer_store.record_route_forward_failure_for_descriptor(
                        &first_middle,
                        now,
                        "missing_endpoint",
                    );
                    continue;
                };
                let Some(url) = Self::blind_relay_probe_url(endpoint) else {
                    attempted = true;
                    peer_store.record_blind_relay_three_hop_probe_result_with_context(
                        now,
                        false,
                        "middle_invalid_endpoint",
                        effective_middle_candidate_count,
                        terminal_candidate_count,
                        3,
                        2,
                    );
                    let _ = peer_store.record_route_forward_failure_for_descriptor(
                        &first_middle,
                        now,
                        "invalid_endpoint",
                    );
                    continue;
                };

                for terminal in terminal_candidates {
                    if request_count >= THREE_HOP_PROBE_REQUEST_LIMIT {
                        break 'first_middle_search;
                    }
                    attempted = true;
                    let terminal_node_id = terminal.node_id();
                    request_count = request_count.saturating_add(1);
                    let preparation_identity = (*identity).clone();
                    let preparation_self_node_id = *self_node_id;
                    let preparation_first_middle = first_middle.clone();
                    let preparation_second_middle = second_middle.clone();
                    let preparation_terminal = terminal.clone();
                    let (request, payload_commitment) =
                        match prepare_peer_blind_relay_http_request_with(move || {
                            Self::build_three_hop_onion_delivery_probe_request(
                                &preparation_identity,
                                &preparation_self_node_id,
                                &preparation_first_middle,
                                &preparation_second_middle,
                                &preparation_terminal,
                                now,
                            )
                            .ok_or(())
                        })
                        .await
                        {
                            Ok(request) => request,
                            Err(BlindRelayRequestPreparationError::Build(())) => {
                                peer_store.record_blind_relay_three_hop_probe_result_with_context(
                                    now,
                                    false,
                                    "onion_route_contract_rejected",
                                    effective_middle_candidate_count,
                                    terminal_candidate_count,
                                    3,
                                    2,
                                );
                                continue;
                            }
                            Err(BlindRelayRequestPreparationError::Local(error)) => {
                                peer_store.record_blind_relay_three_hop_probe_result_with_context(
                                    now,
                                    false,
                                    error.reason_bucket(),
                                    effective_middle_candidate_count,
                                    terminal_candidate_count,
                                    3,
                                    2,
                                );
                                continue;
                            }
                        };

                    let request_started_at = Instant::now();
                    match client
                        .post(&url)
                        .header(reqwest::header::CONTENT_TYPE, "application/json")
                        .body(request.body())
                        .send()
                        .await
                    {
                        Ok(response) if response.status().is_success() => {
                            let observed_at =
                                now.saturating_add(request_started_at.elapsed().as_secs());
                            match decode_bounded_json_response::<PeerBlindRelayResponse>(
                                response,
                                PEER_ACK_RESPONSE_MAX_BYTES,
                            )
                            .await
                            {
                                Ok(ack) if ack.accepted && ack.forwarded => {
                                    match verify_blind_relay_delivery_receipt(
                                        ack.delivery_receipt,
                                        *request.route_id(),
                                        payload_commitment,
                                        terminal_node_id,
                                        observed_at,
                                    )
                                    .await
                                    {
                                        Ok(_) => {
                                            if peer_store
                                                .record_verified_three_hop_probe_delivery(
                                                    &first_middle,
                                                    &second_middle,
                                                    &terminal,
                                                    observed_at,
                                                    effective_middle_candidate_count,
                                                    terminal_candidate_count,
                                                )
                                            {
                                                return TwoHopBlindRelayProbeOutcome {
                                                    attempted: true,
                                                    route_accepted: true,
                                                    terminal_delivery_verified: true,
                                                };
                                            }
                                            continue;
                                        }
                                        Err(
                                            BlindRelayDeliveryReceiptVerificationFailure::Missing
                                            | BlindRelayDeliveryReceiptVerificationFailure::Invalid,
                                        ) => {
                                            peer_store
                                                .record_blind_relay_three_hop_probe_result_with_context(
                                                    observed_at,
                                                    false,
                                                    "onion_receipt_unverified",
                                                    effective_middle_candidate_count,
                                                    terminal_candidate_count,
                                                    3,
                                                    2,
                                                );
                                            let _ = Self::record_onion_route_failure(
                                                peer_store,
                                                &first_middle,
                                                observed_at,
                                                "delivery_receipt_invalid",
                                                OnionRouteFailureAttribution::EndToEnd,
                                            );
                                        }
                                        Err(
                                            BlindRelayDeliveryReceiptVerificationFailure::Unavailable,
                                        ) => {
                                            peer_store
                                                .record_blind_relay_three_hop_probe_result_with_context(
                                                    observed_at,
                                                    false,
                                                    "onion_receipt_verifier_unavailable",
                                                    effective_middle_candidate_count,
                                                    terminal_candidate_count,
                                                    3,
                                                    2,
                                                );
                                        }
                                    }
                                }
                                Ok(_ack) => {
                                    peer_store
                                        .record_blind_relay_three_hop_probe_result_with_context(
                                            observed_at,
                                            false,
                                            "onion_ack_rejected",
                                            effective_middle_candidate_count,
                                            terminal_candidate_count,
                                            3,
                                            2,
                                        );
                                    let _ = Self::record_onion_route_failure(
                                        peer_store,
                                        &first_middle,
                                        observed_at,
                                        "onion_ack_rejected",
                                        OnionRouteFailureAttribution::FirstHop,
                                    );
                                }
                                Err(error) => {
                                    let reason = format!("onion_ack_{}", error.as_str());
                                    peer_store
                                        .record_blind_relay_three_hop_probe_result_with_context(
                                            observed_at,
                                            false,
                                            &reason,
                                            effective_middle_candidate_count,
                                            terminal_candidate_count,
                                            3,
                                            2,
                                        );
                                    let _ = Self::record_onion_route_failure(
                                        peer_store,
                                        &first_middle,
                                        observed_at,
                                        &reason,
                                        OnionRouteFailureAttribution::FirstHop,
                                    );
                                }
                            }
                        }
                        Ok(response) => {
                            let observed_at =
                                now.saturating_add(request_started_at.elapsed().as_secs());
                            let reason = format!("onion_http_{}", response.status().as_u16());
                            peer_store.record_blind_relay_three_hop_probe_result_with_context(
                                observed_at,
                                false,
                                &reason,
                                effective_middle_candidate_count,
                                terminal_candidate_count,
                                3,
                                2,
                            );
                            let _ = Self::record_onion_route_failure(
                                peer_store,
                                &first_middle,
                                observed_at,
                                reason,
                                OnionRouteFailureAttribution::FirstHop,
                            );
                        }
                        Err(error) => {
                            let observed_at =
                                now.saturating_add(request_started_at.elapsed().as_secs());
                            let reason = Self::classify_reqwest_error(
                                "three_hop_onion_delivery_probe",
                                &error,
                            );
                            debug!(
                                reason = %reason,
                                "[DISCOVERY] Three-hop onion delivery probe failed"
                            );
                            peer_store.record_blind_relay_three_hop_probe_result_with_context(
                                observed_at,
                                false,
                                &reason,
                                effective_middle_candidate_count,
                                terminal_candidate_count,
                                3,
                                2,
                            );
                            let _ = Self::record_onion_route_failure(
                                peer_store,
                                &first_middle,
                                observed_at,
                                reason,
                                OnionRouteFailureAttribution::FirstHop,
                            );
                        }
                    }
                }
            }
        }

        if !attempted {
            peer_store.record_blind_relay_three_hop_probe_result_with_context(
                now,
                false,
                if network_diversity_blocked {
                    "no_network_diverse_path"
                } else {
                    "no_distinct_path"
                },
                first_middle_candidate_count,
                0,
                3,
                2,
            );
        }

        TwoHopBlindRelayProbeOutcome {
            attempted,
            route_accepted: false,
            terminal_delivery_verified: false,
        }
    }

    pub(super) fn build_two_hop_onion_delivery_probe_request(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
    ) -> Option<(PeerBlindRelayRequest, [u8; 32])> {
        let middle_node_id = middle.node_id();
        let terminal_node_id = terminal.node_id();
        let route_id = Self::blind_relay_two_hop_probe_route_id(
            now,
            self_node_id,
            &middle_node_id,
            &terminal_node_id,
            b"onion-delivery",
        );
        let chat_envelope = Self::synthetic_two_hop_probe_chat_envelope(
            identity,
            self_node_id,
            &middle_node_id,
            &terminal_node_id,
            route_id,
            now,
        );
        Self::build_two_hop_onion_request(
            identity,
            self_node_id,
            middle,
            terminal,
            &chat_envelope,
            route_id,
            now,
        )
        .ok()
    }

    pub(super) fn build_three_hop_onion_delivery_probe_request(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        first_middle: &SignedNodeDescriptor,
        second_middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
    ) -> Option<(PeerBlindRelayRequest, [u8; 32])> {
        let first_middle_node_id = first_middle.node_id();
        let second_middle_node_id = second_middle.node_id();
        let terminal_node_id = terminal.node_id();
        let route_id = Self::blind_relay_three_hop_probe_route_id(
            now,
            self_node_id,
            &first_middle_node_id,
            &second_middle_node_id,
            &terminal_node_id,
        );
        let chat_envelope = Self::synthetic_three_hop_probe_chat_envelope(
            identity,
            self_node_id,
            &first_middle_node_id,
            &second_middle_node_id,
            &terminal_node_id,
            route_id,
            now,
        );
        Self::build_onion_request(
            identity,
            self_node_id,
            &[first_middle, second_middle, terminal],
            &chat_envelope,
            route_id,
            now,
        )
        .ok()
    }

    pub(super) fn synthetic_two_hop_probe_chat_envelope(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        middle_node_id: &[u8; 32],
        terminal_node_id: &[u8; 32],
        route_id: [u8; 16],
        now: u64,
    ) -> ChatEnvelope {
        Self::synthetic_path_probe_chat_envelope(
            identity,
            self_node_id,
            &[*middle_node_id, *terminal_node_id],
            terminal_node_id,
            &route_id,
            now,
            b"aeronyx:two-hop-onion-delivery-probe:v1",
        )
    }

    pub(super) fn synthetic_three_hop_probe_chat_envelope(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        first_middle_node_id: &[u8; 32],
        second_middle_node_id: &[u8; 32],
        terminal_node_id: &[u8; 32],
        route_id: [u8; 16],
        now: u64,
    ) -> ChatEnvelope {
        Self::synthetic_path_probe_chat_envelope(
            identity,
            self_node_id,
            &[
                *first_middle_node_id,
                *second_middle_node_id,
                *terminal_node_id,
            ],
            terminal_node_id,
            &route_id,
            now,
            b"aeronyx:three-hop-onion-delivery-probe:v1",
        )
    }

    pub(super) fn synthetic_path_probe_chat_envelope(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        path_node_ids: &[[u8; 32]],
        terminal_node_id: &[u8; 32],
        route_id: &[u8; 16],
        now: u64,
        domain: &[u8],
    ) -> ChatEnvelope {
        let receiver = Self::synthetic_path_probe_wallet_id(
            now,
            self_node_id,
            path_node_ids,
            route_id,
            domain,
            b"receiver",
        );
        let nonce_source = Self::synthetic_path_probe_wallet_id(
            now,
            self_node_id,
            path_node_ids,
            route_id,
            domain,
            b"nonce",
        );
        let ciphertext = Self::blind_relay_probe_blob(now, self_node_id, terminal_node_id);
        let mut nonce = [0u8; 24];
        nonce.copy_from_slice(&nonce_source[..24]);
        let mut envelope = ChatEnvelope {
            message_id: *route_id,
            sender: identity.public_key_bytes(),
            receiver,
            timestamp: now,
            ciphertext,
            nonce,
            content_type: ChatContentType::System,
            signature: [0u8; 64],
        };
        envelope.signature = identity.sign(&envelope.sign_data());
        envelope
    }

    pub(super) fn synthetic_path_probe_wallet_id(
        now: u64,
        self_node_id: &[u8; 32],
        path_node_ids: &[[u8; 32]],
        route_id: &[u8; 16],
        domain: &[u8],
        label: &[u8],
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update(label);
        hasher.update(now.to_be_bytes());
        hasher.update(self_node_id);
        for node_id in path_node_ids {
            hasher.update(node_id);
        }
        hasher.update(route_id);
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..32]);
        out
    }

    pub(super) fn blind_relay_probe_route_id(
        now: u64,
        self_node_id: &[u8; 32],
        next_hop: &[u8; 32],
    ) -> [u8; 16] {
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx:blind-relay-probe:route-id:v1");
        hasher.update(now.to_le_bytes());
        hasher.update(&self_node_id[..]);
        hasher.update(&next_hop[..]);
        let digest = hasher.finalize();
        let mut route_id = [0u8; 16];
        route_id.copy_from_slice(&digest[..16]);
        route_id
    }

    pub(super) fn blind_relay_two_hop_probe_route_id(
        now: u64,
        self_node_id: &[u8; 32],
        middle_node_id: &[u8; 32],
        terminal_node_id: &[u8; 32],
        hop_label: &[u8],
    ) -> [u8; 16] {
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx:blind-relay-two-hop-probe:route-id:v1");
        hasher.update(hop_label);
        hasher.update(now.to_be_bytes());
        hasher.update(self_node_id);
        hasher.update(middle_node_id);
        hasher.update(terminal_node_id);
        let digest = hasher.finalize();
        let mut route_id = [0u8; 16];
        route_id.copy_from_slice(&digest[..16]);
        route_id
    }

    pub(super) fn blind_relay_three_hop_probe_route_id(
        now: u64,
        self_node_id: &[u8; 32],
        first_middle_node_id: &[u8; 32],
        second_middle_node_id: &[u8; 32],
        terminal_node_id: &[u8; 32],
    ) -> [u8; 16] {
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx:blind-relay-three-hop-probe:route-id:v1");
        hasher.update(now.to_be_bytes());
        hasher.update(self_node_id);
        hasher.update(first_middle_node_id);
        hasher.update(second_middle_node_id);
        hasher.update(terminal_node_id);
        let digest = hasher.finalize();
        let mut route_id = [0u8; 16];
        route_id.copy_from_slice(&digest[..16]);
        route_id
    }

    pub(super) fn blind_relay_probe_blob(
        now: u64,
        self_node_id: &[u8; 32],
        next_hop: &[u8; 32],
    ) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx:blind-relay-probe:opaque-blob:v1");
        hasher.update(now.to_le_bytes());
        hasher.update(&self_node_id[..]);
        hasher.update(&next_hop[..]);
        hasher.finalize().to_vec()
    }

    pub(super) fn discovery_gossip_url(endpoint: &str) -> Option<String> {
        canonical_peer_http_url(endpoint, "/api/discovery/gossip")
            .ok()
            .map(|url| url.to_string())
    }

    // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Source pins
    // retain configured order and rotate even when a selected peer is offline.
    // No unbounded fanout or extra independent concurrency pool is introduced.
    pub(super) fn private_authorization_source_batch(
        sources: &[[u8; 32]], cursor: &mut usize, concurrency_limit: usize,
    ) -> Vec<[u8; 32]> {
        if sources.is_empty() || sources.len() > 64 || concurrency_limit == 0 {
            return Vec::new();
        }
        let start = *cursor % sources.len();
        let count = concurrency_limit.min(8).min(sources.len());
        let selected = (0..count).map(|offset| sources[(start + offset) % sources.len()]).collect();
        *cursor = (start + count) % sources.len();
        selected
    }

    // [PHALA-AUTHORITY-DELIVERY-CADENCE 2026-10-08 by Codex] Select only
    // within one nonblocking R/P/grant epoch and a post-lock clock sample.
    // Reuse needs a still-valid exact cached grant; missing/expired/rotated
    // authority signs a new candidate, remembered only after a successful ACK.
    // Carry this observation even for an old grant: its issued_at is not the
    // clock floor of the current outbound attempt.
    pub(super) fn private_recipient_authorization_for_gossip(
        peers: &PeerStore,
        identity: &IdentityKeyPair,
        relay_id: [u8; 32],
        purpose: &str,
        floor: u64,
        clock: impl FnOnce() -> u64,
    ) -> Option<(SignedPrivateOnionRecipientAuthorizationV1, SignedNodeDescriptor, SignedNodeDescriptor, u64)> {
        if !matches!(purpose, "blind_vault_pull" | "blind_vault_lease_admission") {
            return None;
        }
        let _epoch = peers.try_private_onion_authority_read_guard()?;
        let now = clock();
        let recipient_id = identity.public_key_bytes();
        if now == 0 || now < floor
            || !peers.has_private_onion_route_identity_pin(&relay_id, &recipient_id)
        {
            return None;
        }
        let (relay, recipient) = peers.get_valid_pair(&relay_id, &recipient_id, now)?;
        if recipient.descriptor.public_endpoint.is_some() || recipient.descriptor.policy.public_discovery {
            return None;
        }
        if let Some((relay, recipient, authorization)) = peers
            .current_private_onion_authority_snapshot_for_purpose_under_guard(
                &relay_id, &recipient_id, purpose, now,
            )
        {
            return Some((authorization, relay, recipient, now));
        }
        let expires_at = now
            .saturating_add(aeronyx_core::protocol::discovery::MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_LIFETIME_SECS_V1)
            .min(relay.descriptor.expires_at)
            .min(recipient.descriptor.expires_at);
        let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay, &recipient, purpose, now, expires_at, identity,
        ).ok()?;
        Some((authorization, relay, recipient, now))
    }

    // [PHALA-AUTHORITY-GOSSIP-FENCE 2026-10-07 by Codex] The real POST
    // entry and regression sources share this exact bundle/target gate. It
    // never selects a replacement or holds a lock across a network await.
    pub(super) fn private_authorization_gossip_admitted_at(
        peers: &PeerStore,
        local: [u8; 32],
        target: &SignedNodeDescriptor,
        message: &NodeDiscoveryMessage,
        url: &str,
        floor: u64,
        clock: impl FnOnce() -> u64,
    ) -> Option<u64> {
        let _authority_epoch = peers.try_private_onion_authority_read_guard()?;
        let now = clock();
        if now == 0 || now < floor { return None; }
        let endpoint = target.descriptor.public_endpoint.as_deref()?;
        if Self::reverse_onion_private_gossip_url(endpoint).as_deref() != Some(url) {
            return None;
        }
        let NodeDiscoveryMessage::PrivateOnionRecipientAuthorizationV1 {
            authorization, relay_descriptor, recipient_descriptor,
        } = message else { return None; };
        peers.private_onion_gossip_bundle_is_current_under_guard(
            local, target, authorization, relay_descriptor, recipient_descriptor, now,
        ).then_some(now)
    }

    // [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] Only operator-pinned
    // reverse-onion authority routes may select a DNS hostname. Generic seed
    // and sampled descriptor selection retain their existing policies.
    pub(super) fn reverse_onion_private_gossip_url(endpoint: &str) -> Option<String> {
        crate::api::reverse_onion_endpoint_supported(endpoint)
            .then(|| Self::discovery_gossip_url(endpoint))?
    }

    // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] A public signature
    // authenticates the descriptor identity, not a new outbound disclosure
    // destination. Require both the configured R identity and fixed origin.
    fn private_recipient_gossip_target_allowed(
        recipient: &crate::config_reverse_onion::ReverseOnionRecipientConfig,
        peer_node_id: Option<[u8; 32]>,
        url: &str,
    ) -> bool {
        let mut relay = [0u8; 32];
        recipient.relay_node_id.len() == 64
            && hex::decode_to_slice(&recipient.relay_node_id, &mut relay).is_ok()
            && relay != [0; 32]
            && peer_node_id == Some(relay)
            && crate::api::reverse_onion_same_origin(&recipient.relay_endpoint, url)
    }

    /// Derives a gossip target from a permissionless signed descriptor.
    ///
    /// Configured bootstrap seeds retain their legacy operator-trusted policy.
    /// PeerStore descriptors sampled as permissionless gossip remain untrusted
    /// and therefore require a public IP literal before outbound requests.
    pub(super) fn discovered_peer_gossip_url(endpoint: &str) -> Option<String> {
        peer_endpoint_is_public_ip(endpoint).then(|| Self::discovery_gossip_url(endpoint))?
    }

    // [PHALA-ATTESTED-DNS-GOSSIP 2026-10-06 by Codex] DNS discovery is
    // available only for an exact signed Phala descriptor with fresh local
    // appraisal. The caller's gossip transport then uses the existing pinned
    // DNS resolver and TLS hostname verification.
    fn attested_discovered_peer_gossip_url(
        peer_store: &PeerStore,
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> Option<String> {
        let endpoint = descriptor.descriptor.public_endpoint.as_deref()?;
        if peer_endpoint_is_public_ip(endpoint) {
            return Self::discovered_peer_gossip_url(endpoint);
        }
        if !peer_store.phala_attested_peer_routes_required()
            || !descriptor
                .descriptor
                .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1)
            || !peer_store.phala_peer_route_is_eligible(descriptor, now)
            || !crate::api::reverse_onion_endpoint_supported(endpoint)
        {
            return None;
        }
        Self::discovery_gossip_url(endpoint)
    }

    /// Adds only current, signed public peers after operator seeds have taken
    /// their reserved places. The deterministic nonce seam keeps the runtime
    /// selection independently testable without opening a network socket.
    pub(super) fn append_sampled_discovered_peer_gossip_urls(
        peer_store: &PeerStore,
        selection: &DiscoveryGossipSampleRequest<'_>,
        seen_urls: &mut HashSet<String>,
        gossip_urls: &mut Vec<String>,
    ) {
        let remaining = selection.round_peer_limit.saturating_sub(gossip_urls.len());
        if remaining == 0 {
            return;
        }
        // [PERMISSIONLESS-GOSSIP-RUNTIME 2026-09-24 by Codex] The sampler
        // sees only the live verified PeerStore, never Stage-A candidates.
        // Filter seed/self URLs before ranking so duplicates do not consume
        // the bounded non-seed budget.
        let sampled = sample_public_gossip_peers_with_policy(
            peer_store,
            selection.now,
            selection.round_nonce,
            remaining,
            &[*selection.self_node_id],
            |endpoint| {
                let safe = peer_endpoint_is_public_ip(endpoint)
                    || crate::api::reverse_onion_endpoint_supported(endpoint);
                safe.then(|| Self::discovery_gossip_url(endpoint))?
            },
            |descriptor| {
                Self::attested_discovered_peer_gossip_url(peer_store, descriptor, selection.now)
                    .is_some()
            },
        );
        for peer in sampled {
            let url = peer.canonical_endpoint;
            if selection.self_gossip_url != Some(url.as_str())
                && seen_urls.insert(url.clone())
            {
                gossip_urls.push(url);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_reverse_onion_private_routes_may_select_dns_gossip_targets() {
        assert!(Server::reverse_onion_private_gossip_url("https://relay.example.net:443").is_some());
        assert!(Server::reverse_onion_private_gossip_url("https://8.8.8.8:8422").is_some());
        assert!(Server::reverse_onion_private_gossip_url("http://8.8.8.8:8422").is_none());
        assert!(Server::discovered_peer_gossip_url("https://relay.example.net:443").is_none());
        assert!(Server::discovered_peer_gossip_url("https://8.8.8.8:443").is_some());
        // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] Generic discovered
        // IP routes keep their existing HTTP compatibility; only the private
        // operator-pinned reverse-onion path is HTTPS-only.
        assert!(Server::discovered_peer_gossip_url("http://8.8.8.8:8422").is_some());
        assert!(Server::reverse_onion_private_gossip_url("http://relay.example.net:8422").is_none());
    }

    // [PHALA-ATTESTED-DNS-GOSSIP 2026-10-06 by Codex]
    #[test]
    fn general_dns_gossip_requires_fresh_phala_appraisal_and_strict_policy() {
        let now = unix_now_secs();
        let identity = IdentityKeyPair::generate();
        let mut body = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 600,
            "1.0.0",
        )
        .with_protocol_features([NodeProtocolFeature::PhalaNodeAttestationV1]);
        body.public_endpoint = Some("https://phala-node.example.net:443".into());
        body.capabilities = vec![NodeCapability::ChatRelay];
        body.policy.public_discovery = true;
        let descriptor = SignedNodeDescriptor::sign(body, &identity).unwrap();
        let store = PeerStore::new();
        store.configure_phala_attested_peer_routes(true, 300);
        store.upsert_verified(descriptor.clone(), now).unwrap();

        assert!(Server::attested_discovered_peer_gossip_url(&store, &descriptor, now).is_none());
        assert!(store.record_phala_peer_attestation(&descriptor, now));
        assert_eq!(
            Server::attested_discovered_peer_gossip_url(&store, &descriptor, now).as_deref(),
            Some("https://phala-node.example.net/api/discovery/gossip")
        );
        assert!(Server::attested_discovered_peer_gossip_url(&store, &descriptor, now + 301).is_none());
    }

    // [PHALA-PRIVATE-RECIPIENT-EGRESS 2026-10-06 by Codex] Authored, unexecuted.
    #[test]
    fn private_recipient_gossip_target_policy_is_pinned_relay_only() {
        let mut config = ServerConfig::default();
        config.discovery.seed_endpoints = vec!["https://seed.example.net".into()];
        config.reverse_onion.recipient.enabled = true;

        let policy = DiscoveryGossipTargetPolicy::for_config(&config, false);
        assert_eq!(policy.round_peer_limit, 1);
        assert!(!policy.include_operator_seeds);
        assert!(!policy.include_cached_peers);

        config.reverse_onion.recipient.enabled = false;
        let public_policy = DiscoveryGossipTargetPolicy::for_config(&config, false);
        assert!(public_policy.include_operator_seeds);
        assert!(public_policy.include_cached_peers);
    }

    // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Authored, not run.
    // Includes negative controls for same-identity endpoint rotation and
    // same-origin identity substitution in both live and recovery modes.
    #[test]
    fn private_recipient_gossip_requires_configured_identity_and_origin() {
        let relay = [0x31; 32];
        let mut recipient = crate::config_reverse_onion::ReverseOnionRecipientConfig::default();
        recipient.enabled = true;
        recipient.relay_node_id = hex::encode(relay);
        recipient.relay_endpoint = "https://relay.example.net".into();
        let same_origin = "https://RELAY.example.net:443/api/discovery/gossip";
        for recovery_only in [false, true] {
            recipient.recovery_only = recovery_only;
            assert!(Server::private_recipient_gossip_target_allowed(
                &recipient, Some(relay), same_origin,
            ));
            for endpoint in ["https://other.example.net/api/discovery/gossip",
                "https://relay.example.net:8443/api/discovery/gossip",
                "http://relay.example.net/api/discovery/gossip",
                "https://relay.internal/api/discovery/gossip"] {
                assert!(!Server::private_recipient_gossip_target_allowed(
                    &recipient, Some(relay), endpoint,
                ));
            }
            for identity in [None, Some([0x32; 32])] {
                assert!(!Server::private_recipient_gossip_target_allowed(
                    &recipient, identity, same_origin,
                ));
            }
        }
        recipient.relay_endpoint = "http://relay.example.net".into();
        assert!(!Server::private_recipient_gossip_target_allowed(
            &recipient, Some(relay), "http://relay.example.net/api/discovery/gossip",
        ));
        recipient.relay_endpoint = "https://relay.example.net".into();
        recipient.relay_node_id = hex::encode([0; 32]);
        assert!(!Server::private_recipient_gossip_target_allowed(
            &recipient, Some([0; 32]), same_origin,
        ));
    }

    #[test]
    fn directory_proof_gossip_result_classifies_privacy_safe_status_buckets() {
        assert_eq!(
            DirectoryProofGossipResult::from_http_status(reqwest::StatusCode::UNPROCESSABLE_ENTITY),
            DirectoryProofGossipResult::EvidenceRejected
        );
        assert_eq!(
            DirectoryProofGossipResult::from_http_status(reqwest::StatusCode::SERVICE_UNAVAILABLE),
            DirectoryProofGossipResult::ReplicaUnavailable
        );
        assert_eq!(
            DirectoryProofGossipResult::from_http_status(reqwest::StatusCode::TOO_MANY_REQUESTS),
            DirectoryProofGossipResult::RateLimited
        );
        assert_eq!(
            DirectoryProofGossipResult::from_http_status(
                reqwest::StatusCode::INTERNAL_SERVER_ERROR
            ),
            DirectoryProofGossipResult::ProtocolRejected
        );
    }

    #[test]
    fn gossip_round_counts_proof_independently_from_legacy_failure() {
        // [GOSSIP-OUTCOME-INTEGRITY 2026-07-28 by Codex] The compatibility
        // failure controls ordinary gossip health but cannot erase proof work.
        let report = DiscoveryPeerGossipReport {
            directory_proof: DirectoryProofGossipOutcome {
                state: DirectoryProofGossipPeerState::Attempted,
                frames_attempted: 1,
                evidence_rejected: 0,
                result: DirectoryProofGossipResult::Accepted,
            },
            legacy_error: Some(DiscoveryGossipFailure {
                phase: DiscoveryGossipPhase::SnapshotStatus,
                kind: DiscoveryGossipFailureKind::Http(500),
            }),
        };
        let mut round = DiscoveryGossipRoundAccumulator::default();

        round.observe(report);

        assert_eq!(round.attempted, 1);
        assert_eq!(round.succeeded, 0);
        assert_eq!(round.directory_proof.accepted, 1);
        assert_eq!(
            round
                .last_failure_reason
                .map(DiscoveryGossipFailure::bucket)
                .as_deref(),
            Some("snapshot_status_http_500")
        );
    }

    #[test]
    fn discovery_gossip_failure_renders_stable_privacy_safe_buckets() {
        assert_eq!(
            DiscoveryGossipFailure {
                phase: DiscoveryGossipPhase::SnapshotStatus,
                kind: DiscoveryGossipFailureKind::Http(503),
            }
            .bucket(),
            "snapshot_status_http_503"
        );
        assert_eq!(
            DiscoveryGossipFailure::from_bounded_response(
                DiscoveryGossipPhase::SnapshotResponse,
                BoundedHttpResponseError::TooLarge,
            )
            .bucket(),
            "snapshot_response_response_too_large"
        );
        assert_eq!(
            DiscoveryGossipFailure::peer_timeout().bucket(),
            "peer_timeout"
        );
    }
}
