// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Wired TrafficTracker into PacketHandler and HeartbeatReporter.
//   spawn_cleanup_task() now accepts + calls traffic_tracker.remove_wallet().
//   HeartbeatReporter receives sessions, traffic, udp via builder methods
//   so it can collect connected_wallets, drain deltas, and enforce
//   membership rules on heartbeat responses.
//   Adds a shared aggregate encrypted VPN message counter for public stats.
//
// What changed vs previous version:
//   1. Added import: use crate::services::traffic_tracker::TrafficTracker;
//   2. In run(): Arc::new(TrafficTracker::new()) created before PacketHandler
//   3. PacketHandler::new() receives Arc::clone(&traffic_tracker)
//   4. HeartbeatReporter gets .with_sessions/.with_traffic_tracker/.with_udp
//   5. spawn_cleanup_task() signature + call site: added traffic_tracker param
//   6. cleanup loop: calls traffic_tracker.remove_wallet(&wallet)
//   7. encrypted_message_counter shared by PacketHandler and VPN health
//   8. Optionally starts a privacy-safe VPN DNS proxy on gateway_ip:53 so
//      commercial clients can resolve domains through the tunnel when Rust
//      owns gateway DNS.
//   9. Optionally hydrates PeerStore from verified discovery bootstrap
//      snapshots when discovery bootstrap is enabled.
//  10. Generates this node's signed discovery descriptor at startup when
//      discovery self-advertisement is enabled.
//  11. Optionally persists verified discovery peers to a local JSON cache so
//      restarts do not depend entirely on remote bootstrap availability.
//  12. Optionally starts outbound discovery gossip to known public peers.
//  13. Optionally relays signed encrypted chat envelopes to discovered
//      `ChatRelay` peers while retaining local pending-storage fallback.
//  14. Optionally starts a public-only discovery API listener that exposes
//      only `/api/discovery/*` and `/api/chat/peer/relay`.
//  15. Optionally contacts configured discovery seed endpoints every gossip
//      round so nodes can recover from stale cached peer endpoints.
//  16. Reports privacy-safe seed endpoint recovery counters for nodeboard.
//  17. Treats stale bootstrap descriptors as benign when newer cache/gossip
//      state already exists, while still warning on rejected descriptors.
//  18. Records privacy-safe outbound gossip health buckets for node stability
//      monitoring without exposing seed endpoint values or peer URLs.
//  19. Reports privacy-safe encrypted chat peer relay health counters.
//  20. Treats memchain.chat_relay.enabled=false as a hard runtime gate while
//      still reporting disabled chat relay telemetry to nodeboard.
//  21. Wires PeerStore discovery summary into local VPN health and operator
//      status so CLI healthchecks, heartbeat, and nodeboard share one
//      privacy-safe peer-discovery readiness contract.
//  22. Saves the verified PeerStore cache once immediately after bootstrap so
//      restart recovery is durable before the first periodic write interval.
//  23. Flushes the verified PeerStore cache once more during graceful shutdown
//      so recently discovered peers are not lost during node upgrades.
//  24. Keeps a previous verified PeerStore cache backup and falls back to it
//      when the primary cache file is missing, unreadable, JSON-corrupted, or
//      contains no usable verified descriptors.
//  25. Loads static bootstrap snapshots before the local PeerStore cache so
//      the most recent verified cache can supersede expired file seed warnings.
//  26. Tags PeerStore imports with source buckets (file/url/cache/backup/self/
//      gossip) so heartbeat/nodeboard can show stale/discovered/cache peers
//      without exposing peer URLs or client traffic.
//  27. Adds outbound discovery gossip jitter/backpressure so node fleets do
//      not retry stale peers in synchronized bursts during incidents.
//  28. Durably fsyncs PeerStore cache writes and the parent directory after
//      atomic rename, reducing peer-cache loss during host crashes/upgrades.
//  29. Records a privacy-safe discovery startup self-check so nodeboard can
//      tell whether cache, gossip, self advertisement, and public endpoint
//      wiring are production-ready without exposing endpoint values.
//  30. Mirrors cache/cache_backup startup load results into dedicated
//      PeerStore cache-load evidence so restart recovery can be audited
//      without exposing cache paths, peer endpoints, or user traffic.
//  31. Adds a compact `discovery_readiness` heartbeat object so backend,
//      nodeboard, and AI maintenance tools can read ChatRelay capability and
//      peer quorum readiness without parsing internal PeerStore structures.
//  32. Validates peer relay ACK bodies before marking a discovered chat relay
//      route healthy, so HTTP 2xx with `accepted=false` is treated as a real
//      encrypted-envelope delivery failure.
//  33. Adds blind relay runtime quality to discovery_readiness so nodeboard,
//      backend, website, and AI maintenance tools can show relay evidence
//      without parsing full PeerStore internals or exposing private metadata.
//  34. Runs low-frequency two-hop blind relay path proofs after successful
//      discovery gossip and reports only aggregate counters/readiness.
//  35. Prefers a real two-hop onion delivery probe before the legacy
//      control-plane onward-envelope probe, proving middle-hop peel/forward
//      and terminal ChatRelay store-and-forward without exposing payloads.
//  36. Prioritizes unproven, non-quarantined route candidates during low-
//      frequency synthetic probes so three-node meshes converge routeability
//      coverage instead of repeatedly probing only the already-proven peer.
//  37. Preserves an existing usable PeerStore cache when the current in-memory
//      view is empty, preventing early-start/shutdown writes from erasing
//      restart recovery evidence before gossip has recovered peers.
//  38. Uses a shorter privacy-safe probe recovery cooldown while two-hop
//      message-delivery proof is not ready, so transient restart failures do
//      not keep healthy nodes in a forming state for a full low-noise cycle.
//  39. Promotes route governance to a top-level heartbeat discovery_status
//      field so backend/nodeboard can read route-pool health without parsing
//      the full internal PeerStore payload.
//  40. Promotes blind relay runtime evidence to a top-level heartbeat
//      discovery_status field so backend/nodeboard can track encrypted relay
//      participation without parsing routes, endpoints, payloads, or peer ids.
//  41. Rebuilds every authenticated owner's isolated vector partition on
//      node-blind/remote MemChain nodes after restart; local-only nodes retain
//      the historical single-owner startup path.
//  42. Verifies content-address integrity for every MemChain record restored
//      into the vector index and additionally verifies the owner's Ed25519
//      signature for node-blind records.
//  43. Mounts the authenticated commitment block range API on node-peer
//      surfaces and rejects range sync frames from ordinary client tunnels.
//  44. Reconciles bounded node-blind commitment blocks at Local-mode startup;
//      announcements contain headers only and never memory payload metadata.
//  45. Gates commitment production behind a default-off coordinator role so
//      follower nodes cannot create independent forks before consensus exists.
//  46. Runs default-off follower catch-up against one pinned coordinator,
//      verifies each complete signed page before append, and backs off on any
//      rollback, fork, signature, continuity, endpoint, or transport failure.
//  47. Publishes bounded privacy-safe commitment sync lifecycle evidence to
//      the local status API and heartbeat without peer or payload metadata.
//  48. Re-verifies the complete persisted commitment chain and membership
//      index before any network listener or follower task can start.
//  49. Publishes the runtime-only verified commitment-chain baseline and keeps
//      it current after transactionally verified appends.
//  50. Requires a signed shared-prefix checkpoint before follower catch-up may
//      declare convergence and reports only aggregate proof outcomes.
//  51. Audits a bounded local vault of exact signature-verified checkpoint
//      response frames before networking and exposes only aggregate vault health.
//  52. Runs bounded low-frequency coordinator witness reconciliation, storing
//      signed peer observations without treating witness count as consensus.
//  53. Keeps the public API startup route inventory aligned with the signed
//      checkpoint witness endpoint used by coordinator reconciliation.
//  54. Reports durable signed checkpoint observation freshness independently
//      from local evidence-vault integrity and request-attempt counters.
//  55. Reports each bounded coordinator witness round as privacy-safe aggregate
//      evidence without turning peer count into consensus or fork choice.
//  56. Verifies or initializes a signed coordinator-local commitment tip anchor
//      after full chain audit and before any block producer can start.
//  57. Checks only operator-pinned external checkpoint witnesses before opening
//      listeners, so permissionless peers cannot manufacture startup authority.
//
// ⚠️ Important Notes for Next Developer:
//   - traffic_tracker is Arc-shared between packet_handler (writes) and
//     heartbeat reporter (drains). Same instance, different usage patterns.
//   - heartbeat is declared `mut` in run() so builder calls can be chained
//     after init_management_reporter() returns.
//   - remove_wallet() is called AFTER sessions.remove() (inside
//     cleanup_expired). Order matters: session must be gone first.
//   - All other logic (VPN, MemChain, ChatRelay, Voice, SuperNode,
//     SaaS pool) remains backward-compatible with the previous version.
//   - Commitment block range frames are node-peer control traffic. Never route
//     them through a client session or add full memory records to the response.
//   - Block Sync v1 is single-writer: only an explicitly configured Local-mode
//     blind coordinator may pack blocks; all other nodes remain followers.
//   - Coordinator witness reconciliation is evidence collection only. It must
//     never mutate the canonical chain, elect a leader, or infer fork choice.
//   - Checkpoint freshness comes only from audited durable signed evidence.
//     Failed attempts and inbound served requests cannot refresh it.
//   - Witness round states are operator evidence only. Never use their counts
//     as votes, quorum, finality, leader election, or fork choice.
//   - Commitment coordinators must confirm SQLite FULL-or-stronger durability
//     before startup audit and before any block producer can start.
//   - The signed commitment tip anchor detects an older/replaced SQLite chain
//     only while the host-side anchor remains current. It does not detect a
//     whole-host snapshot rollback and is not consensus, quorum, or finality.
//   - External startup witnesses are explicit identity trust pins. Discovery
//     may rotate their signed endpoints, but unpinned peers stay evidence-only.
//   - The minimum verified witness count is an operator startup threshold over
//     distinct pins. It is not consensus, quorum, finality, or fork choice.
//   - encrypted_message_counter is aggregate only and never stores payload,
//     destination, DNS, URL, voucher, wallet, or client public IP details.
//   - dns_proxy forwards opaque DNS UDP payloads only; it does not parse,
//     log, store, or report queried domains.
//   - If vpn.dns_proxy_enabled=false, an external gateway DNS listener such as
//     systemd-resolved must own gateway_ip:53. The health endpoint verifies
//     that listener independently.
//
// Last Modified:
//   v2.7.16-WitnessThreshold - Enforced an operator-defined strict witness threshold
//   v2.7.15-ExternalWitnessGuard - Pinned pre-listener checkpoint startup gate
//   v2.7.14-CommitmentTipAnchor - Signed local high-water rollback guard
//   v2.7.11-CheckpointFreshness - Durable proof recency in status and heartbeat
//   v2.7.12-WitnessRoundEvidence - Privacy-safe bounded witness round coverage
//   v2.7.13-CommitmentDurability - Fail-closed coordinator SQLite durability
//   v2.7.9-CheckpointRouteInventory - Advertise checkpoint in startup route inventory
//   v2.7.8-CoordinatorWitness - Low-frequency signed peer checkpoint evidence
//   v2.7.6-EvidenceVault - Durable bounded checkpoint proofs and startup audit
//   v2.7.5-CheckpointProof - Signed cross-node tip reconciliation and convergence gate
//   v2.7.4-BlockIntegrityStatus - Privacy-safe verified chain evidence in status and heartbeat
//   v2.7.3-BlockAudit - Fail-closed startup verification for persisted commitment chains
//   v2.7.2-BlockSyncStatus - Runtime sync state, fault evidence, and heartbeat
//   v2.7.1-BlockFollower - Pinned coordinator catch-up with bounded retry/backoff
//   v2.7.0-BlockSync - Node-blind commitment packing, peer range API, coordinator fork guard
//   v1.2.8-MemChainStartupIntegrity - Reject tampered records during vector index recovery
//   v1.2.7-MemChainBlindVectorRecovery - Restore all owner/model vector partitions on blind storage nodes
//   v1.2.6-BlindRelayRuntimeHeartbeat - Promote aggregate blind relay runtime evidence in discovery_status heartbeat
//   v1.2.5-RouteGovernanceHeartbeat - Promote aggregate route governance in discovery_status heartbeat
//   v1.2.4-BlindRelayProbeRecoveryCooldown - Reprobe faster until two-hop delivery proof is healthy
//   v1.2.3-PeerCacheEmptyOverwriteGuard - Preserve usable cache when current PeerStore is empty
//   v1.2.2-ProbeCoveragePriority - Probe unproven non-quarantined peers before already-proven peers
//   v1.2.1-TwoHopOnionDeliveryProbe - Prefer synthetic onion delivery over control-plane proof
//   v1.2.0-TwoHopRuntimeProof - Low-frequency aggregate two-hop blind relay path proof
//   v1.1.9-BlindRelayProbeCooldown - Rate-limit synthetic relay probes so discovery health checks stay low-noise
//   v1.1.8-BlindRelaySyntheticProbe - Low-frequency opaque route probes after successful discovery gossip
//   v1.2.1-TwoHopSmokeTrigger - Add local-only operator smoke test for real two-hop onion delivery
//   v1.1.7-BlindRelayQualityReadiness - Expose aggregate blind relay quality in discovery readiness
//   v1.1.6-PeerRelayAckValidation - Require accepted peer relay ACK before route success
//   v1.1.5-DiscoveryReadinessHeartbeat - Add compact ChatRelay/quorum readiness to heartbeat
//   v1.1.4-PeerCacheLoadEvidence - Expose cache/backup startup load evidence
//   v1.1.3-DiscoveryStartupSelfCheck - Report discovery startup readiness buckets
//   v1.1.2-DurablePeerCacheFsync - Fsync PeerStore cache temp file and parent directory
//   v1.1.1-DiscoveryGossipBackpressure - Add jitter/backpressure for outbound discovery gossip
//   v1.1.0-CommercialPeerSummary - Tag PeerStore source buckets for nodeboard
//   v1.0.9-DiscoveryCacheSourcePriority - Load peer cache after static bootstrap seeds
//   v1.0.8-DiscoveryPeerCacheUsableFallback - Fallback when primary cache imports no usable peers
//   v1.0.7-DiscoveryPeerCacheBackup - Backup and fallback for PeerStore cache
//   v1.0.6-DiscoveryShutdownCacheFlush - Persist PeerStore cache on shutdown
//   v1.0.5-DiscoveryImmediateCacheSave - Persist PeerStore cache after bootstrap
//   v1.0.4-DiscoveryHealthContract - PeerStore summary in VPN health
//   v1.0.3-DNSOwnership - Honor vpn.dns_proxy_enabled before spawning DNS proxy
//   v1.0.2-DNSProxy - VPN gateway DNS proxy wiring
//   v2.5.3+Security    - Server::new() gains config_path
//   v1.0.0-MultiTenant - SaaS startup branch
//   v1.2.0-MultiDevice - ChatRelayService init
//   v1.0.0-Voice+SessionFix - Voice API, session fixes
//   v1.0.0-Membership  - TrafficTracker wiring
//   v1.0.1-VpnMessageStats - encrypted message counter wiring
//   v0.7.0-DiscoveryChatRelay - Peer-discovered encrypted chat relay fanout
//   v0.7.1-ChatPeerRelayHealth - Peer relay health status in heartbeat
//   v0.7.2-ChatRelayDisabledGate - Honor disabled chat relay config at runtime
//   v0.9.3-DiscoveryGossipHealth - Outbound gossip status and failure buckets
//   v0.9.1-DiscoverySeedStatus - Seed endpoint recovery counters in status
//   v0.9.2-DiscoveryBootstrapStatus - Avoid warning on benign stale bootstrap descriptors
//   v0.9.0-DiscoverySeedEndpoints - Periodic seed endpoint gossip recovery
//   v0.8.0-DiscoveryPublicApi - Optional public-only discovery listener
//   v0.5.0-DiscoveryPeerCache - Optional local PeerStore cache load/writeback
//   v0.6.0-DiscoveryOutboundGossip - Periodic descriptor announce + snapshot sync
//   v0.4.0-DiscoverySelfDescriptor - Generate and register signed self descriptor
//   v0.3.0-DiscoveryBootstrap - PeerStore bootstrap snapshot loading
// ============================================

use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use tokio::io::AsyncWriteExt;
use tokio::sync::{broadcast, mpsc, Mutex as TokioMutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};

use aeronyx_core::protocol::auth::{
    verify_signed_message, DOMAIN_CHAT_ACK, DOMAIN_CHAT_PULL, DOMAIN_DEVICE_REGISTER,
    DOMAIN_WALLET_PRESENCE,
};
use aeronyx_core::protocol::chat::{
    encode_envelope, BlindRelayEnvelope, ChatContentType, ChatEnvelope,
};
use sha2::{Digest, Sha256};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::crypto::transport::{
    DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD,
};
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::ledger::MemoryRecord;
use aeronyx_core::protocol::codec::{
    decode_client_hello, encode_data_packet, encode_server_hello, ProtocolCodec,
};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::messages::CLIENT_HELLO_SIZE;
use aeronyx_core::protocol::{
    build_onion_envelope, DataPacket, MessageType, NodeBootstrapSnapshot, NodeCapability,
    NodeCapacity, NodeDescriptor, NodeDiscoveryMessage, NodePolicy, OnionHop, SignedNodeDescriptor,
};
use aeronyx_transport::traits::{Transport, TunConfig, TunDevice};
use aeronyx_transport::UdpTransport;

#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use rusqlite::OptionalExtension;

use crate::api::auth::{ensure_jwt_secret, generate_secret};
use crate::api::chat_peer::{
    build_chat_peer_router, PeerBlindRelayRequest, PeerBlindRelayResponse, PeerChatRelayRequest,
    PeerChatRelayResponse,
};
use crate::api::discovery::{
    blind_relay_runtime_status_value, build_discovery_router_with_local_status,
    discovery_readiness_status_value, DiscoveryApiPolicy, DiscoveryLocalCapabilityStatus,
    GossipResponse,
};
use crate::api::memchain_peer::{
    build_memchain_peer_router, pull_record_commitment_checkpoint, pull_record_commitment_page,
    reconcile_record_commitment_pinned_witnesses, reconcile_record_commitment_witnesses,
    CommitmentCheckpointRelation, CommitmentReconciliationOutcome,
};
use crate::api::mpi::{build_mpi_router, BaselineSnapshot, Mode, MpiState};
use crate::api::voice::build_voice_router;
use crate::api::vpn_health::{
    build_vpn_health_router, collect_node_operator_status_value, collect_vpn_health_value,
};
use crate::config::{
    DiscoveryConfig, MemChainConfig, MemChainMode, ServerConfig, VectorQuantizationMode,
};
use crate::error::{Result, ServerError};
use crate::handlers::packet::DecryptedPayload;
use crate::handlers::PacketHandler;
use crate::management::{
    reporter::{SessionEventSender, SessionQuality},
    CommandHandler, HeartbeatReporter, ManagementClient, SessionReporter,
};
use crate::miner::ReflectionMiner;
use crate::services::chat_relay::{derive_node_secret, ChatRelayService};
use crate::services::memchain::derive_rawlog_key;
use crate::services::memchain::derive_record_key;
use crate::services::memchain::EmbedEngine;
use crate::services::memchain::NerEngine;
use crate::services::memchain::RerankerEngine;
use crate::services::memchain::{
    ensure_volumes_config, StoragePool, SystemDb, VectorIndexPool, VolumeRouter,
};
#[allow(deprecated)]
use crate::services::memchain::{AofWriter, MemPool, MemoryStorage, VectorIndex};
use crate::services::memchain::{LlmRouter, TaskWorker};
use crate::services::{
    spawn_dns_proxy, HandshakeService, IpPoolService, NodePolicyRuntime, PeerStore, RoutingService,
    SessionManager,
};
// v1.0.0-Membership
use crate::services::deny_list::DenyList;
use crate::services::session::StatsSnapshot;
use crate::services::traffic_tracker::TrafficTracker;
use crate::voucher_verifier::VoucherVerifier;

// ============================================
// Constants
// ============================================

const KEEPALIVE_PACKET_SIZE: usize = 17;
#[allow(dead_code)]
const DISCONNECT_PACKET_MIN_SIZE: usize = 18;
const COMMAND_CHANNEL_BUFFER: usize = 100;
const QUANTIZER_CAL_KEY_PREFIX: &str = "quantizer_cal";
const POOL_EVICTION_INTERVAL_SECS: u64 = 300;
const MINER_SCHEDULER_TICK_SECS: u64 = 60;
const KEEPALIVE_PROBE_INTERVAL_SECS: u64 = 60;
const KEEPALIVE_ACK_TIMEOUT_SECS: u64 = 90;
const CHAT_PEER_RELAY_FANOUT_LIMIT: usize = 3;
const BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS: u64 = 15 * 60;
const BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS: u64 = 60;

/// Return the privacy-safe reason a persisted MemChain record must not enter
/// the in-memory recall index.
///
/// Sighted records retain backward compatibility: their content-addressed ID
/// is required, but legacy zero signatures remain accepted. Node-blind records
/// additionally require the authenticated owner's Ed25519 signature because
/// the node is not allowed to re-sign client ciphertext.
fn memchain_index_rejection_reason(record: &MemoryRecord) -> Option<&'static str> {
    if !record.verify_id() {
        return Some("record_id_mismatch");
    }

    if record.blind {
        let owner_key = match IdentityPublicKey::from_bytes(&record.owner) {
            Ok(key) => key,
            Err(_) => return Some("owner_key_invalid"),
        };
        if owner_key
            .verify(&record.record_id, &record.signature)
            .is_err()
        {
            return Some("owner_signature_invalid");
        }
    }

    None
}

// ============================================
// Server
// ============================================

fn quality_from_stats(snap: StatsSnapshot) -> SessionQuality {
    let rejects = snap.replays_rejected + snap.too_old_rejected;
    let accepted = snap.packets_rx + snap.packets_tx;
    let packet_loss = if accepted + rejects > 0 {
        Some((rejects as f64 / (accepted + rejects) as f64) * 100.0)
    } else {
        None
    };

    SessionQuality {
        last_rx_at: (snap.last_rx_at > 0).then_some(snap.last_rx_at),
        last_tx_at: (snap.last_tx_at > 0).then_some(snap.last_tx_at),
        rtt_ms: (snap.rtt_us > 0).then_some(snap.rtt_us as f64 / 1000.0),
        packet_loss,
        replay_rejections: Some(snap.replays_rejected),
        too_old_rejections: Some(snap.too_old_rejected),
        packets_rx: Some(snap.packets_rx),
        packets_tx: Some(snap.packets_tx),
        keepalive_probes_sent: Some(snap.keepalive_probes_sent),
        keepalive_acks: Some(snap.keepalive_acks),
        keepalive_missed: Some(snap.keepalive_missed),
        keepalive_pending: Some(snap.keepalive_pending),
    }
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Typed startup outcomes keep security policy branches exhaustive.
///
/// These variants describe an operator-pinned evidence policy. They are not
/// network votes, consensus, quorum, finality, leader election, or fork choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentWitnessStartupDecision {
    Verified,
    DegradedUnverified,
    DegradedBelowThreshold,
}

impl CommitmentWitnessStartupDecision {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::DegradedUnverified => "degraded_unverified",
            Self::DegradedBelowThreshold => "degraded_below_threshold",
        }
    }
}

/// Fail-closed reasons produced by authenticated witness evidence or policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentWitnessStartupBlockReason {
    Divergence,
    RemoteAhead,
    Unavailable,
    ThresholdUnmet,
}

impl CommitmentWitnessStartupBlockReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Divergence => "signed_checkpoint_divergence",
            Self::RemoteAhead => "signed_checkpoint_remote_ahead",
            Self::Unavailable => "signed_checkpoint_unavailable",
            Self::ThresholdUnmet => "signed_checkpoint_threshold_unmet",
        }
    }
}

impl std::fmt::Display for CommitmentWitnessStartupBlockReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

fn commitment_witness_startup_decision(
    round: &CommitmentReconciliationOutcome,
    signed_evidence_required: bool,
    minimum_verified_witnesses: usize,
) -> std::result::Result<CommitmentWitnessStartupDecision, CommitmentWitnessStartupBlockReason> {
    if round.diverged > 0 {
        return Err(CommitmentWitnessStartupBlockReason::Divergence);
    }
    if round.remote_ahead > 0 {
        return Err(CommitmentWitnessStartupBlockReason::RemoteAhead);
    }
    let minimum_verified_witnesses = minimum_verified_witnesses.max(1);
    if round.verified < minimum_verified_witnesses {
        if signed_evidence_required {
            return if round.verified == 0 {
                Err(CommitmentWitnessStartupBlockReason::Unavailable)
            } else {
                Err(CommitmentWitnessStartupBlockReason::ThresholdUnmet)
            };
        }
        return if round.verified == 0 {
            Ok(CommitmentWitnessStartupDecision::DegradedUnverified)
        } else {
            Ok(CommitmentWitnessStartupDecision::DegradedBelowThreshold)
        };
    }
    Ok(CommitmentWitnessStartupDecision::Verified)
}

pub struct Server {
    config: ServerConfig,
    identity: IdentityKeyPair,
    config_path: Option<PathBuf>,
    shutdown: Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Server {
    pub fn new(
        config: ServerConfig,
        identity: IdentityKeyPair,
        config_path: Option<PathBuf>,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            config,
            identity,
            config_path,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
        }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting AeroNyx server v{}", env!("CARGO_PKG_VERSION"));

        // Onion routing forward secrecy: initialize the in-memory rotating onion
        // key at startup. It is never persisted, so a restart yields a fresh key
        // and the old secret is unrecoverable. The grace window is tied to the
        // descriptor TTL so a previous key outlives any descriptor still in
        // circulation. See services::onion_keys.
        crate::services::onion_keys::init_shared(
            unix_now_secs(),
            self.config.discovery.descriptor_ttl_secs,
        );

        if self.config.memchain.is_enabled() {
            self.ensure_api_secret_on_disk().await;
        }

        let (ip_pool, sessions, routing) = self.init_services()?;

        let (storage, vector_index, mempool, aof_writer) = if self.config.memchain.is_enabled() {
            let (st, vi, mp, aw) = self.init_memchain().await?;
            (Some(st), Some(vi), Some(mp), Some(aw))
        } else {
            info!("[MEMCHAIN] Disabled (mode=off)");
            (None, None, None, None)
        };
        if let Some(ref commitment_storage) = storage {
            commitment_storage.configure_record_commitment_sync(
                self.config.memchain.commitment_coordinator_enabled,
                self.config.memchain.commitment_sync_enabled,
            );
        }

        let chat_relay_enabled = self.config.memchain.is_chat_relay_enabled();
        let chat_relay: Option<Arc<ChatRelayService>> = if chat_relay_enabled {
            let node_secret = derive_node_secret(&self.identity.to_bytes());
            match ChatRelayService::new(self.config.memchain.chat_relay.clone(), node_secret) {
                Ok(svc) => {
                    info!("[CHAT_RELAY] Service initialized");
                    Some(Arc::new(svc))
                }
                Err(e) => {
                    warn!(error = %e, "[CHAT_RELAY] Init failed — chat relay disabled");
                    None
                }
            }
        } else {
            if self.config.memchain.is_enabled() {
                info!(
                    "[CHAT_RELAY] Disabled by memchain.chat_relay.enabled=false; chat routes remain unavailable"
                );
            }
            None
        };
        let chat_relay_runtime_ready = chat_relay.is_some();

        let peer_store = self.init_peer_store(chat_relay_runtime_ready).await;
        if let Some(ref commitment_storage) = storage {
            self.verify_memchain_commitment_startup_witnesses(commitment_storage, &peer_store)
                .await?;
        }
        let _peer_store_persistence_task =
            self.spawn_peer_store_persistence_task(Arc::clone(&peer_store));
        let _discovery_gossip_task =
            self.spawn_discovery_gossip_task(Arc::clone(&peer_store), chat_relay_runtime_ready);

        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.listen_addr())
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind: {}", e)))?,
        );
        info!("UDP transport listening on {}", self.config.listen_addr());

        #[cfg(target_os = "linux")]
        let tun = self.init_tun().await?;

        let server_pubkey_hex = hex::encode(self.identity.public_key_bytes());
        let mut tasks: Vec<(&str, JoinHandle<()>)> = Vec::new();

        // AeroNyx client readiness requires DNS to be available at the tunnel
        // gateway. When enabled, this proxy forwards opaque UDP DNS bytes only
        // and never records queried domains, DNS contents, destinations, or
        // client IPs. Operators may disable it when systemd-resolved or another
        // hardened host resolver intentionally owns gateway_ip:53.
        if self.config.dns_proxy_enabled() {
            let dns_task = spawn_dns_proxy(self.config.gateway_ip(), self.shutdown_tx.subscribe());
            tasks.push(("dns-proxy", dns_task));
        } else {
            info!(
                gateway_ip = %self.config.gateway_ip(),
                "[DNS] Built-in VPN DNS proxy disabled by vpn.dns_proxy_enabled=false; expecting external gateway DNS listener"
            );
        }

        // v1.0.0-Membership: TrafficTracker must be created before
        // PacketHandler AND before init_management_reporter so both
        // can receive the same Arc.
        let traffic_tracker = Arc::new(TrafficTracker::new());
        let encrypted_message_counter = Arc::new(AtomicU64::new(0));
        // v1.0.0-Membership: DenyList shared between HandshakeService
        // (read: reject denied wallets) and HeartbeatReporter (write: add/remove entries).
        let deny_list = Arc::new(DenyList::new());
        let node_policy = Arc::new(NodePolicyRuntime::default());

        let packet_handler = Arc::new(PacketHandler::new(
            Arc::clone(&sessions),
            Arc::clone(&routing),
            Arc::clone(&traffic_tracker),
            Arc::clone(&encrypted_message_counter),
            Arc::clone(&node_policy),
        ));

        let handshake_service = Arc::new(HandshakeService::new(
            self.identity.clone(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            Arc::clone(&deny_list),
            Arc::clone(&node_policy),
        ));

        // [VOUCHER-P1] Observe-only verifier. It never rejects handshakes in
        // this phase; it records valid/invalid/missing voucher rates first.
        let voucher_verifier = Arc::new(VoucherVerifier::new());

        // init_management_reporter needs udp + traffic_tracker,
        // so it is called here after both are available.
        let session_event_sender = self
            .init_management_reporter(
                &sessions,
                Arc::clone(&ip_pool),
                Arc::clone(&udp),
                Arc::clone(&traffic_tracker),
                Arc::clone(&deny_list),
                Arc::clone(&node_policy),
                Arc::clone(&voucher_verifier),
                Arc::clone(&encrypted_message_counter),
                Arc::clone(&packet_handler),
                Arc::clone(&peer_store),
                storage.clone(),
                chat_relay.clone(),
                chat_relay_enabled,
            )
            .await;

        let udp_task = self.spawn_udp_task(
            Arc::clone(&udp),
            #[cfg(target_os = "linux")]
            Arc::clone(&tun),
            Arc::clone(&handshake_service),
            Arc::clone(&packet_handler),
            Arc::clone(&voucher_verifier),
            Arc::clone(&sessions),
            session_event_sender.clone(),
            mempool.clone(),
            aof_writer.clone(),
            storage.clone(),
            vector_index.clone(),
            self.config.memchain.clone(),
            server_pubkey_hex.clone(),
            chat_relay.clone(),
            Arc::clone(&routing),
            Arc::clone(&peer_store),
        );
        tasks.push(("udp", udp_task));

        #[cfg(target_os = "linux")]
        {
            let tun_task = self.spawn_tun_task(
                Arc::clone(&tun),
                Arc::clone(&udp),
                Arc::clone(&packet_handler),
            );
            tasks.push(("tun", tun_task));
        }

        let cleanup_task = self.spawn_cleanup_task(
            Arc::clone(&sessions),
            Arc::clone(&ip_pool),
            Arc::clone(&routing),
            session_event_sender.clone(),
            chat_relay.clone(),
            Arc::clone(&traffic_tracker),
            Arc::clone(&deny_list),
        );
        tasks.push(("cleanup", cleanup_task));

        let snapshot_task = self.spawn_traffic_snapshot_task(
            Arc::clone(&sessions),
            session_event_sender.clone(),
            self.config.management.session_report_interval_secs,
        );
        tasks.push(("traffic-snapshot", snapshot_task));

        let keepalive_task = self.spawn_keepalive_probe_task(
            Arc::clone(&sessions),
            Arc::clone(&udp),
            Arc::clone(&packet_handler),
            self.config.gateway_ip(),
        );
        tasks.push(("vpn-keepalive", keepalive_task));

        if let Some(ref relay) = chat_relay {
            let routes = Arc::clone(&relay.wallet_routes);
            let mut rx = self.shutdown_tx.subscribe();
            tasks.push((
                "wallet-routes-cleanup",
                tokio::spawn(async move {
                    let mut interval = tokio::time::interval(Duration::from_secs(60));
                    loop {
                        tokio::select! {
                            _ = rx.recv() => break,
                            _ = interval.tick() => {
                                let evicted = routes.cleanup_stale(Duration::from_secs(300));
                                if evicted > 0 {
                                    debug!(evicted, "[CHAT_RELAY] Stale wallet routes evicted");
                                }
                            }
                        }
                    }
                }),
            ));
            info!("[CHAT_RELAY] Wallet route cleanup task started (ttl=300s, interval=60s)");
        }

        if let (Some(ref st), Some(ref vi), Some(ref mp), Some(ref aw)) =
            (&storage, &vector_index, &mempool, &aof_writer)
        {
            let is_saas = self.config.memchain.mode == MemChainMode::Saas;

            let user_weights = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()));

            if !is_saas {
                let owner = self.identity.public_key_bytes();
                if let Some(blob) = st.load_user_weights(&owner).await {
                    if let Some(w) = crate::services::memchain::mvf::WeightVector::from_bytes(&blob)
                    {
                        let mut map = user_weights.write();
                        map.insert(hex::encode(owner), w);
                        info!("[MEMCHAIN] Loaded MVF user weights from SQLite");
                    }
                }
            }

            let mvf_baseline: Option<BaselineSnapshot> = if !is_saas {
                let conn = st.conn_lock().await;
                let raw: Option<Vec<u8>> = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = 'mvf_baseline'",
                        [],
                        |row: &rusqlite::Row<'_>| row.get::<_, Vec<u8>>(0),
                    )
                    .optional()
                    .unwrap_or(None);
                drop(conn);
                raw.and_then(|bytes| {
                    serde_json::from_str::<BaselineSnapshot>(&String::from_utf8_lossy(&bytes)).ok()
                })
            } else {
                None
            };

            let owner_key = self.identity.public_key_bytes();
            let api_secret = self
                .config
                .memchain
                .effective_api_secret()
                .map(|s| s.to_string());

            let embed_engine = self.init_embed_engine();
            let ner_engine = self.init_ner_engine();
            let reranker_engine = self.init_reranker_engine();

            let llm_router: Option<Arc<LlmRouter>> = self.init_llm_router().await;

            if llm_router.is_some() {
                let timeout_secs = self.config.memchain.supernode.worker.task_timeout_secs as i64;
                let recovered = st.reset_stale_processing_tasks(timeout_secs).await;
                if recovered > 0 {
                    info!(recovered, timeout_secs, "[SUPERNODE] Recovered stale tasks");
                }
            }

            let mpi_state = if is_saas {
                self.init_saas_mpi_state(
                    st,
                    owner_key,
                    api_secret,
                    Arc::clone(&user_weights),
                    embed_engine.clone(),
                    ner_engine.clone(),
                    reranker_engine,
                    llm_router.clone(),
                )
                .await?
            } else {
                let mpi = MpiState::local(
                    Arc::clone(st),
                    Arc::clone(vi),
                    self.identity.clone(),
                    parking_lot::RwLock::new(std::collections::HashMap::new()),
                    std::sync::atomic::AtomicBool::new(false),
                    Arc::clone(&user_weights),
                    self.config.memchain.mvf_alpha,
                    self.config.memchain.mvf_enabled,
                    parking_lot::RwLock::new(std::collections::HashMap::new()),
                    parking_lot::RwLock::new(mvf_baseline),
                    owner_key,
                    api_secret,
                    embed_engine.clone(),
                    self.config.memchain.allow_remote_storage,
                    self.config.memchain.blind_storage_enabled,
                    self.config.memchain.max_remote_owners,
                    ner_engine.clone(),
                    self.config.memchain.graph_enabled,
                    self.config.memchain.entropy_filter_enabled,
                    reranker_engine,
                    Some(derive_rawlog_key(&self.identity.to_bytes())),
                    llm_router.clone(),
                );
                Arc::new(mpi)
            };

            if !is_saas {
                let owner_hex = hex::encode(owner_key);
                let identity_records = st
                    .get_active_records(
                        &owner_key,
                        Some(aeronyx_core::ledger::MemoryLayer::Identity),
                        100,
                    )
                    .await;
                if !identity_records.is_empty() {
                    let mut cache = mpi_state.identity_cache.write();
                    cache.insert(owner_hex, identity_records);
                }
                mpi_state
                    .index_ready
                    .store(true, std::sync::atomic::Ordering::Relaxed);

                if self.config.memchain.mvf_enabled && mpi_state.mvf_baseline.read().is_none() {
                    let feedback = st.get_recent_feedback(200).await;
                    if !feedback.is_empty() {
                        let positive = feedback.iter().filter(|(s, _)| *s == 1).count();
                        let rate = positive as f32 / feedback.len() as f32;
                        let now_ts = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64;

                        let baseline = BaselineSnapshot {
                            positive_rate: rate,
                            sample_size: feedback.len(),
                            frozen_at: now_ts,
                        };

                        if let Ok(json) = serde_json::to_string(&baseline) {
                            let conn = st.conn_lock().await;
                            let _ = conn.execute(
                                "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('mvf_baseline', ?1)",
                                rusqlite::params![json.as_bytes()],
                            );
                        }

                        *mpi_state.mvf_baseline.write() = Some(baseline);
                        info!(rate, samples = feedback.len(), "[MVF] Baseline frozen");
                    }
                }
            }

            let api_task = self.start_combined_api(
                self.config.memchain.api_listen_addr,
                Arc::clone(&mpi_state),
                Arc::clone(mp),
                Arc::clone(aw),
                Arc::clone(&ip_pool),
                Arc::clone(&sessions),
                Arc::clone(&udp),
                Arc::clone(&node_policy),
                Arc::clone(&voucher_verifier),
                Arc::clone(&encrypted_message_counter),
                Arc::clone(&packet_handler),
                Arc::clone(&peer_store),
                chat_relay.clone(),
                Arc::clone(&udp),
            );
            tasks.push(("memchain-api", api_task));

            if let Some(sync_task) =
                self.spawn_memchain_commitment_sync_task(Arc::clone(st), Arc::clone(&peer_store))
            {
                tasks.push(("memchain-block-sync", sync_task));
            }
            if let Some(reconciliation_task) = self.spawn_memchain_commitment_reconciliation_task(
                Arc::clone(st),
                Arc::clone(&peer_store),
            ) {
                tasks.push(("memchain-checkpoint-witness", reconciliation_task));
            }

            if let Some(ref router) = llm_router {
                let worker = TaskWorker::new(
                    Arc::clone(st),
                    Arc::clone(router),
                    self.config.memchain.supernode.worker.clone(),
                );
                let worker_shutdown = self.shutdown_tx.subscribe();
                tasks.push((
                    "supernode-worker",
                    tokio::spawn(async move {
                        worker.run(worker_shutdown).await;
                    }),
                ));
                info!(
                    poll_interval = self.config.memchain.supernode.worker.poll_interval_secs,
                    max_concurrent = self.config.memchain.supernode.worker.max_concurrent,
                    "[SUPERNODE] TaskWorker spawned"
                );
            }

            if is_saas {
                if let (Some(ref sp), Some(ref vp)) =
                    (&mpi_state.storage_pool, &mpi_state.vector_pool)
                {
                    let sp_clone = Arc::clone(sp);
                    let vp_clone = Arc::clone(vp);
                    let mut evict_rx = self.shutdown_tx.subscribe();
                    tasks.push((
                        "pool-eviction",
                        tokio::spawn(async move {
                            let mut interval = tokio::time::interval(Duration::from_secs(
                                POOL_EVICTION_INTERVAL_SECS,
                            ));
                            loop {
                                tokio::select! {
                                    _ = evict_rx.recv() => break,
                                    _ = interval.tick() => {
                                        let evicted_s = sp_clone.evict_idle().await;
                                        let evicted_v = vp_clone.evict_idle();
                                        if evicted_s + evicted_v > 0 {
                                            info!(
                                                storage = evicted_s,
                                                vector  = evicted_v,
                                                "[POOL] Evicted idle connections"
                                            );
                                        }
                                    }
                                }
                            }
                        }),
                    ));
                    info!(
                        interval_secs = POOL_EVICTION_INTERVAL_SECS,
                        "[POOL] Eviction timer started"
                    );
                }
            }

            if self.config.memchain.miner_interval_secs > 0 {
                if is_saas {
                    if let (Some(ref sp), Some(ref sys_db)) =
                        (&mpi_state.storage_pool, &mpi_state.system_db)
                    {
                        let scheduler = crate::miner::MinerScheduler::new(
                            Arc::clone(sp),
                            Arc::clone(sys_db),
                            self.config
                                .memchain
                                .saas
                                .as_ref()
                                .map(|s| s.miner_max_owners_per_tick)
                                .unwrap_or(10),
                            self.config
                                .memchain
                                .saas
                                .as_ref()
                                .map(|s| s.miner_max_rounds_per_hour)
                                .unwrap_or(6) as u32,
                            self.identity.clone(),
                            llm_router.clone(),
                            embed_engine.clone(),
                            ner_engine.clone(),
                        )
                        .await;
                        let mut sched_rx = self.shutdown_tx.subscribe();
                        tasks.push((
                            "miner-scheduler",
                            tokio::spawn(async move {
                                let mut interval = tokio::time::interval(Duration::from_secs(
                                    MINER_SCHEDULER_TICK_SECS,
                                ));
                                loop {
                                    tokio::select! {
                                        _ = sched_rx.recv() => break,
                                        _ = interval.tick() => { scheduler.tick().await; }
                                    }
                                }
                            }),
                        ));
                        info!(
                            tick_secs = MINER_SCHEDULER_TICK_SECS,
                            "[MINER] SaaS MinerScheduler started"
                        );
                    }
                } else {
                    let miner = ReflectionMiner::new(
                        self.config.memchain.miner_interval_secs,
                        Arc::clone(st),
                        Arc::clone(vi),
                        self.identity.clone(),
                        Arc::clone(mp),
                        Arc::clone(aw),
                        Arc::clone(&sessions),
                        Arc::clone(&udp),
                    )
                    .with_compaction_threshold(self.config.memchain.compaction_threshold)
                    .with_mvf(self.config.memchain.mvf_enabled, Arc::clone(&user_weights))
                    .with_commitment_coordinator(
                        self.config.memchain.commitment_coordinator_enabled,
                    );

                    let miner = if let Some(ref ee) = embed_engine {
                        miner.with_embed_engine(Arc::clone(ee))
                    } else {
                        miner
                    };
                    let miner = if let Some(ref ne) = ner_engine {
                        miner.with_ner_engine(Arc::clone(ne))
                    } else {
                        miner
                    };
                    let miner = if let Some(ref lr) = llm_router {
                        miner.with_llm_router(Arc::clone(lr))
                    } else {
                        miner
                    };

                    // Reconcile a bounded backlog before announcing startup.
                    // This packs only verified opaque commitments; it never
                    // copies memory payloads to peers. Remaining backlog, if
                    // any, is drained by the normal bounded miner tick.
                    let bootstrap_blocks = miner.pack_commitment_blocks(64).await;
                    if bootstrap_blocks > 0 {
                        info!(
                            blocks = bootstrap_blocks,
                            "[MEMCHAIN_BLOCK] Startup commitment backlog reconciled"
                        );
                    }

                    let miner_shutdown = self.shutdown_tx.subscribe();
                    tasks.push((
                        "miner",
                        tokio::spawn(async move {
                            miner.run(miner_shutdown).await;
                        }),
                    ));
                }
            } else {
                info!("[MINER] Disabled (interval=0)");
            }
        }

        info!("Server started successfully");
        self.wait_for_shutdown().await;
        info!("Shutting down server...");

        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());

        for (name, task) in tasks {
            match tokio::time::timeout(Duration::from_secs(5), task).await {
                Ok(Ok(())) => debug!("Task '{}' completed", name),
                Ok(Err(e)) => warn!("Task '{}' failed: {}", name, e),
                Err(_) => warn!("Task '{}' timed out", name),
            }
        }

        if let Err(e) = udp.shutdown().await {
            warn!("UDP shutdown error: {}", e);
        }

        if let (Some(ref st), Some(ref mp), Some(ref aw)) = (&storage, &mempool, &aof_writer) {
            info!(
                sqlite = st.count().await,
                mempool = mp.count(),
                aof = aw.lock().await.write_count(),
                "Shutdown complete (MemChain stats)"
            );
        } else {
            info!("Shutdown complete");
        }

        Ok(())
    }

    // ============================================
    // Auto-generate api_secret
    // ============================================

    async fn ensure_api_secret_on_disk(&self) {
        if self.config.memchain.effective_api_secret().is_some() {
            return;
        }
        let Some(ref path) = self.config_path else {
            return;
        };
        let secret = generate_secret();
        match crate::api::auth::write_secret_to_config_pub(path, "api_secret", &secret) {
            Ok(()) => {
                info!(path = %path.display(), "[SECURITY] Auto-generated api_secret written to config")
            }
            Err(e) => warn!(error = %e, "[SECURITY] Failed to persist api_secret"),
        }
    }

    // ============================================
    // SaaS MpiState init
    // ============================================

    #[allow(clippy::too_many_arguments)]
    async fn init_saas_mpi_state(
        &self,
        _server_storage: &Arc<MemoryStorage>,
        owner_key: [u8; 32],
        api_secret: Option<String>,
        user_weights: Arc<
            parking_lot::RwLock<
                std::collections::HashMap<String, crate::services::memchain::mvf::WeightVector>,
            >,
        >,
        embed_engine: Option<Arc<EmbedEngine>>,
        ner_engine: Option<Arc<NerEngine>>,
        reranker_engine: Option<Arc<RerankerEngine>>,
        llm_router: Option<Arc<LlmRouter>>,
    ) -> Result<Arc<MpiState>> {
        let saas_cfg = self.config.memchain.saas.as_ref().ok_or_else(|| {
            ServerError::startup_failed("mode=saas requires [memchain.saas] config section")
        })?;

        let data_root = &saas_cfg.data_root;

        tokio::fs::create_dir_all(data_root).await.map_err(|e| {
            ServerError::startup_failed(format!("SaaS data_root '{}': {}", data_root.display(), e))
        })?;

        let system_db = SystemDb::open(&data_root.join("system.db"))
            .await
            .map_err(|e| ServerError::startup_failed(format!("SystemDb: {}", e)))?;

        let volumes_config_path = ensure_volumes_config(data_root)
            .map_err(|e| ServerError::startup_failed(format!("volumes.toml: {}", e)))?;

        let volume_router = VolumeRouter::new(&volumes_config_path, Arc::clone(&system_db))
            .await
            .map_err(|e| ServerError::startup_failed(format!("VolumeRouter: {}", e)))?;

        let storage_pool = StoragePool::new(
            Arc::clone(&volume_router),
            Arc::clone(&system_db),
            saas_cfg.pool_max_connections,
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
        );

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let saturation_threshold = if self.config.memchain.vector_early_termination {
            0.001_f32
        } else {
            0.0_f32
        };

        let vector_pool = VectorIndexPool::new(
            Arc::clone(&volume_router),
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
            quantization_enabled,
            saturation_threshold,
        );

        let jwt_secret = ensure_jwt_secret(
            self.config.memchain.jwt_secret.as_deref(),
            self.config_path.as_deref(),
        )
        .map_err(|e| ServerError::startup_failed(format!("jwt_secret: {}", e)))?;

        info!(
            data_root     = %data_root.display(),
            pool_max      = saas_cfg.pool_max_connections,
            idle_timeout  = saas_cfg.pool_idle_timeout_secs,
            "[SAAS] Infrastructure initialized"
        );

        let mpi_state = Arc::new(MpiState {
            mode: Mode::Saas,
            storage: None,
            vector_index: None,
            identity: self.identity.clone(),
            identity_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            index_ready: std::sync::atomic::AtomicBool::new(true),
            user_weights,
            mvf_alpha: self.config.memchain.mvf_alpha,
            mvf_enabled: self.config.memchain.mvf_enabled,
            session_embeddings: parking_lot::RwLock::new(std::collections::HashMap::new()),
            mvf_baseline: parking_lot::RwLock::new(None),
            owner_key,
            api_secret,
            embed_engine,
            allow_remote_storage: false,
            blind_storage_enabled: self.config.memchain.blind_storage_enabled,
            max_remote_owners: 0,
            ner_engine,
            graph_enabled: self.config.memchain.graph_enabled,
            entropy_filter_enabled: self.config.memchain.entropy_filter_enabled,
            reranker_engine,
            rawlog_key: Some(derive_rawlog_key(&self.identity.to_bytes())),
            llm_router,
            storage_pool: Some(storage_pool),
            vector_pool: Some(vector_pool),
            volume_router: Some(volume_router),
            system_db: Some(system_db),
            jwt_secret: Some(jwt_secret),
            token_ttl_secs: self.config.memchain.token_ttl_secs,
            pool_max_connections: saas_cfg.pool_max_connections,
            pool_idle_timeout_secs: saas_cfg.pool_idle_timeout_secs,
        });

        Ok(mpi_state)
    }

    // ============================================
    // Engine initialization
    // ============================================

    fn init_embed_engine(&self) -> Option<Arc<EmbedEngine>> {
        if !self.config.memchain.embed_enabled {
            info!("[EMBED] Local embedding engine disabled by memchain.embed_enabled=false");
            return None;
        }

        let model_path = &self.config.memchain.embed_model_path;
        match EmbedEngine::load(
            model_path,
            self.config.memchain.embed_max_tokens,
            self.config.memchain.embed_output_dim,
        ) {
            Ok(engine) => {
                info!(model = %model_path, model_type = %engine.model_type(), dim = engine.dim(), "[EMBED] Local embedding engine loaded");
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!(model = %model_path, error = %e, "[EMBED] Unavailable");
                None
            }
        }
    }

    fn init_ner_engine(&self) -> Option<Arc<NerEngine>> {
        if !self.config.memchain.ner_enabled {
            debug!("[NER] Disabled");
            return None;
        }
        let model_path = &self.config.memchain.ner_model_path;
        let threshold = self.config.memchain.ner_confidence_threshold;
        match NerEngine::load(model_path, threshold, 0) {
            Ok(engine) => {
                info!(model = %model_path, threshold, "[NER] Local NER engine loaded");
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!(model = %model_path, error = %e, "[NER] Unavailable");
                None
            }
        }
    }

    fn init_reranker_engine(&self) -> Option<Arc<RerankerEngine>> {
        if !self.config.memchain.reranker_enabled {
            debug!("[RERANKER] Disabled");
            return None;
        }
        let model_path = &self.config.memchain.reranker_model_path;
        let max_seq = self.config.memchain.reranker_max_seq_length;
        match RerankerEngine::load(model_path, max_seq) {
            Ok(engine) => {
                info!(model = %model_path, blend_weight = %RerankerEngine::blend_weight(), "[RERANKER] Cross-encoder loaded");
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!(model = %model_path, error = %e, "[RERANKER] Unavailable");
                None
            }
        }
    }

    // ============================================
    // LlmRouter initialization
    // ============================================

    async fn init_llm_router(&self) -> Option<Arc<LlmRouter>> {
        use crate::config_supernode::ProviderType;
        use crate::services::memchain::{AnthropicProvider, LlmProvider, OpenAiCompatProvider};

        if !self.config.memchain.is_supernode_enabled() {
            debug!("[SUPERNODE] Disabled");
            return None;
        }

        let supernode = &self.config.memchain.supernode;
        if supernode.providers.is_empty() {
            warn!("[SUPERNODE] enabled=true but no providers — disabled");
            return None;
        }

        let mut providers: Vec<(String, String, String, Arc<dyn LlmProvider>)> = Vec::new();

        for provider_cfg in &supernode.providers {
            let api_key: Option<String> = provider_cfg.api_key.as_ref().map(|k| {
                if k.starts_with('$') {
                    std::env::var(&k[1..]).unwrap_or_else(|_| {
                        warn!(key = %k, provider = %provider_cfg.name, "[SUPERNODE] ENV var not set for api_key");
                        String::new()
                    })
                } else { k.clone() }
            });

            let api_base = if provider_cfg.api_base.is_empty()
                && provider_cfg.provider_type == ProviderType::Anthropic
            {
                "https://api.anthropic.com".to_string()
            } else {
                provider_cfg.api_base.clone()
            };

            let provider: Arc<dyn LlmProvider> = match provider_cfg.provider_type {
                ProviderType::OpenaiCompatible => {
                    match OpenAiCompatProvider::new(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        api_key.unwrap_or_default(),
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    ) {
                        Ok(p) => Arc::new(p),
                        Err(e) => {
                            warn!(provider = %provider_cfg.name, error = %e, "[SUPERNODE] OpenAiCompatProvider failed");
                            continue;
                        }
                    }
                }
                ProviderType::Anthropic => {
                    let key = match api_key {
                        Some(k) if !k.is_empty() => k,
                        _ => {
                            warn!(provider = %provider_cfg.name, "[SUPERNODE] Anthropic requires api_key");
                            continue;
                        }
                    };
                    match AnthropicProvider::new(
                        provider_cfg.name.clone(),
                        key,
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    ) {
                        Ok(p) => Arc::new(p),
                        Err(e) => {
                            warn!(provider = %provider_cfg.name, error = %e, "[SUPERNODE] AnthropicProvider failed");
                            continue;
                        }
                    }
                }
            };

            info!(name = %provider_cfg.name, type_ = ?provider_cfg.provider_type, model = %provider_cfg.model, api_base = %api_base, "[SUPERNODE] Provider registered");
            providers.push((
                provider_cfg.name.clone(),
                api_base,
                provider_cfg.model.clone(),
                provider,
            ));
        }

        if providers.is_empty() {
            warn!("[SUPERNODE] All providers failed — disabled");
            return None;
        }

        let router = LlmRouter::new(providers, supernode.routing.clone());
        info!(providers = supernode.providers.len(), fallback = ?supernode.routing.fallback, "[SUPERNODE] LlmRouter initialized");
        Some(Arc::new(router))
    }

    // ============================================
    // MemChain initialization
    // ============================================

    async fn init_memchain(
        &self,
    ) -> Result<(
        Arc<MemoryStorage>,
        Arc<VectorIndex>,
        Arc<MemPool>,
        Arc<TokioMutex<AofWriter>>,
    )> {
        let db_path = &self.config.memchain.db_path;
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("DB dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        let record_key = derive_record_key(&self.identity.to_bytes());
        info!("[MEMCHAIN] Record content encryption enabled");

        let storage = Arc::new(
            MemoryStorage::open(db_path, Some(record_key))
                .map_err(|e| ServerError::startup_failed(format!("SQLite: {}", e)))?,
        );
        let commitment_durability = storage
            .configure_record_commitment_durability(
                self.config.memchain.commitment_coordinator_enabled,
            )
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain commitment durability: {error}"))
            })?;
        info!(
            durability_mode = commitment_durability,
            coordinator = self.config.memchain.commitment_coordinator_enabled,
            "[MEMCHAIN_BLOCK] Commitment durability gate passed"
        );
        let commitment_audit = storage
            .audit_record_commitment_chain()
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain commitment integrity audit: {error}"))
            })?;
        let commitment_integrity = storage.record_commitment_chain_integrity_status();
        info!(
            blocks = commitment_audit.block_count,
            commitments = commitment_audit.commitment_count,
            tip_height = commitment_audit.tip_height,
            duration_ms = commitment_integrity.verification_duration_ms.unwrap_or(0),
            "[MEMCHAIN_BLOCK] Persisted commitment chain audit passed"
        );
        if self.config.memchain.commitment_coordinator_enabled {
            let anchor_state = storage
                .configure_record_commitment_tip_anchor(
                    self.config.memchain.effective_commitment_tip_anchor_path(),
                    &self.identity,
                )
                .await
                .map_err(|error| {
                    ServerError::startup_failed(format!(
                        "MemChain commitment tip rollback guard: {error}"
                    ))
                })?;
            info!(
                state = anchor_state,
                tip_height = commitment_audit.tip_height,
                scope = "local_db_file_rollback_only",
                "[MEMCHAIN_BLOCK] Signed commitment tip rollback guard passed"
            );
        }
        let checkpoint_evidence_audit = storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain checkpoint evidence audit: {error}"))
            })?;
        info!(
            evidence_records = checkpoint_evidence_audit.evidence_records,
            divergence_records = checkpoint_evidence_audit.divergence_evidence_records,
            "[MEMCHAIN_BLOCK] Persisted checkpoint evidence audit passed"
        );

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let vector_index = Arc::new(if quantization_enabled {
            let sat = if self.config.memchain.vector_early_termination {
                0.001_f32
            } else {
                0.0_f32
            };
            info!(
                quantization = "scalar_uint8",
                "[MEMCHAIN] VectorIndex with scalar quantization"
            );
            VectorIndex::with_config(true, sat)
        } else {
            VectorIndex::new()
        });

        let owner = self.identity.public_key_bytes();
        let rebuild_all_owners =
            self.config.memchain.blind_storage_enabled || self.config.memchain.allow_remote_storage;
        let records_with_model = if rebuild_all_owners {
            storage.get_all_records_with_embedding().await
        } else {
            storage.get_records_with_embedding(&owner).await
        };
        let mut rebuilt_owners = std::collections::HashSet::new();
        let mut rebuilt_partitions = std::collections::HashSet::new();
        let mut integrity_rejected = 0usize;
        for (r, model) in records_with_model {
            if r.has_embedding() {
                if let Some(reason) = memchain_index_rejection_reason(&r) {
                    integrity_rejected += 1;
                    warn!(
                        reason,
                        blind = r.blind,
                        "[MEMCHAIN] Persisted record rejected from vector rebuild"
                    );
                    continue;
                }
                rebuilt_owners.insert(r.owner);
                rebuilt_partitions.insert((r.owner, model.clone()));
                vector_index.upsert(
                    r.record_id,
                    r.embedding.clone(),
                    r.layer,
                    r.timestamp,
                    &r.owner,
                    &model,
                );
            }
        }
        let rebuild_count = vector_index.total_vectors();

        info!(
            db = %db_path,
            records = storage.count().await,
            vectors = rebuild_count,
            owners = rebuilt_owners.len(),
            partitions = rebuilt_partitions.len(),
            integrity_rejected,
            rebuild_scope = if rebuild_all_owners { "all_active_owners" } else { "local_owner" },
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );
        if integrity_rejected > 0 {
            warn!(
                integrity_rejected,
                "[MEMCHAIN] Integrity audit quarantined persisted records from recall"
            );
        }

        if quantization_enabled && rebuild_count > 0 {
            // Each owner/model pair is an independent security partition. A
            // blind storage node may host many such partitions, so restoring
            // only the node identity's quantizer would silently degrade remote
            // recall after restart.
            for (partition_owner, model_name) in rebuilt_partitions {
                let owner_hex = hex::encode(partition_owner);
                let cal_key = format!("{}:{}:{}", QUANTIZER_CAL_KEY_PREFIX, owner_hex, model_name);

                let restored = {
                    let conn = storage.conn_lock().await;
                    let cal_data: Option<Vec<u8>> = conn
                        .query_row(
                            "SELECT value FROM chain_state WHERE key = ?1",
                            rusqlite::params![cal_key],
                            |row| row.get::<_, Vec<u8>>(0),
                        )
                        .optional()
                        .unwrap_or(None);
                    drop(conn);
                    if let Some(data) = cal_data {
                        vector_index.restore_quantizer(&partition_owner, &model_name, &data)
                    } else {
                        false
                    }
                };

                if restored {
                    info!(
                        owner = %owner_hex,
                        model = %model_name,
                        "[VECTOR] Quantizer restored"
                    );
                    continue;
                }

                vector_index.calibrate_partition(&partition_owner, &model_name);
                if let Some(cal_bytes) =
                    vector_index.get_quantizer_bytes(&partition_owner, &model_name)
                {
                    let conn = storage.conn_lock().await;
                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES (?1, ?2)",
                        rusqlite::params![cal_key, cal_bytes.as_slice()],
                    );
                    drop(conn);
                    info!(
                        owner = %owner_hex,
                        model = %model_name,
                        "[VECTOR] Quantizer calibrated and persisted"
                    );
                }
            }
        }

        let aof_path = &self.config.memchain.aof_path;
        if let Some(parent) = std::path::Path::new(aof_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("AOF dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        let (existing_facts, last_block) = AofWriter::replay(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF replay: {}", e)))?;

        let mempool = Arc::new(MemPool::new());
        let mut loaded = 0u64;
        for fact in existing_facts {
            if mempool.add_fact(fact) {
                loaded += 1;
            }
        }

        let aof_writer = AofWriter::open(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF open: {}", e)))?;
        aof_writer.set_chain_state(last_block.as_ref());
        let aof_writer = Arc::new(TokioMutex::new(aof_writer));

        info!(
            facts = loaded,
            "[MEMCHAIN] Legacy MemPool + AOF initialized"
        );
        Ok((storage, vector_index, mempool, aof_writer))
    }

    // ============================================
    // Combined API Server
    // ============================================

    fn start_combined_api(
        &self,
        listen_addr: std::net::SocketAddr,
        mpi_state: Arc<MpiState>,
        _mempool: Arc<MemPool>,
        _aof_writer: Arc<TokioMutex<AofWriter>>,
        ip_pool: Arc<IpPoolService>,
        sessions: Arc<SessionManager>,
        _udp: Arc<UdpTransport>,
        node_policy: Arc<NodePolicyRuntime>,
        voucher_verifier: Arc<VoucherVerifier>,
        encrypted_message_counter: Arc<AtomicU64>,
        packet_handler: Arc<PacketHandler>,
        peer_store: Arc<PeerStore>,
        chat_relay: Option<Arc<ChatRelayService>>,
        udp: Arc<UdpTransport>,
    ) -> JoinHandle<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut shutdown_rx_vpn = self.shutdown_tx.subscribe();
        let mut shutdown_rx_public = self.shutdown_tx.subscribe();
        let vpn_listen_addr: std::net::SocketAddr = format!("100.64.0.1:{}", listen_addr.port())
            .parse()
            .unwrap_or_else(|_| "100.64.0.1:8421".parse().unwrap());
        let vpn_health_config = self.config.clone();
        let discovery_api_policy = DiscoveryApiPolicy::from_config(&self.config.discovery);
        let chat_relay_runtime_ready = chat_relay.is_some();
        let local_capability_status = Self::discovery_local_capability_status_for_runtime(
            &vpn_health_config,
            chat_relay_runtime_ready,
        );
        let public_api_listen_addr = self.config.discovery.public_api_listen_addr;
        let node_identity = Arc::new(self.identity.clone());
        let peer_http_client = Arc::new(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
        let smoke_peer_store = Arc::clone(&peer_store);
        let smoke_node_identity = Arc::clone(&node_identity);
        let smoke_peer_http_client = Arc::clone(&peer_http_client);
        let smoke_local_capability_status = local_capability_status.clone();
        let commitment_storage = mpi_state.storage.clone();

        tokio::spawn(async move {
            if let Some(public_addr) = public_api_listen_addr {
                let public_app = Self::build_public_discovery_router(
                    Arc::clone(&peer_store),
                    discovery_api_policy.clone(),
                    chat_relay.clone(),
                    Arc::clone(&sessions),
                    Arc::clone(&udp),
                    Arc::clone(&node_identity),
                    Arc::clone(&peer_http_client),
                    local_capability_status.clone(),
                    commitment_storage.clone(),
                );
                tokio::spawn(async move {
                    Self::serve_public_discovery_api(public_addr, public_app, shutdown_rx_public)
                        .await;
                });
            }

            let app = build_mpi_router(mpi_state)
                .merge(build_voice_router(Arc::clone(&sessions)))
                .merge(build_vpn_health_router(
                    vpn_health_config,
                    Arc::clone(&ip_pool),
                    Arc::clone(&sessions),
                    node_policy,
                    voucher_verifier,
                    encrypted_message_counter,
                    packet_handler,
                    Arc::clone(&peer_store),
                ))
                .merge(build_chat_peer_router(
                    chat_relay,
                    Arc::clone(&sessions),
                    udp,
                    Arc::clone(&peer_store),
                    Arc::clone(&node_identity),
                    Arc::clone(&peer_http_client),
                ))
                // Local/VPN-only operator smoke trigger. The public discovery API
                // intentionally does not expose this route; it actively sends a
                // synthetic two-hop onion delivery probe and returns aggregate
                // counters only, never route ids, selected hops, receiver keys,
                // endpoints, encrypted payload bytes, or social graph metadata.
                .route(
                    "/api/discovery/smoke/two-hop",
                    axum::routing::post(move || {
                        let peer_store = Arc::clone(&smoke_peer_store);
                        let identity = Arc::clone(&smoke_node_identity);
                        let client = Arc::clone(&smoke_peer_http_client);
                        let local_capabilities = smoke_local_capability_status.clone();
                        async move {
                            let before_at = unix_now_secs();
                            let before_status = peer_store.status(before_at);
                            let before_runtime = blind_relay_runtime_status_value(
                                before_at,
                                &before_status,
                                &local_capabilities,
                            );
                            let self_node_id = identity.public_key_bytes();
                            let accepted = Self::probe_two_hop_blind_relay_path(
                                &client,
                                &peer_store,
                                &identity,
                                &self_node_id,
                                before_at,
                            )
                            .await;
                            let after_at = unix_now_secs();
                            let after_status = peer_store.status(after_at);
                            let after_runtime = blind_relay_runtime_status_value(
                                after_at,
                                &after_status,
                                &local_capabilities,
                            );
                            axum::Json(serde_json::json!({
                                "success": accepted,
                                "contract_version": "two_hop_smoke.v1",
                                "source": "rust_local_operator_smoke",
                                "scope": "local_or_vpn_operator_api_only",
                                "probe": {
                                    "type": "two_hop_onion_delivery",
                                    "payload": "synthetic_opaque_ciphertext",
                                    "ack_boundary": "terminal_chat_relay_store_or_online_delivery",
                                },
                                "before": before_runtime,
                                "after": after_runtime,
                                "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
                                "privacy_boundary": "operator smoke returns aggregate counters only; no endpoints, route ids, selected hops, receiver keys, encrypted payloads, client IPs, destinations, DNS contents, private keys, wallet-level traffic, or social graph metadata",
                            }))
                        }
                    }),
                )
                .merge(build_discovery_router_with_local_status(
                    Arc::clone(&peer_store),
                    discovery_api_policy,
                    local_capability_status,
                ));
            let app = if let Some(storage) = commitment_storage {
                app.merge(build_memchain_peer_router(
                    storage,
                    peer_store,
                    node_identity,
                ))
            } else {
                app
            };

            let listener = match tokio::net::TcpListener::bind(listen_addr).await {
                Ok(l) => {
                    info!("[API] MemChain API on http://{}", listen_addr);
                    l
                }
                Err(e) => {
                    error!("[API] Bind failed {}: {}", listen_addr, e);
                    return;
                }
            };

            match tokio::net::TcpListener::bind(vpn_listen_addr).await {
                Ok(vpn_listener) => {
                    info!(
                        "[API] Voice API also available on http://{} (VPN clients only)",
                        vpn_listen_addr
                    );
                    let app_clone = app.clone();
                    tokio::spawn(async move {
                        let server = axum::serve(vpn_listener, app_clone).with_graceful_shutdown(
                            async move {
                                let _ = shutdown_rx_vpn.recv().await;
                            },
                        );
                        if let Err(e) = server.await {
                            error!("[API] VPN listener error: {}", e);
                        }
                        info!("[API] VPN listener stopped");
                    });
                }
                Err(e) => {
                    warn!(
                        "[API] VPN listener on {} not ready yet ({}), will retry every 10s",
                        vpn_listen_addr, e
                    );
                    let app_clone = app.clone();
                    tokio::spawn(async move {
                        let mut interval =
                            tokio::time::interval(std::time::Duration::from_secs(10));
                        interval.tick().await;
                        loop {
                            tokio::select! {
                                _ = shutdown_rx_vpn.recv() => { debug!("[API] VPN listener retry task shutting down"); break; }
                                _ = interval.tick() => {
                                    match tokio::net::TcpListener::bind(vpn_listen_addr).await {
                                        Ok(vpn_listener) => {
                                            info!("[API] VPN listener bound on {} (TUN is now up)", vpn_listen_addr);
                                            let server = axum::serve(vpn_listener, app_clone)
                                                .with_graceful_shutdown(async { std::future::pending::<()>().await });
                                            if let Err(e) = server.await { error!("[API] VPN listener error: {}", e); }
                                            break;
                                        }
                                        Err(e) => { debug!("[API] VPN listener retry failed ({}): {}", vpn_listen_addr, e); }
                                    }
                                }
                            }
                        }
                    });
                }
            }

            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
                info!("[API] Shutdown signal received");
            });
            if let Err(e) = server.await {
                error!("[API] Server error: {}", e);
            }
            info!("[API] Stopped");
        })
    }

    fn build_public_discovery_router(
        peer_store: Arc<PeerStore>,
        discovery_api_policy: DiscoveryApiPolicy,
        chat_relay: Option<Arc<ChatRelayService>>,
        sessions: Arc<SessionManager>,
        udp: Arc<UdpTransport>,
        node_identity: Arc<IdentityKeyPair>,
        peer_http_client: Arc<reqwest::Client>,
        local_capability_status: DiscoveryLocalCapabilityStatus,
        commitment_storage: Option<Arc<MemoryStorage>>,
    ) -> axum::Router {
        let block_peer_store = Arc::clone(&peer_store);
        let block_identity = Arc::clone(&node_identity);
        let app = build_discovery_router_with_local_status(
            Arc::clone(&peer_store),
            discovery_api_policy,
            local_capability_status,
        )
        .merge(build_chat_peer_router(
            chat_relay,
            sessions,
            udp,
            Arc::clone(&peer_store),
            node_identity,
            peer_http_client,
        ));
        if let Some(storage) = commitment_storage {
            app.merge(build_memchain_peer_router(
                storage,
                block_peer_store,
                block_identity,
            ))
        } else {
            app
        }
    }

    async fn serve_public_discovery_api(
        listen_addr: SocketAddr,
        app: axum::Router,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let listener = match tokio::net::TcpListener::bind(listen_addr).await {
            Ok(listener) => {
                info!(
                    "[DISCOVERY] Public node API on http://{} (routes: /api/discovery/*, /api/chat/peer/*, /api/memchain/peer/block-range, /api/memchain/peer/checkpoint)",
                    listen_addr
                );
                listener
            }
            Err(error) => {
                error!(
                    "[DISCOVERY] Public discovery API bind failed {}: {}",
                    listen_addr, error
                );
                return;
            }
        };

        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = shutdown_rx.recv().await;
            info!("[DISCOVERY] Public discovery API shutdown signal received");
        });
        if let Err(error) = server.await {
            error!("[DISCOVERY] Public discovery API error: {}", error);
        }
        info!("[DISCOVERY] Public discovery API stopped");
    }

    // ============================================
    // Management Reporter
    // ============================================

    async fn init_management_reporter(
        &self,
        sessions: &Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        udp: Arc<UdpTransport>,
        traffic_tracker: Arc<TrafficTracker>,
        deny_list: Arc<DenyList>,
        node_policy: Arc<NodePolicyRuntime>,
        voucher_verifier: Arc<VoucherVerifier>,
        encrypted_message_counter: Arc<AtomicU64>,
        packet_handler: Arc<PacketHandler>,
        peer_store: Arc<PeerStore>,
        memchain_storage: Option<Arc<MemoryStorage>>,
        chat_relay: Option<Arc<ChatRelayService>>,
        chat_relay_enabled: bool,
    ) -> SessionEventSender {
        info!("Initializing management reporting...");

        let mgmt_client = Arc::new(ManagementClient::new(
            self.config.management.clone(),
            self.identity.clone(),
        ));
        info!("Node ID: {}", mgmt_client.node_id());

        let public_ip = self.resolve_public_ip().await;

        let (session_reporter, event_tx) = SessionReporter::new(Arc::clone(&mgmt_client));
        let session_event_sender = SessionEventSender::new(event_tx);

        let (cmd_tx, cmd_rx) = mpsc::channel(COMMAND_CHANNEL_BUFFER);
        let cmd_handler = CommandHandler::new(cmd_rx, Arc::clone(&mgmt_client))
            .with_session_control(Arc::clone(sessions), session_event_sender.clone())
            .with_deny_list(Arc::clone(&deny_list))
            .with_node_policy(Arc::clone(&node_policy));
        let cmd_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            cmd_handler.run(cmd_shutdown).await;
        });

        let memchain_status_fn: Option<crate::management::reporter::MemChainStatusFn> =
            if self.config.memchain.is_enabled() {
                let allow_remote = self.config.memchain.allow_remote_storage;
                let max_owners = self.config.memchain.max_remote_owners;
                let pinned_witnesses_configured =
                    self.config.memchain.commitment_witness_node_ids.len();
                let witness_scope = if pinned_witnesses_configured == 0 {
                    "permissionless_evidence"
                } else {
                    "operator_pinned"
                };
                let startup_evidence_required =
                    self.config.memchain.commitment_witness_startup_required;
                let startup_minimum_verified = self.config.memchain.commitment_witness_min_verified;
                let commitment_storage = memchain_storage.clone();
                Some(Box::new(move || {
                    let record_commitment_integrity = commitment_storage.as_ref().map(|storage| {
                        let status = storage.record_commitment_chain_integrity_status();
                        crate::management::client::RecordCommitmentIntegrityHeartbeatStatus {
                            contract_version: status.contract_version,
                            state: status.state.to_string(),
                            baseline_verified_at: status.baseline_verified_at,
                            last_verified_at: status.last_verified_at,
                            verification_duration_ms: status.verification_duration_ms,
                            verified_block_count: status.verified_block_count,
                            verified_commitment_count: status.verified_commitment_count,
                            verified_tip_height: status.verified_tip_height,
                            durability_mode: status.durability_mode,
                            rollback_guard_state: status.rollback_guard_state,
                            rollback_guard_height: status.rollback_guard_height,
                            rollback_guard_last_verified_at: status.rollback_guard_last_verified_at,
                            rollback_guard_last_persisted_at: status
                                .rollback_guard_last_persisted_at,
                            rollback_guard_write_failures_total: status
                                .rollback_guard_write_failures_total,
                        }
                    });
                    let record_commitment_sync = commitment_storage.as_ref().map(|storage| {
                        let status = storage.record_commitment_sync_status();
                        crate::management::client::RecordCommitmentSyncHeartbeatStatus {
                            contract_version: status.contract_version,
                            role: status.role,
                            state: status.state,
                            enabled: status.enabled,
                            last_attempt_at: status.last_attempt_at,
                            last_success_at: status.last_success_at,
                            last_failure_at: status.last_failure_at,
                            last_recovered_at: status.last_recovered_at,
                            next_poll_at: status.next_poll_at,
                            consecutive_failures: status.consecutive_failures,
                            last_error_code: status.last_error_code,
                            remote_tip_height: status.remote_tip_height,
                            pages_received_total: status.pages_received_total,
                            blocks_received_total: status.blocks_received_total,
                            failure_events_total: status.failure_events_total,
                            recovery_events_total: status.recovery_events_total,
                        }
                    });
                    let record_commitment_checkpoint = commitment_storage.as_ref().map(|storage| {
                        let status = storage.record_commitment_checkpoint_status();
                        crate::management::client::RecordCommitmentCheckpointHeartbeatStatus {
                            contract_version: status.contract_version,
                            witness_scope,
                            pinned_witnesses_configured,
                            startup_evidence_required,
                            startup_minimum_verified,
                            state: status.state,
                            last_checked_at: status.last_checked_at,
                            last_converged_at: status.last_converged_at,
                            last_divergence_at: status.last_divergence_at,
                            last_failure_at: status.last_failure_at,
                            last_served_at: status.last_served_at,
                            local_tip_height: status.local_tip_height,
                            remote_tip_height: status.remote_tip_height,
                            proofs_verified_total: status.proofs_verified_total,
                            proofs_failed_total: status.proofs_failed_total,
                            divergences_total: status.divergences_total,
                            requests_served_total: status.requests_served_total,
                            evidence_state: status.evidence_state,
                            evidence_records: status.evidence_records,
                            divergence_evidence_records: status.divergence_evidence_records,
                            last_evidence_at: status.last_evidence_at,
                            observation_freshness: status.observation_freshness,
                            observation_age_seconds: status.observation_age_seconds,
                            freshness_window_seconds: status.freshness_window_seconds,
                            last_round_state: status.last_round_state,
                            last_round_at: status.last_round_at,
                            last_round_eligible: status.last_round_eligible,
                            last_round_attempted: status.last_round_attempted,
                            last_round_verified: status.last_round_verified,
                            last_round_failed: status.last_round_failed,
                            last_round_converged: status.last_round_converged,
                            last_round_remote_ahead: status.last_round_remote_ahead,
                            last_round_remote_behind: status.last_round_remote_behind,
                            last_round_diverged: status.last_round_diverged,
                            evidence_persistence_failures_total: status
                                .evidence_persistence_failures_total,
                        }
                    });
                    Some(crate::management::client::MemChainHeartbeatStatus {
                        enabled: true,
                        allow_remote_storage: allow_remote,
                        max_remote_owners: max_owners,
                        current_remote_owners: 0,
                        record_commitment_integrity,
                        record_commitment_sync,
                        record_commitment_checkpoint,
                    })
                }))
            } else {
                None
            };

        // Note: .with_sessions / .with_traffic_tracker / .with_udp are
        // injected here — all three are available at this call site.
        let mut heartbeat = HeartbeatReporter::new(Arc::clone(&mgmt_client), public_ip)
            .with_command_sender(cmd_tx)
            .with_sessions(Arc::clone(sessions))
            .with_traffic_tracker(Arc::clone(&traffic_tracker))
            .with_udp(Arc::clone(&udp))
            .with_deny_list(Arc::clone(&deny_list))
            .with_node_policy(Arc::clone(&node_policy));

        if let Some(f) = memchain_status_fn {
            heartbeat = heartbeat.with_memchain_status(f);
        }

        let vpn_health_config = self.config.clone();
        let vpn_health_ip_pool = Arc::clone(&ip_pool);
        let vpn_health_sessions = Arc::clone(sessions);
        let vpn_health_policy = Arc::clone(&node_policy);
        let vpn_health_verifier = Arc::clone(&voucher_verifier);
        let vpn_health_message_counter = Arc::clone(&encrypted_message_counter);
        let vpn_health_packet_handler = Arc::clone(&packet_handler);
        let vpn_health_peer_store = Arc::clone(&peer_store);
        heartbeat = heartbeat.with_vpn_health_status(Box::new(move || {
            let config = vpn_health_config.clone();
            let ip_pool = Arc::clone(&vpn_health_ip_pool);
            let sessions = Arc::clone(&vpn_health_sessions);
            let node_policy = Arc::clone(&vpn_health_policy);
            let verifier = Arc::clone(&vpn_health_verifier);
            let message_counter = Arc::clone(&vpn_health_message_counter);
            let packet_handler = Arc::clone(&vpn_health_packet_handler);
            let peer_store = Arc::clone(&vpn_health_peer_store);
            Box::pin(async move {
                Some(
                    collect_vpn_health_value(
                        config,
                        ip_pool,
                        sessions,
                        node_policy,
                        verifier,
                        message_counter,
                        packet_handler,
                        peer_store,
                    )
                    .await,
                )
            })
        }));

        let operator_status_config = self.config.clone();
        let operator_status_ip_pool = Arc::clone(&ip_pool);
        let operator_status_sessions = Arc::clone(sessions);
        let operator_status_policy = Arc::clone(&node_policy);
        let operator_status_verifier = Arc::clone(&voucher_verifier);
        let operator_status_message_counter = Arc::clone(&encrypted_message_counter);
        let operator_status_packet_handler = Arc::clone(&packet_handler);
        let operator_status_peer_store = Arc::clone(&peer_store);
        heartbeat = heartbeat.with_operator_status(Box::new(move || {
            let config = operator_status_config.clone();
            let ip_pool = Arc::clone(&operator_status_ip_pool);
            let sessions = Arc::clone(&operator_status_sessions);
            let node_policy = Arc::clone(&operator_status_policy);
            let verifier = Arc::clone(&operator_status_verifier);
            let message_counter = Arc::clone(&operator_status_message_counter);
            let packet_handler = Arc::clone(&operator_status_packet_handler);
            let peer_store = Arc::clone(&operator_status_peer_store);
            Box::pin(async move {
                Some(
                    collect_node_operator_status_value(
                        config,
                        ip_pool,
                        sessions,
                        node_policy,
                        verifier,
                        message_counter,
                        packet_handler,
                        peer_store,
                    )
                    .await,
                )
            })
        }));

        let discovery_status_config = self.config.clone();
        let discovery_status_peer_store = Arc::clone(&peer_store);
        let discovery_chat_relay_runtime_ready = chat_relay.is_some();
        heartbeat = heartbeat.with_discovery_status(Box::new(move || {
            let config = discovery_status_config.clone();
            let peer_store = Arc::clone(&discovery_status_peer_store);
            Box::pin(async move {
                let now = unix_now_secs();
                let status = peer_store.status(now);
                let local_capabilities = Self::discovery_local_capability_status_for_runtime(
                    &config,
                    discovery_chat_relay_runtime_ready,
                );
                let discovery_readiness =
                    discovery_readiness_status_value(&status, &local_capabilities);
                let route_governance = serde_json::json!(&status.route_governance);
                let blind_relay_runtime =
                    blind_relay_runtime_status_value(now, &status, &local_capabilities);
                let signed_peer_records = peer_store.export_signed_peer_records_for_heartbeat(
                    now,
                    Some(config.discovery.max_snapshot_limit),
                );
                Some(serde_json::json!({
                    "generated_at": now,
                    "peer_store": status,
                    "route_governance": route_governance,
                    "blind_relay_runtime": blind_relay_runtime,
                    "signed_peer_records": signed_peer_records,
                    "local_capabilities": local_capabilities,
                    "discovery_readiness": discovery_readiness,
                    "source": "rust_peer_store",
                    "privacy_boundary": "aggregate node discovery counters plus signed node-level discovery descriptors for central verification; no client IPs, destinations, DNS contents, packet payloads, chat plaintext, voucher secrets, private keys, or wallet-level traffic"
                }))
            })
        }));

        if let Some(relay) = chat_relay.as_ref() {
            let chat_relay_status: Arc<ChatRelayService> = Arc::clone(relay);
            heartbeat = heartbeat.with_chat_relay_status(Box::new(move || {
                let relay = Arc::clone(&chat_relay_status);
                Box::pin(async move {
                    let now = unix_now_secs();
                    Some(serde_json::json!({
                        "generated_at": now,
                        "peer_relay": relay.peer_status(),
                        "source": "rust_chat_relay_service",
                        "privacy_boundary": "aggregate encrypted chat peer relay counters only; no message ids, wallet ids, client IPs, destinations, DNS contents, packet payloads, chat plaintext, ciphertext, private keys, voucher secrets, or per-user traffic"
                    }))
                })
            }));
        } else if !chat_relay_enabled {
            heartbeat = heartbeat.with_chat_relay_status(Box::new(move || {
                Box::pin(async move {
                    let now = unix_now_secs();
                    Some(serde_json::json!({
                        "generated_at": now,
                        "peer_relay": {
                            "enabled": false,
                            "outbound_attempted_total": 0,
                            "outbound_accepted_total": 0,
                            "outbound_failed_total": 0,
                            "outbound_rounds": 0,
                            "last_outbound_attempted": 0,
                            "last_outbound_accepted": 0,
                            "last_outbound_failed": 0,
                            "last_outbound_status": null,
                            "last_outbound_failure_reason": null,
                            "consecutive_outbound_failures": 0,
                            "last_outbound_success_at": null,
                            "last_outbound_at": null,
                            "inbound_accepted_total": 0,
                            "inbound_duplicate_total": 0,
                            "inbound_delivered_online_total": 0,
                            "inbound_stored_pending_total": 0,
                            "inbound_rejected_total": 0,
                            "last_inbound_status": null,
                            "last_inbound_failure_reason": null,
                            "last_inbound_at": null
                        },
                        "source": "rust_chat_relay_disabled_config",
                        "privacy_boundary": "aggregate encrypted chat peer relay counters only; no message ids, wallet ids, client IPs, destinations, DNS contents, packet payloads, chat plaintext, ciphertext, private keys, voucher secrets, or per-user traffic"
                    }))
                })
            }));
        }

        let sess = Arc::clone(sessions);
        let hb_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            heartbeat
                .run(move || sess.count() as u32, hb_shutdown)
                .await;
        });

        let sr_shutdown = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            session_reporter.run(sr_shutdown).await;
        });

        info!("[MANAGEMENT] Reporting started");
        session_event_sender
    }

    // ============================================
    // Public IP Resolution
    // ============================================

    async fn resolve_public_ip(&self) -> String {
        if let Some(ip) = self.config.network.public_ip() {
            return ip.to_string();
        }

        let services = [
            "https://api.ipify.org",
            "https://ifconfig.me/ip",
            "https://ipinfo.io/ip",
            "http://169.254.169.254/latest/meta-data/public-ipv4",
            "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
        ];

        if let Ok(client) = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            for url in &services {
                let mut req = client.get(*url);
                if url.contains("metadata.google.internal") {
                    req = req.header("Metadata-Flavor", "Google");
                }
                if let Ok(resp) = req.send().await {
                    if resp.status().is_success() {
                        if let Ok(body) = resp.text().await {
                            let ip_str = body.trim();
                            if ip_str.len() <= 45 {
                                if let Ok(addr) = ip_str.parse::<std::net::IpAddr>() {
                                    let is_private = match addr {
                                        std::net::IpAddr::V4(v4) => {
                                            v4.is_loopback()
                                                || v4.is_private()
                                                || v4.is_unspecified()
                                        }
                                        std::net::IpAddr::V6(v6) => {
                                            v6.is_loopback() || v6.is_unspecified()
                                        }
                                    };
                                    if !is_private {
                                        info!(ip = %addr, source = %url, "[NET] Public IP detected");
                                        return addr.to_string();
                                    }
                                    warn!(ip = %addr, source = %url, "[NET] Ignoring private/loopback IP");
                                }
                            }
                        }
                    }
                }
            }
        }

        let fallback = self.config.listen_addr().ip().to_string();
        warn!(ip = %fallback, "[NET] Fallback to listen address");
        fallback
    }

    // ============================================
    // Core Services
    // ============================================

    async fn init_peer_store(&self, chat_relay_runtime_ready: bool) -> Arc<PeerStore> {
        let peer_store = Arc::new(PeerStore::new());
        peer_store.set_max_peers(Some(self.config.discovery.max_peers));
        peer_store.configure_bootstrap_status(
            self.config.discovery.enabled,
            self.config.discovery.peer_cache_path.is_some(),
            self.config.discovery.gossip_enabled,
            self.config.discovery.seed_endpoints.len(),
        );
        let now = unix_now_secs();
        let (self_check_status, self_check_detail) =
            Self::discovery_startup_self_check(&self.config);
        peer_store.record_startup_self_check(now, self_check_status, self_check_detail.clone());
        if self_check_status == "warning" {
            warn!(
                detail = %self_check_detail,
                "[DISCOVERY] Startup self-check warning"
            );
        } else {
            info!(
                status = %self_check_status,
                detail = %self_check_detail,
                "[DISCOVERY] Startup self-check complete"
            );
        }
        if !self.config.discovery.enabled {
            info!("[DISCOVERY] Bootstrap disabled");
            peer_store.record_bootstrap_source(now, "config", "skipped", "discovery_enabled=false");
            return peer_store;
        }

        if let Some(path) = &self.config.discovery.bootstrap_snapshot_path {
            match tokio::fs::read(path).await {
                Ok(bytes) => {
                    Self::import_bootstrap_snapshot_bytes(&peer_store, "file", path, &bytes, now);
                }
                Err(e) => {
                    peer_store.record_bootstrap_source(now, "file", "failed", "read_failed");
                    warn!(
                        source = %path,
                        error = %e,
                        "[DISCOVERY] Failed to read bootstrap snapshot"
                    );
                }
            }
        }

        if let Some(url) = &self.config.discovery.bootstrap_snapshot_url {
            match reqwest::Client::builder()
                .timeout(Duration::from_secs(
                    self.config.discovery.fetch_timeout_secs,
                ))
                .build()
            {
                Ok(client) => match client.get(url).send().await {
                    Ok(response) => {
                        let status = response.status();
                        if status.is_success() {
                            match response.bytes().await {
                                Ok(bytes) => {
                                    Self::import_bootstrap_snapshot_bytes(
                                        &peer_store,
                                        "url",
                                        url,
                                        bytes.as_ref(),
                                        now,
                                    );
                                }
                                Err(e) => {
                                    peer_store.record_bootstrap_source(
                                        now,
                                        "url",
                                        "failed",
                                        "body_read_failed",
                                    );
                                    warn!(
                                        source = %url,
                                        error = %e,
                                        "[DISCOVERY] Failed to read bootstrap response body"
                                    );
                                }
                            }
                        } else {
                            peer_store.record_bootstrap_source(
                                now,
                                "url",
                                "failed",
                                format!("http_status={status}"),
                            );
                            warn!(
                                source = %url,
                                status = %status,
                                "[DISCOVERY] Bootstrap URL returned non-success status"
                            );
                        }
                    }
                    Err(e) => {
                        peer_store.record_bootstrap_source(now, "url", "failed", "fetch_failed");
                        warn!(
                            source = %url,
                            error = %e,
                            "[DISCOVERY] Failed to fetch bootstrap snapshot"
                        );
                    }
                },
                Err(e) => {
                    peer_store.record_bootstrap_source(
                        now,
                        "url",
                        "failed",
                        "http_client_build_failed",
                    );
                    warn!(
                        error = %e,
                        "[DISCOVERY] Failed to build bootstrap HTTP client"
                    );
                }
            }
        }

        if let Some(path) = &self.config.discovery.peer_cache_path {
            self.load_peer_cache(&peer_store, path, now).await;
        }

        if self.config.discovery.advertise_self {
            crate::services::onion_keys::tick_rotation(now);
            match self.build_self_discovery_descriptor_with_runtime(now, chat_relay_runtime_ready) {
                Ok(descriptor) => match peer_store
                    .upsert_verified_from_source(descriptor, now, "self")
                {
                    Ok(true) => {
                        peer_store.record_self_descriptor_status(now, "success", "registered");
                        info!("[DISCOVERY] Self descriptor registered");
                    }
                    Ok(false) => {
                        peer_store.record_self_descriptor_status(now, "success", "already_current");
                        info!("[DISCOVERY] Self descriptor already current");
                    }
                    Err(e) => {
                        peer_store.record_self_descriptor_status(
                            now,
                            "failed",
                            "peer_store_rejected",
                        );
                        warn!(
                            error = %e,
                            "[DISCOVERY] Self descriptor rejected by local PeerStore"
                        );
                    }
                },
                Err(e) => {
                    peer_store.record_self_descriptor_status(now, "failed", "build_failed");
                    warn!(
                        error = %e,
                        "[DISCOVERY] Failed to build self descriptor"
                    );
                }
            }
        }

        let snapshot = peer_store.snapshot(now);
        info!(
            total_peers = snapshot.total_peers,
            valid_peers = snapshot.valid_peers,
            public_peers = snapshot.public_peers,
            public_exit_peers = snapshot.public_exit_peers,
            "[DISCOVERY] PeerStore bootstrap complete"
        );

        if let Some(path) = &self.config.discovery.peer_cache_path {
            let cache_save_at = unix_now_secs();
            if let Err(e) =
                Self::persist_peer_store_cache_once(&peer_store, path, cache_save_at).await
            {
                warn!(
                    source = %path,
                    error = %e,
                    "[DISCOVERY] Failed to persist initial peer cache snapshot"
                );
            } else {
                debug!(
                    source = %path,
                    "[DISCOVERY] Initial peer cache snapshot persisted"
                );
            }
        }
        peer_store
    }

    fn discovery_startup_self_check(config: &ServerConfig) -> (&'static str, String) {
        let discovery = &config.discovery;
        if !discovery.enabled {
            return ("skipped", "discovery_enabled=false".to_string());
        }

        let mut missing = Vec::new();
        if !discovery.advertise_self {
            missing.push("self_advertisement");
        }
        if discovery.peer_cache_path.is_none() {
            missing.push("peer_cache_path");
        }
        if !discovery.gossip_enabled {
            missing.push("gossip_enabled");
        }
        if discovery.gossip_enabled && discovery.seed_endpoints.is_empty() {
            missing.push("seed_endpoints");
        }
        let public_endpoint_configured =
            discovery.public_endpoint.is_some() || config.network.public_endpoint.is_some();
        if !public_endpoint_configured {
            missing.push("public_endpoint");
        }
        if discovery.public_api_listen_addr.is_none() {
            missing.push("public_api_listener");
        }
        if discovery.descriptor_ttl_secs < discovery.gossip_interval_secs.saturating_mul(2) {
            missing.push("descriptor_ttl_for_gossip");
        }

        if missing.is_empty() {
            (
                "ready",
                "cache,gossip,self_advertisement,public_endpoint,public_api_listener configured"
                    .to_string(),
            )
        } else {
            ("warning", format!("missing={}", missing.join(",")))
        }
    }

    async fn load_peer_cache(&self, peer_store: &PeerStore, path: &str, now: u64) {
        match tokio::fs::read(path).await {
            Ok(bytes) => {
                if !Self::import_bootstrap_snapshot_bytes(peer_store, "cache", path, &bytes, now) {
                    self.load_peer_cache_backup(peer_store, path, now).await;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                peer_store.record_bootstrap_source(now, "cache", "missing", "file_not_found");
                info!(
                    source = %path,
                    "[DISCOVERY] Peer cache not found; starting with bootstrap only"
                );
                self.load_peer_cache_backup(peer_store, path, now).await;
            }
            Err(e) => {
                peer_store.record_bootstrap_source(now, "cache", "failed", "read_failed");
                warn!(
                    source = %path,
                    error = %e,
                    "[DISCOVERY] Failed to read peer cache"
                );
                self.load_peer_cache_backup(peer_store, path, now).await;
            }
        }
    }

    async fn load_peer_cache_backup(&self, peer_store: &PeerStore, path: &str, now: u64) {
        let backup_path = Self::peer_cache_backup_path(path);
        let backup_source = backup_path.to_string_lossy().to_string();
        match tokio::fs::read(&backup_path).await {
            Ok(bytes) => {
                if Self::import_bootstrap_snapshot_bytes(
                    peer_store,
                    "cache_backup",
                    &backup_source,
                    &bytes,
                    now,
                ) {
                    info!(
                        source = %backup_source,
                        "[DISCOVERY] Peer cache backup restored"
                    );
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                peer_store.record_bootstrap_source(
                    now,
                    "cache_backup",
                    "missing",
                    "file_not_found",
                );
            }
            Err(e) => {
                peer_store.record_bootstrap_source(now, "cache_backup", "failed", "read_failed");
                warn!(
                    source = %backup_source,
                    error = %e,
                    "[DISCOVERY] Failed to read peer cache backup"
                );
            }
        }
    }

    fn spawn_peer_store_persistence_task(
        &self,
        peer_store: Arc<PeerStore>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.discovery.enabled {
            return None;
        }
        let Some(path) = self.config.discovery.peer_cache_path.clone() else {
            return None;
        };

        let interval_secs = self.config.discovery.peer_cache_write_interval_secs;
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();

        Some(tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(interval_secs));
            let persist_snapshot =
                |peer_store: Arc<PeerStore>, path: String, reason: &'static str| async move {
                    let now = unix_now_secs();
                    match Self::persist_peer_store_cache_once(&peer_store, &path, now).await {
                        Ok(()) => {
                            debug!(
                                source = %path,
                                reason = reason,
                                "[DISCOVERY] Peer cache snapshot persisted"
                            );
                        }
                        Err(e) => {
                            warn!(
                                source = %path,
                                reason = reason,
                                error = %e,
                                "[DISCOVERY] Failed to persist peer cache snapshot"
                            );
                        }
                    }
                };

            loop {
                tokio::select! {
                    _ = rx.recv() => {
                        persist_snapshot(Arc::clone(&peer_store), path.clone(), "shutdown").await;
                        break;
                    }
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) {
                            persist_snapshot(Arc::clone(&peer_store), path.clone(), "shutdown_flag").await;
                            break;
                        }
                        persist_snapshot(Arc::clone(&peer_store), path.clone(), "interval").await;
                    }
                }
            }
        }))
    }

    /// Verifies operator-pinned external checkpoint evidence before listeners.
    ///
    /// A pinned audited follower may attest that its canonical chain copy is
    /// beyond, or inconsistent with, the local audited tip. Such positive
    /// evidence fails closed. Network absence fails open unless the operator
    /// explicitly enables `commitment_witness_startup_required`, preserving
    /// backward-compatible availability while allowing strict deployments.
    async fn verify_memchain_commitment_startup_witnesses(
        &self,
        storage: &MemoryStorage,
        peer_store: &PeerStore,
    ) -> Result<()> {
        if !self.config.memchain.commitment_coordinator_enabled {
            return Ok(());
        }

        let witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        if witness_node_ids.is_empty() {
            info!(
                "[MEMCHAIN_BLOCK] External startup witness guard not configured; local signed tip guard remains active"
            );
            return Ok(());
        }

        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(1)
            .build()
            .map_err(|_| {
                ServerError::startup_failed(
                    "MemChain external witness rollback guard: HTTP client init failed",
                )
            })?;
        let round = reconcile_record_commitment_pinned_witnesses(
            storage,
            peer_store,
            &self.identity,
            &client,
            &witness_node_ids,
        )
        .await;
        let decision = commitment_witness_startup_decision(
            &round,
            self.config.memchain.commitment_witness_startup_required,
            self.config.memchain.commitment_witness_min_verified,
        )
        .map_err(|reason| {
            ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {reason}"
            ))
        })?;

        match decision {
            CommitmentWitnessStartupDecision::DegradedUnverified
            | CommitmentWitnessStartupDecision::DegradedBelowThreshold => {
                warn!(
                    configured = witness_node_ids.len(),
                    eligible = round.eligible_witnesses,
                    attempted = round.attempted,
                    verified = round.verified,
                    failed = round.failed,
                    minimum_verified = self.config.memchain.commitment_witness_min_verified,
                    strict = self.config.memchain.commitment_witness_startup_required,
                    decision = decision.as_str(),
                    "[MEMCHAIN_BLOCK] External startup witness guard is below the configured evidence threshold; continuing in availability mode"
                );
            }
            CommitmentWitnessStartupDecision::Verified => {
                info!(
                    configured = witness_node_ids.len(),
                    eligible = round.eligible_witnesses,
                    attempted = round.attempted,
                    verified = round.verified,
                    converged = round.converged,
                    remote_behind = round.remote_behind,
                    minimum_verified = self.config.memchain.commitment_witness_min_verified,
                    strict = self.config.memchain.commitment_witness_startup_required,
                    "[MEMCHAIN_BLOCK] External startup witness rollback guard passed"
                );
            }
        }
        Ok(())
    }

    /// Starts low-frequency signed checkpoint evidence collection on the
    /// configured Block Sync v1 coordinator.
    ///
    /// Discovered encrypted-storage peers act only as witnesses. Their signed
    /// observations are independently verified and stored, but peer count is
    /// never interpreted as votes, quorum, finality, or fork choice. A remote
    /// ahead/diverged result is operator evidence and cannot mutate the local
    /// canonical chain.
    fn spawn_memchain_commitment_reconciliation_task(
        &self,
        storage: Arc<MemoryStorage>,
        peer_store: Arc<PeerStore>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.memchain.commitment_coordinator_enabled {
            return None;
        }

        const INITIAL_DELAY_SECS: u64 = 15;
        const MIN_INTERVAL_SECS: u64 = 300;
        const MAX_WITNESSES_PER_ROUND: usize = 3;
        let client = match reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(1)
            .build()
        {
            Ok(client) => client,
            Err(_) => {
                error!(
                    "[MEMCHAIN_BLOCK] Coordinator witness reconciliation disabled: HTTP client init failed"
                );
                return None;
            }
        };
        let identity = self.identity.clone();
        let interval_secs = self
            .config
            .memchain
            .commitment_sync_interval_secs
            .max(MIN_INTERVAL_SECS);
        let pinned_witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        let witness_scope = if pinned_witness_node_ids.is_empty() {
            "permissionless_evidence"
        } else {
            "operator_pinned"
        };
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        Some(tokio::spawn(async move {
            let mut next_delay = Duration::from_secs(INITIAL_DELAY_SECS);
            let mut consecutive_unverified_rounds = 0u32;
            info!(
                interval_secs,
                max_witnesses = MAX_WITNESSES_PER_ROUND,
                witness_scope,
                "[MEMCHAIN_BLOCK] Coordinator witness reconciliation started"
            );

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = tokio::time::sleep(next_delay) => {}
                }

                let round = tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    outcome = async {
                        if pinned_witness_node_ids.is_empty() {
                            reconcile_record_commitment_witnesses(
                                &storage,
                                &peer_store,
                                &identity,
                                &client,
                                MAX_WITNESSES_PER_ROUND,
                            )
                            .await
                        } else {
                            reconcile_record_commitment_pinned_witnesses(
                                &storage,
                                &peer_store,
                                &identity,
                                &client,
                                &pinned_witness_node_ids,
                            )
                            .await
                        }
                    } => outcome,
                };
                next_delay = Duration::from_secs(interval_secs);

                if round.attempted == 0 {
                    consecutive_unverified_rounds = 0;
                    debug!(
                        eligible_witnesses = round.eligible_witnesses,
                        "[MEMCHAIN_BLOCK] Coordinator witness round waiting for eligible peers"
                    );
                    continue;
                }
                if round.verified == 0 {
                    consecutive_unverified_rounds = consecutive_unverified_rounds.saturating_add(1);
                    if consecutive_unverified_rounds == 1
                        || consecutive_unverified_rounds.is_power_of_two()
                    {
                        warn!(
                            attempted = round.attempted,
                            failed = round.failed,
                            consecutive_unverified_rounds,
                            "[MEMCHAIN_BLOCK] Coordinator witness round established no signed evidence"
                        );
                    }
                    continue;
                }

                consecutive_unverified_rounds = 0;
                if round.diverged > 0 || round.remote_ahead > 0 {
                    warn!(
                        attempted = round.attempted,
                        verified = round.verified,
                        converged = round.converged,
                        remote_ahead = round.remote_ahead,
                        remote_behind = round.remote_behind,
                        diverged = round.diverged,
                        failed = round.failed,
                        "[MEMCHAIN_BLOCK] Coordinator witness round found signed chain attention evidence"
                    );
                } else {
                    debug!(
                        attempted = round.attempted,
                        verified = round.verified,
                        converged = round.converged,
                        remote_behind = round.remote_behind,
                        failed = round.failed,
                        "[MEMCHAIN_BLOCK] Coordinator witness round complete"
                    );
                }
            }

            info!("[MEMCHAIN_BLOCK] Coordinator witness reconciliation stopped");
        }))
    }

    /// Starts the default-off Block Sync v1 follower.
    ///
    /// The configured coordinator identity is the sole trust root. Discovery
    /// resolves only that identity's signed endpoint, while the pull helper
    /// verifies the response signer, every block proposer, and full chain
    /// continuity before SQLite is changed. There is deliberately no peer
    /// fallback or longest-chain selection in this pre-consensus phase.
    fn spawn_memchain_commitment_sync_task(
        &self,
        storage: Arc<MemoryStorage>,
        peer_store: Arc<PeerStore>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.memchain.commitment_sync_enabled {
            return None;
        }
        let Some(coordinator_node_id) = self.config.memchain.commitment_sync_coordinator_node_id()
        else {
            let now = unix_now_secs();
            storage.record_commitment_sync_failure(
                now,
                "invalid_pinned_coordinator",
                1,
                now.saturating_add(600),
            );
            error!(
                "[MEMCHAIN_BLOCK] Follower sync disabled at runtime: invalid pinned coordinator"
            );
            return None;
        };
        if self.identity.public_key_bytes() == coordinator_node_id {
            let now = unix_now_secs();
            storage.record_commitment_sync_failure(
                now,
                "coordinator_self_reference",
                1,
                now.saturating_add(600),
            );
            error!(
                "[MEMCHAIN_BLOCK] Follower sync disabled at runtime: coordinator cannot follow itself"
            );
            return None;
        }

        let client = match reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(1)
            .build()
        {
            Ok(client) => client,
            Err(_) => {
                let now = unix_now_secs();
                storage.record_commitment_sync_failure(
                    now,
                    "http_client_init_failed",
                    1,
                    now.saturating_add(600),
                );
                error!(
                    "[MEMCHAIN_BLOCK] Follower sync disabled at runtime: HTTP client init failed"
                );
                return None;
            }
        };
        let identity = self.identity.clone();
        let base_interval_secs = self.config.memchain.commitment_sync_interval_secs;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        Some(tokio::spawn(async move {
            const MAX_PAGES_PER_ROUND: usize = 8;
            const MAX_BACKOFF_SECS: u64 = 600;

            let mut consecutive_failures = 0u32;
            let mut next_delay = Duration::from_secs(0);
            info!(
                interval_secs = base_interval_secs,
                max_pages_per_round = MAX_PAGES_PER_ROUND,
                "[MEMCHAIN_BLOCK] Pinned coordinator follower started"
            );

            loop {
                if !next_delay.is_zero() {
                    tokio::select! {
                        _ = shutdown_rx.recv() => break,
                        _ = tokio::time::sleep(next_delay) => {}
                    }
                }

                storage.record_commitment_sync_attempt(unix_now_secs());
                let round_future = async {
                    let mut inserted = 0usize;
                    let mut remote_tip_height = 0u64;
                    for _ in 0..MAX_PAGES_PER_ROUND {
                        let outcome = pull_record_commitment_page(
                            &storage,
                            &peer_store,
                            &identity,
                            &coordinator_node_id,
                            &client,
                        )
                        .await?;
                        let verified_blocks = outcome
                            .inserted
                            .saturating_add(outcome.already_present)
                            .try_into()
                            .unwrap_or(u64::MAX);
                        storage.record_commitment_sync_page_success(
                            unix_now_secs(),
                            verified_blocks,
                            outcome.remote_tip_height,
                            outcome.has_more,
                        );
                        inserted = inserted.saturating_add(outcome.inserted);
                        remote_tip_height = outcome.remote_tip_height;
                        if !outcome.has_more {
                            let checkpoint = match pull_record_commitment_checkpoint(
                                &storage,
                                &peer_store,
                                &identity,
                                &coordinator_node_id,
                                &client,
                            )
                            .await
                            {
                                Ok(checkpoint) => checkpoint,
                                Err(reason) => {
                                    storage.record_commitment_checkpoint_failure(unix_now_secs());
                                    return Err(reason);
                                }
                            };
                            let checked_at = unix_now_secs();
                            storage.record_commitment_checkpoint_verified(
                                checked_at,
                                checkpoint.relation.as_str(),
                                checkpoint.local_tip_height,
                                checkpoint.remote_tip_height,
                            );
                            match checkpoint.relation {
                                CommitmentCheckpointRelation::Converged => {
                                    storage.record_commitment_sync_checkpoint_success(
                                        checked_at,
                                        checkpoint.remote_tip_height,
                                    );
                                    return Ok((inserted, checkpoint.remote_tip_height, false));
                                }
                                CommitmentCheckpointRelation::RemoteAhead => {
                                    remote_tip_height = checkpoint.remote_tip_height;
                                    continue;
                                }
                                CommitmentCheckpointRelation::RemoteBehind => {
                                    return Err("signed_checkpoint_remote_behind".to_string());
                                }
                                CommitmentCheckpointRelation::Diverged => {
                                    return Err("signed_checkpoint_divergence".to_string());
                                }
                            }
                        }
                    }
                    Ok((inserted, remote_tip_height, true))
                };
                let round: std::result::Result<(usize, u64, bool), String> = tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    result = round_future => result,
                };

                match round {
                    Ok((inserted, remote_tip_height, backlog_remaining)) => {
                        consecutive_failures = 0;
                        if inserted > 0 {
                            info!(
                                blocks = inserted,
                                tip_height = remote_tip_height,
                                backlog_remaining,
                                "[MEMCHAIN_BLOCK] Follower catch-up advanced"
                            );
                        } else {
                            debug!(
                                tip_height = remote_tip_height,
                                "[MEMCHAIN_BLOCK] Follower is current"
                            );
                        }
                        // Keep each round bounded but drain a verified backlog
                        // promptly without turning the normal loop into polling.
                        next_delay = Duration::from_secs(if backlog_remaining {
                            1
                        } else {
                            base_interval_secs
                        });
                        storage.schedule_next_commitment_sync_poll(
                            unix_now_secs().saturating_add(next_delay.as_secs()),
                        );
                    }
                    Err(reason) => {
                        consecutive_failures = consecutive_failures.saturating_add(1);
                        let shift = consecutive_failures.saturating_sub(1).min(5);
                        let multiplier = 1u64 << shift;
                        let retry_secs = base_interval_secs
                            .saturating_mul(multiplier)
                            .min(MAX_BACKOFF_SECS);
                        next_delay = Duration::from_secs(retry_secs);
                        let failed_at = unix_now_secs();
                        storage.record_commitment_sync_failure(
                            failed_at,
                            &reason,
                            consecutive_failures,
                            failed_at.saturating_add(retry_secs),
                        );
                        if consecutive_failures == 1 || consecutive_failures.is_power_of_two() {
                            warn!(
                                consecutive_failures,
                                retry_secs,
                                reason = %reason,
                                "[MEMCHAIN_BLOCK] Follower catch-up failed closed"
                            );
                        }
                    }
                }
            }
            storage.stop_record_commitment_sync();
            info!("[MEMCHAIN_BLOCK] Pinned coordinator follower stopped");
        }))
    }

    fn spawn_discovery_gossip_task(
        &self,
        peer_store: Arc<PeerStore>,
        chat_relay_runtime_ready: bool,
    ) -> Option<JoinHandle<()>> {
        if !self.config.discovery.enabled || !self.config.discovery.gossip_enabled {
            return None;
        }

        let config = self.config.clone();
        let identity = self.identity.clone();
        let self_node_id = identity.public_key_bytes();
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(config.discovery.fetch_timeout_secs))
            .pool_max_idle_per_host(8)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                warn!(
                    error = %e,
                    "[DISCOVERY] Failed to build outbound gossip HTTP client"
                );
                return None;
            }
        };

        Some(tokio::spawn(async move {
            let mut run_immediately = true;
            let mut last_blind_relay_probe_at = 0u64;
            let mut last_two_hop_blind_relay_probe_at = 0u64;
            loop {
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
                // Rotate the onion key on the discovery cadence (no-op until the
                // rotation period elapses). Forward secrecy — see onion_keys.
                crate::services::onion_keys::tick_rotation(now);
                let Ok(self_descriptor) = Self::build_self_discovery_descriptor_for_runtime(
                    &config,
                    &identity,
                    now,
                    chat_relay_runtime_ready,
                ) else {
                    warn!("[DISCOVERY] Skipping outbound gossip; self descriptor build failed");
                    peer_store.record_gossip_round(
                        now,
                        0,
                        0,
                        0,
                        Some("self_descriptor_build_failed".to_string()),
                    );
                    continue;
                };

                let consecutive_failures = peer_store.consecutive_gossip_failures();
                let backpressure_active = Self::discovery_gossip_backpressure_active(
                    &config.discovery,
                    consecutive_failures,
                );
                let peer_limit = usize::from(config.discovery.gossip_peer_limit);
                let seed_limit = config.discovery.seed_endpoints.len().max(1);
                let round_peer_limit = if backpressure_active {
                    peer_limit.min(seed_limit)
                } else {
                    peer_limit
                };
                let include_cached_peers =
                    !backpressure_active || config.discovery.seed_endpoints.is_empty();
                let self_gossip_url = config
                    .discovery
                    .public_endpoint
                    .as_deref()
                    .or(config.network.public_endpoint.as_deref())
                    .and_then(Self::discovery_gossip_url);
                let mut seen_urls = HashSet::new();
                let mut gossip_urls = Vec::new();

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

                let mut attempted = 0usize;
                let mut succeeded = 0usize;
                let mut last_failure_reason: Option<String> = None;

                let seed_attempted = gossip_urls.len();

                if include_cached_peers && gossip_urls.len() < round_peer_limit {
                    let snapshot = peer_store.export_bootstrap_snapshot(
                        now,
                        now,
                        true,
                        Some(round_peer_limit),
                    );

                    for peer in snapshot.peers {
                        if gossip_urls.len() >= round_peer_limit {
                            break;
                        }
                        if peer.node_id() == self_node_id {
                            continue;
                        }
                        let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
                            continue;
                        };
                        let Some(url) = Self::discovery_gossip_url(endpoint) else {
                            continue;
                        };
                        if self_gossip_url.as_deref() == Some(url.as_str()) {
                            continue;
                        }
                        if !seen_urls.insert(url.clone()) {
                            continue;
                        }
                        gossip_urls.push(url);
                    }
                }

                for url in gossip_urls {
                    attempted += 1;
                    match Self::gossip_with_peer(
                        &client,
                        &peer_store,
                        &url,
                        self_descriptor.clone(),
                        now,
                        config.discovery.gossip_peer_limit,
                    )
                    .await
                    {
                        Ok(()) => succeeded += 1,
                        Err(e) => {
                            last_failure_reason = Some(e.clone());
                            debug!(
                                peer = %url,
                                error = %e,
                                backpressure_active,
                                "[DISCOVERY] Outbound gossip peer sync failed"
                            );
                        }
                    }
                }

                if attempted > 0 {
                    info!(
                        attempted,
                        succeeded,
                        backpressure_active,
                        "[DISCOVERY] Outbound gossip round complete"
                    );
                }
                peer_store.record_gossip_round(
                    now,
                    attempted,
                    succeeded,
                    seed_attempted,
                    last_failure_reason,
                );
                if chat_relay_runtime_ready && succeeded > 0 {
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
                        if Self::probe_blind_relay_candidate(
                            &client,
                            &peer_store,
                            &identity,
                            &self_node_id,
                            probe_now,
                        )
                        .await
                        {
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
                        if Self::probe_two_hop_blind_relay_path(
                            &client,
                            &peer_store,
                            &identity,
                            &self_node_id,
                            probe_now,
                        )
                        .await
                        {
                            last_two_hop_blind_relay_probe_at = probe_now;
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
        }))
    }

    fn blind_relay_probe_cooldown_secs(discovery: &DiscoveryConfig) -> u64 {
        discovery
            .gossip_interval_secs
            .saturating_mul(3)
            .max(BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS)
    }

    fn blind_relay_probe_recovery_cooldown_secs(discovery: &DiscoveryConfig) -> u64 {
        discovery.gossip_interval_secs.clamp(
            BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS,
            BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS,
        )
    }

    fn blind_relay_probe_cooldown_secs_for_status(
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

    fn prioritize_probe_candidates(
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

    fn discovery_gossip_backpressure_active(
        discovery: &DiscoveryConfig,
        consecutive_failures: u64,
    ) -> bool {
        consecutive_failures >= discovery.gossip_backpressure_failure_threshold
    }

    fn discovery_gossip_schedule(
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

    fn discovery_gossip_jitter_seconds(
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

    async fn gossip_with_peer(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        url: &str,
        self_descriptor: SignedNodeDescriptor,
        now: u64,
        limit: u16,
    ) -> std::result::Result<(), String> {
        let announce_response = client
            .post(url)
            .json(&NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: self_descriptor,
            })
            .send()
            .await
            .map_err(|e| Self::classify_reqwest_error("announce_request", &e))?;

        announce_response
            .error_for_status()
            .map_err(|e| Self::classify_reqwest_error("announce_status", &e))?;

        let snapshot_response = client
            .post(url)
            .json(&NodeDiscoveryMessage::SnapshotRequest {
                requested_at: now,
                limit: Some(limit),
            })
            .send()
            .await
            .map_err(|e| Self::classify_reqwest_error("snapshot_request", &e))?;

        let response = snapshot_response
            .error_for_status()
            .map_err(|e| Self::classify_reqwest_error("snapshot_status", &e))?
            .json::<GossipResponse>()
            .await
            .map_err(|e| Self::classify_reqwest_error("snapshot_decode", &e))?;

        if let Some(message) = response.response {
            peer_store.apply_discovery_message(&message, now);
        }
        peer_store.mark_gossip_at(now);

        Ok(())
    }

    async fn probe_blind_relay_candidate(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        now: u64,
    ) -> bool {
        let mut candidates = peer_store.route_probe_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            8,
            &[*self_node_id],
        );
        Self::prioritize_probe_candidates(peer_store, now, &mut candidates);
        let Some(candidate) = candidates.into_iter().next() else {
            return false;
        };

        let next_hop = candidate.node_id();
        let Some(endpoint) = candidate.descriptor.public_endpoint.as_deref() else {
            peer_store.record_blind_relay_probe_result(now, false, "missing_endpoint");
            peer_store.record_route_forward_failure(&next_hop, now, "missing_endpoint");
            return true;
        };
        let Some(url) = Self::blind_relay_probe_url(endpoint) else {
            peer_store.record_blind_relay_probe_result(now, false, "invalid_endpoint");
            peer_store.record_route_forward_failure(&next_hop, now, "invalid_endpoint");
            return true;
        };

        let envelope = BlindRelayEnvelope {
            route_id: Self::blind_relay_probe_route_id(now, self_node_id, &next_hop),
            next_hop,
            ttl: 1,
            encrypted_blob: Self::blind_relay_probe_blob(now, self_node_id, &next_hop),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(identity);
        let request = PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: *self_node_id,
            onward_envelope: None,
            onward_descriptor_hint: None,
        };

        match client.post(url).json(&request).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<PeerBlindRelayResponse>().await {
                    Ok(ack) if ack.accepted => {
                        peer_store.record_blind_relay_probe_result(now, true, "accepted");
                        peer_store.record_route_forward_success(&next_hop, now);
                    }
                    Ok(_ack) => {
                        peer_store.record_blind_relay_probe_result(now, false, "ack_rejected");
                        peer_store.record_route_forward_failure(&next_hop, now, "ack_rejected");
                    }
                    Err(error) => {
                        debug!(
                            error = %error,
                            "[DISCOVERY] Blind relay probe ACK decode failed"
                        );
                        peer_store.record_blind_relay_probe_result(now, false, "ack_decode");
                        peer_store.record_route_forward_failure(&next_hop, now, "ack_decode");
                    }
                }
            }
            Ok(response) => {
                let reason = format!("http_{}", response.status().as_u16());
                peer_store.record_blind_relay_probe_result(now, false, &reason);
                peer_store.record_route_forward_failure(&next_hop, now, reason);
            }
            Err(error) => {
                let reason = Self::classify_reqwest_error("blind_relay_probe", &error);
                debug!(
                    reason = %reason,
                    "[DISCOVERY] Blind relay probe failed"
                );
                peer_store.record_blind_relay_probe_result(now, false, &reason);
                peer_store.record_route_forward_failure(&next_hop, now, reason);
            }
        }
        true
    }

    async fn probe_two_hop_blind_relay_path(
        client: &reqwest::Client,
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        now: u64,
    ) -> bool {
        let mut middle_candidates = peer_store.route_probe_candidates_with_capability_excluding(
            NodeCapability::OnionMiddle,
            now,
            8,
            &[*self_node_id],
        );
        Self::prioritize_probe_candidates(peer_store, now, &mut middle_candidates);
        if middle_candidates.is_empty() {
            return false;
        }

        let middle_candidate_count = middle_candidates.len();
        let mut attempted = false;
        for middle in middle_candidates {
            let middle_node_id = middle.node_id();
            let mut terminal_candidates = peer_store
                .route_probe_candidates_with_capability_excluding(
                    NodeCapability::ChatRelay,
                    now,
                    8,
                    &[*self_node_id, middle_node_id],
                );
            Self::prioritize_probe_candidates(peer_store, now, &mut terminal_candidates);
            let terminal_candidate_count = terminal_candidates.len();
            if terminal_candidates.is_empty() {
                continue;
            }

            for terminal in terminal_candidates {
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
                    peer_store.record_route_forward_failure(
                        &middle_node_id,
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
                    peer_store.record_route_forward_failure(
                        &middle_node_id,
                        now,
                        "invalid_endpoint",
                    );
                    continue;
                };

                // Milestone 2 probe: prefer a real onion-wrapped ChatEnvelope
                // delivery over the older onward-envelope control-plane probe.
                // The middle hop peels exactly one layer and forwards the next
                // onion layer; the terminal hop must decode the final
                // ChatEnvelope and hand it to ChatRelay store-and-forward before
                // ACKing. The payload remains opaque and synthetic, and no
                // route id, receiver, endpoint, or ciphertext is reported.
                if let Some(request) = Self::build_two_hop_onion_delivery_probe_request(
                    identity,
                    self_node_id,
                    &middle,
                    &terminal,
                    now,
                ) {
                    match client.post(&url).json(&request).send().await {
                        Ok(response) if response.status().is_success() => {
                            match response.json::<PeerBlindRelayResponse>().await {
                                Ok(ack) if ack.accepted && ack.forwarded => {
                                    peer_store
                                        .record_blind_relay_two_hop_probe_result_with_context(
                                            now,
                                            true,
                                            "onion_terminal_delivered",
                                            middle_candidate_count,
                                            terminal_candidate_count,
                                            2,
                                            1,
                                        );
                                    peer_store.record_route_forward_success(&middle_node_id, now);
                                    return true;
                                }
                                Ok(_ack) => {
                                    peer_store
                                        .record_blind_relay_two_hop_probe_result_with_context(
                                            now,
                                            false,
                                            "onion_ack_rejected",
                                            middle_candidate_count,
                                            terminal_candidate_count,
                                            2,
                                            1,
                                        );
                                    peer_store.record_route_forward_failure(
                                        &middle_node_id,
                                        now,
                                        "onion_ack_rejected",
                                    );
                                }
                                Err(error) => {
                                    debug!(
                                        error = %error,
                                        "[DISCOVERY] Two-hop onion delivery probe ACK decode failed"
                                    );
                                    peer_store
                                        .record_blind_relay_two_hop_probe_result_with_context(
                                            now,
                                            false,
                                            "onion_ack_decode",
                                            middle_candidate_count,
                                            terminal_candidate_count,
                                            2,
                                            1,
                                        );
                                    peer_store.record_route_forward_failure(
                                        &middle_node_id,
                                        now,
                                        "onion_ack_decode",
                                    );
                                }
                            }
                        }
                        Ok(response) => {
                            let reason = format!("onion_http_{}", response.status().as_u16());
                            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                now,
                                false,
                                &reason,
                                middle_candidate_count,
                                terminal_candidate_count,
                                2,
                                1,
                            );
                            peer_store.record_route_forward_failure(&middle_node_id, now, reason);
                        }
                        Err(error) => {
                            let reason = Self::classify_reqwest_error(
                                "two_hop_onion_delivery_probe",
                                &error,
                            );
                            debug!(
                                reason = %reason,
                                "[DISCOVERY] Two-hop onion delivery probe failed"
                            );
                            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                now,
                                false,
                                &reason,
                                middle_candidate_count,
                                terminal_candidate_count,
                                2,
                                1,
                            );
                            peer_store.record_route_forward_failure(&middle_node_id, now, reason);
                        }
                    }
                } else {
                    peer_store.record_blind_relay_two_hop_probe_result_with_context(
                        now,
                        false,
                        "onion_kem_unavailable",
                        middle_candidate_count,
                        terminal_candidate_count,
                        2,
                        1,
                    );
                }

                let outer_envelope = BlindRelayEnvelope {
                    route_id: Self::blind_relay_two_hop_probe_route_id(
                        now,
                        self_node_id,
                        &middle_node_id,
                        &terminal_node_id,
                        b"outer",
                    ),
                    next_hop: middle_node_id,
                    ttl: 2,
                    encrypted_blob: Self::blind_relay_probe_blob(
                        now,
                        self_node_id,
                        &middle_node_id,
                    ),
                    timestamp: now,
                    signature: [0u8; 64],
                }
                .sign_with(identity);
                let onward_envelope = BlindRelayEnvelope {
                    route_id: Self::blind_relay_two_hop_probe_route_id(
                        now,
                        self_node_id,
                        &middle_node_id,
                        &terminal_node_id,
                        b"onward",
                    ),
                    next_hop: terminal_node_id,
                    ttl: 1,
                    encrypted_blob: Self::blind_relay_probe_blob(
                        now,
                        self_node_id,
                        &terminal_node_id,
                    ),
                    timestamp: now,
                    signature: [0u8; 64],
                }
                .sign_with(identity);
                let request = PeerBlindRelayRequest {
                    envelope: outer_envelope,
                    previous_hop_node_id: *self_node_id,
                    onward_envelope: Some(onward_envelope),
                    onward_descriptor_hint: Some(terminal.clone()),
                };

                match client.post(&url).json(&request).send().await {
                    Ok(response) if response.status().is_success() => {
                        match response.json::<PeerBlindRelayResponse>().await {
                            Ok(ack) if ack.accepted && ack.forwarded => {
                                peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                    now,
                                    true,
                                    "accepted",
                                    middle_candidate_count,
                                    terminal_candidate_count,
                                    2,
                                    1,
                                );
                                peer_store.record_route_forward_success(&middle_node_id, now);
                                return true;
                            }
                            Ok(_ack) => {
                                peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                    now,
                                    false,
                                    "ack_rejected",
                                    middle_candidate_count,
                                    terminal_candidate_count,
                                    2,
                                    1,
                                );
                                peer_store.record_route_forward_failure(
                                    &middle_node_id,
                                    now,
                                    "ack_rejected",
                                );
                            }
                            Err(error) => {
                                debug!(
                                    error = %error,
                                    "[DISCOVERY] Two-hop blind relay proof ACK decode failed"
                                );
                                peer_store.record_blind_relay_two_hop_probe_result_with_context(
                                    now,
                                    false,
                                    "ack_decode",
                                    middle_candidate_count,
                                    terminal_candidate_count,
                                    2,
                                    1,
                                );
                                peer_store.record_route_forward_failure(
                                    &middle_node_id,
                                    now,
                                    "ack_decode",
                                );
                            }
                        }
                    }
                    Ok(response) => {
                        let reason = format!("http_{}", response.status().as_u16());
                        peer_store.record_blind_relay_two_hop_probe_result_with_context(
                            now,
                            false,
                            &reason,
                            middle_candidate_count,
                            terminal_candidate_count,
                            2,
                            1,
                        );
                        peer_store.record_route_forward_failure(&middle_node_id, now, reason);
                    }
                    Err(error) => {
                        let reason =
                            Self::classify_reqwest_error("two_hop_blind_relay_probe", &error);
                        debug!(
                            reason = %reason,
                            "[DISCOVERY] Two-hop blind relay proof failed"
                        );
                        peer_store.record_blind_relay_two_hop_probe_result_with_context(
                            now,
                            false,
                            &reason,
                            middle_candidate_count,
                            terminal_candidate_count,
                            2,
                            1,
                        );
                        peer_store.record_route_forward_failure(&middle_node_id, now, reason);
                    }
                }
            }
        }

        if !attempted {
            peer_store.record_blind_relay_two_hop_probe_result_with_context(
                now,
                false,
                "no_distinct_path",
                middle_candidate_count,
                0,
                2,
                1,
            );
        }
        true
    }

    fn build_two_hop_onion_delivery_probe_request(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        middle: &SignedNodeDescriptor,
        terminal: &SignedNodeDescriptor,
        now: u64,
    ) -> Option<PeerBlindRelayRequest> {
        let middle_node_id = middle.node_id();
        let terminal_node_id = terminal.node_id();
        let middle_kem_pub = middle.descriptor.x25519_kem_public()?;
        let terminal_kem_pub = terminal.descriptor.x25519_kem_public()?;
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
        let encoded_chat = encode_envelope(&chat_envelope).ok()?;
        let path = [
            OnionHop {
                node_id: middle_node_id,
                kem_pub: middle_kem_pub,
            },
            OnionHop {
                node_id: terminal_node_id,
                kem_pub: terminal_kem_pub,
            },
        ];
        let envelope =
            build_onion_envelope(&path, &encoded_chat, route_id, 2, now, identity).ok()?;

        Some(PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: *self_node_id,
            onward_envelope: None,
            onward_descriptor_hint: None,
        })
    }

    fn synthetic_two_hop_probe_chat_envelope(
        identity: &IdentityKeyPair,
        self_node_id: &[u8; 32],
        middle_node_id: &[u8; 32],
        terminal_node_id: &[u8; 32],
        route_id: [u8; 16],
        now: u64,
    ) -> ChatEnvelope {
        let receiver = Self::synthetic_two_hop_probe_wallet_id(
            now,
            self_node_id,
            middle_node_id,
            terminal_node_id,
            &route_id,
            b"receiver",
        );
        let nonce_source = Self::synthetic_two_hop_probe_wallet_id(
            now,
            self_node_id,
            middle_node_id,
            terminal_node_id,
            &route_id,
            b"nonce",
        );
        let ciphertext = Self::blind_relay_probe_blob(now, self_node_id, terminal_node_id);
        let mut nonce = [0u8; 24];
        nonce.copy_from_slice(&nonce_source[..24]);
        let mut envelope = ChatEnvelope {
            message_id: route_id,
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

    fn synthetic_two_hop_probe_wallet_id(
        now: u64,
        self_node_id: &[u8; 32],
        middle_node_id: &[u8; 32],
        terminal_node_id: &[u8; 32],
        route_id: &[u8; 16],
        label: &[u8],
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx:two-hop-onion-delivery-probe:v1");
        hasher.update(label);
        hasher.update(now.to_be_bytes());
        hasher.update(self_node_id);
        hasher.update(middle_node_id);
        hasher.update(terminal_node_id);
        hasher.update(route_id);
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..32]);
        out
    }

    fn blind_relay_probe_url(endpoint: &str) -> Option<String> {
        let endpoint = endpoint.trim().trim_end_matches('/');
        if endpoint.is_empty() {
            return None;
        }
        let base = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("http://{endpoint}")
        };
        Some(format!("{base}/api/chat/peer/blind-relay"))
    }

    fn blind_relay_probe_route_id(
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

    fn blind_relay_two_hop_probe_route_id(
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

    fn blind_relay_probe_blob(now: u64, self_node_id: &[u8; 32], next_hop: &[u8; 32]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"aeronyx:blind-relay-probe:opaque-blob:v1");
        hasher.update(now.to_le_bytes());
        hasher.update(&self_node_id[..]);
        hasher.update(&next_hop[..]);
        hasher.finalize().to_vec()
    }

    fn classify_reqwest_error(phase: &str, error: &reqwest::Error) -> String {
        if error.is_timeout() {
            return format!("{phase}_timeout");
        }
        if error.is_connect() {
            return format!("{phase}_connect");
        }
        if error.is_status() {
            if let Some(status) = error.status() {
                return format!("{phase}_http_{}", status.as_u16());
            }
            return format!("{phase}_http_status");
        }
        if error.is_decode() {
            return format!("{phase}_decode");
        }
        if error.is_body() {
            return format!("{phase}_body");
        }
        if error.is_request() {
            return format!("{phase}_request");
        }
        format!("{phase}_unknown")
    }

    fn discovery_gossip_url(endpoint: &str) -> Option<String> {
        let endpoint = endpoint.trim().trim_end_matches('/');
        if endpoint.is_empty() {
            return None;
        }
        let base = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("http://{endpoint}")
        };
        Some(format!("{base}/api/discovery/gossip"))
    }

    async fn relay_chat_envelope_to_discovered_peers(
        client: Option<&reqwest::Client>,
        relay: Option<&ChatRelayService>,
        peer_store: &PeerStore,
        self_node_id: &[u8; 32],
        envelope: &ChatEnvelope,
    ) -> usize {
        let now = unix_now_secs();
        let Some(client) = client else {
            if let Some(relay) = relay {
                relay.record_peer_relay_outbound(
                    now,
                    0,
                    0,
                    Some("peer_http_client_unavailable".to_string()),
                );
            }
            return 0;
        };

        let mut attempted = 0usize;
        let mut accepted = 0usize;
        let mut last_failure_reason: Option<String> = None;

        let excluded_node_ids = [*self_node_id];
        for peer in peer_store.route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            CHAT_PEER_RELAY_FANOUT_LIMIT,
            &excluded_node_ids,
        ) {
            let peer_node_id = peer.node_id();
            let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
                peer_store.record_route_forward_failure(&peer_node_id, now, "missing_endpoint");
                continue;
            };
            let Some(url) = Self::chat_peer_relay_url(endpoint) else {
                peer_store.record_route_forward_failure(&peer_node_id, now, "invalid_endpoint");
                continue;
            };

            attempted += 1;
            let response = client
                .post(&url)
                .json(&PeerChatRelayRequest {
                    envelope: envelope.clone(),
                })
                .send()
                .await;

            match response {
                Ok(response) if response.status().is_success() => {
                    match response.json::<PeerChatRelayResponse>().await {
                        Ok(ack) if ack.accepted => {
                            accepted += 1;
                            peer_store.record_route_forward_success(&peer_node_id, now);
                        }
                        Ok(_ack) => {
                            let reason = "peer_relay_ack_rejected".to_string();
                            peer_store.record_route_forward_failure(
                                &peer_node_id,
                                now,
                                reason.clone(),
                            );
                            last_failure_reason = Some(reason);
                            debug!(
                                peer = %url,
                                "[CHAT_RELAY] Peer relay ACK rejected encrypted envelope"
                            );
                        }
                        Err(error) => {
                            let reason = Self::classify_reqwest_error("peer_relay_ack", &error);
                            peer_store.record_route_forward_failure(
                                &peer_node_id,
                                now,
                                reason.clone(),
                            );
                            last_failure_reason = Some(reason);
                            debug!(
                                peer = %url,
                                error = %error,
                                "[CHAT_RELAY] Peer relay ACK decode failed"
                            );
                        }
                    }
                }
                Ok(response) => {
                    let reason = format!("peer_relay_http_{}", response.status().as_u16());
                    peer_store.record_route_forward_failure(&peer_node_id, now, reason.clone());
                    last_failure_reason = Some(reason);
                    debug!(
                        peer = %url,
                        status = %response.status(),
                        "[CHAT_RELAY] Peer relay rejected encrypted envelope"
                    );
                }
                Err(error) => {
                    let reason = Self::classify_reqwest_error("peer_relay_request", &error);
                    peer_store.record_route_forward_failure(&peer_node_id, now, reason.clone());
                    last_failure_reason = Some(reason);
                    debug!(
                        peer = %url,
                        error = %error,
                        "[CHAT_RELAY] Peer relay request failed"
                    );
                }
            }
        }

        if let Some(relay) = relay {
            relay.record_peer_relay_outbound(now, attempted, accepted, last_failure_reason);
        }

        if attempted > 0 {
            debug!(
                attempted,
                accepted,
                id = %hex::encode(envelope.message_id),
                "[CHAT_RELAY] Peer relay fanout complete"
            );
        }

        accepted
    }

    fn chat_peer_relay_url(endpoint: &str) -> Option<String> {
        let endpoint = endpoint.trim().trim_end_matches('/');
        if endpoint.is_empty() {
            return None;
        }
        let base = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("http://{endpoint}")
        };
        Some(format!("{base}/api/chat/peer/relay"))
    }

    async fn save_peer_store_cache_snapshot(
        peer_store: &PeerStore,
        path: &str,
        now: u64,
    ) -> Result<()> {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }

        let snapshot = peer_store.export_peer_cache_snapshot(now);
        let bytes = snapshot.to_json_pretty()?;
        let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
        let backup_path = Self::peer_cache_backup_path(path.to_string_lossy().as_ref());

        let mut tmp_file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .await?;
        tmp_file.write_all(&bytes).await?;
        tmp_file.flush().await?;
        tmp_file.sync_all().await?;
        drop(tmp_file);

        if tokio::fs::metadata(&path).await.is_ok() {
            if tokio::fs::copy(&path, &backup_path).await.is_ok() {
                let _ = Self::sync_file_for_durability(&backup_path).await;
            }
        }
        tokio::fs::rename(&tmp_path, &path).await?;
        Self::sync_parent_dir_for_durability(&path).await?;
        Ok(())
    }

    async fn sync_file_for_durability(path: &PathBuf) -> Result<()> {
        let file = tokio::fs::OpenOptions::new().read(true).open(path).await?;
        file.sync_all().await?;
        Ok(())
    }

    async fn sync_parent_dir_for_durability(path: &PathBuf) -> Result<()> {
        let Some(parent) = path.parent() else {
            return Ok(());
        };
        if parent.as_os_str().is_empty() {
            return Ok(());
        }

        match tokio::fs::File::open(parent).await {
            Ok(dir) => {
                dir.sync_all().await?;
                Ok(())
            }
            Err(e) if cfg!(target_os = "windows") => {
                debug!(
                    source = %path.display(),
                    error = %e,
                    "[DISCOVERY] Parent directory fsync unavailable on this platform"
                );
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn peer_cache_backup_path(path: &str) -> PathBuf {
        PathBuf::from(format!("{path}.bak"))
    }

    async fn persist_peer_store_cache_once(
        peer_store: &PeerStore,
        path: &str,
        now: u64,
    ) -> Result<()> {
        let pending_snapshot = peer_store.export_bootstrap_snapshot(now, now, false, None);
        if pending_snapshot.verified_count_at(now) == 0
            && Self::peer_cache_has_usable_recovery_snapshot(path, now).await
        {
            peer_store.record_cache_save_status(now, "skipped", "preserved_existing_snapshot");
            return Ok(());
        }

        match Self::save_peer_store_cache_snapshot(peer_store, path, now).await {
            Ok(()) => {
                peer_store.record_cache_save_status(now, "success", "snapshot_persisted");
                Ok(())
            }
            Err(e) => {
                peer_store.record_cache_save_status(now, "failed", "write_failed");
                Err(e)
            }
        }
    }

    async fn peer_cache_has_usable_recovery_snapshot(path: &str, now: u64) -> bool {
        let primary = PathBuf::from(path);
        if Self::cache_snapshot_file_has_usable_peers(&primary, now).await {
            return true;
        }

        let backup = Self::peer_cache_backup_path(path);
        Self::cache_snapshot_file_has_usable_peers(&backup, now).await
    }

    async fn cache_snapshot_file_has_usable_peers(path: &PathBuf, now: u64) -> bool {
        let Ok(bytes) = tokio::fs::read(path).await else {
            return false;
        };
        let Ok(snapshot) = NodeBootstrapSnapshot::from_json_bytes(&bytes) else {
            return false;
        };

        snapshot.verified_count_at(now) > 0
    }

    fn build_self_discovery_descriptor(&self, now: u64) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for(&self.config, &self.identity, now)
    }

    fn build_self_discovery_descriptor_with_runtime(
        &self,
        now: u64,
        chat_relay_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime(
            &self.config,
            &self.identity,
            now,
            chat_relay_runtime_ready,
        )
    }

    fn build_self_discovery_descriptor_for(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
    ) -> Result<SignedNodeDescriptor> {
        Self::build_self_discovery_descriptor_for_runtime(
            config,
            identity,
            now,
            config.memchain.is_chat_relay_enabled(),
        )
    }

    fn build_self_discovery_descriptor_for_runtime(
        config: &ServerConfig,
        identity: &IdentityKeyPair,
        now: u64,
        chat_relay_runtime_ready: bool,
    ) -> Result<SignedNodeDescriptor> {
        let ttl = config.discovery.descriptor_ttl_secs;
        let expires_at = now.saturating_add(ttl);
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            now,
            now,
            expires_at,
            env!("CARGO_PKG_VERSION"),
        );

        descriptor.public_endpoint = config
            .discovery
            .public_endpoint
            .as_ref()
            .or(config.network.public_endpoint.as_ref())
            .map(|endpoint| endpoint.trim().to_string())
            .filter(|endpoint| !endpoint.is_empty());

        descriptor.capabilities =
            Self::discovery_capabilities_for_runtime(config, chat_relay_runtime_ready);
        descriptor.capacity = NodeCapacity {
            max_sessions: u32::try_from(config.max_sessions()).unwrap_or(u32::MAX),
            max_bps: None,
            max_pps: None,
        };
        descriptor.policy = NodePolicy {
            allows_public_exit: false,
            public_discovery: config.discovery.public_discovery,
            region: config
                .discovery
                .region
                .as_ref()
                .map(|region| region.trim().to_string())
                .filter(|region| !region.is_empty()),
        };

        // Onion routing: publish the node's CURRENT rotating onion KEM public
        // key (forward secrecy — see services::onion_keys). NOT identity-derived:
        // the X25519 public is not recoverable from the Ed25519 node_id, and a
        // rotating key bounds exposure of past routing metadata if a key leaks.
        descriptor = descriptor.with_x25519_kem(crate::services::onion_keys::current_public_key());

        SignedNodeDescriptor::sign(descriptor, identity).map_err(ServerError::from)
    }

    fn discovery_capabilities_for_runtime(
        config: &ServerConfig,
        chat_relay_runtime_ready: bool,
    ) -> Vec<NodeCapability> {
        let mut capabilities = vec![NodeCapability::PrivacyRelay];
        let advertises_peer_api = Self::discovery_peer_api_ready_for(config);

        if config.memchain.is_chat_relay_enabled()
            && chat_relay_runtime_ready
            && advertises_peer_api
        {
            capabilities.push(NodeCapability::ChatRelay);
        }
        if config.discovery.advertise_onion_middle && advertises_peer_api {
            capabilities.push(NodeCapability::OnionMiddle);
        }
        if config.memchain.is_enabled() {
            capabilities.push(NodeCapability::EncryptedStorage);
        }
        if config.memchain.is_supernode_enabled() {
            capabilities.push(NodeCapability::AgentRelay);
        }

        capabilities
    }

    fn discovery_peer_api_ready_for(config: &ServerConfig) -> bool {
        config.discovery.public_api_listen_addr.is_some()
            && config
                .discovery
                .public_endpoint
                .as_deref()
                .map(str::trim)
                .map(|endpoint| !endpoint.is_empty())
                .unwrap_or(false)
    }

    fn discovery_local_capability_status_for(
        config: &ServerConfig,
    ) -> DiscoveryLocalCapabilityStatus {
        Self::discovery_local_capability_status_for_runtime(
            config,
            config.memchain.is_chat_relay_enabled(),
        )
    }

    fn discovery_local_capability_status_for_runtime(
        config: &ServerConfig,
        chat_relay_runtime_ready: bool,
    ) -> DiscoveryLocalCapabilityStatus {
        let capabilities =
            Self::discovery_capabilities_for_runtime(config, chat_relay_runtime_ready);
        DiscoveryLocalCapabilityStatus::new(
            config.memchain.is_chat_relay_enabled(),
            Self::discovery_peer_api_ready_for(config),
            chat_relay_runtime_ready,
            capabilities.contains(&NodeCapability::ChatRelay),
        )
    }

    fn import_bootstrap_snapshot_bytes(
        peer_store: &PeerStore,
        source_kind: &'static str,
        source: &str,
        bytes: &[u8],
        now: u64,
    ) -> bool {
        match NodeBootstrapSnapshot::from_json_bytes(bytes) {
            Ok(snapshot) => {
                let report = if matches!(source_kind, "cache" | "cache_backup") {
                    peer_store.load_peer_cache_snapshot_from_source(&snapshot, now, source_kind)
                } else {
                    peer_store.load_bootstrap_snapshot_from_source(&snapshot, now, source_kind)
                };
                peer_store.record_bootstrap_source(
                    now,
                    source_kind,
                    if report.rejected > 0 {
                        "warning"
                    } else {
                        "success"
                    },
                    format!(
                        "total={} inserted={} unchanged={} stale={} rejected={}",
                        report.total,
                        report.inserted,
                        report.unchanged,
                        report.stale,
                        report.rejected
                    ),
                );
                info!(
                    source_kind,
                    source = %source,
                    total = report.total,
                    inserted = report.inserted,
                    unchanged = report.unchanged,
                    stale = report.stale,
                    rejected = report.rejected,
                    "[DISCOVERY] Bootstrap snapshot imported"
                );
                report.inserted > 0 || report.unchanged > 0
            }
            Err(e) => {
                peer_store.record_bootstrap_source(now, source_kind, "failed", "json_rejected");
                warn!(
                    source_kind,
                    source = %source,
                    error = %e,
                    "[DISCOVERY] Bootstrap snapshot rejected"
                );
                false
            }
        }
    }

    fn init_services(
        &self,
    ) -> Result<(Arc<IpPoolService>, Arc<SessionManager>, Arc<RoutingService>)> {
        let (network, prefix) = self.config.parse_ip_range()?;
        let ip_pool = Arc::new(IpPoolService::new(
            network,
            prefix,
            self.config.gateway_ip(),
        )?);
        let sessions = Arc::new(SessionManager::new(
            self.config.max_sessions(),
            Duration::from_secs(self.config.session_timeout_secs()),
        ));
        let routing = Arc::new(RoutingService::new());
        info!(
            capacity = ip_pool.capacity(),
            max_sessions = self.config.max_sessions(),
            "Services initialized"
        );
        Ok((ip_pool, sessions, routing))
    }

    #[cfg(target_os = "linux")]
    async fn init_tun(&self) -> Result<Arc<LinuxTun>> {
        let (_network, prefix_len) = self.config.parse_ip_range()?;
        let cfg = TunConfig::new(self.config.device_name())
            .with_address(self.config.gateway_ip())
            .with_netmask(prefix_to_netmask(prefix_len))
            .with_mtu(self.config.mtu());
        let tun = LinuxTun::create(cfg)
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN: {}", e)))?;
        tun.up()
            .await
            .map_err(|e| ServerError::startup_failed(format!("TUN up: {}", e)))?;
        info!(
            "TUN '{}' initialized @ {}",
            tun.name(),
            self.config.gateway_ip()
        );
        Ok(Arc::new(tun))
    }

    // ============================================
    // UDP Task
    // ============================================

    #[allow(clippy::too_many_arguments)]
    fn spawn_udp_task(
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
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let udp_reply = Arc::clone(&udp);
        let self_node_id = self.identity.public_key_bytes();

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            let crypto = DefaultTransportCrypto::new();
            let chat_peer_client = reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .ok();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    result = udp.recv(&mut buf) => {
                        match result {
                            Ok((len, source)) => {
                                if shutdown.load(Ordering::SeqCst) { break; }
                                let data = &buf[..len];

                                match ProtocolCodec::peek_message_type(data) {
                                    Ok(MessageType::ClientHello) => {
                                        let extension = if data.len() > CLIENT_HELLO_SIZE {
                                            data[CLIENT_HELLO_SIZE..].to_vec()
                                        } else {
                                            Vec::new()
                                        };
                                        let verifier = Arc::clone(&voucher_verifier);
                                        if !verifier.accept_client_hello_extension(extension).await {
                                            warn!(
                                                client = %source.addr,
                                                "[VOUCHER] rejected ClientHello with invalid voucher"
                                            );
                                            continue;
                                        }

                                        if let Ok(hello) = decode_client_hello(data) {
                                            match handshake.process(&hello, source.addr) {
                                                Ok(result) => {
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
                                                Err(e) => warn!("[HANDSHAKE] Failed {}: {}", source.addr, e),
                                            }
                                        }
                                    }
                                    Ok(MessageType::Keepalive) => {
                                        if len >= KEEPALIVE_PACKET_SIZE {
                                            let mut sid = [0u8; 16];
                                            sid.copy_from_slice(&data[1..17]);
                                            if let Some(id) = SessionId::from_bytes(&sid) {
                                                if let Some(s) = sessions.get(&id) { s.touch(); }
                                            }
                                        }
                                    }
                                    Ok(MessageType::Data) | Err(_) => {
                                        match packet_handler.handle_udp_packet(data) {
                                            Ok((_sess, DecryptedPayload::Vpn(pkt))) => {
                                                #[cfg(target_os = "linux")]
                                                { let _ = tun.write(&pkt).await; }
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
                                            Ok((session, DecryptedPayload::VoiceSignal { discriminant, target_wallet, payload })) => {
                                                let signal_name = match discriminant { 31 => "Offer", 32 => "Answer", 33 => "End", _ => "Unknown" };
                                                match target_wallet {
                                                    None => { warn!(src = %session.virtual_ip, disc = discriminant, "[VOICE_SIGNAL] {} — missing 'to' field, dropped", signal_name); }
                                                    Some(ref wallet_hex) => {
                                                        let target_bytes = hex::decode(wallet_hex).ok().and_then(|b| {
                                                            if b.len() == 32 { let mut arr = [0u8; 32]; arr.copy_from_slice(&b); Some(arr) } else { None }
                                                        });
                                                        match target_bytes {
                                                            None => { warn!(wallet = %wallet_hex, "[VOICE_SIGNAL] {} — invalid pubkey hex, dropped", signal_name); }
                                                            Some(pk) => {
                                                                let target = sessions.get_by_wallet(&pk).or_else(|| {
                                                                    sessions.all_sessions().into_iter().find(|s| s.client_public_key.to_bytes() == pk)
                                                                });
                                                                match target {
                                                                    None => { debug!(wallet = %&wallet_hex[..8], "[VOICE_SIGNAL] {} — target offline, dropped", signal_name); }
                                                                    Some(target_session) => {
                                                                        let counter = target_session.next_tx_counter();
                                                                        let mut encrypted = vec![0u8; payload.len() + ENCRYPTION_OVERHEAD];
                                                                        match crypto.encrypt(&target_session.session_key, counter, target_session.id.as_bytes(), &payload, &mut encrypted) {
                                                                            Ok(len) => {
                                                                                encrypted.truncate(len);
                                                                                let pkt   = aeronyx_core::protocol::DataPacket::new(*target_session.id.as_bytes(), counter, encrypted);
                                                                                let bytes = aeronyx_core::protocol::codec::encode_data_packet(&pkt).to_vec();
                                                                                let _ = udp_reply.send(&bytes, &target_session.client_endpoint).await;
                                                                                debug!(src = %session.virtual_ip, dst = %target_session.virtual_ip, signal = signal_name, "[VOICE_SIGNAL] Forwarded");
                                                                            }
                                                                            Err(e) => { warn!(error = %e, "[VOICE_SIGNAL] Re-encrypt failed"); }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            Ok((session, DecryptedPayload::Voice { dst_ip, payload })) => {
                                                if let Some(target_sid) = routing.lookup(dst_ip) {
                                                    if let Some(target) = sessions.get(&target_sid) {
                                                        let counter = target.next_tx_counter();
                                                        let mut encrypted = vec![0u8; payload.len() + ENCRYPTION_OVERHEAD];
                                                        match crypto.encrypt(&target.session_key, counter, target.id.as_bytes(), &payload, &mut encrypted) {
                                                            Ok(len) => {
                                                                encrypted.truncate(len);
                                                                let pkt   = aeronyx_core::protocol::DataPacket::new(*target.id.as_bytes(), counter, encrypted);
                                                                let bytes = aeronyx_core::protocol::codec::encode_data_packet(&pkt).to_vec();
                                                                let _ = udp_reply.send(&bytes, &target.client_endpoint).await;
                                                                trace!(src = %session.virtual_ip, dst = %dst_ip, "[VOICE] Relayed voice packet");
                                                            }
                                                            Err(e) => { warn!(dst_ip = %dst_ip, "[VOICE] Re-encrypt failed: {}", e); }
                                                        }
                                                    } else {
                                                        debug!(dst_ip = %dst_ip, "[VOICE] Target session not found (disconnected?)");
                                                    }
                                                } else {
                                                    debug!(dst_ip = %dst_ip, "[VOICE] No route to dst_ip (peer offline)");
                                                }
                                            }
                                            Ok((session, DecryptedPayload::MemChain(msg))) => {
                                                if let (Some(ref mp), Some(ref aw)) = (&mempool, &aof_writer) {
                                                    Self::handle_memchain_message(
                                                        msg, mp, aw, &storage, &vector_index,
                                                        &memchain_config, &server_pubkey_hex,
                                                        &session, &udp_reply, &crypto,
                                                        &sessions, &chat_relay, &peer_store,
                                                        &self_node_id, chat_peer_client.as_ref(),
                                                    ).await;
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                if !shutdown.load(Ordering::SeqCst) { error!("UDP recv error: {}", e); }
                            }
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // MemChain Message Handler (unchanged)
    // ============================================

    #[allow(clippy::too_many_arguments)]
    async fn handle_memchain_message(
        msg: MemChainMessage,
        mempool: &Arc<MemPool>,
        aof_writer: &Arc<TokioMutex<AofWriter>>,
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
        chat_peer_client: Option<&reqwest::Client>,
    ) {
        match msg {
            MemChainMessage::BroadcastFact(fact) => {
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
                if let Some(ref st) = storage {
                    if st.insert(&record, "p2p-remote").await {
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
            }
            MemChainMessage::SyncRequest { last_known_hash } => {
                let facts = mempool.get_facts_after(last_known_hash);
                let resp = MemChainMessage::SyncResponse { facts };
                Self::send_to_session(&resp, session, udp, crypto).await;
            }
            MemChainMessage::SyncResponse { facts } => {
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
                if let Some(ref st) = storage {
                    let records = st.query_by_owner_after(&owner, after_timestamp).await;
                    let resp = MemChainMessage::SyncRecordResponse { records };
                    Self::send_to_session(&resp, session, udp, crypto).await;
                }
            }
            MemChainMessage::SyncRecordResponse { records } => {
                if let Some(ref st) = storage {
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
                        let _ = st.insert(&record, "p2p-sync").await;
                    }
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
                    warn!(receiver = %hex::encode(&envelope.receiver[..4]), "[CHAT_RELAY] Envelope sig failed — dropped");
                    return;
                }
                let Some(ref relay) = chat_relay else {
                    warn!(receiver = %hex::encode(&envelope.receiver[..4]), "[CHAT_RELAY] Relay unavailable — dropped");
                    return;
                };
                if relay.is_online_duplicate(&envelope.message_id) {
                    debug!(id = %hex::encode(envelope.message_id), "[CHAT_RELAY] Online duplicate — dropped");
                    return;
                }
                relay.wallet_routes.announce(
                    &envelope.sender,
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
                            Self::send_to_session(
                                &MemChainMessage::ChatRelay(envelope.clone()),
                                &target_session,
                                udp,
                                crypto,
                            )
                            .await;
                            all_failed = false;
                        } else {
                            relay.wallet_routes.remove_session(target_sid);
                            debug!(session = %target_sid, "[CHAT_RELAY] Pruned stale route during delivery");
                        }
                    }
                    if all_failed {
                        Self::relay_chat_envelope_to_discovered_peers(
                            chat_peer_client,
                            Some(relay.as_ref()),
                            peer_store,
                            self_node_id,
                            &envelope,
                        )
                        .await;
                        if let Err(e) = relay.store_pending(&envelope) {
                            warn!(error = %e, receiver = %hex::encode(&receiver[..4]), "[CHAT_RELAY] Fallback store_pending failed");
                        } else {
                            debug!(receiver = %hex::encode(&receiver[..4]), "[CHAT_RELAY] All routes stale — stored for offline delivery");
                        }
                    } else {
                        debug!(receiver = %hex::encode(&receiver[..4]), devices = device_count, "[CHAT_RELAY] Online delivery to {} device(s)", device_count);
                    }
                } else {
                    Self::relay_chat_envelope_to_discovered_peers(
                        chat_peer_client,
                        Some(relay.as_ref()),
                        peer_store,
                        self_node_id,
                        &envelope,
                    )
                    .await;
                    if let Err(e) = relay.store_pending(&envelope) {
                        warn!(error = %e, receiver = %hex::encode(&receiver[..4]), "[CHAT_RELAY] store_pending failed");
                    } else {
                        debug!(receiver = %hex::encode(&receiver[..4]), id = %hex::encode(envelope.message_id), "[CHAT_RELAY] Stored for offline delivery");
                    }
                }
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
                relay
                    .wallet_routes
                    .announce(&wallet, session.id.clone(), session.client_endpoint);
                match relay.pull_pending(&wallet, after_timestamp, &cursor, limit) {
                    Ok((messages, has_more)) => {
                        let envelopes: Vec<_> = messages.into_iter().map(|m| m.envelope).collect();
                        let resp = MemChainMessage::ChatPullResponse {
                            envelopes,
                            has_more,
                        };
                        Self::send_to_session(&resp, session, udp, crypto).await;
                    }
                    Err(e) => {
                        warn!(error = %e, wallet = %hex::encode(&wallet[..4]), "[CHAT_RELAY] pull_pending failed");
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
                if let Err(e) = verify_result {
                    warn!(wallet = %hex::encode(&wallet[..4]), error = %e, "[CHAT_RELAY] ChatAck sig failed");
                    return;
                }
                match relay.ack_messages(&message_ids, &wallet) {
                    Ok(deleted) => {
                        debug!(deleted, wallet = %hex::encode(&wallet[..4]), "[CHAT_RELAY] ChatAck processed");
                    }
                    Err(e) => {
                        warn!(error = %e, wallet = %hex::encode(&wallet[..4]), "[CHAT_RELAY] ack_messages failed");
                    }
                }
            }
            MemChainMessage::DeviceRegister {
                device_id,
                device_name,
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
                if let Err(e) = verify_result {
                    warn!(session = %session.id, wallet = %hex::encode(&wallet_pubkey[..4]), error = %e, "[CHAT_RELAY] DeviceRegister sig failed");
                    return;
                }
                let name_display = if device_name.len() > 64 {
                    let mut end = 64;
                    while !device_name.is_char_boundary(end) {
                        end -= 1;
                    }
                    &device_name[..end]
                } else {
                    &device_name
                };
                let Some(ref relay) = chat_relay else {
                    return;
                };
                relay.wallet_routes.announce(
                    &wallet_pubkey,
                    session.id.clone(),
                    session.client_endpoint,
                );
                sessions.register_device(&wallet_pubkey, device_id, session.id.clone());
                info!(session_id = %session.id, wallet = %hex::encode(&wallet_pubkey[..4]), device_id = %hex::encode(device_id), device_name = %name_display, "[CHAT_RELAY] Device registered");
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
                        info!(count, wallet = %hex::encode(&wallet_pubkey[..4]), "[CHAT_RELAY] Delivered pending messages on register");
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!(error = %e, wallet = %hex::encode(&wallet_pubkey[..4]), "[CHAT_RELAY] pull_pending on register failed");
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
                if let Err(e) = verify_result {
                    debug!(wallet = %hex::encode(&wallet_pubkey[..4]), error = %e, "[CHAT_RELAY] WalletPresence sig failed");
                    return;
                }
                relay.wallet_routes.announce(
                    &wallet_pubkey,
                    session.id.clone(),
                    session.client_endpoint,
                );
                debug!(wallet = %hex::encode(&wallet_pubkey[..4]), "[CHAT_RELAY] WalletPresence: route refreshed");
            }
            _ => {
                debug!("[MEMCHAIN] Unhandled message variant");
            }
        }
    }

    async fn send_to_session(
        msg: &MemChainMessage,
        session: &Arc<crate::services::Session>,
        udp: &Arc<UdpTransport>,
        crypto: &DefaultTransportCrypto,
    ) {
        let plaintext = match encode_memchain(msg) {
            Ok(p) => p,
            Err(e) => {
                error!("[MEMCHAIN_TX] Encode: {}", e);
                return;
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
            Err(e) => {
                error!("[MEMCHAIN_TX] Encrypt: {}", e);
                return;
            }
        };
        encrypted.truncate(len);
        let pkt = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
        let bytes = encode_data_packet(&pkt).to_vec();
        let _ = udp.send(&bytes, &session.client_endpoint).await;
    }

    // ============================================
    // TUN Task
    // ============================================

    #[cfg(target_os = "linux")]
    fn spawn_tun_task(
        &self,
        tun: Arc<LinuxTun>,
        udp: Arc<UdpTransport>,
        handler: Arc<PacketHandler>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    result = tun.read(&mut buf) => {
                        match result {
                            Ok(len) => {
                                if shutdown.load(Ordering::SeqCst) { break; }
                                if let Ok((enc, ep)) = handler.handle_tun_packet(&buf[..len]) {
                                    let _ = udp.send(&enc, &ep).await;
                                }
                            }
                            Err(e) => { if !shutdown.load(Ordering::SeqCst) { error!("TUN: {}", e); } }
                        }
                    }
                }
            }
        })
    }

    // ============================================
    // Traffic Snapshot Task
    // ============================================

    fn spawn_traffic_snapshot_task(
        &self,
        sessions: Arc<SessionManager>,
        events: SessionEventSender,
        interval_secs: u64,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        let interval_secs = interval_secs.clamp(10, 300);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
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

    fn spawn_keepalive_probe_task(
        &self,
        sessions: Arc<SessionManager>,
        udp: Arc<UdpTransport>,
        packet_handler: Arc<PacketHandler>,
        gateway_ip: Ipv4Addr,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
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

    // ============================================
    // Cleanup Task
    // ============================================

    fn spawn_cleanup_task(
        &self,
        sessions: Arc<SessionManager>,
        ip_pool: Arc<IpPoolService>,
        routing: Arc<RoutingService>,
        events: SessionEventSender,
        chat_relay: Option<Arc<ChatRelayService>>,
        traffic_tracker: Arc<TrafficTracker>,
        deny_list: Arc<DenyList>,
    ) -> JoinHandle<()> {
        let shutdown = Arc::clone(&self.shutdown);
        let mut rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(Duration::from_secs(60));
            loop {
                tokio::select! {
                    _ = rx.recv() => break,
                    _ = timer.tick() => {
                        if shutdown.load(Ordering::SeqCst) { break; }

                        for (sid, vip, wallet, bytes_rx, bytes_tx, snap) in sessions.cleanup_expired() {
                            routing.remove_route(vip);
                            events.session_ended(
                                &sid.to_string(),
                                Some(wallet.clone()),
                                Some(vip.to_string()),
                                bytes_rx,
                                bytes_tx,
                                quality_from_stats(snap),
                            );
                            if let Some(ref relay) = chat_relay {
                                relay.wallet_routes.remove_session(&sid);
                            }
                            traffic_tracker.remove_wallet(&wallet);
                        }

                        for ip in sessions.drain_cooldown_pool() {
                            ip_pool.release(ip);
                        }

                        // Evict expired deny list entries (QuotaExceeded whose
                        // month has rolled over). NoPremiumAccess entries are
                        // permanent and are only removed by handle_membership_response.
                        deny_list.cleanup();
                    }
                }
            }
        })
    }

    // ============================================
    // Shutdown
    // ============================================

    async fn wait_for_shutdown(&self) {
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl+C listener failed");
        info!("Shutdown signal received");
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());
    }
}

fn prefix_to_netmask(prefix_len: u8) -> Ipv4Addr {
    if prefix_len == 0 {
        return Ipv4Addr::new(0, 0, 0, 0);
    }

    Ipv4Addr::from(u32::MAX << (32 - prefix_len))
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("listen", &self.config.listen_addr())
            .field("tun", &self.config.device_name())
            .field("mode", &self.config.memchain.mode)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        commitment_witness_startup_decision, memchain_index_rejection_reason, prefix_to_netmask,
        unix_now_secs, CommitmentWitnessStartupBlockReason, CommitmentWitnessStartupDecision,
        Server, BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS,
    };
    use crate::api::memchain_peer::CommitmentReconciliationOutcome;
    use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
    use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};
    use aeronyx_core::protocol::chat::{ChatContentType, ChatEnvelope};
    use aeronyx_core::protocol::onion::is_onion_blob;
    use aeronyx_core::protocol::{
        NodeBootstrapSnapshot, NodeCapability, NodeCapacity, NodeDescriptor, NodeDiscoveryMessage,
        SignedNodeDescriptor,
    };
    use axum::{routing::post, Json, Router};
    use std::net::Ipv4Addr;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::net::TcpListener;

    use crate::api::chat_peer::{PeerChatRelayRequest, PeerChatRelayResponse};
    use crate::api::discovery::GossipResponse;
    use crate::config::{DiscoveryConfig, ServerConfig};
    use crate::services::{PeerStore, PeerStoreImportReport};

    #[test]
    fn commitment_witness_startup_gate_enforces_threshold_and_conflicts() {
        let unavailable = CommitmentReconciliationOutcome::default();
        assert_eq!(
            commitment_witness_startup_decision(&unavailable, false, 1),
            Ok(CommitmentWitnessStartupDecision::DegradedUnverified)
        );
        assert_eq!(
            commitment_witness_startup_decision(&unavailable, true, 1),
            Err(CommitmentWitnessStartupBlockReason::Unavailable)
        );

        let converged = CommitmentReconciliationOutcome {
            verified: 1,
            converged: 1,
            ..CommitmentReconciliationOutcome::default()
        };
        assert_eq!(
            commitment_witness_startup_decision(&converged, true, 1),
            Ok(CommitmentWitnessStartupDecision::Verified)
        );
        assert_eq!(
            commitment_witness_startup_decision(&converged, false, 2),
            Ok(CommitmentWitnessStartupDecision::DegradedBelowThreshold)
        );
        assert_eq!(
            commitment_witness_startup_decision(&converged, true, 2),
            Err(CommitmentWitnessStartupBlockReason::ThresholdUnmet)
        );

        let two_converged = CommitmentReconciliationOutcome {
            verified: 2,
            converged: 2,
            ..CommitmentReconciliationOutcome::default()
        };
        assert_eq!(
            commitment_witness_startup_decision(&two_converged, true, 2),
            Ok(CommitmentWitnessStartupDecision::Verified)
        );

        let remote_ahead = CommitmentReconciliationOutcome {
            verified: 1,
            remote_ahead: 1,
            ..CommitmentReconciliationOutcome::default()
        };
        assert_eq!(
            commitment_witness_startup_decision(&remote_ahead, false, 2),
            Err(CommitmentWitnessStartupBlockReason::RemoteAhead)
        );

        let diverged = CommitmentReconciliationOutcome {
            verified: 1,
            diverged: 1,
            ..CommitmentReconciliationOutcome::default()
        };
        assert_eq!(
            commitment_witness_startup_decision(&diverged, false, 2),
            Err(CommitmentWitnessStartupBlockReason::Divergence)
        );
    }

    fn signed_test_chat_envelope(now: u64) -> ChatEnvelope {
        let sender = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [0x55; 16],
            sender: sender.public_key_bytes(),
            receiver: [0x66; 32],
            timestamp: now,
            ciphertext: b"opaque encrypted payload".to_vec(),
            nonce: [0x77; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        envelope.signature = sender.sign(&envelope.sign_data());
        envelope
    }

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

    fn signed_chat_relay_peer_descriptor(
        endpoint: String,
        sequence: u64,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        let peer_identity = IdentityKeyPair::generate();
        let mut descriptor = NodeDescriptor::new(
            peer_identity.public_key_bytes(),
            sequence,
            sequence,
            expires_at,
            "test-peer",
        );
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        SignedNodeDescriptor::sign(descriptor, &peer_identity).unwrap()
    }

    fn signed_probe_peer_descriptor(
        endpoint: String,
        sequence: u64,
        expires_at: u64,
        capabilities: Vec<NodeCapability>,
        kem_public: [u8; 32],
    ) -> SignedNodeDescriptor {
        let peer_identity = IdentityKeyPair::generate();
        let mut descriptor = NodeDescriptor::new(
            peer_identity.public_key_bytes(),
            sequence,
            sequence,
            expires_at,
            "test-onion-peer",
        )
        .with_x25519_kem(kem_public);
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = capabilities;
        descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        SignedNodeDescriptor::sign(descriptor, &peer_identity).unwrap()
    }

    #[test]
    fn two_hop_onion_delivery_probe_request_uses_onion_blob_and_signed_probe_envelope() {
        let source = IdentityKeyPair::generate();
        let self_node_id = source.public_key_bytes();
        let now = 1_800_000_000;
        let middle = signed_probe_peer_descriptor(
            "http://198.51.100.10:8422".to_string(),
            now,
            now + 300,
            vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
            [0x21; 32],
        );
        let terminal = signed_probe_peer_descriptor(
            "http://198.51.100.11:8422".to_string(),
            now + 1,
            now + 300,
            vec![NodeCapability::ChatRelay],
            [0x22; 32],
        );

        let request = Server::build_two_hop_onion_delivery_probe_request(
            &source,
            &self_node_id,
            &middle,
            &terminal,
            now,
        )
        .expect("descriptors with KEM keys should build an onion delivery probe");
        let source_public = IdentityPublicKey::from_bytes(&self_node_id).unwrap();

        assert!(request.onward_envelope.is_none());
        assert!(request.onward_descriptor_hint.is_none());
        assert_eq!(request.previous_hop_node_id, self_node_id);
        assert_eq!(request.envelope.next_hop, middle.node_id());
        assert_eq!(request.envelope.ttl, 2);
        assert!(is_onion_blob(&request.envelope.encrypted_blob));
        request
            .envelope
            .verify_signature_from(&source_public)
            .expect("entry node signs the outer blind relay envelope");

        let synthetic_chat = Server::synthetic_two_hop_probe_chat_envelope(
            &source,
            &self_node_id,
            &middle.node_id(),
            &terminal.node_id(),
            request.envelope.route_id,
            now,
        );
        synthetic_chat
            .verify_signature()
            .expect("synthetic terminal ChatEnvelope must verify like user chat");
        assert_eq!(synthetic_chat.message_id, request.envelope.route_id);
        assert_eq!(synthetic_chat.sender, self_node_id);
        assert_ne!(synthetic_chat.receiver, [0u8; 32]);
        assert_eq!(synthetic_chat.content_type, ChatContentType::System);
    }

    #[test]
    fn two_hop_onion_delivery_probe_request_requires_published_kem_keys() {
        let source = IdentityKeyPair::generate();
        let self_node_id = source.public_key_bytes();
        let now = 1_800_000_000;
        let middle = signed_probe_peer_descriptor(
            "http://198.51.100.10:8422".to_string(),
            now,
            now + 300,
            vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
            [0u8; 32],
        );
        let terminal = signed_probe_peer_descriptor(
            "http://198.51.100.11:8422".to_string(),
            now + 1,
            now + 300,
            vec![NodeCapability::ChatRelay],
            [0x22; 32],
        );

        assert!(Server::build_two_hop_onion_delivery_probe_request(
            &source,
            &self_node_id,
            &middle,
            &terminal,
            now,
        )
        .is_none());
    }

    #[test]
    fn prefix_to_netmask_supports_vpn_pool_expansion() {
        assert_eq!(prefix_to_netmask(22), Ipv4Addr::new(255, 255, 252, 0));
        assert_eq!(prefix_to_netmask(24), Ipv4Addr::new(255, 255, 255, 0));
        assert_eq!(prefix_to_netmask(0), Ipv4Addr::new(0, 0, 0, 0));
    }

    #[test]
    fn discovery_startup_self_check_reports_commercial_readiness_buckets() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.gossip_enabled = true;

        let (status, detail) = Server::discovery_startup_self_check(&config);

        assert_eq!(status, "warning");
        assert!(detail.contains("peer_cache_path"));
        assert!(detail.contains("seed_endpoints"));
        assert!(detail.contains("public_endpoint"));
        assert!(detail.contains("public_api_listener"));
        assert!(!detail.contains("https://"));
        assert!(!detail.contains("/root/"));

        config.discovery.peer_cache_path = Some("/root/private/peer-cache.json".to_string());
        config.discovery.seed_endpoints = vec!["https://seed.example.com".to_string()];
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());

        let (status, detail) = Server::discovery_startup_self_check(&config);

        assert_eq!(status, "ready");
        assert!(detail.contains("cache"));
        assert!(!detail.contains("seed.example.com"));
        assert!(!detail.contains("/root/private"));
    }

    #[test]
    fn blind_relay_probe_cooldown_uses_recovery_interval_until_stability_window_is_ready() {
        let mut discovery = DiscoveryConfig::default();
        discovery.gossip_interval_secs = 60;

        let store = PeerStore::new();
        assert_eq!(
            Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_000,),
            super::BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS
        );

        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_010,
            true,
            "onion_terminal_delivered",
            2,
            1,
            2,
            1,
        );
        assert_eq!(
            Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_020,),
            super::BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS
        );

        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_030,
            true,
            "onion_terminal_delivered",
            2,
            1,
            2,
            1,
        );
        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_040,
            true,
            "onion_terminal_delivered",
            2,
            1,
            2,
            1,
        );
        assert_eq!(
            Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_050,),
            BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS
        );

        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_060,
            false,
            "request_error",
            2,
            1,
            2,
            1,
        );
        assert_eq!(
            Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_070,),
            super::BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS
        );
    }

    fn self_discovery_descriptor_uses_privacy_safe_public_metadata() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());
        config.discovery.region = Some("us-central".to_string());
        config.discovery.descriptor_ttl_secs = 900;
        config.discovery.public_discovery = false;

        let identity = IdentityKeyPair::generate();
        let node_id = identity.public_key_bytes();
        let server = Server::new(config, identity, None);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let signed = server.build_self_discovery_descriptor(now).unwrap();

        assert!(signed.verify_at(now + 1).is_ok());
        assert_eq!(signed.descriptor.node_id, node_id);
        assert_eq!(signed.descriptor.sequence, now);
        assert_eq!(signed.descriptor.issued_at, now);
        assert_eq!(signed.descriptor.expires_at, now + 900);
        assert_eq!(
            signed.descriptor.public_endpoint.as_deref(),
            Some("node.example.com:443")
        );
        assert_eq!(
            signed.descriptor.policy.region.as_deref(),
            Some("us-central")
        );
        assert!(!signed.descriptor.policy.allows_public_exit);
        assert!(!signed.descriptor.policy.public_discovery);
        assert!(signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::PrivacyRelay));
        assert_eq!(
            signed.descriptor.capacity.max_sessions,
            server.config.max_sessions() as u32
        );
    }

    #[test]
    fn self_discovery_descriptor_can_fallback_to_network_endpoint() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.network.public_endpoint = Some("198.51.100.10:51820".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let signed = server
            .build_self_discovery_descriptor(1_800_000_000)
            .unwrap();

        assert_eq!(
            signed.descriptor.public_endpoint.as_deref(),
            Some("198.51.100.10:51820")
        );
    }

    #[test]
    fn self_discovery_descriptor_requires_peer_api_endpoint_for_chat_relay_capability() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.network.public_endpoint = Some("198.51.100.10:51820".to_string());
        config.memchain.chat_relay.enabled = true;

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let signed = server
            .build_self_discovery_descriptor(1_800_000_000)
            .unwrap();

        assert_eq!(
            signed.descriptor.public_endpoint.as_deref(),
            Some("198.51.100.10:51820")
        );
        assert!(!signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::ChatRelay));
        assert!(!signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::OnionMiddle));
    }

    #[test]
    fn self_discovery_descriptor_requires_peer_api_listener_for_chat_relay_capability() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.memchain.chat_relay.enabled = true;

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let signed = server
            .build_self_discovery_descriptor(1_800_000_000)
            .unwrap();

        assert!(!signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::ChatRelay));
        assert!(!signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::OnionMiddle));
    }

    #[test]
    fn self_discovery_descriptor_requires_chat_relay_runtime_for_chat_relay_capability() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.memchain.chat_relay.enabled = true;

        let signed = Server::build_self_discovery_descriptor_for_runtime(
            &config,
            &IdentityKeyPair::generate(),
            1_800_000_000,
            false,
        )
        .unwrap();

        assert!(!signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::ChatRelay));
        assert!(signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::PrivacyRelay));
    }

    #[test]
    fn local_capability_status_is_ready_when_chat_relay_is_configured_and_advertised() {
        let mut config = ServerConfig::default();
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.memchain.chat_relay.enabled = true;

        let status = Server::discovery_local_capability_status_for(&config);

        assert_eq!(status.status, "ready");
        assert!(status.chat_relay_configured);
        assert!(status.blind_relay_endpoint_ready);
        assert!(status.chat_relay_runtime_ready);
        assert!(status.safe_to_advertise_chat_relay);
        assert!(status.advertised_chat_relay_capability);
        assert!(status.capability_config_consistent);
    }

    #[test]
    fn local_capability_status_reports_disabled_without_chat_relay_config() {
        let mut config = ServerConfig::default();
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());

        let status = Server::discovery_local_capability_status_for(&config);

        assert_eq!(status.status, "disabled");
        assert!(!status.chat_relay_configured);
        assert!(status.blind_relay_endpoint_ready);
        assert!(!status.chat_relay_runtime_ready);
        assert!(!status.safe_to_advertise_chat_relay);
        assert!(!status.advertised_chat_relay_capability);
        assert!(status.capability_config_consistent);
    }

    #[test]
    fn local_capability_status_reports_misconfigured_when_chat_relay_lacks_peer_api() {
        let mut config = ServerConfig::default();
        config.memchain.chat_relay.enabled = true;

        let status = Server::discovery_local_capability_status_for(&config);

        assert_eq!(status.status, "misconfigured");
        assert!(status.chat_relay_configured);
        assert!(!status.blind_relay_endpoint_ready);
        assert!(status.chat_relay_runtime_ready);
        assert!(!status.safe_to_advertise_chat_relay);
        assert!(!status.advertised_chat_relay_capability);
        assert!(status.capability_config_consistent);
    }

    #[test]
    fn local_capability_status_reports_misconfigured_when_chat_relay_runtime_is_missing() {
        let mut config = ServerConfig::default();
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.memchain.chat_relay.enabled = true;

        let status = Server::discovery_local_capability_status_for_runtime(&config, false);

        assert_eq!(status.status, "misconfigured");
        assert!(status.chat_relay_configured);
        assert!(status.blind_relay_endpoint_ready);
        assert!(!status.chat_relay_runtime_ready);
        assert!(!status.safe_to_advertise_chat_relay);
        assert!(!status.advertised_chat_relay_capability);
        assert!(status.capability_config_consistent);
        assert!(status
            .advertisement_blockers
            .contains(&"chat_relay_runtime_not_ready"));
    }

    #[test]
    fn discovery_readiness_includes_blind_relay_runtime_quality_without_private_metadata() {
        let mut config = ServerConfig::default();
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.memchain.chat_relay.enabled = true;

        let peer_store = PeerStore::new();
        peer_store.record_blind_relay_forwarded(1_700_000_010, 1);
        let status = peer_store.status(1_700_000_020);
        let local_capabilities = Server::discovery_local_capability_status_for(&config);
        let readiness =
            crate::api::discovery::discovery_readiness_status_value(&status, &local_capabilities);
        let blind_relay_runtime = readiness
            .get("blind_relay_runtime")
            .expect("blind relay runtime readiness object");
        let route_governance = readiness
            .get("route_governance")
            .expect("route governance readiness object");
        let protocol_foundation = readiness
            .get("protocol_foundation")
            .expect("protocol foundation readiness object");

        assert_eq!(protocol_foundation["status"], "forming");
        assert_eq!(protocol_foundation["stage"], "single_hop_relay_ready");
        assert_eq!(protocol_foundation["checks_total"], 4);
        assert_eq!(protocol_foundation["checks_passed"], 2);
        assert_eq!(protocol_foundation["local_relay_ready"], true);
        assert_eq!(protocol_foundation["blind_relay_ready"], true);
        assert_eq!(
            protocol_foundation["privacy_invariant"],
            "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status"
        );
        assert_eq!(blind_relay_runtime["status"], "ready");
        assert_eq!(blind_relay_runtime["runtime_ready"], true);
        assert_eq!(blind_relay_runtime["quality_ready"], true);
        assert_eq!(blind_relay_runtime["accepted_total"], 1);
        assert_eq!(blind_relay_runtime["forward_failed"], 0);
        assert_eq!(blind_relay_runtime["last_event_age_seconds"], 10);
        assert!(blind_relay_runtime["last_probe_age_seconds"].is_null());
        assert_eq!(route_governance["contract_version"], "route_governance.v1");
        assert_eq!(route_governance["status"], "forming");
        assert_eq!(route_governance["route_pool_ready"], false);
        assert_eq!(route_governance["quality_ready"], false);
        assert_eq!(route_governance["candidates_total"], 0);
        assert!(route_governance["average_score"].is_null());

        let serialized = serde_json::to_string(&readiness).unwrap();
        assert!(!serialized.contains("https://node.example.com"));
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("payload_b64"));
        assert!(!serialized.contains("client_ip"));
    }

    #[test]
    fn self_discovery_descriptor_can_advertise_onion_middle_when_explicitly_enabled() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("https://node.example.com".to_string());
        config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
        config.discovery.advertise_onion_middle = true;
        config.memchain.chat_relay.enabled = true;

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let signed = server
            .build_self_discovery_descriptor(1_800_000_000)
            .unwrap();

        assert!(signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::ChatRelay));
        assert!(signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::OnionMiddle));
        assert!(signed
            .descriptor
            .capabilities
            .contains(&NodeCapability::PrivacyRelay));
    }

    #[test]
    fn discovery_gossip_url_normalizes_endpoint_forms() {
        assert_eq!(
            Server::discovery_gossip_url("198.51.100.10:51820").as_deref(),
            Some("http://198.51.100.10:51820/api/discovery/gossip")
        );
        assert_eq!(
            Server::discovery_gossip_url("https://node.example.com").as_deref(),
            Some("https://node.example.com/api/discovery/gossip")
        );
        assert_eq!(Server::discovery_gossip_url("   "), None);
    }

    #[test]
    fn discovery_gossip_schedule_applies_jitter_and_backpressure() {
        let mut discovery = DiscoveryConfig {
            enabled: true,
            gossip_enabled: true,
            gossip_interval_secs: 60,
            gossip_jitter_percent: 10,
            gossip_backpressure_failure_threshold: 3,
            gossip_failure_backoff_max_secs: 300,
            ..DiscoveryConfig::default()
        };
        let node_id = [0x42; 32];

        let (_delay, backpressure_active, delay_secs, jitter_secs) =
            Server::discovery_gossip_schedule(&discovery, &node_id, 1_800_000_000, 0);
        assert!(!backpressure_active);
        assert!((54..=66).contains(&delay_secs));
        assert!((-6..=6).contains(&jitter_secs));

        let (_delay, backpressure_active, delay_secs, _jitter_secs) =
            Server::discovery_gossip_schedule(&discovery, &node_id, 1_800_000_060, 5);
        assert!(backpressure_active);
        assert!((216..=264).contains(&delay_secs));

        discovery.gossip_jitter_percent = 0;
        let (_delay, backpressure_active, delay_secs, jitter_secs) =
            Server::discovery_gossip_schedule(&discovery, &node_id, 1_800_000_120, 8);
        assert!(backpressure_active);
        assert_eq!(delay_secs, 300);
        assert_eq!(jitter_secs, 0);
    }

    #[test]
    fn blind_relay_probe_cooldown_keeps_synthetic_checks_low_frequency() {
        let mut discovery = DiscoveryConfig {
            gossip_interval_secs: 60,
            ..DiscoveryConfig::default()
        };

        assert_eq!(
            Server::blind_relay_probe_cooldown_secs(&discovery),
            BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS
        );

        discovery.gossip_interval_secs = 600;
        assert_eq!(Server::blind_relay_probe_cooldown_secs(&discovery), 1_800);
    }

    #[test]
    fn blind_relay_probe_priority_prefers_unproven_non_quarantined_peer() {
        let store = PeerStore::new();
        let now = 1_800_000_000;

        let proven = signed_probe_peer_descriptor(
            "https://proven.example".to_string(),
            1,
            now + 300,
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            [0x31; 32],
        );
        let unproven = signed_probe_peer_descriptor(
            "https://unproven.example".to_string(),
            2,
            now + 300,
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            [0x32; 32],
        );
        let quarantined = signed_probe_peer_descriptor(
            "https://quarantined.example".to_string(),
            3,
            now + 300,
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
            [0x33; 32],
        );

        let proven_id = proven.node_id();
        let unproven_id = unproven.node_id();
        let quarantined_id = quarantined.node_id();
        store.upsert_verified(proven.clone(), now).unwrap();
        store.upsert_verified(unproven.clone(), now).unwrap();
        store.upsert_verified(quarantined.clone(), now).unwrap();
        store.record_route_forward_success(&proven_id, now + 1);
        store.record_route_forward_failure(&quarantined_id, now + 1, "request_failed");
        store.record_route_forward_failure(&quarantined_id, now + 2, "request_failed");
        store.record_route_forward_failure(&quarantined_id, now + 3, "request_failed");

        let mut candidates = vec![proven, quarantined, unproven];
        Server::prioritize_probe_candidates(&store, now + 4, &mut candidates);

        assert_eq!(candidates[0].node_id(), unproven_id);
        assert_eq!(candidates[1].node_id(), proven_id);
        assert_eq!(candidates[2].node_id(), quarantined_id);
    }

    #[test]
    fn chat_peer_relay_url_normalizes_endpoint_forms() {
        assert_eq!(
            Server::chat_peer_relay_url("198.51.100.10:8421").as_deref(),
            Some("http://198.51.100.10:8421/api/chat/peer/relay")
        );
        assert_eq!(
            Server::chat_peer_relay_url("https://node.example.com/").as_deref(),
            Some("https://node.example.com/api/chat/peer/relay")
        );
        assert_eq!(Server::chat_peer_relay_url("   "), None);
    }

    #[tokio::test]
    async fn discovered_chat_relay_peer_receives_encrypted_envelope_fanout() {
        let received = Arc::new(AtomicUsize::new(0));
        let received_for_handler = Arc::clone(&received);
        let app = Router::new().route(
            "/api/chat/peer/relay",
            post(move |Json(request): Json<PeerChatRelayRequest>| {
                let received_for_handler = Arc::clone(&received_for_handler);
                async move {
                    assert_eq!(request.envelope.message_id, [0x55; 16]);
                    received_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerChatRelayResponse {
                        accepted: true,
                        duplicate: false,
                        delivered_online: 0,
                        stored_pending: true,
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let mock_peer = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let peer_store = PeerStore::new();
        let peer_descriptor = signed_chat_relay_peer_descriptor(endpoint, now, now + 300);
        let peer_node_id = peer_descriptor.node_id();
        let peer_prefix = hex::encode(&peer_node_id[..4]);
        peer_store
            .upsert_verified(peer_descriptor, now)
            .expect("mock peer descriptor should verify");

        let client = reqwest::Client::new();
        let accepted = Server::relay_chat_envelope_to_discovered_peers(
            Some(&client),
            None,
            &peer_store,
            &[0x99; 32],
            &signed_test_chat_envelope(now),
        )
        .await;

        assert_eq!(accepted, 1);
        assert_eq!(received.load(AtomicOrdering::SeqCst), 1);
        let route_status = peer_store.route_candidate_status(now + 1);
        let row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == peer_prefix)
            .expect("mock peer should remain in route candidate status");
        assert_eq!(row.route_health, "healthy");
        assert_eq!(row.route_consecutive_failures, 0);
        assert_eq!(row.last_route_success_at, Some(now));
        mock_peer.abort();
    }

    #[tokio::test]
    async fn discovered_chat_relay_peer_rejected_ack_marks_route_failure() {
        let received = Arc::new(AtomicUsize::new(0));
        let received_for_handler = Arc::clone(&received);
        let app = Router::new().route(
            "/api/chat/peer/relay",
            post(move |Json(request): Json<PeerChatRelayRequest>| {
                let received_for_handler = Arc::clone(&received_for_handler);
                async move {
                    assert_eq!(request.envelope.message_id, [0x55; 16]);
                    received_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerChatRelayResponse {
                        accepted: false,
                        duplicate: false,
                        delivered_online: 0,
                        stored_pending: false,
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let mock_peer = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let peer_store = PeerStore::new();
        let peer_descriptor = signed_chat_relay_peer_descriptor(endpoint, now, now + 300);
        let peer_node_id = peer_descriptor.node_id();
        let peer_prefix = hex::encode(&peer_node_id[..4]);
        peer_store
            .upsert_verified(peer_descriptor, now)
            .expect("mock peer descriptor should verify");

        let client = reqwest::Client::new();
        let accepted = Server::relay_chat_envelope_to_discovered_peers(
            Some(&client),
            None,
            &peer_store,
            &[0x99; 32],
            &signed_test_chat_envelope(now),
        )
        .await;

        assert_eq!(accepted, 0);
        assert_eq!(received.load(AtomicOrdering::SeqCst), 1);
        let route_status = peer_store.route_candidate_status(now + 1);
        let row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == peer_prefix)
            .expect("mock peer should remain in route candidate status");
        assert_eq!(row.route_health, "degraded");
        assert_eq!(row.route_consecutive_failures, 1);
        assert_eq!(
            row.last_route_failure_reason.as_deref(),
            Some("peer_relay_ack_rejected")
        );
        assert_eq!(row.last_route_success_at, None);
        mock_peer.abort();
    }

    #[tokio::test]
    async fn outbound_gossip_imports_snapshot_response_from_peer() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_handler = Arc::clone(&calls);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let remote_descriptor =
            signed_chat_relay_peer_descriptor("http://127.0.0.1:9".to_string(), now, now + 300);
        let remote_node_id = remote_descriptor.node_id();
        let snapshot_response = NodeDiscoveryMessage::SnapshotResponse {
            snapshot: NodeBootstrapSnapshot::new(now, vec![remote_descriptor.clone()]),
        };
        let app = Router::new().route(
            "/api/discovery/gossip",
            post(move |Json(message): Json<NodeDiscoveryMessage>| {
                let calls_for_handler = Arc::clone(&calls_for_handler);
                let snapshot_response = snapshot_response.clone();
                async move {
                    calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    let response = match message {
                        NodeDiscoveryMessage::DescriptorAnnounce { .. } => GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        },
                        NodeDiscoveryMessage::SnapshotRequest { .. } => GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: Some(snapshot_response),
                        },
                        NodeDiscoveryMessage::SnapshotResponse { .. } => GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        },
                    };
                    Json(response)
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/api/discovery/gossip",
            listener.local_addr().unwrap()
        );
        let mock_peer = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let peer_store = PeerStore::new();
        let self_descriptor =
            signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
        let client = reqwest::Client::new();

        Server::gossip_with_peer(&client, &peer_store, &url, self_descriptor, now, 8)
            .await
            .expect("mock discovery peer should accept gossip exchange");

        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
        assert!(peer_store.get_valid(&remote_node_id, now + 1).is_some());
        assert_eq!(peer_store.status(now + 1).runtime.last_gossip_at, Some(now));
        mock_peer.abort();
    }

    #[tokio::test]
    async fn peer_store_cache_persists_verified_snapshot_json() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let now = unix_now_secs();
        let signed = server.build_self_discovery_descriptor(now).unwrap();
        let peer_store = Arc::new(PeerStore::new());
        assert!(peer_store.upsert_verified(signed, now).unwrap());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-{unique}.json"));
        let path_str = path.to_string_lossy().to_string();

        Server::save_peer_store_cache_snapshot(&peer_store, &path_str, now)
            .await
            .unwrap();

        let bytes = tokio::fs::read(&path).await.unwrap();
        let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();

        assert_eq!(snapshot.peers.len(), 1);
        assert_eq!(snapshot.verified_count_at(now + 1), 1);

        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn peer_store_cache_can_restore_verified_peers_after_restart() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let now: u64 = 1_800_000_000;
        let signed = server.build_self_discovery_descriptor(now).unwrap();
        let original_store = Arc::new(PeerStore::new());
        assert!(original_store.upsert_verified(signed.clone(), now).unwrap());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-restore-{unique}.json"));
        let path_str = path.to_string_lossy().to_string();

        Server::save_peer_store_cache_snapshot(&original_store, &path_str, now)
            .await
            .unwrap();

        let restored_store = PeerStore::new();
        let bytes = tokio::fs::read(&path).await.unwrap();
        Server::import_bootstrap_snapshot_bytes(&restored_store, "cache", &path_str, &bytes, now);

        assert_eq!(restored_store.len(), 1);
        assert!(restored_store
            .get_valid(&signed.node_id(), now + 1)
            .is_some());

        let status = restored_store.status(now + 1);
        assert_eq!(status.snapshot.valid_peers, 1);
        assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.last_cache_load_source.as_deref(),
            Some("cache")
        );
        assert_eq!(
            status.bootstrap.last_cache_load_status.as_deref(),
            Some("success")
        );
        assert_eq!(status.bootstrap.last_cache_load_at, Some(now));
        assert!(status
            .recent_audit_events
            .iter()
            .any(|event| event.action == "bootstrap_source"));

        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn peer_store_cache_retains_expired_signed_peers_after_restart() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let now: u64 = 1_800_000_000;
        let peer_key = IdentityKeyPair::generate();
        let expired_descriptor = NodeDescriptor::new(
            peer_key.public_key_bytes(),
            7,
            now.saturating_sub(2_000),
            now.saturating_sub(1_000),
            "aeronyx-test-expired",
        );
        let expired = SignedNodeDescriptor::sign(expired_descriptor, &peer_key).unwrap();
        let node_id = expired.node_id();
        let original_store = Arc::new(PeerStore::new());
        let cache_snapshot = NodeBootstrapSnapshot::new(now, vec![expired]);
        let report =
            original_store.load_peer_cache_snapshot_from_source(&cache_snapshot, now, "cache");
        assert_eq!(report.inserted, 1);
        assert_eq!(original_store.status(now).peer_summary.expired_peers, 1);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("aeronyx-peer-cache-expired-restore-{unique}.json"));
        let path_str = path.to_string_lossy().to_string();

        Server::save_peer_store_cache_snapshot(&original_store, &path_str, now)
            .await
            .unwrap();

        let restored_store = PeerStore::new();
        let bytes = tokio::fs::read(&path).await.unwrap();
        Server::import_bootstrap_snapshot_bytes(&restored_store, "cache", &path_str, &bytes, now);

        assert_eq!(restored_store.len(), 1);
        assert!(restored_store.get_valid(&node_id, now).is_none());
        let status = restored_store.status(now);
        assert_eq!(status.snapshot.valid_peers, 0);
        assert_eq!(status.peer_summary.expired_peers, 1);
        assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("success")
        );

        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn peer_store_cache_falls_back_to_backup_when_primary_is_corrupt() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let now = unix_now_secs();
        let signed = server.build_self_discovery_descriptor(now).unwrap();
        let original_store = Arc::new(PeerStore::new());
        assert!(original_store.upsert_verified(signed.clone(), now).unwrap());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("aeronyx-peer-cache-backup-restore-{unique}.json"));
        let path_str = path.to_string_lossy().to_string();
        let backup_path = Server::peer_cache_backup_path(&path_str);

        Server::save_peer_store_cache_snapshot(&original_store, &path_str, now)
            .await
            .unwrap();
        tokio::fs::copy(&path, &backup_path).await.unwrap();
        tokio::fs::write(&path, b"{not-json").await.unwrap();

        let restored_store = PeerStore::new();
        server
            .load_peer_cache(&restored_store, &path_str, now + 1)
            .await;

        assert!(restored_store
            .get_valid(&signed.node_id(), now + 1)
            .is_some());

        let status = restored_store.status(now + 1);
        assert_eq!(status.snapshot.valid_peers, 1);
        assert_eq!(
            status.bootstrap.last_source_kind.as_deref(),
            Some("cache_backup")
        );
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.last_cache_load_source.as_deref(),
            Some("cache_backup")
        );
        assert_eq!(
            status.bootstrap.last_cache_load_status.as_deref(),
            Some("success")
        );
        assert_eq!(status.bootstrap.last_cache_load_at, Some(now + 1));

        let _ = tokio::fs::remove_file(path).await;
        let _ = tokio::fs::remove_file(backup_path).await;
    }

    #[tokio::test]
    async fn peer_store_cache_falls_back_to_backup_when_primary_has_no_usable_peers() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let now = unix_now_secs();
        let signed = server.build_self_discovery_descriptor(now).unwrap();
        let original_store = Arc::new(PeerStore::new());
        assert!(original_store.upsert_verified(signed.clone(), now).unwrap());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("aeronyx-peer-cache-empty-restore-{unique}.json"));
        let path_str = path.to_string_lossy().to_string();
        let backup_path = Server::peer_cache_backup_path(&path_str);

        Server::save_peer_store_cache_snapshot(&original_store, &path_str, now)
            .await
            .unwrap();
        tokio::fs::copy(&path, &backup_path).await.unwrap();

        let empty_snapshot = NodeBootstrapSnapshot::new(now, Vec::new());
        tokio::fs::write(&path, empty_snapshot.to_json_pretty().unwrap())
            .await
            .unwrap();

        let restored_store = PeerStore::new();
        server
            .load_peer_cache(&restored_store, &path_str, now + 1)
            .await;

        assert!(restored_store
            .get_valid(&signed.node_id(), now + 1)
            .is_some());

        let status = restored_store.status(now + 1);
        assert_eq!(status.snapshot.valid_peers, 1);
        assert_eq!(
            status.bootstrap.last_source_kind.as_deref(),
            Some("cache_backup")
        );
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("success")
        );

        let _ = tokio::fs::remove_file(path).await;
        let _ = tokio::fs::remove_file(backup_path).await;
    }

    #[tokio::test]
    async fn peer_store_immediate_cache_save_records_restart_recovery_evidence() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.peer_cache_path = Some(
            std::env::temp_dir()
                .join(format!(
                    "aeronyx-peer-cache-immediate-{}.json",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ))
                .to_string_lossy()
                .to_string(),
        );
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config.clone(), IdentityKeyPair::generate(), None);
        let peer_store = server.init_peer_store(false).await;
        let path = config.discovery.peer_cache_path.unwrap();

        let bytes = tokio::fs::read(&path)
            .await
            .expect("initial PeerStore cache should be saved during bootstrap");
        let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
        let now = unix_now_secs();
        assert!(snapshot.verified_count_at(now) >= 1);

        let status = peer_store.status(now);
        assert_eq!(
            status.bootstrap.last_cache_save_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.last_cache_save_detail.as_deref(),
            Some("snapshot_persisted")
        );
        assert!(status.stability.restart_recovery_configured);

        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn peer_store_cache_save_preserves_existing_recovery_snapshot_when_current_view_empty() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let now = 1_800_000_000;
        let signed = server.build_self_discovery_descriptor(now).unwrap();
        let recovery_store = Arc::new(PeerStore::new());
        assert!(recovery_store.upsert_verified(signed.clone(), now).unwrap());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-preserve-{unique}.json"));
        let path_str = path.to_string_lossy().to_string();

        Server::save_peer_store_cache_snapshot(&recovery_store, &path_str, now)
            .await
            .unwrap();

        let empty_store = Arc::new(PeerStore::new());
        Server::persist_peer_store_cache_once(&empty_store, &path_str, now + 1)
            .await
            .unwrap();

        let bytes = tokio::fs::read(&path).await.unwrap();
        let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
        assert_eq!(snapshot.verified_count_at(now + 2), 1);
        assert_eq!(snapshot.peers[0].node_id(), signed.node_id());

        let status = empty_store.status(now + 2);
        assert_eq!(
            status.bootstrap.last_cache_save_status.as_deref(),
            Some("skipped")
        );
        assert_eq!(
            status.bootstrap.last_cache_save_detail.as_deref(),
            Some("preserved_existing_snapshot")
        );

        let _ = tokio::fs::remove_file(path).await;
        let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    }

    #[tokio::test]
    async fn peer_store_cache_source_supersedes_expired_static_bootstrap_warning() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let bootstrap_path =
            std::env::temp_dir().join(format!("aeronyx-expired-bootstrap-{unique}.json"));
        let cache_path = std::env::temp_dir().join(format!("aeronyx-fresh-cache-{unique}.json"));
        let bootstrap_path_str = bootstrap_path.to_string_lossy().to_string();
        let cache_path_str = cache_path.to_string_lossy().to_string();

        let now = unix_now_secs();
        let expired =
            signed_chat_relay_peer_descriptor("http://127.0.0.1:9".to_string(), now - 600, now - 1);
        let expired_snapshot = NodeBootstrapSnapshot::new(now - 600, vec![expired]);
        tokio::fs::write(&bootstrap_path, expired_snapshot.to_json_pretty().unwrap())
            .await
            .unwrap();

        let fresh =
            signed_chat_relay_peer_descriptor("http://127.0.0.1:10".to_string(), now, now + 300);
        let cache_store = Arc::new(PeerStore::new());
        assert!(cache_store.upsert_verified(fresh.clone(), now).unwrap());
        Server::save_peer_store_cache_snapshot(&cache_store, &cache_path_str, now)
            .await
            .unwrap();

        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.advertise_self = false;
        config.discovery.bootstrap_snapshot_path = Some(bootstrap_path_str.clone());
        config.discovery.peer_cache_path = Some(cache_path_str.clone());
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config, IdentityKeyPair::generate(), None);
        let restored_store = server.init_peer_store(false).await;
        let status = restored_store.status(now + 1);

        assert!(restored_store
            .get_valid(&fresh.node_id(), now + 1)
            .is_some());
        assert_eq!(status.snapshot.valid_peers, 1);
        assert_eq!(status.runtime.rejected, 1);
        assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("success")
        );
        assert_eq!(status.bootstrap.recovery_status.as_deref(), Some("success"));

        let _ = tokio::fs::remove_file(bootstrap_path).await;
        let _ = tokio::fs::remove_file(cache_path).await;
        let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&cache_path_str)).await;
    }

    #[tokio::test]
    async fn peer_store_persistence_task_flushes_cache_on_shutdown() {
        let mut config = ServerConfig::default();
        config.discovery.enabled = true;
        config.discovery.peer_cache_write_interval_secs = 3600;
        config.discovery.peer_cache_path = Some(
            std::env::temp_dir()
                .join(format!(
                    "aeronyx-peer-cache-shutdown-{}.json",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ))
                .to_string_lossy()
                .to_string(),
        );
        config.discovery.public_endpoint = Some("node.example.com:443".to_string());

        let server = Server::new(config.clone(), IdentityKeyPair::generate(), None);
        let now = unix_now_secs();
        let signed = server.build_self_discovery_descriptor(now).unwrap();
        let peer_store = Arc::new(PeerStore::new());
        assert!(peer_store.upsert_verified(signed, now).unwrap());

        let path = config.discovery.peer_cache_path.unwrap();
        let handle = server
            .spawn_peer_store_persistence_task(Arc::clone(&peer_store))
            .expect("peer cache persistence should be enabled");
        server
            .shutdown_tx
            .send(())
            .expect("shutdown receiver should be subscribed");
        handle.await.expect("peer cache task should stop cleanly");

        let bytes = tokio::fs::read(&path)
            .await
            .expect("shutdown should flush PeerStore cache");
        let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
        assert_eq!(snapshot.verified_count_at(now + 1), 1);

        let status = peer_store.status(now + 1);
        assert_eq!(
            status.bootstrap.last_cache_save_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.last_cache_save_detail.as_deref(),
            Some("snapshot_persisted")
        );

        let _ = tokio::fs::remove_file(path).await;
    }
}
