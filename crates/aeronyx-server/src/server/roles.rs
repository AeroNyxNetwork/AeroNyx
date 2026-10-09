// ============================================
// File: crates/aeronyx-server/src/server/roles.rs
// ============================================
// Creation Reason:
//   [NODE-ROLES 2026-10-09 by Claude] `Server::run` threaded two dozen loose
//   `Arc`s and `Option`s between startup steps, and `start_combined_api` took
//   22 positional arguments. Nothing in the types said which values belong
//   together, which role owns them, or what a role needs from another one.
//
// Main Functionality:
//   One struct per role, holding exactly what that role brought up:
//   - `MemoryStores`: MemChain storage (absent when `memchain.mode = off`).
//   - `DataPlane`: the UDP transport and the session state the API reads.
//   - `Messaging`: chat relay, anonymous mailbox and its VPN-side source.
//   - `Directory`: directory chain/replica stores and replica sync runtime.
//
// Important Note for Next Developer:
//   - These are handles, not services: construction, ordering and failure
//     policy stay in `Server::run`. A role reads another role only through a
//     field listed here, so a new cross-role dependency shows up in review.
//   - This is step 1 of the role split (see the node architecture plan of
//     2026-10-09). Behaviour is unchanged.
// ============================================

use super::*;

/// MemChain storage. Present only when `memchain.mode != off`.
pub(super) struct MemoryStores {
    pub(super) storage: Arc<MemoryStorage>,
    pub(super) vector_index: Arc<VectorIndex>,
    pub(super) mempool: Arc<MemPool>,
    pub(super) aof_writer: Arc<TokioMutex<AofWriter>>,
}

/// The VPN data plane: transport, sessions and the packet/handshake services.
///
/// UDP is bound even with `vpn.enabled = false`: management heartbeats and
/// the miner send through it. TUN exists only on Linux with VPN enabled.
pub(super) struct DataPlane {
    pub(super) udp: Arc<UdpTransport>,
    #[cfg(target_os = "linux")]
    pub(super) tun: Option<Arc<LinuxTun>>,
    pub(super) ip_pool: Arc<IpPoolService>,
    pub(super) sessions: Arc<SessionManager>,
    pub(super) routing: Arc<RoutingService>,
    pub(super) traffic_tracker: Arc<TrafficTracker>,
    pub(super) encrypted_message_counter: Arc<AtomicU64>,
    /// Written by the management heartbeat, read by the handshake service.
    pub(super) deny_list: Arc<DenyList>,
    pub(super) node_policy: Arc<NodePolicyRuntime>,
    pub(super) packet_handler: Arc<PacketHandler>,
    pub(super) handshake_service: Arc<HandshakeService>,
    pub(super) voucher_verifier: Arc<VoucherVerifier>,
}

/// Chat relay, the anonymous mailbox terminal and its VPN-side source.
pub(super) struct Messaging {
    pub(super) chat_relay: Option<Arc<ChatRelayService>>,
    pub(super) anonymous_mailbox: Option<Arc<SqliteAnonymousMailboxStore>>,
    pub(super) anonymous_mailbox_source: Option<Arc<AnonymousMailboxSourceCoordinator>>,
    /// Whether the mailbox cleanup loop is registered as a required task;
    /// readiness is reported only when it is.
    pub(super) anonymous_mailbox_cleanup_supervised: bool,
    pub(super) anonymous_mailbox_readiness: AnonymousMailboxReadinessProjection,
}

/// Directory chain and replica state.
pub(super) struct Directory {
    pub(super) chain_store: Option<Arc<DirectoryChainStore>>,
    pub(super) replica_store: Option<Arc<DirectoryReplicaStore>>,
    pub(super) replica_sync_runtime: Arc<DirectoryReplicaSyncRuntime>,
}

/// What the memory role prepared before the node API starts, handed on to
/// the tasks it spawns afterwards.
pub(super) struct MemoryRuntime {
    pub(super) is_saas: bool,
    pub(super) mpi_state: Arc<MpiState>,
    pub(super) user_weights:
        Arc<parking_lot::RwLock<HashMap<String, crate::services::memchain::mvf::WeightVector>>>,
    pub(super) embed_engine: Option<Arc<EmbedEngine>>,
    pub(super) ner_engine: Option<Arc<NerEngine>>,
    /// Handed to the node API, which announces new tips to followers.
    pub(super) commitment_sync_tip_notifier: Option<mpsc::Sender<u64>>,
    pub(super) commitment_sync_tip_rx: mpsc::Receiver<u64>,
    pub(super) commitment_tip_tx: mpsc::Sender<u64>,
    pub(super) commitment_tip_rx: mpsc::Receiver<u64>,
}
