// ============================================
// File: crates/aeronyx-server/src/services/mod.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Registered `deny_list` submodule and re-exported DenyList + DenyReason.
//   Registered `dns_proxy` so AeroNyx clients can resolve DNS through the gateway.
//   Registered `peer_store` for Phase 1 decentralized node discovery.
//   Re-exported `PeerStoreImportReport` for Phase 2 bootstrap snapshot loading.
//   Re-exported PeerStoreStatus for nodeboard discovery status.
//   [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Re-exported the aggregate,
//   path-free relay custody maintenance audit verification receipt.
//
// Last Modified:
//   v0.18.0-CustodyAuditVerify - Re-exported host-local audit verification
//   v0.17.0-BlindVaultPutFailureClass - Re-exported the privacy-safe retry
//     classification used by multi-hop anonymous ciphertext delivery
//   v0.16.0-RouteDomainCertificateRecovery - [ROUTE-DOMAIN-CERTIFICATE-RECOVERY
//     2026-08-03 by Codex] Re-exported the bounded, identity-blind local cache
//     recovery report and schema contract
//   v0.15.0-RouteDomainCertificateIngress - [ROUTE-DOMAIN-CERTIFICATE-INGRESS
//     2026-08-03 by Codex] Re-exported focused attestor-policy and certificate-
//     import errors without widening the signed-descriptor error contract
//   v0.14.0-DNSTransactionalStartup - [DNS-STARTUP-READINESS 2026-07-30 by
//     Codex] Re-exported the pre-bound production DNS startup path while
//     retaining the legacy spawn API
//   v0.13.0-BlindVaultIssuerRuntime - Re-exported the monotonic issuer
//     installation outcome and aggregate runtime status
//   v0.12.0-DirectoryWitnessThreshold - Re-exported the audited checkpoint
//     target used to retry only current pinned witnesses below threshold
//   v0.11.0-DirectoryWitnessOutcomeTelemetry - Re-exported bounded durable and
//     process-lifetime witness outcome telemetry types
//   v0.10.0-DirectoryObservationCheckpoint - Re-exported the bounded append
//     report used by the synchronization coordinator
//   v0.9.0-DirectoryReplicaIncidentEvidence - Re-exported bounded incident
//     summary, page, and independently verified evidence types
//   v0.8.0-DirectoryReplicaConvergence - Re-exported bounded multi-source
//     observation evidence snapshots
//   v0.7.0-DirectoryReplicaStatus - Re-exported replica status/runtime types
//   v0.6.0-DirectoryReplicaStore - Registered producer-isolated remote replicas
//   v0.5.0-DirectorySyncReads - Re-exported audit-gated Directory Chain pages
//   v0.4.0-DirectoryChainStore - Registered transactional local directory ledger
//   v0.3.0-DiscoveryStatus - Re-exported PeerStoreStatus
//   v0.2.0-DiscoveryPhase2 - Re-exported PeerStoreImportReport
//   v0.1.0-DiscoveryPhase1 - Added peer_store submodule
//   v1.2.0-DNSProxy - Added VPN gateway DNS proxy
//   v1.1.0-ChatRelay - Added chat_relay submodule
//   v1.0.0-Membership - Added deny_list submodule + traffic_tracker
//   v1.0.0-BlindVaultService - Added anonymous encrypted-object storage

pub mod blind_vault;
pub mod chat_relay;
pub mod deny_list;
pub mod directory_chain;
pub mod directory_replica;
pub mod dns_proxy;
pub mod handshake;
pub mod ip_pool;
pub mod memchain;
pub mod node_policy;
pub mod onion_keys;
pub mod peer_store;
pub mod routing;
pub mod session;
pub mod traffic_tracker;
pub mod wallet_routes;

// Re-export primary types
// [BLIND-VAULT-ISSUER-RUNTIME 2026-07-23 by Codex] Keep the authenticated
// control-plane contract on the public services boundary.
// [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] Relay/API boundaries consume
// the service-owned coarse class instead of matching storage internals twice.
pub use blind_vault::{
    BlindVaultCleanupReport, BlindVaultIssuerInstallOutcome, BlindVaultIssuerRuntimeStatus,
    BlindVaultLeaseProvisionOutcome, BlindVaultPullPage, BlindVaultPutFailureClass,
    BlindVaultService, BlindVaultServiceError, BlindVaultStatus, BlindVaultStoredObject,
    SharedBlindVaultService,
};
// [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Expose only aggregate,
// host-local maintenance contracts and authenticated path-free plans; artifact
// paths and private identity metadata remain confined to the relay service.
pub use chat_relay::{
    derive_node_secret, ChatRelayBackupAuditVerificationReceipt, ChatRelayBackupPruneReceipt,
    ChatRelayBackupPruneRequest, ChatRelayBackupRetentionReceipt, ChatRelayRestorePlanReceipt,
    ChatRelayRestoreReadinessReceipt, ChatRelayService, CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
    CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS,
};
pub use deny_list::{DenyList, DenyReason};
pub use directory_chain::{
    DirectoryChainAppendReport, DirectoryChainAudit, DirectoryChainPage, DirectoryChainStore,
    DirectoryChainStoreError,
};
pub(crate) use directory_replica::DirectoryReplicaGossipAnnouncement;
pub(crate) use directory_replica::MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE;
pub use directory_replica::{
    DirectoryObservationCheckpointAppendReport, DirectoryObservationWitnessDecision,
    DirectoryObservationWitnessOutcome, DirectoryObservationWitnessOutcomeCounters,
    DirectoryObservationWitnessOutcomeSnapshot, DirectoryObservationWitnessTarget,
    DirectoryReplicaAudit, DirectoryReplicaImportReport, DirectoryReplicaIncidentEvidence,
    DirectoryReplicaIncidentPage, DirectoryReplicaIncidentSummary,
    DirectoryReplicaObservationConvergenceSnapshot, DirectoryReplicaProducerSnapshot,
    DirectoryReplicaResolutionCommand, DirectoryReplicaResolutionReport, DirectoryReplicaStore,
    DirectoryReplicaStoreError, DirectoryReplicaStoreSnapshot, DirectoryReplicaSyncObservation,
    DirectoryReplicaSyncRuntime, DirectoryReplicaTip,
};
pub use dns_proxy::{spawn_dns_proxy, start_dns_proxy};
pub use handshake::HandshakeService;
pub use ip_pool::IpPoolService;
pub use memchain::{AofWriter, MemPool};
pub use node_policy::{
    NodePolicyEnforcementSnapshot, NodePolicyPlacementSnapshot, NodePolicyRuntime,
    NodePolicySnapshot,
};
pub use peer_store::{
    PeerStore, PeerStoreError, PeerStoreImportReport, PeerStoreRouteDomainCertificateCacheReport,
    PeerStoreSnapshot, PeerStoreStatus, RouteDomainAttestorPolicyError,
    RouteDomainCertificateImportError,
};
pub use routing::RoutingService;
// [SESSION-TERMINATION 2026-08-15 by Codex] Export the owned lifecycle result
// so UDP and timeout termination share one external-resource finalizer.
pub use session::{Session, SessionManager, SessionState, SessionTermination};
pub use wallet_routes::WalletRouteCache;
