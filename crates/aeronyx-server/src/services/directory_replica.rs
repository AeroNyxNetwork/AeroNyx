// ============================================================================
// File: crates/aeronyx-server/src/services/directory_replica.rs
// ============================================================================
//! # Directory Chain Replica Store
//!
//! ## Creation Reason
//! Directory Sync responses are producer attestations, not global consensus.
//! A node therefore needs a durable producer-scoped replica namespace instead
//! of inserting remote blocks into its own locally produced Directory Chain.
//!
//! ## Main Functionality
//! - Stores each remote producer chain under an independent `SQLite` namespace.
//! - Re-verifies signed response evidence, blocks, commitments, and descriptor
//!   objects before one atomic import transaction.
//! - Makes exact repeated pages idempotent.
//! - Persists signed fork/rollback evidence and permanently quarantines only
//!   the producer that authored conflicting chain claims.
//! - Records authenticated descriptor equivocation without blaming an honest
//!   chain producer that merely observed the conflicting public descriptors.
//! - Audits every replica chain and index before the node may synchronize.
//! - Persists bounded producer retry state across process restarts and clears
//!   it atomically with the next authenticated successful page.
//! - Computes a bounded recent-window intersection of exact descriptor
//!   commitments across non-quarantined configured producer replicas.
//! - Exposes low-cost aggregate snapshots and privacy-safe synchronization
//!   observations, including bounded retry state, without re-running a full
//!   cryptographic audit per API read.
//! - Exports bounded incident summaries and re-verified signed evidence for
//!   authenticated operator review without adding an automatic recovery path.
//! - [DIRECTORY-STREAMING-AUDIT 2026-08-31 by Codex] Audits each producer
//!   block-by-block so restart and evidence export do not materialize the full
//!   retained block, commitment, and descriptor history in memory at once.
//! - [DIRECTORY-AUDIT-SNAPSHOT 2026-08-31 by Codex] Holds every full-history
//!   audit inside one deferred SQLite read transaction and rejects oversized
//!   persisted block/descriptor blobs before materializing them into Rust.
//! - Resolves quarantine only through a node-identity-signed, host-local,
//!   compare-and-swap command while retaining the accepted prefix and every
//!   incident and resolution as an append-only audit trail.
//! - Persists local-identity-signed, hash-linked observation checkpoints only
//!   after a complete configured producer set yields a recomputable overlap.
//! - Independently recomputes checkpoints received from pinned observers and
//!   persists only canonical, accepted, externally signed witness receipts.
//! - Persists privacy-safe aggregate witness outcomes separately from signed
//!   receipts so operators can distinguish unavailable evidence from faults.
//! - Persists every local witness pin/threshold change as a node-identity-
//!   signed, hash-linked policy epoch with a metadata-anchored durable head.
//! - [ROUTE-DOMAIN-POLICY-HISTORY 2026-08-03 by Codex] Persists canonical
//!   route-domain pin changes as a separate node-signed, hash-linked history;
//!   the opaque grouping remains local policy, not an identity or AS proof.
//! - [ROUTE-DOMAIN-ATTESTOR-HISTORY 2026-08-03 by Codex] Persists canonical
//!   attestor pins, threshold, and strict-mode changes as an independent signed
//!   history without exposing trust-root identities in aggregate status.
//! - Retains opaque policy heads observed by independent nodes and accepted
//!   signed external anchor receipts without exposing policy member identities.
//! - Selects only forward-moving, mature, unwitnessed checkpoints for external
//!   recomputation so asymmetric sync schedules cannot chase the moving tip.
//! - Keeps recurring witness selection history-bounded by verifying only the
//!   candidate, its predecessor, the latest receipt set, and durable outcome;
//!   startup and explicit audits still verify the complete retained history.
//! - Exports bounded producer blocks and descriptors only after a complete
//!   producer-scoped audit inside the same `SQLite` read transaction for
//!   signed carrier use, without rescanning unrelated producer histories.
//! - Retains a bounded, durable registry for permissionless full-node mirrors
//!   without granting those producers checkpoint, witness, or policy authority.
//! - Gates public carrier recovery reads on that durable mirror registry while
//!   preserving exact producer signatures, object hashes, and quarantine state.
//! - [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] Distinguishes a
//!   producer-signed tip from a carrier-reported prefix, so untrusted carrier
//!   metadata cannot create durable producer incidents while signed block
//!   conflicts remain fail-closed.
//! - [DIRECTORY-CONFLICT-VERIFICATION 2026-09-01 by Codex] Requires every
//!   conflicting block to pass producer signature and retained-predecessor
//!   verification before it may create durable fork evidence or quarantine.
//! - [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] Rebuilds compact descriptor
//!   inclusion proofs from one fully audited producer namespace and one exact
//!   selected producer block without granting the carrier authority.
//! - [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] Selects one bounded,
//!   live public descriptor from retained replica evidence, then rebuilds its
//!   exact audited inclusion proof for mixed-version outbound gossip.
//! - Distinguishes a lagging carrier's unavailable range from a malformed
//!   request so recovery can continue without weakening contract validation.
//! - Records only aggregate routeability and signed-region-hint diversity for
//!   the latest bounded carrier selection, never peer identities or endpoints.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Retains process-only aggregate
//!   witness-carrier selection/outcome telemetry without any peer, route,
//!   endpoint, request, checkpoint, frame, or signature metadata.
//! - [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Separates target cooldown
//!   and local overload outcomes without storing identity-bearing dimensions.
//! - [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] Retains mutually
//!   exclusive process-lifetime Directory synchronization transport outcomes
//!   without accepting peer, endpoint, request, status-code, or frame data.
//! - [DIRECTORY-TRANSPORT-WINDOW 2026-07-28 by Codex] Retains a fixed-size
//!   outcome-class window so recent instability cannot be hidden by one final
//!   success, without adding any identity-bearing metric dimension.
//! - [DIRECTORY-TRANSPORT-LIFECYCLE 2026-07-29 by Codex] Owns transport health
//!   policy and degradation/recovery transitions inside the runtime, while
//!   keeping all wall-clock diagnostic timestamps monotonic.
//! - [WITNESS-TERMINAL-STATE 2026-07-29 by Codex] Retains the exact latest
//!   aggregate terminal outcome for witness recovery and carrier service so
//!   same-second completions cannot be misordered by status presentation.
//! - Re-verifies one carrier-returned retained mirror anchor without importing
//!   it, enabling a production smoke test with no authority or storage change.
//! - Builds operator-scoped portable observation certificates only from the
//!   latest fully re-verified checkpoint receipt set and current witness pins.
//!
//! ## Calling Relationships
//! - `server.rs` opens this store beside `DirectoryChainStore` at startup.
//! - `api/directory_replica_sync.rs` verifies and downloads bounded peer pages,
//!   then calls `import_verified_page` from a blocking worker.
//! - `api/directory_replica_status.rs` reads low-cost audited snapshots and
//!   requests one bounded full-audit portable certificate on operator demand.
//! - `api/directory_chain_peer.rs` serves local history plus audited, registry-
//!   gated replica recovery evidence; witness and policy routes remain pinned.
//!
//! ## Main Logical Flow
//! 1. Open the existing Directory Chain `SQLite` file and initialize only the
//!    `directory_replica_*` tables.
//! 2. Pin schema, chain id, and the local node identity in replica metadata.
//! 3. Audit every accepted producer prefix and all durable incident and
//!    operator-resolution evidence.
//! 4. Re-verify the signed range-response frame and exact descriptor objects.
//! 5. Atomically append a contiguous producer prefix, clear its retry state,
//!    or persist quarantine without mutating another producer namespace.
//! 6. Derive recent multi-source observation evidence without choosing a fork,
//!    producer, quorum, or globally finalized height.
//! 7. Sign and append a checkpoint only when every configured producer has an
//!    eligible prefix; re-derive every historical root during startup audit.
//! 8. Select only mature forward-moving checkpoints for external witnessing,
//!    recompute them from locally retained exact producer prefixes, and audit
//!    every accepted receipt again on restart.
//! 9. Audit bounded aggregate witness outcome counters without retaining peer
//!    identity, endpoint, request id, signature, or checkpoint hash metadata.
//! 10. Keep periodic selection cost independent of retained history while
//!     retaining complete fail-closed audits at startup and operator request.
//! 11. Canonicalize the configured witness set and append a policy epoch only
//!     when pins or threshold change; verify the complete policy chain before
//!     synchronization or any listener starts.
//! 12. Exchange only opaque epoch/digest policy heads with pinned witnesses;
//!     reject rollback, same-epoch conflict, and non-contiguous progression.
//! 13. Admit dynamically discovered mirrors into a capacity-bounded registry;
//!     promote them out of mirror status atomically if later operator-pinned.
//! 14. Permit public recovery reads only for producers still present in that
//!     registry, while preserving the general reader for pinned authority use;
//!     re-audit the complete requested producer namespace transactionally.
//! 15. For operator carrier smoke, audit one retained local anchor, verify the
//!     carrier frame and all producer/object evidence, then discard the page.
//! 16. For operator certificate export, re-verify the latest receipt set,
//!     exact checkpoint, current pins, threshold, and every signature without
//!     mutating retained evidence.
//! 17. Record only aggregate witness availability-recovery counters in memory;
//!     durable witness truth remains the independently signed receipt store.
//! 18. For replica descriptor proofs, audit one producer namespace, load the
//!     exact block and descriptor in the same read transaction, rebuild and
//!     independently re-verify the original producer-signed proof.
//! 19. For outbound proof gossip, scan only a bounded recent candidate window,
//!     retain only currently live authenticated descriptors, rotate selection,
//!     and run the complete producer audit before returning one announcement.
//! 20. Reduce each completed coordinator-owned HTTP exchange to one aggregate
//!     transport outcome without retaining any peer- or request-level dimension.
//! 21. Maintain a fixed 32-outcome health window beside lifetime totals so
//!     status reflects recent behavior rather than only the final completion.
//! 22. Classify health and record aggregate degradation/recovery transitions
//!     in the runtime so API presentation cannot drift from service policy.
//! 23. Preserve witness recovery and carrier terminal-event order independently
//!     from second-granularity timestamps and verify mutually exclusive totals.
//! 24. Canonicalize route-domain attestor pins and append a signed policy epoch
//!     only when pins, threshold, or strict mode change; audit the complete
//!     local trust-root history before route listeners start.
//!
//! ## Privacy Invariant
//! Replica tables contain only public signed node descriptors, public
//! descriptor commitments, signed Directory Chain blocks, and signed incident
//! evidence. They must never contain client identities, IPs, routes, selected
//! hops, message ids, payloads, ciphertext, Memory Chain records, DNS contents,
//! destinations, private keys, or wallet traffic.
//!
//! ## Important Note for Next Developer
//! - Never merge remote blocks into `directory_chain_blocks`.
//! - A Full-node Mirror producer is untrusted replicated evidence. Never include
//!   mirror registry membership in observation checkpoints or authority policy.
//! - Never auto-delete, auto-rewind, or auto-select through a quarantined fork.
//! - A producer-signed block fork/rollback quarantines that producer. A signed
//!   descriptor equivocation is evidence about the descriptor owner and does
//!   not automatically quarantine the observing producer.
//! - Keep all limits synchronized with the core Directory Sync V1 contract.
//! - Observation convergence is a local digest over independently verified
//!   producer evidence. It is never consensus, voting, fork choice, or finality.
//! - Observation checkpoints preserve that exact boundary. They are one
//!   observer's signed evidence and must never be presented as global blocks.
//! - A witness receipt proves one external recomputation of one exact local
//!   checkpoint. It is not a vote, quorum, fork choice, consensus, or finality.
//! - A policy anchor receipt proves only external retention of an opaque local
//!   policy head. It neither reveals nor approves policy membership.
//! - Witness outcome telemetry is aggregate diagnostic evidence only. Never add
//!   witness identities, endpoints, request ids, signatures, or hashes to it.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Recovery telemetry must remain
//!   process-only and aggregate. It is neither durable evidence nor peer
//!   reputation and must not influence witness policy or checkpoint truth.
//! - [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] Transport outcomes
//!   are process diagnostics only. Never add peer, endpoint, producer, carrier,
//!   URL, status code, frame, payload, or user-plane dimensions to them.
//! - [DIRECTORY-TRANSPORT-WINDOW 2026-07-28 by Codex] Keep the recent window
//!   fixed-size and outcome-only. It must never retain timestamps per request,
//!   identities, endpoints, operations, frames, hashes, or payload data.
//! - [DIRECTORY-TRANSPORT-LIFECYCLE 2026-07-29 by Codex] Lifecycle timestamps
//!   are aggregate transition diagnostics only. They are process-local, must
//!   never open durable security incidents, and must not affect routing.
//! - [WITNESS-TERMINAL-STATE 2026-07-29 by Codex] Witness availability health
//!   must use the service-owned latest terminal outcome, never timestamp
//!   comparison. Multiple completions may legitimately share one Unix second.
//! - Witness policy epochs describe only this operator's local evidence target.
//!   They are not a validator set, vote, quorum, fork choice, consensus, or
//!   finality, and public status must never expose their full member identities.
//! - Route-domain attestor epochs are local verification trust roots. They do
//!   not prove ASN/operator/geographic independence, consensus, honest behavior,
//!   or Sybil resistance; public status must remain aggregate-only.
//! - A carrier may export only non-quarantined producer evidence retained in
//!   this store. The receiver must still verify producer and carrier signatures.
//! - Public carrier reads must call `audited_mirror_evidence_*`; never use the
//!   general authority reader to bypass durable mirror membership.
//! - Replica descriptor proofs preserve that same namespace boundary. Mirror
//!   retention permits transport only and never grants producer or chain
//!   selection authority.
//! - Producer-scoped export audits may skip unrelated chain histories, but must
//!   still verify metadata, mirror admission when required, and every block,
//!   commitment, object, signature, linkage, index, and tip for the target.
//! - Incident evidence export is read-only. Quarantine resolution requires a
//!   separately authenticated, audited compare-and-swap command boundary.
//! - [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] Certificate export
//!   must re-verify the selected checkpoint and predecessor, receipt signatures,
//!   current witness membership, canonical order, and threshold on every read.
//!   It is evidence, never vote, quorum, consensus, fork choice, or finality.
//! - Never expose [`DirectoryReplicaStore::resolve_quarantine`] through the
//!   peer or public HTTP routers. It belongs only to the host-local CLI, whose
//!   caller must also possess the node identity key and database permissions.
//! - [DIRECTORY-AUDIT-OWNERSHIP 2026-08-12 by Codex] Public and local peer
//!   listeners must obtain their audit permit from the shared process runtime.
//!   Never create one independent admission gate per listener.
//!
//! ## Last Modified
//! v0.45.0-DirectoryConflictVerification - Rejected unverified same-page fork
//! payloads atomically before durable incident or retained-prefix mutation.
//! v0.44.0-DirectoryMirrorProvenance - Separated producer-signed tips from
//! carrier-reported prefixes and retained durable mirror resume cursors.
//! v0.43.0-DirectoryAuditSnapshot - Made streaming audits snapshot-consistent
//! and added SQL-side admission for persisted block and descriptor payloads.
//! v0.42.0-DirectoryStreamingProducerAudit - Bounded producer audit memory by
//! streaming blocks and loading only one protocol-limited commitment set.
//! v0.41.0-DirectoryAuditOwnership - Moved the full-history audit admission
//! permit into the shared process runtime so listeners share one CPU boundary.
//! v0.40.0-RouteDomainAttestorPolicyHistory - Added schema v12 canonical signed
//! local attestor/quorum epochs with atomic migration and startup audit.
//! v0.39.0-RouteDomainPolicyHistory - Added schema v11 canonical signed local
//! route-domain policy epochs and fail-closed startup audit/reconciliation.
//! v0.38.0-WitnessTerminalState - Made witness recovery and carrier health
//! service-owned, order-preserving, and independently counter-auditable.
//! v0.37.0-DirectoryTransportLifecycle - Centralized transport health policy,
//! tracked bounded aggregate degraded/recovered transitions, and prevented
//! diagnostic timestamps from regressing after a wall-clock rollback.
//! v0.36.0-DirectoryTransportWindow - Added a bounded recent-outcome window
//! so one recovery success cannot erase evidence of current transport churn.
//! v0.35.0-DirectoryTransportTelemetry - Added process-only, mutually exclusive
//! coordinator transport outcomes with no peer- or request-level dimensions.
//! v0.34.0-DirectoryProofDiversity - Rotated gossip evidence across producers
//! before selecting descriptors so alternate proofs cannot repeat one anchor
//! namespace
//! v0.33.0-DirectoryProofMaturity - Restricted gossip proof selection to
//! operator-policy-mature audited blocks so publication cannot outrun replicas
//! v0.32.0-DirectoryProofGossipPublisher - Added bounded live replica
//! descriptor selection with exact audited inclusion-proof reconstruction.
//! v0.31.0-ReplicaDescriptorInclusionProof - Added transactionally audited,
//! exact-block compact proof export for pinned and retained mirror namespaces
//! v0.27.0-WitnessCarrierAdmissionTelemetry - Added process-only mutually
//! exclusive target-cooldown and local-overload outcome counters.
//! v0.26.0-WitnessCarrierServiceTelemetry - Added process-only carrier-side
//! request outcomes without retaining request, identity, route, or frame data
//! v0.25.0-BoundedWitnessCarrierTelemetry - Added process-only aggregate
//! availability-recovery telemetry with no authority or persistence changes
//! v0.24.0-PortableObservationCertificate - Added current-pin, threshold-gated,
//! fail-closed portable checkpoint evidence export
//! v0.23.0-ReadOnlyCarrierSmoke - Added retained-anchor carrier verification
//! with no replica import, authority mutation, or incident-state mutation
//! v0.22.0-SignedMirrorCarrierTelemetry - Separated signed carrier capability
//! evidence from unadvertised compatibility fallback
//! v0.21.0-MirrorSourceDiversityTelemetry - Added privacy-safe aggregate
//! routeability and signed-region-hint carrier selection observations
//! v0.20.1-MirrorCarrierRangeAvailability - Added a typed unavailable-range
//! result for audited carrier reads so lagging mirrors remain safely retryable
//! v0.20.0-MirrorBoundedCatchUp - Added aggregate converged/catching-up mirror
//! outcomes and truthful multi-page request telemetry
//! v0.19.1-ProducerScopedExportAudit - Isolated transactional carrier audits
//! by producer without weakening target-chain verification
//! v0.19.0-MirrorRecovery - Added registry-gated public recovery reads and carrier import tests
//! v0.18.0-FullNodeMirror - Added schema v9 bounded non-authoritative mirror registry
//! v0.17.0-DirectoryPolicyHeadAnchor - Added schema v8 opaque external policy-head anchors and signed receipts
//! v0.16.0-DirectoryWitnessPolicyEpoch - Added schema v7 signed hash-linked local witness policy history, metadata-head partial-deletion protection, startup reconciliation, and tamper tests
//! v0.15.0-DirectoryWitnessFailureDrills - Locked partial-receipt restart recovery and current-pin rotation fail-closed behavior with deterministic state-machine coverage
//! v0.14.0-DirectoryWitnessThreshold - Added configurable pinned-witness corroboration targets
//! v0.13.0-DirectoryBoundedWitnessSelectionAudit - Bounded recurring selection verification without weakening startup audit
//! v0.12.0-DirectoryMatureWitnessScheduling - Added audited mature unwitnessed checkpoint selection
//! v0.11.0-DirectoryWitnessCapabilityNegotiation - Clarified peer-unavailable witness semantics for rolling upgrades
//! v0.10.0-DirectoryWitnessOutcomeTelemetry - Added schema v6 privacy-safe durable and runtime witness outcome buckets
//! v0.9.0-DirectoryEvidenceCarrier - Added transactional audited producer evidence export and carrier-frame audit
//! v0.30.0-PortableCertificateImport - Added schema v10 with a bounded,
//! node-signed, hash-linked import history for third-party observation
//! certificates and complete restart audit.
//! v0.8.0-DirectoryObservationWitness - Added schema v5, independent checkpoint recomputation, and receipt audit
//! v0.7.0-DirectoryObservationCheckpoints - Added schema v4, append-only signed
//! checkpoints, exact-prefix root recomputation, and startup tamper detection.
//! v0.6.0-DirectoryReplicaQuarantineResolution - Added schema v3, signed local
//! operator resolution commands, exact incident/tip CAS, and linked immutable
//! resolution auditing without deleting or rewinding accepted evidence.
//! v0.5.0-DirectoryReplicaIncidentEvidence - Added bounded incident pagination
//! and fail-closed, signature-reverified evidence export for local operators.
//! v0.4.0-DirectoryReplicaObservationConvergence - Added bounded recent-window
//! multi-source commitment overlap and a deterministic local observation root.
//! v0.3.0-DirectoryReplicaDurableRetry - Added an atomic schema v1-to-v2
//! migration and audited restart-durable producer retry state.
//! v0.2.2-DirectoryReplicaRetryRuntime - Added producer-local retry boundaries
//! and backoff skip counters to process-lifetime synchronization telemetry.
//! v0.2.1-DirectoryReplicaModuleSplit - Updated transport and status ownership.
//! v0.2.0-DirectoryReplicaStatus - Added aggregate status snapshots and shared
//! synchronization observations for bounded catch-up visibility.
//! v0.1.0-DirectoryReplicaStore - Initial producer-isolated replica persistence.
// ============================================================================

// [DIRECTORY-REPLICA-TRANSPORT-MODULE 2026-09-23 by Codex] Keep the
// process-only transport health state machine isolated from replica storage.

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod evidence_export;
mod observation_certificate;
mod observation_checkpoint;
mod observation_convergence;
mod observation_witness;
mod quarantine_resolution;
mod retry_state;
mod schema;
mod store_access;
mod verified_import;

mod transport;
// [DIRECTORY-REPLICA-POLICY-MODULE 2026-09-24 by Codex] Keep signed policy
// reconciliation, durable anchor receipts, and full-history verification in
// one private submodule without changing the service module registry.
#[path = "directory_replica_policy.rs"]
mod policy;
use policy::{
    validate_observation_witness_policy_members, validate_route_domain_attestor_policy_members,
    validate_route_domain_policy_assignments,
};

use transport::{latest_runtime_timestamp, DirectoryReplicaTransportRuntime};
pub use transport::{
    DirectoryReplicaTransportHealth, DirectoryReplicaTransportOutcome,
    DirectoryReplicaTransportSnapshot, DIRECTORY_REPLICA_TRANSPORT_DEGRADED_CONSECUTIVE_FAILURES,
    DIRECTORY_REPLICA_TRANSPORT_DEGRADED_FAILURE_PERCENT,
    DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::config::PinnedRouteDomainAssignment;
use crate::services::directory_chain::parallel_try_map;

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_observation_certificate, decode_directory_sync_message,
    directory_block_range_response_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_replica_block_range_response_signing_bytes, encode_directory_observation_certificate,
    encode_directory_sync_message, DirectoryCommitmentBlockV1, DirectoryCommitmentValidationError,
    DirectoryDescriptorCommitmentV1, DirectoryDescriptorInclusionProofV1,
    DirectoryObservationCertificateV1, DirectoryObservationCheckpointV1, DirectoryObservationTipV1,
    DirectoryObservationWitnessReceiptV1, DirectorySyncMessage, SignedNodeDescriptor,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1, DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
    DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1, DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
    MAX_DIRECTORY_COMMITMENTS_PER_BLOCK, MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES,
    MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1, MAX_DIRECTORY_SYNC_BLOCKS_V1,
};
use bincode::Options;
use parking_lot::Mutex;
use rusqlite::{
    params, params_from_iter, types::Value, Connection, OptionalExtension, Transaction,
    TransactionBehavior,
};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

const DIRECTORY_REPLICA_SCHEMA_VERSION: i64 = 12;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V11: i64 = 11;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V10: i64 = 10;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V9: i64 = 9;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V8: i64 = 8;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V7: i64 = 7;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V6: i64 = 6;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V5: i64 = 5;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V4: i64 = 4;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V3: i64 = 3;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V2: i64 = 2;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V1: i64 = 1;
const MAX_DIRECTORY_BLOCK_BYTES: u64 = 64 * 1024;
const MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES: u64 = 32 * 1024;
const MAX_DIRECTORY_SYNC_EVIDENCE_BYTES: usize = 512 * 1024;
const DIRECTORY_REPLICA_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// One shared full-history audit protects CPU without weakening verification.
const MAX_DIRECTORY_AUDITS_IN_FLIGHT: usize = 1;
const RESPONSE_TIMESTAMP_SKEW_SECS: u64 = 60;
const MAX_DIRECTORY_REPLICA_FAILURE_REASON_BYTES: usize = 96;
const DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS: u64 = 32;
const MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS: usize = 16;
const MAX_DIRECTORY_REPLICA_INCIDENT_KIND_BYTES: usize = 64;
const MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES: u64 = 4 * 1024;
const MAX_DIRECTORY_OBSERVATION_WITNESS_BYTES: usize = 2 * 1024;
const MAX_DIRECTORY_POLICY_ANCHOR_BYTES: usize = 2 * 1024;
const MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS: usize = 16;
const MAX_DIRECTORY_ROUTE_DOMAIN_POLICY_ASSIGNMENTS: usize = 256;
const MAX_DIRECTORY_ROUTE_DOMAIN_ATTESTOR_POLICY_MEMBERS: usize = 16;
/// Maximum third-party observation certificates retained by one local store.
pub(crate) const MAX_DIRECTORY_OBSERVATION_CERTIFICATE_IMPORTS: usize = 4_096;
const DIRECTORY_OBSERVATION_CERTIFICATE_IMPORT_TIMESTAMP_SKEW_SECS: u64 = 60;
const DIRECTORY_REPLICA_RESOLUTION_ACTION: &str = "resume_existing_prefix";
const DIRECTORY_REPLICA_RESOLUTION_TIMESTAMP_SKEW_SECS: u64 = 60;
/// Maximum incident summaries returned by one operator API read.
pub(crate) const MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE: usize = 50;
/// Maximum producer failure streak retained in memory and audited `SQLite`.
pub(crate) const DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES: u64 = 64;
/// Maximum durable retry delay accepted by the replica store and scheduler.
pub(crate) const DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS: u64 = 30 * 60;
/// Hard implementation ceiling for durable permissionless mirror namespaces.
pub(crate) const MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS: usize = 64;
/// Maximum recent public descriptor rows inspected for one gossip proof.
///
/// [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] This ceiling keeps one
/// periodic selection independent of retained history. The selected producer
/// is still fully audited before any proof leaves the process.
const DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryReplicaImportMode {
    PinnedAuthority,
    FullNodeMirror {
        descriptor_sequence: u64,
        max_producers: usize,
    },
}

/// Durable scheduling cursor for one non-authoritative mirror namespace.
///
/// Registry membership is local availability state only. It grants no
/// producer, checkpoint, witness, policy, or carrier authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectoryRetainedMirrorCursor {
    pub(crate) producer: [u8; 32],
    pub(crate) descriptor_sequence: u64,
}

/// Authenticated source of the range response's advertised tip fields.
///
/// [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] Producer responses may
/// establish tip-level contradictions. Carrier responses authenticate only
/// what that carrier reported; the embedded producer-signed blocks remain the
/// sole producer evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryRangeTipProvenance {
    ProducerSigned,
    CarrierReported,
}

/// Producer-authenticated evidence of two different blocks at one page height.
///
/// [DIRECTORY-CONFLICT-VERIFICATION 2026-09-01 by Codex] This typed preflight
/// result carries hashes only after both page claims verify against the same
/// durable predecessor. Neither hash becomes an accepted fork choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifiedPageBlockFork {
    height: u64,
    first_hash: [u8; 32],
    conflicting_hash: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryReplicaEvidenceScope {
    AnyAudited,
    RetainedMirror,
}

/// Persisted protocol payloads that require SQL-side size admission before
/// SQLite may materialize their bytes into a Rust allocation.
///
/// [DIRECTORY-AUDIT-SNAPSHOT 2026-08-31 by Codex] Keeping the byte ceiling and
/// typed failure text on this domain enum prevents audit, proof, and export
/// readers from applying different limits to the same durable representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistedReplicaBlobKind {
    Block,
    Descriptor,
}

impl PersistedReplicaBlobKind {
    const fn max_bytes(self) -> u64 {
        match self {
            Self::Block => MAX_DIRECTORY_BLOCK_BYTES,
            Self::Descriptor => MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES,
        }
    }

    const fn length_field(self) -> &'static str {
        match self {
            Self::Block => "persisted replica block byte length",
            Self::Descriptor => "persisted replica descriptor byte length",
        }
    }

    const fn oversized_message(self) -> &'static str {
        match self {
            Self::Block => "replica block exceeds its byte limit",
            Self::Descriptor => "replica descriptor object exceeds its byte limit",
        }
    }

    const fn missing_message(self) -> &'static str {
        match self {
            Self::Block => "admitted replica block payload is missing",
            Self::Descriptor => "replica commitment is missing its descriptor payload",
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryReplicaAuditTestEvent {
    BlockVerified(u64),
    BlobMaterialized(PersistedReplicaBlobKind),
}

#[cfg(test)]
thread_local! {
    static DIRECTORY_REPLICA_AUDIT_TEST_OBSERVER:
        std::cell::RefCell<Option<Box<dyn Fn(DirectoryReplicaAuditTestEvent)>>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
fn set_directory_replica_audit_test_observer(
    observer: Option<Box<dyn Fn(DirectoryReplicaAuditTestEvent)>>,
) {
    DIRECTORY_REPLICA_AUDIT_TEST_OBSERVER.with(|current| {
        *current.borrow_mut() = observer;
    });
}

#[cfg(test)]
fn notify_directory_replica_audit_test_observer(event: DirectoryReplicaAuditTestEvent) {
    DIRECTORY_REPLICA_AUDIT_TEST_OBSERVER.with(|observer| {
        if let Some(observer) = observer.borrow().as_ref() {
            observer(event);
        }
    });
}

/// Failures returned by the producer-isolated replica store.
#[derive(Debug, thiserror::Error)]
pub enum DirectoryReplicaStoreError {
    /// Filesystem setup failed.
    #[error("directory replica filesystem error: {0}")]
    Io(#[from] std::io::Error),
    /// `SQLite` rejected a schema, query, or transaction operation.
    #[error("directory replica sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A protocol object could not be encoded or decoded safely.
    #[error("directory replica codec error: {0}")]
    Codec(String),
    /// A descriptor object did not reproduce its signed commitment.
    #[error("directory replica descriptor error: {0}")]
    Descriptor(String),
    /// A block failed the canonical Directory Chain V1 contract.
    #[error("directory replica block validation error: {0}")]
    Block(#[from] DirectoryCommitmentValidationError),
    /// Durable metadata, chain, index, or evidence is inconsistent.
    #[error("directory replica integrity error: {0}")]
    Integrity(String),
    /// A bounded import request violates the V1 transport contract.
    #[error("directory replica request error: {0}")]
    Request(String),
    /// The producer is durably isolated pending operator review.
    #[error("directory producer is quarantined: {0}")]
    Quarantined(String),
    /// The configured durable permissionless mirror namespace ceiling is full.
    #[error("directory full-node mirror capacity reached")]
    MirrorCapacity,
    /// A public recovery read requested a namespace not retained as a mirror.
    #[error("directory full-node mirror namespace is not retained")]
    MirrorNotRetained,
    /// [MIRROR-CARRIER 2026-07-24 by Codex] The requested producer range is
    /// valid, but this carrier has not retained that height yet. Keep this
    /// distinct from malformed requests so callers may try another carrier
    /// without making protocol-contract failures retryable.
    #[error(
        "directory replica range from height {from_height} is beyond retained tip {tip_height}"
    )]
    RangeNotRetained {
        /// First block height requested by the authenticated peer.
        from_height: u64,
        /// Highest producer block fully audited on this carrier.
        tip_height: u64,
    },
}

/// Operator-owned trust anchors for one portable observation certificate.
///
/// [PORTABLE-CERTIFICATE-IMPORT 2026-07-26 by Codex] Valid signatures prove
/// authorship, not authority. Every verifier and importer therefore supplies a
/// pinned observer, a bounded witness allowlist, and a local minimum. The
/// certificate's self-declared threshold can never weaken this local policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryObservationCertificateTrustPolicy {
    expected_observer: [u8; 32],
    allowed_witnesses: Vec<[u8; 32]>,
    minimum_witnesses: u16,
}

impl DirectoryObservationCertificateTrustPolicy {
    /// Builds one canonical pinned trust policy.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError::Request`] for zero identities,
    /// duplicate witnesses, observer/witness overlap, or an invalid threshold.
    pub fn new(
        expected_observer: [u8; 32],
        mut allowed_witnesses: Vec<[u8; 32]>,
        minimum_witnesses: u16,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        if expected_observer == [0u8; 32]
            || allowed_witnesses.is_empty()
            || allowed_witnesses.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryReplicaStoreError::Request(
                "portable certificate trust policy identity set is invalid".to_string(),
            ));
        }
        allowed_witnesses.sort_unstable();
        if allowed_witnesses
            .iter()
            .any(|witness| *witness == [0u8; 32] || *witness == expected_observer)
            || allowed_witnesses
                .windows(2)
                .any(|witnesses| witnesses[0] == witnesses[1])
            || minimum_witnesses == 0
            || usize::from(minimum_witnesses) > allowed_witnesses.len()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "portable certificate trust policy is not canonical".to_string(),
            ));
        }
        Ok(Self {
            expected_observer,
            allowed_witnesses,
            minimum_witnesses,
        })
    }

    /// Pinned observer node identity.
    #[must_use]
    pub const fn expected_observer(&self) -> [u8; 32] {
        self.expected_observer
    }

    /// Canonically sorted pinned witness identities.
    #[must_use]
    pub fn allowed_witnesses(&self) -> &[[u8; 32]] {
        &self.allowed_witnesses
    }

    /// Locally required distinct witness count.
    #[must_use]
    pub const fn minimum_witnesses(&self) -> u16 {
        self.minimum_witnesses
    }

    fn verify(
        &self,
        certificate: &DirectoryObservationCertificateV1,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if certificate.checkpoint.observer != self.expected_observer {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate observer does not match the pinned observer".to_string(),
            ));
        }
        if certificate.receipts.iter().any(|receipt| {
            self.allowed_witnesses
                .binary_search(&receipt.responder)
                .is_err()
        }) {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate contains a witness outside the allowed set".to_string(),
            ));
        }
        if certificate.receipts.len() < usize::from(self.minimum_witnesses) {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate does not satisfy the local witness threshold".to_string(),
            ));
        }
        Ok(())
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationCertificateTrustPolicy-v1");
        hasher.update(AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        hasher.update(self.expected_observer);
        hasher.update(self.minimum_witnesses.to_le_bytes());
        hasher.update(
            u64::try_from(self.allowed_witnesses.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for witness in &self.allowed_witnesses {
            hasher.update(witness);
        }
        hasher.finalize().into()
    }
}

/// Fully verified portable observation certificate and transport bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedDirectoryObservationCertificate {
    /// Decoded canonical certificate.
    pub certificate: DirectoryObservationCertificateV1,
    /// Stable certificate identity over checkpoint and exact receipts.
    pub certificate_id: [u8; 32],
    /// SHA-256 of the exact canonical frame supplied by the operator.
    pub certificate_sha256: [u8; 32],
    /// Stable digest of the local pinned trust policy.
    pub policy_digest: [u8; 32],
    /// Host time used for signature and timestamp validation.
    pub verified_at: u64,
}

/// Verifies exact bytes, canonical encoding, signatures, bindings, and pins.
///
/// # Errors
/// Returns [`DirectoryReplicaStoreError::Request`] for a malformed, oversized,
/// non-canonical, mistimed, incorrectly signed, or locally untrusted frame.
pub fn verify_directory_observation_certificate_frame(
    frame: &[u8],
    expected_sha256: &[u8; 32],
    trust_policy: &DirectoryObservationCertificateTrustPolicy,
    verified_at: u64,
) -> Result<VerifiedDirectoryObservationCertificate, DirectoryReplicaStoreError> {
    if verified_at == 0
        || frame.is_empty()
        || frame.len() > MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES
    {
        return Err(DirectoryReplicaStoreError::Request(
            "observation certificate frame size or verification time is invalid".to_string(),
        ));
    }
    let certificate_sha256: [u8; 32] = Sha256::digest(frame).into();
    if &certificate_sha256 != expected_sha256 {
        return Err(DirectoryReplicaStoreError::Request(
            "observation certificate frame SHA-256 mismatch".to_string(),
        ));
    }
    let certificate = decode_directory_observation_certificate(frame).map_err(|error| {
        DirectoryReplicaStoreError::Request(format!(
            "observation certificate decode failed: {error}"
        ))
    })?;
    let canonical_frame =
        encode_directory_observation_certificate(&certificate).map_err(|error| {
            DirectoryReplicaStoreError::Request(format!(
                "observation certificate canonical encoding failed: {error}"
            ))
        })?;
    if canonical_frame != frame {
        return Err(DirectoryReplicaStoreError::Request(
            "observation certificate frame is not canonically encoded".to_string(),
        ));
    }
    certificate
        .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, verified_at)
        .map_err(|error| {
            DirectoryReplicaStoreError::Request(format!(
                "observation certificate signature verification failed: {error}"
            ))
        })?;
    trust_policy.verify(&certificate)?;
    let certificate_id = certificate.hash();
    Ok(VerifiedDirectoryObservationCertificate {
        certificate,
        certificate_id,
        certificate_sha256,
        policy_digest: trust_policy.digest(),
        verified_at,
    })
}

/// Result of one durable host-local certificate import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryObservationCertificateImportReport {
    /// True only when a new signed import row was appended.
    pub inserted: bool,
    /// Local append-only import sequence.
    pub import_sequence: u64,
    /// Hash-linked digest of the local signed import row.
    pub import_digest: [u8; 32],
    /// Stable identity of the imported portable certificate.
    pub certificate_id: [u8; 32],
    /// SHA-256 of the exact imported certificate frame.
    pub certificate_sha256: [u8; 32],
    /// External observer represented by this certificate.
    pub observer: [u8; 32],
    /// External observer checkpoint sequence.
    pub checkpoint_sequence: u64,
    /// External observer checkpoint hash.
    pub checkpoint_hash: [u8; 32],
    /// Number of certificates retained after the operation.
    pub retained_certificates: u64,
    /// Host time at which the frame and local pins were verified.
    pub verified_at: u64,
}

/// Aggregate result of a complete replica startup audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryReplicaAudit {
    /// Number of producer namespaces.
    pub producers: u64,
    /// Producer namespaces admitted only as non-authoritative mirrors.
    pub mirror_producers: u64,
    /// Number of producer namespaces currently quarantined.
    pub quarantined_producers: u64,
    /// Number of verified remote blocks.
    pub blocks: u64,
    /// Number of commitments exactly matched to block payloads.
    pub commitments: u64,
    /// Number of durable authenticated incidents.
    pub incidents: u64,
    /// Number of node-identity-signed operator resolutions.
    pub resolutions: u64,
    /// Number of audited observer-signed convergence checkpoints.
    pub observation_checkpoints: u64,
    /// Latest audited checkpoint sequence, or zero when none exists.
    pub observation_checkpoint_sequence: u64,
    /// Latest audited checkpoint hash, or zero when none exists.
    pub observation_checkpoint_hash: [u8; 32],
    /// Latest audited checkpoint timestamp, or zero when none exists.
    pub observation_checkpoint_observed_at: u64,
    /// Number of independently signed accepted witness receipts.
    pub observation_checkpoint_witnesses: u64,
    /// Latest local checkpoint sequence with at least one accepted witness.
    pub observation_checkpoint_witnessed_sequence: u64,
    /// Distinct witnesses retained for the latest witnessed sequence.
    pub observation_checkpoint_latest_witnesses: u64,
    /// Audited privacy-safe witness attempt aggregates.
    pub observation_witness_outcomes: DirectoryObservationWitnessOutcomeSnapshot,
    /// Number of audited local witness-policy epochs.
    pub observation_witness_policy_epochs: u64,
    /// Current local witness-policy epoch, or zero before reconciliation.
    pub observation_witness_policy_epoch: u64,
    /// Timestamp bound into the current local witness policy.
    pub observation_witness_policy_activated_at: u64,
    /// Number of operator-pinned witnesses in the current policy.
    pub observation_witness_policy_members: u64,
    /// External receipt threshold in the current local policy.
    pub observation_witness_policy_threshold: u64,
    /// Signed external anchor receipts retained for local policy epochs.
    pub observation_witness_policy_anchor_receipts: u64,
    /// Opaque foreign policy heads this node retains for independent observers.
    pub observation_witness_remote_policy_anchors: u64,
    /// Number of audited local route-domain policy epochs.
    pub route_domain_policy_epochs: u64,
    /// Current local route-domain policy epoch, or zero before first use.
    pub route_domain_policy_epoch: u64,
    /// Timestamp bound into the current route-domain policy.
    pub route_domain_policy_activated_at: u64,
    /// Number of opaque node-to-domain assignments in the current policy.
    pub route_domain_policy_assignments: u64,
    /// Whether current multi-hop selection requires complete pinned coverage.
    pub route_domain_policy_strict: bool,
    /// Number of audited local route-domain attestor-policy epochs.
    pub route_domain_attestor_policy_epochs: u64,
    /// Current local route-domain attestor-policy epoch, or zero before use.
    pub route_domain_attestor_policy_epoch: u64,
    /// Timestamp bound into the current route-domain attestor policy.
    pub route_domain_attestor_policy_activated_at: u64,
    /// Number of locally pinned route-domain attestors.
    pub route_domain_attestor_policy_members: u64,
    /// Locally required distinct valid route-domain attestations.
    pub route_domain_attestor_policy_threshold: u64,
    /// Whether current multi-hop selection requires attested route domains.
    pub route_domain_attestor_policy_strict: bool,
    /// Third-party portable observation certificates in the audited import log.
    pub imported_observation_certificates: u64,
    /// Latest node-signed certificate-import sequence, or zero when empty.
    pub imported_observation_certificate_sequence: u64,
    /// Latest node-signed certificate-import digest, or zero when empty.
    pub imported_observation_certificate_head: [u8; 32],
    /// Number of audited producer-local retry rows.
    pub retry_states: u64,
}

/// Low-cost aggregate view of the already audited replica namespace.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectoryReplicaStoreSnapshot {
    /// Number of producer namespaces currently persisted.
    pub producers: u64,
    /// Producer namespaces retained only as non-authoritative mirrors.
    pub mirror_producers: u64,
    /// Number of producer namespaces blocked by durable quarantine.
    pub quarantined_producers: u64,
    /// Number of verified remote blocks retained across all producers.
    pub blocks: u64,
    /// Number of verified descriptor commitments retained across all producers.
    pub commitments: u64,
    /// Number of durable authenticated incidents.
    pub incidents: u64,
    /// Number of durable signed quarantine resolutions.
    pub resolutions: u64,
    /// Number of durable observer-signed convergence checkpoints.
    pub observation_checkpoints: u64,
    /// Latest checkpoint sequence, or zero when none exists.
    pub observation_checkpoint_sequence: u64,
    /// Latest checkpoint hash, or zero when none exists.
    pub observation_checkpoint_hash: [u8; 32],
    /// Latest checkpoint timestamp, or zero when none exists.
    pub observation_checkpoint_observed_at: u64,
    /// Number of independently signed accepted witness receipts.
    pub observation_checkpoint_witnesses: u64,
    /// Latest local checkpoint sequence with at least one accepted witness.
    pub observation_checkpoint_witnessed_sequence: u64,
    /// Distinct witnesses retained for the latest witnessed sequence.
    pub observation_checkpoint_latest_witnesses: u64,
    /// Audited privacy-safe witness attempt aggregates.
    pub observation_witness_outcomes: DirectoryObservationWitnessOutcomeSnapshot,
    /// Number of durable, signed local witness-policy epochs.
    pub observation_witness_policy_epochs: u64,
    /// Current local witness-policy epoch, or zero before reconciliation.
    pub observation_witness_policy_epoch: u64,
    /// Timestamp bound into the current local witness policy.
    pub observation_witness_policy_activated_at: u64,
    /// Number of operator-pinned witnesses in the current policy.
    pub observation_witness_policy_members: u64,
    /// External receipt threshold in the current local policy.
    pub observation_witness_policy_threshold: u64,
    /// Signed external anchor receipts retained for local policy epochs.
    pub observation_witness_policy_anchor_receipts: u64,
    /// Opaque foreign policy heads this node retains for independent observers.
    pub observation_witness_remote_policy_anchors: u64,
    /// Third-party portable observation certificates retained after audit.
    pub imported_observation_certificates: u64,
    /// Latest local certificate-import sequence, or zero when empty.
    pub imported_observation_certificate_sequence: u64,
    /// Latest local certificate-import digest, or zero when empty.
    pub imported_observation_certificate_head: [u8; 32],
    /// Per-producer accepted-prefix summaries for local operator presentation.
    pub producer_snapshots: Vec<DirectoryReplicaProducerSnapshot>,
}

/// Bounded, locally recomputable overlap across verified producer replicas.
///
/// This snapshot compares exact commitment hashes from each eligible
/// producer's most recent block window. It does not assign voting weight,
/// choose a chain, or create a globally finalized checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectoryReplicaObservationConvergenceSnapshot {
    /// Unique producer pins supplied by the validated node configuration.
    pub configured_producers: u64,
    /// Configured producers with a non-empty, non-quarantined accepted prefix.
    pub eligible_producers: u64,
    /// Configured producers that have not supplied an accepted block yet.
    pub pending_producers: u64,
    /// Configured producers excluded because signed evidence quarantined them.
    pub excluded_quarantined_producers: u64,
    /// Maximum number of recent blocks inspected per eligible producer.
    pub window_blocks: u64,
    /// Commitment observations across all eligible producer windows.
    pub recent_commitments: u64,
    /// Unique commitment hashes across all eligible producer windows.
    pub distinct_recent_commitments: u64,
    /// Commitments observed by at least two eligible producer chains.
    pub multi_source_recent_commitments: u64,
    /// Commitments observed by every eligible producer when at least two exist.
    pub all_eligible_source_recent_commitments: u64,
    /// Deterministic digest of eligible tips and their exact common commitments.
    pub observation_root: Option<[u8; 32]>,
}

/// Result of attempting to append one complete observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryObservationCheckpointAppendReport {
    /// Whether a new checkpoint was written. An unchanged root is idempotent.
    pub appended: bool,
    /// Latest checkpoint sequence after the transaction.
    pub sequence: u64,
    /// Latest checkpoint hash after the transaction.
    pub checkpoint_hash: [u8; 32],
    /// Timestamp bound into the latest checkpoint.
    pub observed_at: u64,
    /// Number of configured producer tips bound into the checkpoint.
    pub producer_count: u16,
    /// Recomputable multi-source overlap root.
    pub observation_root: [u8; 32],
}

/// Audited mature checkpoint that has not reached its configured corroboration
/// target among the current operator-pinned witnesses.
///
/// The retained witness identities are public node signing keys required only
/// to avoid duplicate outbound requests. They must never be exposed by public
/// status or interpreted as voting weight, consensus membership, or finality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryObservationWitnessTarget {
    /// Canonical observer-signed checkpoint requiring more external evidence.
    pub checkpoint: DirectoryObservationCheckpointV1,
    /// Current pinned witnesses with an audited accepted receipt for this row.
    pub witnessed_by: Vec<[u8; 32]>,
    /// Required number of distinct pinned witness receipts.
    pub minimum_witnesses: usize,
}

/// Result of independently evaluating an external observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessDecision {
    /// Every exact producer prefix exists locally and the root recomputes.
    Accepted,
    /// At least one exact referenced producer prefix is not retained locally.
    EvidenceUnavailable,
    /// Retained producer evidence conflicts or recomputes a different root.
    EvidenceConflict,
}

/// Stable privacy-safe result bucket for one outbound witness attempt.
///
/// The enum deliberately excludes peer identity, endpoint, request id,
/// signature, checkpoint hash, transport text, and response body data. New
/// variants require a schema migration and additive status-contract review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessOutcome {
    /// A canonical accepted receipt was verified and durably retained.
    Accepted,
    /// The witness does not yet retain every exact referenced producer prefix.
    EvidenceUnavailable,
    /// Locally retained evidence conflicts with the observed checkpoint.
    EvidenceConflict,
    /// The witness is not admitted, reachable, or serving the optional route.
    PeerUnavailable,
    /// The bounded outbound request failed before a verifiable frame arrived.
    TransportFailure,
    /// A received frame failed canonical contract or signature verification.
    VerificationFailure,
    /// A verified accepted receipt could not be durably retained.
    PersistenceFailure,
}

/// Aggregate counters for a bounded set of witness attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessOutcomeCounters {
    /// Canonical accepted receipts durably retained.
    pub accepted: u64,
    /// Witnesses missing at least one exact producer prefix.
    pub evidence_unavailable: u64,
    /// Witnesses whose retained evidence conflicts with the checkpoint.
    pub evidence_conflict: u64,
    /// Witnesses unavailable at admission, endpoint, or capability validation.
    pub peer_unavailable: u64,
    /// Bounded outbound transport failures.
    pub transport_failures: u64,
    /// Canonical contract or signature verification failures.
    pub verification_failures: u64,
    /// Verified receipts rejected by durable persistence.
    pub persistence_failures: u64,
}

impl DirectoryObservationWitnessOutcomeCounters {
    fn from_outcomes(outcomes: &[DirectoryObservationWitnessOutcome]) -> Self {
        let mut counters = Self::default();
        for outcome in outcomes {
            counters.record(*outcome);
        }
        counters
    }

    fn record(&mut self, outcome: DirectoryObservationWitnessOutcome) {
        let counter = match outcome {
            DirectoryObservationWitnessOutcome::Accepted => &mut self.accepted,
            DirectoryObservationWitnessOutcome::EvidenceUnavailable => {
                &mut self.evidence_unavailable
            }
            DirectoryObservationWitnessOutcome::EvidenceConflict => &mut self.evidence_conflict,
            DirectoryObservationWitnessOutcome::PeerUnavailable => &mut self.peer_unavailable,
            DirectoryObservationWitnessOutcome::TransportFailure => &mut self.transport_failures,
            DirectoryObservationWitnessOutcome::VerificationFailure => {
                &mut self.verification_failures
            }
            DirectoryObservationWitnessOutcome::PersistenceFailure => {
                &mut self.persistence_failures
            }
        };
        *counter = counter.saturating_add(1);
    }

    fn checked_add(self, other: Self) -> Result<Self, DirectoryReplicaStoreError> {
        let add = |left: u64, right: u64| {
            left.checked_add(right).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness outcome counter exhausted".to_string(),
                )
            })
        };
        Ok(Self {
            accepted: add(self.accepted, other.accepted)?,
            evidence_unavailable: add(self.evidence_unavailable, other.evidence_unavailable)?,
            evidence_conflict: add(self.evidence_conflict, other.evidence_conflict)?,
            peer_unavailable: add(self.peer_unavailable, other.peer_unavailable)?,
            transport_failures: add(self.transport_failures, other.transport_failures)?,
            verification_failures: add(self.verification_failures, other.verification_failures)?,
            persistence_failures: add(self.persistence_failures, other.persistence_failures)?,
        })
    }

    const fn saturating_add(self, other: Self) -> Self {
        Self {
            accepted: self.accepted.saturating_add(other.accepted),
            evidence_unavailable: self
                .evidence_unavailable
                .saturating_add(other.evidence_unavailable),
            evidence_conflict: self
                .evidence_conflict
                .saturating_add(other.evidence_conflict),
            peer_unavailable: self.peer_unavailable.saturating_add(other.peer_unavailable),
            transport_failures: self
                .transport_failures
                .saturating_add(other.transport_failures),
            verification_failures: self
                .verification_failures
                .saturating_add(other.verification_failures),
            persistence_failures: self
                .persistence_failures
                .saturating_add(other.persistence_failures),
        }
    }

    /// Total attempts represented by these mutually exclusive buckets.
    #[must_use]
    pub const fn attempts(self) -> u64 {
        self.accepted
            .saturating_add(self.evidence_unavailable)
            .saturating_add(self.evidence_conflict)
            .saturating_add(self.peer_unavailable)
            .saturating_add(self.transport_failures)
            .saturating_add(self.verification_failures)
            .saturating_add(self.persistence_failures)
    }

    /// Non-accepted attempts represented by these buckets.
    #[must_use]
    pub const fn failures(self) -> u64 {
        self.attempts().saturating_sub(self.accepted)
    }
}

/// Audited aggregate witness telemetry retained across restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessOutcomeSnapshot {
    /// Completed bounded witness rounds.
    pub rounds: u64,
    /// Cumulative mutually exclusive attempt outcomes.
    pub totals: DirectoryObservationWitnessOutcomeCounters,
    /// Latest local checkpoint sequence evaluated by a witness round.
    pub last_checkpoint_sequence: u64,
    /// Timestamp of the latest completed witness round.
    pub last_round_at: Option<u64>,
    /// Latest round containing at least one accepted receipt.
    pub last_success_at: Option<u64>,
    /// Latest round containing at least one non-accepted attempt.
    pub last_failure_at: Option<u64>,
    /// Mutually exclusive outcomes from only the latest completed round.
    pub last_round: DirectoryObservationWitnessOutcomeCounters,
    /// Process-only failures while persisting this telemetry itself.
    /// Durable snapshots always keep this field at zero.
    pub telemetry_persistence_failures: u64,
}

impl DirectoryObservationWitnessOutcomeSnapshot {
    fn next_durable_round(
        self,
        checkpoint_sequence: u64,
        observed_at: u64,
        round: DirectoryObservationWitnessOutcomeCounters,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        if checkpoint_sequence < self.last_checkpoint_sequence
            || self
                .last_round_at
                .is_some_and(|last_round_at| observed_at < last_round_at)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome round regressed".to_string(),
            ));
        }
        Ok(Self {
            rounds: self.rounds.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness outcome round counter exhausted".to_string(),
                )
            })?,
            totals: self.totals.checked_add(round)?,
            last_checkpoint_sequence: checkpoint_sequence,
            last_round_at: Some(observed_at),
            last_success_at: if round.accepted > 0 {
                Some(observed_at)
            } else {
                self.last_success_at
            },
            last_failure_at: if round.failures() > 0 {
                Some(observed_at)
            } else {
                self.last_failure_at
            },
            last_round: round,
            telemetry_persistence_failures: 0,
        })
    }
}

/// Persisted aggregate state for one producer namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaProducerSnapshot {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Timestamp signed into the accepted tip block.
    pub tip_timestamp: u64,
    /// Whether imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable authenticated incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Last time this namespace metadata changed locally.
    pub updated_at: u64,
    /// Verified blocks retained for this producer.
    pub blocks: u64,
    /// Verified commitments retained for this producer.
    pub commitments: u64,
    /// Durable incidents attributed to this producer response stream.
    pub incidents: u64,
    /// Signed operator resolutions retained for this producer.
    pub resolutions: u64,
}

/// Bounded metadata for one startup-audited Directory Replica incident.
///
/// The summary intentionally excludes the potentially large signed response
/// frame. Call [`DirectoryReplicaStore::incident_evidence`] for an independent,
/// fail-closed verification immediately before exporting that frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentSummary {
    /// Content-addressed incident identifier used as the pagination cursor.
    pub incident_digest: [u8; 32],
    /// Producer that signed the conflicting Directory Sync response.
    pub producer: [u8; 32],
    /// Identity whose chain or descriptor assertion conflicts.
    pub subject_node_id: [u8; 32],
    /// Stable internal incident classification.
    pub kind: String,
    /// Conflicting block or advertised tip height.
    pub height: u64,
    /// Previously accepted local claim.
    pub local_hash: [u8; 32],
    /// Conflicting producer-signed remote claim.
    pub remote_hash: [u8; 32],
    /// Local Unix timestamp at which the signed evidence was persisted.
    pub observed_at: u64,
    /// Whether this producer remains quarantined at read time.
    pub producer_quarantined: bool,
}

/// Deterministic cursor page of incident metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentPage {
    /// Incident summaries ordered by ascending content digest.
    pub incidents: Vec<DirectoryReplicaIncidentSummary>,
    /// Last returned digest when another page exists.
    pub next_cursor: Option<[u8; 32]>,
}

/// Complete independently verifiable evidence for one durable incident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentEvidence {
    /// Validated incident metadata and current quarantine state.
    pub summary: DirectoryReplicaIncidentSummary,
    /// Exact canonical producer-signed `BlockRangeResponseV1` bytes.
    pub evidence_frame: Vec<u8>,
    /// SHA-256 digest of `evidence_frame` for transport/file verification.
    pub evidence_sha256: [u8; 32],
}

/// Current accepted prefix and isolation state for one producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaTip {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Accepted tip hash, or zero for an empty prefix.
    pub tip_hash: [u8; 32],
    /// Accepted tip timestamp, or zero for an empty prefix.
    pub tip_timestamp: u64,
    /// Whether further imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Exact unresolved incident when quarantined.
    pub active_incident_digest: Option<[u8; 32]>,
    /// Latest signed resolution in this producer's linked audit history.
    pub last_resolution_digest: Option<[u8; 32]>,
}

/// One bounded page exported from a fully audited producer replica.
///
/// The carrier signs transport metadata separately. Every block in this page
/// remains signed by the original producer and is re-verified by the receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaEvidencePage {
    /// Contiguous producer-signed blocks in ascending height order.
    pub blocks: Vec<DirectoryCommitmentBlockV1>,
    /// Audited accepted producer tip height at export time.
    pub tip_height: u64,
    /// Audited accepted producer tip hash at export time.
    pub tip_hash: [u8; 32],
}

/// One producer-authenticated descriptor proof ready for outbound gossip.
///
/// This value contains only public node-directory evidence. It deliberately
/// excludes carrier identity, selected routes, endpoints outside the signed
/// descriptor, user data, messages, ciphertext, and traffic observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectoryReplicaGossipAnnouncement {
    /// Original Directory block producer.
    pub(crate) producer: [u8; 32],
    /// Exact producer-signed block selected from the audited local replica.
    pub(crate) block_hash: [u8; 32],
    /// Exact authenticated descriptor object hash.
    pub(crate) descriptor_hash: [u8; 32],
    /// Compact producer-signed inclusion proof and descriptor object.
    pub(crate) proof: DirectoryDescriptorInclusionProofV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryReplicaGossipCandidate {
    producer: [u8; 32],
    block_hash: [u8; 32],
    descriptor_hash: [u8; 32],
    descriptor: SignedNodeDescriptor,
}

impl DirectoryReplicaTip {
    const fn empty(producer: [u8; 32]) -> Self {
        Self {
            producer,
            tip_height: 0,
            tip_hash: [0u8; 32],
            tip_timestamp: 0,
            quarantined: false,
            quarantine_kind: None,
            active_incident_digest: None,
            last_resolution_digest: None,
        }
    }
}

/// Node-identity-signed command that resumes one exact quarantined prefix.
///
/// The command cannot select a fork, delete evidence, or rewind a chain. Its
/// compare-and-swap fields bind one immutable incident to the exact prefix and
/// previous resolution history inspected by the host-local operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaResolutionCommand {
    /// Random operator command identifier; unique across this replica store.
    pub command_id: [u8; 16],
    /// Immutable incident explicitly approved by the operator.
    pub incident_digest: [u8; 32],
    /// Producer namespace whose existing prefix may resume synchronization.
    pub producer: [u8; 32],
    /// Accepted prefix height observed before signing.
    pub expected_tip_height: u64,
    /// Accepted prefix hash observed before signing.
    pub expected_tip_hash: [u8; 32],
    /// Quarantine classification observed before signing.
    pub expected_quarantine_kind: String,
    /// Previous linked resolution, or `None` for this producer's first one.
    pub previous_resolution_digest: Option<[u8; 32]>,
    /// Host timestamp at which the operator approved the command.
    pub resolved_at: u64,
    /// Local node identity that must match replica metadata.
    pub resolver_node_id: [u8; 32],
    /// Ed25519 signature over every command field and the fixed action.
    pub signature: [u8; 64],
}

impl DirectoryReplicaResolutionCommand {
    /// Constructs and signs one exact `resume_existing_prefix` command.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when any bounded command field is
    /// invalid. Signing never reads or modifies the replica database.
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        identity: &IdentityKeyPair,
        command_id: [u8; 16],
        incident_digest: [u8; 32],
        producer: [u8; 32],
        expected_tip_height: u64,
        expected_tip_hash: [u8; 32],
        expected_quarantine_kind: String,
        previous_resolution_digest: Option<[u8; 32]>,
        resolved_at: u64,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut command = Self {
            command_id,
            incident_digest,
            producer,
            expected_tip_height,
            expected_tip_hash,
            expected_quarantine_kind,
            previous_resolution_digest,
            resolved_at,
            resolver_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        command.validate_unsigned_fields()?;
        command.signature = identity.sign(&command.signing_bytes());
        Ok(command)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.command_id == [0u8; 16]
            || self.incident_digest == [0u8; 32]
            || self.producer == [0u8; 32]
            || self.resolver_node_id == [0u8; 32]
            || self.resolved_at == 0
            || (self.expected_tip_height == 0 && self.expected_tip_hash != [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution command contains an invalid sentinel".to_string(),
            ));
        }
        validate_incident_kind(&self.expected_quarantine_kind)
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(320);
        bytes.extend_from_slice(b"AeroNyx-DirectoryReplicaResolution-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.command_id);
        bytes.extend_from_slice(&self.incident_digest);
        bytes.extend_from_slice(&self.producer);
        bytes.extend_from_slice(&self.expected_tip_height.to_le_bytes());
        bytes.extend_from_slice(&self.expected_tip_hash);
        bytes.extend_from_slice(&(self.expected_quarantine_kind.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.expected_quarantine_kind.as_bytes());
        match self.previous_resolution_digest {
            Some(digest) => {
                bytes.push(1);
                bytes.extend_from_slice(&digest);
            }
            None => bytes.push(0),
        }
        bytes.extend_from_slice(&self.resolved_at.to_le_bytes());
        bytes.extend_from_slice(&self.resolver_node_id);
        bytes.extend_from_slice(&(DIRECTORY_REPLICA_RESOLUTION_ACTION.len() as u64).to_le_bytes());
        bytes.extend_from_slice(DIRECTORY_REPLICA_RESOLUTION_ACTION.as_bytes());
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Durable result of one successful compare-and-swap resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaResolutionReport {
    /// Content address of the signed resolution audit record.
    pub resolution_digest: [u8; 32],
    /// Unique command identifier supplied by the operator CLI.
    pub command_id: [u8; 16],
    /// Producer namespace that resumed its already accepted prefix.
    pub producer: [u8; 32],
    /// Prefix height retained without rewind or fork selection.
    pub retained_tip_height: u64,
    /// Prefix hash retained without modification.
    pub retained_tip_hash: [u8; 32],
    /// Signed operator approval timestamp.
    pub resolved_at: u64,
}

/// Result of one verified, atomic bounded-page import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaImportReport {
    /// New blocks committed by this transaction.
    pub blocks_inserted: u64,
    /// Exact existing blocks accepted idempotently.
    pub blocks_already_present: u64,
    /// New descriptor commitments committed by this transaction.
    pub commitments_inserted: u64,
    /// Newly recorded same-node/same-sequence descriptor conflicts.
    pub descriptor_equivocations: u64,
    /// Accepted producer prefix height after import.
    pub tip_height: u64,
    /// Accepted producer prefix hash after import.
    pub tip_hash: [u8; 32],
}

/// Restart-durable producer-local synchronization failure state.
///
/// The state contains bounded control-plane scheduling metadata only. It never
/// contains endpoints, response bodies, descriptors, routes, payloads, client
/// identifiers, private keys, wallet traffic, or social graph data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaRetryState {
    /// Remote producer identity used as the local scheduling key.
    pub producer: [u8; 32],
    /// Consecutive failures since the last authenticated successful page.
    pub consecutive_failures: u64,
    /// Earliest Unix timestamp at which another pull may begin.
    pub retry_not_before: Option<u64>,
    /// Timestamp of the most recent failed pull.
    pub last_failure_at: u64,
    /// Stable bounded internal failure bucket.
    pub last_failure_reason: String,
    /// Number of timer rounds skipped while durable backoff was active.
    pub backoff_skips: u64,
}

/// Terminal result of one bounded witness availability-recovery operation.
///
/// [WITNESS-TERMINAL-STATE 2026-07-29 by Codex] Event order is represented by
/// the last assigned enum, not inferred from second-granularity timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessRecoveryOutcome {
    /// An exact carrier envelope was verified before inner witness validation.
    Recovered,
    /// Every selected availability route was exhausted.
    Exhausted,
    /// A canonical, signature, admission, or target contract check stopped closed.
    FailedClosed,
}

impl DirectoryObservationWitnessRecoveryOutcome {
    /// Stable public outcome label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recovered => "recovered",
            Self::Exhausted => "exhausted",
            Self::FailedClosed => "failed_closed",
        }
    }
}

/// Current process-local witness availability-recovery health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessRecoveryHealth {
    /// No recovery operation has reached a terminal result.
    Standby,
    /// The latest terminal recovery result verified a carrier envelope.
    Recovered,
    /// The latest terminal recovery result exhausted every bounded route.
    Exhausted,
    /// The latest terminal recovery result stopped closed.
    FailedClosed,
    /// Mutually exclusive attempt counters no longer sum to attempts.
    Inconsistent,
}

impl DirectoryObservationWitnessRecoveryHealth {
    /// Stable public status label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standby => "standby",
            Self::Recovered => "recovered",
            Self::Exhausted => "exhausted",
            Self::FailedClosed => "failed_closed",
            Self::Inconsistent => "inconsistent",
        }
    }
}

/// Process-lifetime aggregate checkpoint-witness carrier telemetry.
///
/// [WITNESS-CARRIER 2026-07-26 by Codex] These counters describe bounded
/// availability recovery only. They intentionally omit observer, witness,
/// carrier, endpoint, route, descriptor-sequence, checkpoint-hash, and frame
/// data and are never interpreted as authority or reputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessRecoverySnapshot {
    /// Bounded carrier selections after a direct-path availability failure.
    pub selections: u64,
    /// Candidates visible to the latest selection.
    pub latest_candidates: u64,
    /// Latest candidates with fresh local routeability evidence.
    pub latest_routeable_candidates: u64,
    /// Latest candidates skipped by descriptor-sequence capability memory.
    pub latest_capability_cached_unavailable: u64,
    /// Capacity-bounded carriers selected by the latest recovery.
    pub latest_selected: u64,
    /// Total carrier HTTP attempts during this process lifetime.
    pub attempts: u64,
    /// Carrier envelopes successfully verified before inner witness validation.
    pub succeeded: u64,
    /// Explicit optional carrier- or target-route absence outcomes.
    pub capability_unavailable: u64,
    /// Availability-only carrier or target transport failures.
    pub transport_failures: u64,
    /// Recoveries exhausted without a verified carrier envelope.
    pub exhausted: u64,
    /// Canonical, signature, admission, or target-contract failures that stopped closed.
    pub failed_closed: u64,
    /// Latest selection or carrier attempt timestamp.
    pub last_attempt_at: Option<u64>,
    /// Latest verified carrier-envelope timestamp.
    pub last_success_at: Option<u64>,
    /// Latest exhausted or fail-closed recovery timestamp.
    pub last_failure_at: Option<u64>,
    /// Latest terminal recovery result in exact process event order.
    pub last_outcome: Option<DirectoryObservationWitnessRecoveryOutcome>,
}

impl DirectoryObservationWitnessRecoverySnapshot {
    /// Returns the sum of all mutually exclusive carrier-attempt buckets.
    #[must_use]
    pub const fn attempt_outcomes(&self) -> u64 {
        self.succeeded
            .saturating_add(self.capability_unavailable)
            .saturating_add(self.transport_failures)
            .saturating_add(self.failed_closed)
    }

    /// Verifies that each recorded carrier attempt entered exactly one bucket.
    #[must_use]
    pub const fn attempt_outcomes_consistent(&self) -> bool {
        self.attempt_outcomes() == self.attempts
    }

    /// Classifies current recovery health from exact terminal event order.
    #[must_use]
    pub const fn health(&self) -> DirectoryObservationWitnessRecoveryHealth {
        if !self.attempt_outcomes_consistent() {
            DirectoryObservationWitnessRecoveryHealth::Inconsistent
        } else {
            match self.last_outcome {
                None => DirectoryObservationWitnessRecoveryHealth::Standby,
                Some(DirectoryObservationWitnessRecoveryOutcome::Recovered) => {
                    DirectoryObservationWitnessRecoveryHealth::Recovered
                }
                Some(DirectoryObservationWitnessRecoveryOutcome::Exhausted) => {
                    DirectoryObservationWitnessRecoveryHealth::Exhausted
                }
                Some(DirectoryObservationWitnessRecoveryOutcome::FailedClosed) => {
                    DirectoryObservationWitnessRecoveryHealth::FailedClosed
                }
            }
        }
    }
}

/// Process-lifetime aggregate telemetry for this node acting as a witness carrier.
///
/// [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] One authenticated request is
/// reduced to exactly one terminal outcome before entering this snapshot.
/// Requester, witness, endpoint, route, descriptor, checkpoint, frame, digest,
/// signature, and user-plane data are deliberately not accepted by this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessCarrierSnapshot {
    /// Authenticated pinned-requester requests completed by this process.
    pub requests: u64,
    /// Requests that returned an independently verified target-witness frame.
    pub forwarded: u64,
    /// Authenticated requests rejected because the target was not operator-pinned.
    pub policy_rejected: u64,
    /// Authenticated requests whose inner frame failed canonical or signature checks.
    pub invalid_requests: u64,
    /// Requests whose pinned target descriptor, endpoint, or transport was unavailable.
    pub target_unavailable: u64,
    /// Requests whose target explicitly lacked the witness route.
    pub target_capability_unavailable: u64,
    /// Requests explicitly rejected by the target witness.
    pub target_rejected: u64,
    /// Requests whose successful target response failed bounds or verification.
    pub target_invalid_response: u64,
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Requests skipped while
    /// the current target descriptor remained in process-only cooldown.
    pub target_cooling_down: u64,
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Requests rejected
    /// immediately because all bounded local carrier slots were busy.
    pub local_overloaded: u64,
    /// Requests that could not create the bounded local transport client.
    pub local_failures: u64,
    /// Latest authenticated carrier request completion timestamp.
    pub last_request_at: Option<u64>,
    /// Latest independently verified target response timestamp.
    pub last_forwarded_at: Option<u64>,
    /// Latest non-success carrier request timestamp.
    pub last_failure_at: Option<u64>,
    /// Latest request result in exact process event order.
    pub last_outcome: Option<DirectoryObservationWitnessCarrierOutcome>,
}

/// Stable mutually exclusive carrier-side request outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessCarrierOutcome {
    /// Exact target-witness response verified and returned.
    Forwarded,
    /// Target identity was outside the carrier's operator pins.
    PolicyRejected,
    /// Inner observer request failed canonical or signature verification.
    InvalidRequest,
    /// Target descriptor, endpoint, body stream, or transport was unavailable.
    TargetUnavailable,
    /// Target explicitly did not implement the witness route.
    TargetCapabilityUnavailable,
    /// Target rejected the otherwise valid forwarded request.
    TargetRejected,
    /// Target returned an oversized, malformed, or wrongly signed success body.
    TargetInvalidResponse,
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Current target
    /// descriptor remained inside a process-only failure cooldown.
    TargetCoolingDown,
    /// Every bounded local outbound carrier slot was already occupied.
    LocalOverloaded,
    /// Carrier could not initialize its bounded no-proxy transport client.
    LocalFailure,
}

impl DirectoryObservationWitnessCarrierOutcome {
    /// Stable public outcome label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forwarded => "forwarded",
            Self::PolicyRejected => "policy_rejected",
            Self::InvalidRequest => "invalid_request",
            Self::TargetUnavailable => "target_unavailable",
            Self::TargetCapabilityUnavailable => "target_capability_unavailable",
            Self::TargetRejected => "target_rejected",
            Self::TargetInvalidResponse => "target_invalid_response",
            Self::TargetCoolingDown => "target_cooling_down",
            Self::LocalOverloaded => "local_overloaded",
            Self::LocalFailure => "local_failure",
        }
    }
}

/// Current process-local health of this node acting as a witness carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessCarrierHealth {
    /// No authenticated carrier request has completed.
    Standby,
    /// The latest authenticated request forwarded an exact verified frame.
    Active,
    /// The latest authenticated request did not forward a verified frame.
    Degraded,
    /// Mutually exclusive request counters no longer sum to requests.
    Inconsistent,
}

impl DirectoryObservationWitnessCarrierHealth {
    /// Stable public status label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standby => "standby",
            Self::Active => "active",
            Self::Degraded => "degraded",
            Self::Inconsistent => "inconsistent",
        }
    }
}

impl DirectoryObservationWitnessCarrierSnapshot {
    /// Returns the sum of all mutually exclusive request outcome buckets.
    #[must_use]
    pub const fn terminal_outcomes(&self) -> u64 {
        self.forwarded
            .saturating_add(self.policy_rejected)
            .saturating_add(self.invalid_requests)
            .saturating_add(self.target_unavailable)
            .saturating_add(self.target_capability_unavailable)
            .saturating_add(self.target_rejected)
            .saturating_add(self.target_invalid_response)
            .saturating_add(self.target_cooling_down)
            .saturating_add(self.local_overloaded)
            .saturating_add(self.local_failures)
    }

    /// Verifies that each completed request entered exactly one outcome bucket.
    #[must_use]
    pub const fn terminal_outcomes_consistent(&self) -> bool {
        self.terminal_outcomes() == self.requests
    }

    /// Classifies carrier health from exact terminal request order.
    #[must_use]
    pub const fn health(&self) -> DirectoryObservationWitnessCarrierHealth {
        if !self.terminal_outcomes_consistent() {
            DirectoryObservationWitnessCarrierHealth::Inconsistent
        } else {
            match self.last_outcome {
                None => DirectoryObservationWitnessCarrierHealth::Standby,
                Some(DirectoryObservationWitnessCarrierOutcome::Forwarded) => {
                    DirectoryObservationWitnessCarrierHealth::Active
                }
                Some(_) => DirectoryObservationWitnessCarrierHealth::Degraded,
            }
        }
    }
}

/// Runtime-only synchronization observation for one pinned producer.
///
/// These fields intentionally contain no endpoint, full response, descriptor,
/// route, payload, client, or wallet metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaSyncObservation {
    /// Pinned producer identity used only for internal status correlation.
    pub producer: [u8; 32],
    /// Most recent bounded pull attempt.
    pub last_attempt_at: Option<u64>,
    /// Most recent authenticated successful page.
    pub last_success_at: Option<u64>,
    /// Most recent rejected or failed page.
    pub last_failure_at: Option<u64>,
    /// Stable privacy-safe reason code from the most recent failure.
    pub last_failure_reason: Option<String>,
    /// Earliest Unix timestamp at which this producer may be attempted again.
    pub retry_not_before: Option<u64>,
    /// Signed remote tip height most recently observed.
    pub remote_tip_height: Option<u64>,
    /// Accepted local replica height after the most recent success.
    pub local_tip_height: u64,
    /// Whether the most recent signed response indicated additional pages.
    pub has_more: bool,
    /// Consecutive failed attempts since the last successful page.
    pub consecutive_failures: u64,
    /// Total bounded attempts during this process lifetime.
    pub total_attempts: u64,
    /// Total authenticated pages accepted during this process lifetime.
    pub successful_pages: u64,
    /// Total failed attempts during this process lifetime.
    pub failed_attempts: u64,
    /// Total scheduled rounds skipped while this producer was in backoff.
    pub backoff_skips: u64,
    /// Total new blocks committed during this process lifetime.
    pub blocks_inserted: u64,
    /// Total new commitments committed during this process lifetime.
    pub commitments_inserted: u64,
    /// Total HTTP requests consumed by authenticated successful pages.
    pub requests_sent: u64,
}

/// Aggregate process-lifetime Full-node Mirror scheduling telemetry.
///
/// This intentionally omits identities, endpoints, descriptor hashes, routes,
/// and response details. Mirror observations are diagnostic transport evidence,
/// never authority, consensus, fork choice, voting, or finality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryFullNodeMirrorRuntimeSnapshot {
    /// Completed bounded mirror-selection rounds.
    pub rounds: u64,
    /// Valid public candidates considered by the latest round.
    pub last_round_candidates: u64,
    /// Capacity-bounded candidates selected by the latest round.
    pub last_round_selected: u64,
    /// Selected producers that completed without a terminal failure.
    pub last_round_succeeded: u64,
    /// Selected candidates that failed or were rejected in the latest round.
    pub last_round_failed: u64,
    /// Selected producers that reached their authenticated signed tip.
    pub last_round_converged: u64,
    /// Selected producers that advanced but still have authenticated lag.
    pub last_round_catching_up: u64,
    /// Authenticated pages accepted in the latest round.
    pub last_round_pages_succeeded: u64,
    /// Successful HTTP requests consumed by the latest round.
    pub last_round_requests_sent: u64,
    /// Authenticated mirror pages accepted during this process lifetime.
    pub pages_succeeded: u64,
    /// Successful mirror HTTP requests during this process lifetime.
    pub requests_sent: u64,
    /// Failed bounded mirror attempts during this process lifetime.
    pub attempts_failed: u64,
    /// Direct-page failures that entered bounded carrier recovery.
    pub recovery_attempts: u64,
    /// Mirror pages recovered through an independently authenticated carrier.
    pub recovery_succeeded: u64,
    /// Bounded carrier recovery attempts that exhausted or failed closed.
    pub recovery_failed: u64,
    /// Valid non-quarantined public carriers considered by the latest recovery.
    pub last_recovery_carrier_candidates: u64,
    /// Latest candidates with fresh local routeability evidence.
    pub last_recovery_routeable_carrier_candidates: u64,
    /// Latest candidates with a signed Directory Mirror carrier capability.
    pub last_recovery_explicit_capability_candidates: u64,
    /// Latest candidates retained only as unadvertised compatibility fallback.
    ///
    /// These peers may be old or may have deliberately opted out. They remain
    /// discoverable but are never classified as capability supporting until a
    /// signed descriptor explicitly says so.
    pub last_recovery_unadvertised_compatibility_candidates: u64,
    /// Latest valid carriers skipped for the exact signed descriptor sequence
    /// after an explicit optional replica-endpoint absence response.
    pub last_recovery_capability_cached_unavailable: u64,
    /// Capacity-bounded carriers selected by the latest recovery.
    pub last_recovery_carriers_selected: u64,
    /// Latest selected carriers with fresh local routeability evidence.
    pub last_recovery_routeable_carriers_selected: u64,
    /// Latest selected carriers with explicit signed capability.
    pub last_recovery_explicit_capability_selected: u64,
    /// Latest selected unadvertised compatibility fallbacks.
    pub last_recovery_unadvertised_compatibility_selected: u64,
    /// Latest selected carriers that supplied a non-empty signed region hint.
    pub last_recovery_selected_region_hints: u64,
    /// Distinct signed region hints in the latest bounded selection.
    ///
    /// This is an availability hint only, not operator, ASN, jurisdiction,
    /// identity, Sybil-resistance, consensus, or finality evidence.
    pub last_recovery_distinct_region_hints: u64,
    /// Timestamp of the latest completed mirror round.
    pub last_round_at: Option<u64>,
    /// Timestamp of the latest authenticated mirror import.
    pub last_success_at: Option<u64>,
    /// Timestamp of the latest failed mirror attempt.
    pub last_failure_at: Option<u64>,
    /// Timestamp of the latest bounded carrier recovery attempt.
    pub last_recovery_attempt_at: Option<u64>,
    /// Timestamp of the latest successful carrier recovery.
    pub last_recovery_success_at: Option<u64>,
    /// Timestamp of the latest failed or exhausted carrier recovery.
    pub last_recovery_failure_at: Option<u64>,
}

impl DirectoryReplicaSyncObservation {
    fn new(producer: [u8; 32]) -> Self {
        Self {
            producer,
            last_attempt_at: None,
            last_success_at: None,
            last_failure_at: None,
            last_failure_reason: None,
            retry_not_before: None,
            remote_tip_height: None,
            local_tip_height: 0,
            has_more: false,
            consecutive_failures: 0,
            total_attempts: 0,
            successful_pages: 0,
            failed_attempts: 0,
            backoff_skips: 0,
            blocks_inserted: 0,
            commitments_inserted: 0,
            requests_sent: 0,
        }
    }
}

/// Shared process-lifetime synchronization and witness telemetry.
#[derive(Debug, Default)]
pub struct DirectoryReplicaSyncRuntime {
    observations: Mutex<HashMap<[u8; 32], DirectoryReplicaSyncObservation>>,
    directory_sync_transport: Mutex<DirectoryReplicaTransportRuntime>,
    observation_witness: Mutex<DirectoryObservationWitnessOutcomeSnapshot>,
    observation_witness_recovery: Mutex<DirectoryObservationWitnessRecoverySnapshot>,
    observation_witness_carrier: Mutex<DirectoryObservationWitnessCarrierSnapshot>,
    full_node_mirror: Mutex<DirectoryFullNodeMirrorRuntimeSnapshot>,
    directory_audit_admission: OnceLock<Arc<Semaphore>>,
}

impl DirectoryReplicaSyncRuntime {
    /// Returns the process-shared fail-fast admission gate for history audits.
    ///
    /// [DIRECTORY-AUDIT-OWNERSHIP 2026-08-12 by Codex] Both public and local
    /// routers are built from this runtime. Lazy initialization keeps
    /// compatibility constructors cheap while ensuring their production
    /// clones converge on exactly one permit.
    pub(crate) fn directory_audit_admission(&self) -> Arc<Semaphore> {
        Arc::clone(
            self.directory_audit_admission
                .get_or_init(|| Arc::new(Semaphore::new(MAX_DIRECTORY_AUDITS_IN_FLIGHT))),
        )
    }

    /// Records exactly one completed coordinator-owned transport outcome.
    ///
    /// The caller has already reduced the exchange to a privacy-safe class;
    /// this boundary cannot receive peer, endpoint, request, or payload data.
    pub fn record_directory_sync_transport_outcome(
        &self,
        outcome: DirectoryReplicaTransportOutcome,
        completed_at: u64,
    ) {
        self.directory_sync_transport
            .lock()
            .record(outcome, completed_at);
    }

    /// Returns process-lifetime aggregate Directory sync transport telemetry.
    #[must_use]
    pub fn directory_sync_transport_snapshot(&self) -> DirectoryReplicaTransportSnapshot {
        self.directory_sync_transport.lock().snapshot()
    }

    /// Registers configured pins so status reports can distinguish pending from
    /// disabled before the first low-frequency synchronization round.
    pub fn register_producers(&self, producers: &[[u8; 32]]) {
        let mut observations = self.observations.lock();
        for producer in producers {
            if *producer != [0u8; 32] {
                observations
                    .entry(*producer)
                    .or_insert_with(|| DirectoryReplicaSyncObservation::new(*producer));
            }
        }
    }

    /// Restores audited retry boundaries before the coordinator starts.
    ///
    /// Process-lifetime attempt/page counters remain zero after restart; only
    /// the active failure streak, retry boundary, reason, and skip count are
    /// restored because those fields control request pressure.
    pub fn restore_retry_states(&self, states: &[DirectoryReplicaRetryState]) {
        let mut observations = self.observations.lock();
        for state in states {
            let observation = observations
                .entry(state.producer)
                .or_insert_with(|| DirectoryReplicaSyncObservation::new(state.producer));
            observation.last_attempt_at = Some(state.last_failure_at);
            observation.last_failure_at = Some(state.last_failure_at);
            observation.last_failure_reason = Some(state.last_failure_reason.clone());
            observation.retry_not_before = state.retry_not_before;
            observation.consecutive_failures = state.consecutive_failures;
            observation.backoff_skips = state.backoff_skips;
        }
        drop(observations);
    }

    /// Records the beginning of one bounded page request.
    pub fn record_attempt(&self, producer: [u8; 32], attempted_at: u64) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(attempted_at);
        observation.total_attempts = observation.total_attempts.saturating_add(1);
    }

    /// Records one authenticated page after its atomic import completes.
    #[allow(clippy::too_many_arguments)]
    pub fn record_success(
        &self,
        producer: [u8; 32],
        succeeded_at: u64,
        local_tip_height: u64,
        remote_tip_height: u64,
        has_more: bool,
        blocks_inserted: u64,
        commitments_inserted: u64,
        requests_sent: u32,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(succeeded_at);
        observation.last_success_at = Some(succeeded_at);
        observation.remote_tip_height = Some(remote_tip_height);
        observation.local_tip_height = local_tip_height;
        observation.has_more = has_more;
        observation.consecutive_failures = 0;
        observation.retry_not_before = None;
        observation.successful_pages = observation.successful_pages.saturating_add(1);
        observation.blocks_inserted = observation.blocks_inserted.saturating_add(blocks_inserted);
        observation.commitments_inserted = observation
            .commitments_inserted
            .saturating_add(commitments_inserted);
        observation.requests_sent = observation
            .requests_sent
            .saturating_add(u64::from(requests_sent));
    }

    /// Records one stable failure code without retaining peer endpoints,
    /// response bodies, or underlying transport error strings.
    pub fn record_failure(
        &self,
        producer: [u8; 32],
        failed_at: u64,
        reason: &str,
        retry_not_before: Option<u64>,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(failed_at);
        observation.last_failure_at = Some(failed_at);
        observation.last_failure_reason = Some(reason.chars().take(96).collect());
        observation.retry_not_before = retry_not_before.map(|value| value.max(failed_at));
        observation.consecutive_failures = observation
            .consecutive_failures
            .saturating_add(1)
            .min(DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES);
        observation.failed_attempts = observation.failed_attempts.saturating_add(1);
        drop(observations);
    }

    /// Returns the future retry boundary for one producer, if backoff is active.
    #[must_use]
    pub fn deferred_retry_until(&self, producer: &[u8; 32], now: u64) -> Option<u64> {
        self.observations
            .lock()
            .get(producer)
            .and_then(|observation| observation.retry_not_before)
            .filter(|retry_at| *retry_at > now)
    }

    /// Returns the current consecutive failure count for backoff calculation.
    #[must_use]
    pub fn consecutive_failures(&self, producer: &[u8; 32]) -> u64 {
        self.observations
            .lock()
            .get(producer)
            .map_or(0, |observation| observation.consecutive_failures)
    }

    /// Records one timer tick intentionally skipped by producer-local backoff.
    pub fn record_backoff_skip(&self, producer: [u8; 32]) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.backoff_skips = observation.backoff_skips.saturating_add(1);
        drop(observations);
    }

    /// Records one bounded outbound witness round using mutually exclusive,
    /// privacy-safe outcome buckets.
    ///
    /// `telemetry_durable` describes only the aggregate telemetry write. An
    /// accepted attempt already means its signed receipt was persisted.
    pub fn record_observation_witness_round(
        &self,
        checkpoint_sequence: u64,
        observed_at: u64,
        outcomes: &[DirectoryObservationWitnessOutcome],
        telemetry_durable: bool,
    ) {
        if checkpoint_sequence == 0 || observed_at == 0 || outcomes.is_empty() {
            return;
        }
        let round = DirectoryObservationWitnessOutcomeCounters::from_outcomes(outcomes);
        let mut snapshot = self.observation_witness.lock();
        snapshot.rounds = snapshot.rounds.saturating_add(1);
        snapshot.totals = snapshot.totals.saturating_add(round);
        snapshot.last_checkpoint_sequence = checkpoint_sequence;
        snapshot.last_round_at = Some(observed_at);
        if round.accepted > 0 {
            snapshot.last_success_at = Some(observed_at);
        }
        if round.failures() > 0 {
            snapshot.last_failure_at = Some(observed_at);
        }
        snapshot.last_round = round;
        if !telemetry_durable {
            snapshot.telemetry_persistence_failures =
                snapshot.telemetry_persistence_failures.saturating_add(1);
        }
    }

    /// Returns process-lifetime aggregate witness telemetry.
    #[must_use]
    pub fn observation_witness_snapshot(&self) -> DirectoryObservationWitnessOutcomeSnapshot {
        *self.observation_witness.lock()
    }

    /// Records aggregate properties of one bounded witness-carrier selection.
    ///
    /// Identity-bearing candidate data is reduced to counts before this
    /// boundary and is never retained by runtime status.
    pub fn record_observation_witness_recovery_selection(
        &self,
        candidates: u64,
        routeable_candidates: u64,
        capability_cached_unavailable: u64,
        selected: u64,
        attempted_at: u64,
    ) {
        if attempted_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.selections = snapshot.selections.saturating_add(1);
        snapshot.latest_candidates = candidates;
        snapshot.latest_routeable_candidates = routeable_candidates.min(candidates);
        snapshot.latest_capability_cached_unavailable =
            capability_cached_unavailable.min(candidates);
        snapshot.latest_selected = selected.min(candidates);
        snapshot.last_attempt_at = Some(latest_runtime_timestamp(
            snapshot.last_attempt_at,
            attempted_at,
        ));
    }

    /// Records one carrier transport attempt using mutually exclusive buckets.
    pub fn record_observation_witness_recovery_attempt(
        &self,
        succeeded: bool,
        capability_unavailable: bool,
        completed_at: u64,
    ) {
        if completed_at == 0 || (succeeded && capability_unavailable) {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.attempts = snapshot.attempts.saturating_add(1);
        snapshot.last_attempt_at = Some(latest_runtime_timestamp(
            snapshot.last_attempt_at,
            completed_at,
        ));
        if succeeded {
            snapshot.succeeded = snapshot.succeeded.saturating_add(1);
            snapshot.last_success_at = Some(latest_runtime_timestamp(
                snapshot.last_success_at,
                completed_at,
            ));
            snapshot.last_outcome = Some(DirectoryObservationWitnessRecoveryOutcome::Recovered);
        } else if capability_unavailable {
            snapshot.capability_unavailable = snapshot.capability_unavailable.saturating_add(1);
        } else {
            snapshot.transport_failures = snapshot.transport_failures.saturating_add(1);
        }
    }

    /// Records one bounded recovery that exhausted every selected carrier.
    pub fn record_observation_witness_recovery_exhausted(&self, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.exhausted = snapshot.exhausted.saturating_add(1);
        snapshot.last_failure_at = Some(latest_runtime_timestamp(
            snapshot.last_failure_at,
            completed_at,
        ));
        snapshot.last_outcome = Some(DirectoryObservationWitnessRecoveryOutcome::Exhausted);
    }

    /// Records a carrier or target contract failure that stopped closed.
    pub fn record_observation_witness_recovery_failed_closed(&self, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.attempts = snapshot.attempts.saturating_add(1);
        snapshot.failed_closed = snapshot.failed_closed.saturating_add(1);
        snapshot.last_attempt_at = Some(latest_runtime_timestamp(
            snapshot.last_attempt_at,
            completed_at,
        ));
        snapshot.last_failure_at = Some(latest_runtime_timestamp(
            snapshot.last_failure_at,
            completed_at,
        ));
        snapshot.last_outcome = Some(DirectoryObservationWitnessRecoveryOutcome::FailedClosed);
    }

    /// Returns process-lifetime aggregate witness-carrier telemetry.
    #[must_use]
    pub fn observation_witness_recovery_snapshot(
        &self,
    ) -> DirectoryObservationWitnessRecoverySnapshot {
        *self.observation_witness_recovery.lock()
    }

    /// Records one authenticated carrier request as exactly one aggregate outcome.
    ///
    /// The caller must perform pinned requester authentication first. No
    /// identity-bearing or frame-bearing value crosses this runtime boundary.
    pub fn record_observation_witness_carrier_outcome(
        &self,
        outcome: DirectoryObservationWitnessCarrierOutcome,
        completed_at: u64,
    ) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_carrier.lock();
        snapshot.requests = snapshot.requests.saturating_add(1);
        snapshot.last_request_at = Some(latest_runtime_timestamp(
            snapshot.last_request_at,
            completed_at,
        ));
        snapshot.last_outcome = Some(outcome);
        match outcome {
            DirectoryObservationWitnessCarrierOutcome::Forwarded => {
                snapshot.forwarded = snapshot.forwarded.saturating_add(1);
                snapshot.last_forwarded_at = Some(latest_runtime_timestamp(
                    snapshot.last_forwarded_at,
                    completed_at,
                ));
            }
            DirectoryObservationWitnessCarrierOutcome::PolicyRejected => {
                snapshot.policy_rejected = snapshot.policy_rejected.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::InvalidRequest => {
                snapshot.invalid_requests = snapshot.invalid_requests.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable => {
                snapshot.target_unavailable = snapshot.target_unavailable.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable => {
                snapshot.target_capability_unavailable =
                    snapshot.target_capability_unavailable.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetRejected => {
                snapshot.target_rejected = snapshot.target_rejected.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse => {
                snapshot.target_invalid_response =
                    snapshot.target_invalid_response.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetCoolingDown => {
                snapshot.target_cooling_down = snapshot.target_cooling_down.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::LocalOverloaded => {
                snapshot.local_overloaded = snapshot.local_overloaded.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::LocalFailure => {
                snapshot.local_failures = snapshot.local_failures.saturating_add(1);
            }
        }
        if outcome != DirectoryObservationWitnessCarrierOutcome::Forwarded {
            snapshot.last_failure_at = Some(latest_runtime_timestamp(
                snapshot.last_failure_at,
                completed_at,
            ));
        }
    }

    /// Returns process-lifetime aggregate telemetry for this node as a carrier.
    #[must_use]
    pub fn observation_witness_carrier_snapshot(
        &self,
    ) -> DirectoryObservationWitnessCarrierSnapshot {
        *self.observation_witness_carrier.lock()
    }

    /// Records one aggregate bounded non-authoritative mirror round.
    pub fn record_full_node_mirror_round(
        &self,
        candidates: usize,
        selected: usize,
        succeeded: usize,
        completed_at: u64,
    ) {
        self.record_full_node_mirror_catch_up_round(
            candidates,
            selected,
            succeeded,
            0,
            selected.saturating_sub(succeeded),
            u64::try_from(succeeded).unwrap_or(u64::MAX),
            0,
            completed_at,
        );
    }

    /// Records one aggregate bounded multi-page mirror catch-up round.
    ///
    /// [MIRROR-CATCHUP 2026-07-24 by Codex] Outcome buckets are mutually
    /// exclusive at producer level. Page/request totals remain aggregate-only
    /// and cannot reveal which public producer or carrier served the data.
    #[allow(clippy::too_many_arguments)]
    pub fn record_full_node_mirror_catch_up_round(
        &self,
        candidates: usize,
        selected: usize,
        converged: usize,
        catching_up: usize,
        failed: usize,
        pages_succeeded: u64,
        requests_sent: u64,
        completed_at: u64,
    ) {
        if completed_at == 0
            || converged.saturating_add(catching_up).saturating_add(failed) != selected
        {
            return;
        }
        let succeeded = converged.saturating_add(catching_up);
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.rounds = snapshot.rounds.saturating_add(1);
        snapshot.last_round_candidates = u64::try_from(candidates).unwrap_or(u64::MAX);
        snapshot.last_round_selected = u64::try_from(selected).unwrap_or(u64::MAX);
        snapshot.last_round_succeeded = u64::try_from(succeeded).unwrap_or(u64::MAX);
        snapshot.last_round_failed = u64::try_from(failed).unwrap_or(u64::MAX);
        snapshot.last_round_converged = u64::try_from(converged).unwrap_or(u64::MAX);
        snapshot.last_round_catching_up = u64::try_from(catching_up).unwrap_or(u64::MAX);
        snapshot.last_round_pages_succeeded = pages_succeeded;
        snapshot.last_round_requests_sent = requests_sent;
        snapshot.pages_succeeded = snapshot.pages_succeeded.saturating_add(pages_succeeded);
        snapshot.requests_sent = snapshot.requests_sent.saturating_add(requests_sent);
        snapshot.attempts_failed = snapshot
            .attempts_failed
            .saturating_add(snapshot.last_round_failed);
        snapshot.last_round_at = Some(completed_at);
        if pages_succeeded > 0 {
            snapshot.last_success_at = Some(completed_at);
        }
        if failed > 0 {
            snapshot.last_failure_at = Some(completed_at);
        }
    }

    /// Records one privacy-safe carrier recovery outcome.
    ///
    /// Carrier identities, endpoints, routes, producer identities, response
    /// bodies, and failure strings are intentionally excluded.
    pub fn record_full_node_mirror_recovery(&self, succeeded: bool, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.recovery_attempts = snapshot.recovery_attempts.saturating_add(1);
        snapshot.last_recovery_attempt_at = Some(completed_at);
        if succeeded {
            snapshot.recovery_succeeded = snapshot.recovery_succeeded.saturating_add(1);
            snapshot.last_recovery_success_at = Some(completed_at);
        } else {
            snapshot.recovery_failed = snapshot.recovery_failed.saturating_add(1);
            snapshot.last_recovery_failure_at = Some(completed_at);
        }
    }

    /// Records aggregate properties of the latest bounded carrier selection.
    ///
    /// [MIRROR-CAPABILITY 2026-07-24 by Codex] Capability and signed-region
    /// values are reduced to counts before this boundary. This method never
    /// receives identities, endpoint URLs, region strings, producer
    /// identities, descriptor sequences, or selected order.
    pub fn record_full_node_mirror_carrier_selection(
        &self,
        candidates: u64,
        routeable_candidates: u64,
        explicit_capability_candidates: u64,
        unadvertised_compatibility_candidates: u64,
        capability_cached_unavailable: u64,
        selected: u64,
        selected_routeable: u64,
        selected_explicit_capability: u64,
        selected_unadvertised_compatibility: u64,
        selected_region_hints: u64,
        distinct_region_hints: u64,
    ) {
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.last_recovery_carrier_candidates = candidates;
        snapshot.last_recovery_routeable_carrier_candidates = routeable_candidates.min(candidates);
        snapshot.last_recovery_explicit_capability_candidates =
            explicit_capability_candidates.min(candidates);
        snapshot.last_recovery_unadvertised_compatibility_candidates =
            unadvertised_compatibility_candidates.min(
                candidates.saturating_sub(snapshot.last_recovery_explicit_capability_candidates),
            );
        snapshot.last_recovery_capability_cached_unavailable =
            capability_cached_unavailable.min(candidates);
        snapshot.last_recovery_carriers_selected = selected.min(candidates);
        snapshot.last_recovery_routeable_carriers_selected = selected_routeable
            .min(snapshot.last_recovery_carriers_selected)
            .min(snapshot.last_recovery_routeable_carrier_candidates);
        snapshot.last_recovery_explicit_capability_selected =
            selected_explicit_capability.min(snapshot.last_recovery_carriers_selected);
        snapshot.last_recovery_unadvertised_compatibility_selected =
            selected_unadvertised_compatibility.min(
                snapshot
                    .last_recovery_carriers_selected
                    .saturating_sub(snapshot.last_recovery_explicit_capability_selected),
            );
        snapshot.last_recovery_selected_region_hints =
            selected_region_hints.min(snapshot.last_recovery_carriers_selected);
        snapshot.last_recovery_distinct_region_hints =
            distinct_region_hints.min(snapshot.last_recovery_selected_region_hints);
    }

    /// Returns aggregate permissionless mirror telemetry without peer metadata.
    #[must_use]
    pub fn full_node_mirror_snapshot(&self) -> DirectoryFullNodeMirrorRuntimeSnapshot {
        *self.full_node_mirror.lock()
    }

    /// Returns producer observations in deterministic identity order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DirectoryReplicaSyncObservation> {
        let mut observations = self
            .observations
            .lock()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        observations.sort_by_key(|observation| observation.producer);
        observations
    }
}

/// One commitment row read during an audit, checked for everything except the
/// descriptor signature, which is verified later in a parallel batch.
///
/// [PARALLEL-DIRECTORY-AUDIT 2026-10-10 by Claude]
#[derive(Debug)]
pub(super) struct PendingReplicaCommitment {
    pub(super) commitment: DirectoryDescriptorCommitmentV1,
    pub(super) object_node_id: [u8; 32],
    pub(super) object_sequence: u64,
    pub(super) descriptor_blob: Vec<u8>,
}

/// Replica blocks read and verified per batch during an audit.
const AUDIT_REPLICA_BLOCK_BATCH: usize = 512;

#[derive(Debug)]
struct StoredReplicaBlockRow {
    height: i64,
    block_hash: Vec<u8>,
    prev_block_hash: Vec<u8>,
    produced_at: i64,
    commitment_count: i64,
    block_blob: Vec<u8>,
}

#[derive(Debug)]
struct QuarantineIncident<'a> {
    kind: &'a str,
    height: u64,
    local_hash: [u8; 32],
    remote_hash: [u8; 32],
    evidence_frame: &'a [u8],
}

#[derive(Debug)]
struct StoredResolutionRow {
    digest: Vec<u8>,
    command_id: Vec<u8>,
    incident_digest: Vec<u8>,
    producer: Vec<u8>,
    action: String,
    expected_tip_height: i64,
    expected_tip_hash: Vec<u8>,
    expected_quarantine_kind: String,
    previous_resolution_digest: Option<Vec<u8>>,
    resolved_at: i64,
    resolver_node_id: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationCheckpointRow {
    sequence: i64,
    checkpoint_hash: Vec<u8>,
    previous_checkpoint_hash: Vec<u8>,
    observed_at: i64,
    observation_root: Vec<u8>,
    producer_count: i64,
    checkpoint_blob: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationWitnessRow {
    checkpoint_hash: Vec<u8>,
    checkpoint_sequence: i64,
    observer: Vec<u8>,
    witness_node_id: Vec<u8>,
    witnessed_at: i64,
    response_blob: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationWitnessOutcomeRow {
    rounds: i64,
    attempts: i64,
    totals: [i64; 7],
    last_checkpoint_sequence: i64,
    last_round_at: i64,
    last_success_at: Option<i64>,
    last_failure_at: Option<i64>,
    last_round_attempts: i64,
    last_round: [i64; 7],
    updated_at: i64,
}

#[derive(Debug)]
struct StoredObservationWitnessPolicyRow {
    epoch: i64,
    policy_digest: Vec<u8>,
    previous_policy_digest: Vec<u8>,
    activated_at: i64,
    witness_threshold: i64,
    witness_count: i64,
    witness_node_ids: Vec<u8>,
    signer_node_id: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Debug)]
struct StoredRouteDomainPolicyRow {
    epoch: i64,
    policy_digest: Vec<u8>,
    previous_policy_digest: Vec<u8>,
    activated_at: i64,
    strict_required: i64,
    assignment_count: i64,
    assignments: Vec<u8>,
    signer_node_id: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Debug)]
struct StoredRouteDomainAttestorPolicyRow {
    epoch: i64,
    policy_digest: Vec<u8>,
    previous_policy_digest: Vec<u8>,
    activated_at: i64,
    strict_required: i64,
    attestor_threshold: i64,
    attestor_count: i64,
    attestor_node_ids: Vec<u8>,
    signer_node_id: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationWitnessPolicyAnchorReceiptRow {
    policy_epoch: i64,
    policy_digest: Vec<u8>,
    observer: Vec<u8>,
    witness_node_id: Vec<u8>,
    witnessed_at: i64,
    response_blob: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationCertificateImportRow {
    import_sequence: i64,
    import_digest: Vec<u8>,
    previous_import_digest: Vec<u8>,
    certificate_id: Vec<u8>,
    observer: Vec<u8>,
    checkpoint_sequence: i64,
    checkpoint_hash: Vec<u8>,
    checkpoint_observed_at: i64,
    certificate_sha256: Vec<u8>,
    certificate_frame: Vec<u8>,
    policy_digest: Vec<u8>,
    policy_minimum_witnesses: i64,
    policy_witness_count: i64,
    policy_witness_node_ids: Vec<u8>,
    verified_at: i64,
    importer_node_id: Vec<u8>,
    signature: Vec<u8>,
}

/// One local-node-signed link in the imported certificate history.
///
/// The signature does not promote foreign evidence into consensus. It proves
/// only which exact bytes and local trust policy this node accepted, and where
/// that decision sits in this node's append-only history.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryObservationCertificateImportEntry {
    import_sequence: u64,
    previous_import_digest: [u8; 32],
    certificate_id: [u8; 32],
    observer: [u8; 32],
    checkpoint_sequence: u64,
    checkpoint_hash: [u8; 32],
    checkpoint_observed_at: u64,
    certificate_sha256: [u8; 32],
    policy_digest: [u8; 32],
    verified_at: u64,
    importer_node_id: [u8; 32],
    signature: [u8; 64],
}

impl DirectoryObservationCertificateImportEntry {
    fn sign(
        identity: &IdentityKeyPair,
        import_sequence: u64,
        previous_import_digest: [u8; 32],
        verified: &VerifiedDirectoryObservationCertificate,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let checkpoint = &verified.certificate.checkpoint;
        let mut entry = Self {
            import_sequence,
            previous_import_digest,
            certificate_id: verified.certificate_id,
            observer: checkpoint.observer,
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash: checkpoint.hash(),
            checkpoint_observed_at: checkpoint.observed_at,
            certificate_sha256: verified.certificate_sha256,
            policy_digest: verified.policy_digest,
            verified_at: verified.verified_at,
            importer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        entry.validate_unsigned_fields()?;
        entry.signature = identity.sign(&entry.signing_bytes());
        Ok(entry)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.import_sequence == 0
            || self.certificate_id == [0u8; 32]
            || self.observer == [0u8; 32]
            || self.checkpoint_sequence == 0
            || self.checkpoint_hash == [0u8; 32]
            || self.checkpoint_observed_at == 0
            || self.certificate_sha256 == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.verified_at == 0
            || self.importer_node_id == [0u8; 32]
            || self.checkpoint_observed_at
                > self
                    .verified_at
                    .saturating_add(DIRECTORY_OBSERVATION_CERTIFICATE_IMPORT_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation certificate import contains an invalid sentinel".to_string(),
            ));
        }
        Ok(())
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(360);
        bytes.extend_from_slice(b"AeroNyx-DirectoryObservationCertificateImport-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.import_sequence.to_le_bytes());
        bytes.extend_from_slice(&self.previous_import_digest);
        bytes.extend_from_slice(&self.certificate_id);
        bytes.extend_from_slice(&self.observer);
        bytes.extend_from_slice(&self.checkpoint_sequence.to_le_bytes());
        bytes.extend_from_slice(&self.checkpoint_hash);
        bytes.extend_from_slice(&self.checkpoint_observed_at.to_le_bytes());
        bytes.extend_from_slice(&self.certificate_sha256);
        bytes.extend_from_slice(&self.policy_digest);
        bytes.extend_from_slice(&self.verified_at.to_le_bytes());
        bytes.extend_from_slice(&self.importer_node_id);
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ObservationCertificateImportAudit {
    imports: u64,
    head: [u8; 32],
}

/// One node-identity-signed, hash-linked local witness admission policy.
///
/// This is operator configuration history, not a network vote, validator set,
/// fork-choice rule, consensus object, or finality certificate. Full member
/// identities remain in the host-local database and are never returned by the
/// public aggregate status endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryObservationWitnessPolicyEpoch {
    epoch: u64,
    previous_policy_digest: [u8; 32],
    activated_at: u64,
    witness_node_ids: Vec<[u8; 32]>,
    minimum_witnesses: usize,
    signer_node_id: [u8; 32],
    signature: [u8; 64],
}

impl DirectoryObservationWitnessPolicyEpoch {
    fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        witness_node_ids: Vec<[u8; 32]>,
        minimum_witnesses: usize,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            witness_node_ids,
            minimum_witnesses,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_observation_witness_policy_members(&self.witness_node_ids, self.minimum_witnesses)
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192 + self.witness_node_ids.len() * 32);
        bytes.extend_from_slice(b"AeroNyx-DirectoryObservationWitnessPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.extend_from_slice(&(self.minimum_witnesses as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.witness_node_ids.len() as u64).to_le_bytes());
        for witness_node_id in &self.witness_node_ids {
            bytes.extend_from_slice(witness_node_id);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling validated runtime pins into the signed local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectoryObservationWitnessPolicyReconcileReport {
    /// True only when pins or threshold created a new durable epoch.
    pub(crate) appended: bool,
    /// Current local policy epoch after reconciliation.
    pub(crate) epoch: u64,
    /// Content digest of the current signed policy.
    pub(crate) policy_digest: [u8; 32],
    /// Timestamp bound into the current policy.
    pub(crate) activated_at: u64,
    /// Number of canonical current witness pins.
    pub(crate) witness_members: u64,
    /// Required distinct external receipts.
    pub(crate) minimum_witnesses: u64,
}

/// One node-identity-signed, hash-linked local route-domain policy epoch.
///
/// [ROUTE-DOMAIN-POLICY-HISTORY 2026-08-03 by Codex] Assignments are opaque
/// operator-reviewed failure-domain groups. The signature proves what this
/// node configured and when; it does not prove legal ownership, ASN identity,
/// physical independence, honest operation, consensus, or Sybil resistance.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryRouteDomainPolicyEpoch {
    epoch: u64,
    previous_policy_digest: [u8; 32],
    activated_at: u64,
    strict_required: bool,
    assignments: Vec<PinnedRouteDomainAssignment>,
    signer_node_id: [u8; 32],
    signature: [u8; 64],
}

impl DirectoryRouteDomainPolicyEpoch {
    fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        strict_required: bool,
        assignments: Vec<PinnedRouteDomainAssignment>,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            strict_required,
            assignments,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_route_domain_policy_assignments(&self.assignments, self.strict_required)
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192 + self.assignments.len() * 48);
        bytes.extend_from_slice(b"AeroNyx-DirectoryRouteDomainPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.push(u8::from(self.strict_required));
        bytes.extend_from_slice(&(self.assignments.len() as u64).to_le_bytes());
        for assignment in &self.assignments {
            bytes.extend_from_slice(&assignment.node_id);
            bytes.extend_from_slice(&assignment.route_domain);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling runtime route-domain pins into signed local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DirectoryRouteDomainPolicyReconcileReport {
    pub(crate) appended: bool,
    pub(crate) epoch: u64,
    pub(crate) policy_digest: [u8; 32],
    pub(crate) activated_at: u64,
    pub(crate) assignments: u64,
    pub(crate) strict_required: bool,
}

/// One node-identity-signed, hash-linked local route-domain attestor policy.
///
/// [ROUTE-DOMAIN-ATTESTOR-HISTORY 2026-08-03 by Codex] These identities are
/// host-local trust roots for validating portable route-domain attestations.
/// A valid signature proves only authorship under a locally pinned key; this
/// policy does not prove operator independence, geography, ASN ownership,
/// consensus, honest behavior, or Sybil resistance.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryRouteDomainAttestorPolicyEpoch {
    epoch: u64,
    previous_policy_digest: [u8; 32],
    activated_at: u64,
    strict_required: bool,
    attestor_node_ids: Vec<[u8; 32]>,
    minimum_attestors: usize,
    signer_node_id: [u8; 32],
    signature: [u8; 64],
}

impl DirectoryRouteDomainAttestorPolicyEpoch {
    fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        strict_required: bool,
        attestor_node_ids: Vec<[u8; 32]>,
        minimum_attestors: usize,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            strict_required,
            attestor_node_ids,
            minimum_attestors,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_route_domain_attestor_policy_members(
            &self.attestor_node_ids,
            self.minimum_attestors,
            self.strict_required,
        )
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(200 + self.attestor_node_ids.len() * 32);
        bytes.extend_from_slice(b"AeroNyx-DirectoryRouteDomainAttestorPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.push(u8::from(self.strict_required));
        bytes.extend_from_slice(&(self.minimum_attestors as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.attestor_node_ids.len() as u64).to_le_bytes());
        for attestor_node_id in &self.attestor_node_ids {
            bytes.extend_from_slice(attestor_node_id);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling runtime route-domain attestor pins into local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DirectoryRouteDomainAttestorPolicyReconcileReport {
    pub(crate) appended: bool,
    pub(crate) epoch: u64,
    pub(crate) policy_digest: [u8; 32],
    pub(crate) activated_at: u64,
    pub(crate) attestors: u64,
    pub(crate) minimum_attestors: u64,
    pub(crate) strict_required: bool,
}

/// Privacy-bounded current local policy head exported to pinned witnesses.
///
/// The digest commits to the complete node-signed local policy, while member
/// identities remain host-local and never enter the anchor protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectoryObservationWitnessPolicyAnchor {
    pub(crate) epoch: u64,
    pub(crate) previous_policy_digest: [u8; 32],
    pub(crate) policy_digest: [u8; 32],
}

/// Result of evaluating one authenticated foreign policy-head anchor request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectoryObservationWitnessPolicyAnchorDecision {
    /// Exact head was already retained or appended durably.
    Accepted,
    /// Request regressed below the latest retained observer epoch.
    Rollback,
    /// The same epoch was previously retained with another digest.
    Conflict,
    /// A forward request did not link to the immediately retained head.
    HistoryGap,
}

impl DirectoryObservationWitnessPolicyAnchorDecision {
    #[must_use]
    pub(crate) const fn outcome(self) -> u8 {
        match self {
            Self::Accepted => DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            Self::Rollback => DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
            Self::Conflict => DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
            Self::HistoryGap => DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifiedObservationWitness {
    sequence: u64,
    checkpoint_hash: [u8; 32],
    observer: [u8; 32],
    response_timestamp: u64,
    receipt: DirectoryObservationWitnessReceiptV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ObservationCheckpointTip {
    sequence: u64,
    checkpoint_hash: [u8; 32],
    observed_at: u64,
    producer_count: u16,
    observation_root: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ObservationWitnessAudit {
    witnesses: u64,
    latest_sequence: u64,
    latest_witnesses: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ObservationWitnessPolicyAudit {
    epochs: u64,
    current: Option<DirectoryObservationWitnessPolicyEpoch>,
    current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct RouteDomainPolicyAudit {
    epochs: u64,
    current: Option<DirectoryRouteDomainPolicyEpoch>,
    current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct RouteDomainAttestorPolicyAudit {
    epochs: u64,
    current: Option<DirectoryRouteDomainAttestorPolicyEpoch>,
    current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct VerifiedObservationWitnessSet {
    sequence: u64,
    witness_node_ids: Vec<[u8; 32]>,
    receipts: Vec<DirectoryObservationWitnessReceiptV1>,
}

#[derive(Debug, Default)]
struct AuditedResolutionIndex {
    commands: HashMap<[u8; 32], DirectoryReplicaResolutionCommand>,
    by_producer: HashMap<[u8; 32], HashSet<[u8; 32]>>,
    resolved_incidents: HashMap<[u8; 32], HashSet<[u8; 32]>>,
}

/// Durable producer-scoped replica namespace.
pub struct DirectoryReplicaStore {
    connection: Mutex<Connection>,
    path: PathBuf,
    local_node_id: [u8; 32],
}

fn validate_incident_kind(kind: &str) -> Result<(), DirectoryReplicaStoreError> {
    if kind.is_empty()
        || kind.len() > MAX_DIRECTORY_REPLICA_INCIDENT_KIND_BYTES
        || !matches!(
            kind,
            "signed_tip_rollback"
                | "signed_tip_fork"
                | "signed_empty_range_gap"
                | "signed_block_fork"
                | "descriptor_sequence_equivocation"
        )
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "replica incident kind is invalid".to_string(),
        ));
    }
    Ok(())
}

fn observation_convergence_root(
    eligible_tips: &[DirectoryReplicaTip],
    occurrence_by_commitment: &BTreeMap<[u8; 32], u64>,
) -> [u8; 32] {
    debug_assert!(eligible_tips.len() >= 2);
    let eligible_count = eligible_tips.len() as u64;
    let common_count = occurrence_by_commitment
        .values()
        .filter(|occurrence| **occurrence == eligible_count)
        .count() as u64;
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaObservationConvergence-v1");
    hasher.update(AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
    hasher.update(DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS.to_le_bytes());
    hasher.update(eligible_count.to_le_bytes());
    for tip in eligible_tips {
        hasher.update(tip.producer);
        hasher.update(tip.tip_height.to_le_bytes());
        hasher.update(tip.tip_hash);
    }
    hasher.update(common_count.to_le_bytes());
    for (commitment, occurrence) in occurrence_by_commitment {
        if *occurrence == eligible_count {
            hasher.update(commitment);
        }
    }
    hasher.finalize().into()
}

fn verify_incident_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    verify_signed_range_response_evidence(frame, expected_producer).map(|_| ())
}

struct VerifiedRangeResponseEvidence {
    tip_provenance: DirectoryRangeTipProvenance,
    response_timestamp: u64,
    blocks: Vec<DirectoryCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
    tip_hash: [u8; 32],
}

fn verify_signed_range_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<VerifiedRangeResponseEvidence, DirectoryReplicaStoreError> {
    let message = decode_directory_sync_message(frame)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
    if encode_directory_sync_message(&message)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
        != frame
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence frame is not canonical".to_string(),
        ));
    }
    match message {
        DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id,
            request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID || responder != *expected_producer {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "range evidence belongs to another chain or producer".to_string(),
                ));
            }
            let signing_bytes = directory_block_range_response_signing_bytes(
                &request_id,
                &responder,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&responder)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "range evidence producer signature is invalid".to_string(),
                    )
                })?;
            Ok(VerifiedRangeResponseEvidence {
                tip_provenance: DirectoryRangeTipProvenance::ProducerSigned,
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
            })
        }
        DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id,
            request_id,
            producer,
            carrier,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                || producer != *expected_producer
                || carrier == [0u8; 32]
                || carrier == producer
                || blocks.iter().any(|block| block.header.producer != producer)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "carrier range evidence belongs to another chain or producer".to_string(),
                ));
            }
            let signing_bytes = directory_replica_block_range_response_signing_bytes(
                &chain_id,
                &request_id,
                &producer,
                &carrier,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&carrier)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "carrier range evidence signature is invalid".to_string(),
                    )
                })?;
            Ok(VerifiedRangeResponseEvidence {
                tip_provenance: DirectoryRangeTipProvenance::CarrierReported,
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
            })
        }
        _ => Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence is not a supported block-range response".to_string(),
        )),
    }
}

fn verify_range_response_evidence(
    frame: &[u8],
    producer: &[u8; 32],
    expected_blocks: &[DirectoryCommitmentBlockV1],
    expected_tip_height: u64,
    expected_tip_hash: &[u8; 32],
    observed_at: u64,
) -> Result<VerifiedRangeResponseAdmission, DirectoryReplicaStoreError> {
    let verified = verify_signed_range_response_evidence(frame, producer)?;
    if verified.blocks != expected_blocks
        || verified.tip_height != expected_tip_height
        || verified.tip_hash != *expected_tip_hash
        || verified.response_timestamp.abs_diff(observed_at) > RESPONSE_TIMESTAMP_SKEW_SECS
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "signed range evidence does not match the import".to_string(),
        ));
    }
    Ok(VerifiedRangeResponseAdmission {
        has_more: verified.has_more,
        tip_provenance: verified.tip_provenance,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifiedRangeResponseAdmission {
    has_more: bool,
    tip_provenance: DirectoryRangeTipProvenance,
}

fn validate_page_tip_contract(
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    if tip_height == 0 && *tip_hash != [0u8; 32] {
        return Err(DirectoryReplicaStoreError::Integrity(
            "empty advertised tip must use the zero hash".to_string(),
        ));
    }
    let Some(last) = blocks.last() else {
        if has_more {
            return Err(DirectoryReplicaStoreError::Integrity(
                "an empty response cannot advertise more pages".to_string(),
            ));
        }
        return Ok(());
    };
    if last.header.height > tip_height
        || (has_more && last.header.height >= tip_height)
        || (!has_more && (last.header.height != tip_height || last.hash() != *tip_hash))
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "range pagination fields contradict the signed tip".to_string(),
        ));
    }
    Ok(())
}

fn validate_exact_descriptor_objects<'a>(
    blocks: &[DirectoryCommitmentBlockV1],
    objects: &'a [SignedNodeDescriptor],
) -> Result<HashMap<[u8; 32], &'a SignedNodeDescriptor>, DirectoryReplicaStoreError> {
    let required = blocks
        .iter()
        .flat_map(|block| block.commitments.iter().map(|entry| entry.descriptor_hash))
        .collect::<Vec<_>>();
    let required_set = required.iter().copied().collect::<HashSet<_>>();
    if required_set.len() != required.len() || objects.len() != required.len() {
        return Err(DirectoryReplicaStoreError::Request(
            "descriptor objects must exactly cover unique page commitments".to_string(),
        ));
    }
    let mut mapped = HashMap::with_capacity(objects.len());
    for descriptor in objects {
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
        if !required_set.contains(&commitment.descriptor_hash)
            || mapped
                .insert(commitment.descriptor_hash, descriptor)
                .is_some()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "descriptor response contains an extra or duplicate object".to_string(),
            ));
        }
    }
    Ok(mapped)
}

fn incident_digest(
    producer: &[u8; 32],
    subject_node_id: &[u8; 32],
    incident: &QuarantineIncident<'_>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaIncident-v1");
    hasher.update(producer);
    hasher.update(subject_node_id);
    hasher.update((incident.kind.len() as u64).to_le_bytes());
    hasher.update(incident.kind.as_bytes());
    hasher.update(incident.height.to_le_bytes());
    hasher.update(incident.local_hash);
    hasher.update(incident.remote_hash);
    hasher.update((incident.evidence_frame.len() as u64).to_le_bytes());
    hasher.update(incident.evidence_frame);
    hasher.finalize().into()
}

fn encode_observation_checkpoint(
    checkpoint: &DirectoryObservationCheckpointV1,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES)
        .serialize(checkpoint)
        .map_err(|error| {
            DirectoryReplicaStoreError::Codec(format!(
                "encode directory observation checkpoint: {error}"
            ))
        })
}

fn decode_observation_checkpoint(
    bytes: &[u8],
) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
    if bytes.is_empty()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX)
            > MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES
    {
        return Err(DirectoryReplicaStoreError::Codec(
            "directory observation checkpoint size is invalid".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES)
        .deserialize(bytes)
        .map_err(|error| {
            DirectoryReplicaStoreError::Codec(format!(
                "decode directory observation checkpoint: {error}"
            ))
        })
}

fn encode_block(block: &DirectoryCommitmentBlockV1) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .serialize(block)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn decode_block(bytes: &[u8]) -> Result<DirectoryCommitmentBlockV1, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_DIRECTORY_BLOCK_BYTES) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica block exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn encode_descriptor_object(
    descriptor: &SignedNodeDescriptor,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .serialize(descriptor)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn decode_descriptor_object(
    bytes: &[u8],
) -> Result<SignedNodeDescriptor, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| {
        length > MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES
    }) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica descriptor object exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

/// Converts a SQL-size-admitted payload without ever accepting a missing,
/// oversized, or length-inconsistent durable value.
///
/// The supplying query must project `length(blob)` and
/// `CASE WHEN length(blob) <= ? THEN blob END`. SQLite can answer `length()`
/// for a BLOB without first returning the BLOB bytes; the `CASE` keeps an
/// oversized payload out of `row.get::<Vec<u8>>()` entirely. The codec limits
/// above remain an independent second layer after this storage admission.
fn materialize_admitted_replica_blob(
    stored_length: Option<i64>,
    admitted_blob: Option<Vec<u8>>,
    kind: PersistedReplicaBlobKind,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    let stored_length = stored_length
        .ok_or_else(|| DirectoryReplicaStoreError::Integrity(kind.missing_message().to_string()))?;
    let stored_length = nonnegative_i64_to_u64(stored_length, kind.length_field())?;
    if stored_length > kind.max_bytes() {
        return Err(DirectoryReplicaStoreError::Codec(
            kind.oversized_message().to_string(),
        ));
    }
    let admitted_blob = admitted_blob
        .ok_or_else(|| DirectoryReplicaStoreError::Integrity(kind.missing_message().to_string()))?;
    let materialized_length = u64::try_from(admitted_blob.len()).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!(
            "{} exceeds platform bounds",
            kind.length_field()
        ))
    })?;
    if materialized_length != stored_length {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{} changed during materialization",
            kind.length_field()
        )));
    }
    #[cfg(test)]
    notify_directory_replica_audit_test_observer(DirectoryReplicaAuditTestEvent::BlobMaterialized(
        kind,
    ));
    Ok(admitted_blob)
}

fn validate_retry_state_fields(
    producer: &[u8; 32],
    local_node_id: &[u8; 32],
    consecutive_failures: u64,
    retry_not_before: Option<u64>,
    last_failure_at: u64,
    last_failure_reason: &str,
) -> Result<(), &'static str> {
    if *producer == [0u8; 32] || producer == local_node_id {
        return Err("replica retry producer is invalid");
    }
    if consecutive_failures == 0
        || consecutive_failures > DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
    {
        return Err("replica retry failure count is invalid");
    }
    if last_failure_at == 0 {
        return Err("replica retry failure timestamp is invalid");
    }
    if retry_not_before.is_some_and(|retry_at| {
        retry_at < last_failure_at
            || retry_at.saturating_sub(last_failure_at) > DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS
    }) {
        return Err("replica retry boundary is invalid");
    }
    if last_failure_reason.is_empty()
        || last_failure_reason.len() > MAX_DIRECTORY_REPLICA_FAILURE_REASON_BYTES
        || !last_failure_reason
            .bytes()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == b'_')
    {
        return Err("replica retry failure reason is invalid");
    }
    Ok(())
}

fn bytes32(bytes: &[u8], field: &str) -> Result<[u8; 32], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 32 bytes"))
    })
}

fn bytes16(bytes: &[u8], field: &str) -> Result<[u8; 16], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 16 bytes"))
    })
}

fn bytes64(bytes: &[u8], field: &str) -> Result<[u8; 64], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 64 bytes"))
    })
}

fn u64_to_i64(value: u64, field: &str) -> Result<i64, DirectoryReplicaStoreError> {
    i64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} exceeds SQLite integer range"))
    })
}

fn positive_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryReplicaStoreError> {
    if value <= 0 {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{field} must be positive"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

fn nonnegative_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryReplicaStoreError> {
    if value < 0 {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{field} must not be negative"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

#[cfg(test)]
mod tests {
    mod mirror_schema;
    mod observation;
    mod policy;
    mod quarantine;
    mod remaining;
    mod retry;
    mod store;
    mod sync;
    mod witness;
    use super::*;
    use crate::services::DirectoryChainStore;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        directory_block_range_response_signing_bytes, encode_directory_sync_message, NodeDescriptor,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use tempfile::TempDir;

    const NOW: u64 = 1_700_000_100;

    struct DirectoryReplicaAuditObserverGuard;

    impl Drop for DirectoryReplicaAuditObserverGuard {
        fn drop(&mut self) {
            set_directory_replica_audit_test_observer(None);
        }
    }

    fn observe_directory_replica_audit(
        observer: impl Fn(DirectoryReplicaAuditTestEvent) + 'static,
    ) -> DirectoryReplicaAuditObserverGuard {
        set_directory_replica_audit_test_observer(Some(Box::new(observer)));
        DirectoryReplicaAuditObserverGuard
    }

    fn descriptor(identity: &IdentityKeyPair, sequence: u64) -> SignedNodeDescriptor {
        SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                identity.public_key_bytes(),
                sequence,
                NOW - 10,
                NOW + 3_600,
                "replica-test",
            ),
            identity,
        )
        .unwrap()
    }

    fn block(
        producer: &IdentityKeyPair,
        height: u64,
        previous: [u8; 32],
        object: &SignedNodeDescriptor,
    ) -> DirectoryCommitmentBlockV1 {
        DirectoryCommitmentBlockV1::new_signed(
            height,
            NOW + height,
            previous,
            vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(object).unwrap()],
            producer,
        )
        .unwrap()
    }

    fn response_frame(
        producer: &IdentityKeyPair,
        blocks: Vec<DirectoryCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let responder = producer.public_key_bytes();
        let signing = directory_block_range_response_signing_bytes(
            &request_id,
            &responder,
            NOW + 20,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_directory_sync_message(&DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            responder,
            response_timestamp: NOW + 20,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: producer.sign(&signing),
        })
        .unwrap()
    }

    fn carrier_response_frame(
        producer: &IdentityKeyPair,
        carrier: &IdentityKeyPair,
        blocks: Vec<DirectoryCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let producer_id = producer.public_key_bytes();
        let carrier_id = carrier.public_key_bytes();
        let signing = directory_replica_block_range_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &producer_id,
            &carrier_id,
            NOW + 20,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_directory_sync_message(&DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            producer: producer_id,
            carrier: carrier_id,
            response_timestamp: NOW + 20,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: carrier.sign(&signing),
        })
        .unwrap()
    }

    fn accepted_observation_witness_response(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        checkpoint: &DirectoryObservationCheckpointV1,
        request_seed: u8,
    ) -> DirectorySyncMessage {
        let request_id = [request_seed; 16];
        let checkpoint_hash = checkpoint.hash();
        let responder = witness.public_key_bytes();
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &responder,
            NOW + 22,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder,
            response_timestamp: NOW + 22,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        }
    }

    fn portable_observation_certificate_fixture(
        observer: &IdentityKeyPair,
        witnesses: &[&IdentityKeyPair],
        checkpoint_sequence: u64,
        checkpoint_salt: u8,
        verified_at: u64,
    ) -> (
        Vec<u8>,
        [u8; 32],
        DirectoryObservationCertificateTrustPolicy,
    ) {
        let producer_a = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x42; 32]).unwrap();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            checkpoint_sequence,
            verified_at - 2,
            [checkpoint_salt; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: checkpoint_sequence + 10,
                    tip_hash: [checkpoint_salt.wrapping_add(1); 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: checkpoint_sequence + 11,
                    tip_hash: [checkpoint_salt.wrapping_add(2); 32],
                },
            ],
            [checkpoint_salt.wrapping_add(3); 32],
            observer,
        )
        .unwrap();
        let receipts = witnesses
            .iter()
            .enumerate()
            .map(|(index, witness)| {
                let request_seed = u8::try_from(index).unwrap().wrapping_add(0x50);
                let request_id = [request_seed; 16];
                let checkpoint_hash = checkpoint.hash();
                let responder = witness.public_key_bytes();
                let signing_bytes = directory_observation_witness_response_signing_bytes(
                    &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                    &request_id,
                    &observer.public_key_bytes(),
                    checkpoint.sequence,
                    &checkpoint_hash,
                    &responder,
                    verified_at - 1,
                    DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                );
                DirectoryObservationWitnessReceiptV1 {
                    chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                    request_id,
                    observer: observer.public_key_bytes(),
                    checkpoint_sequence: checkpoint.sequence,
                    checkpoint_hash,
                    responder,
                    response_timestamp: verified_at - 1,
                    outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                    signature: witness.sign(&signing_bytes),
                }
            })
            .collect::<Vec<_>>();
        let certificate = DirectoryObservationCertificateV1::new_verified(
            checkpoint,
            u16::try_from(witnesses.len()).unwrap(),
            receipts,
            verified_at,
        )
        .unwrap();
        let frame = encode_directory_observation_certificate(&certificate).unwrap();
        let frame_sha256 = Sha256::digest(&frame).into();
        let trust_policy = DirectoryObservationCertificateTrustPolicy::new(
            observer.public_key_bytes(),
            witnesses
                .iter()
                .map(|witness| witness.public_key_bytes())
                .collect(),
            u16::try_from(witnesses.len()).unwrap(),
        )
        .unwrap();
        (frame, frame_sha256, trust_policy)
    }

    fn import_replica_block(
        store: &DirectoryReplicaStore,
        producer: &IdentityKeyPair,
        object: &SignedNodeDescriptor,
        replica_block: &DirectoryCommitmentBlockV1,
        request_id: [u8; 16],
    ) {
        let frame = response_frame(
            producer,
            vec![replica_block.clone()],
            false,
            replica_block.header.height,
            replica_block.hash(),
            request_id,
        );
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(replica_block),
                std::slice::from_ref(object),
                replica_block.header.height,
                replica_block.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
    }

    fn resolution_command(
        resolver: &IdentityKeyPair,
        incident_digest: [u8; 32],
        tip: &DirectoryReplicaTip,
        command_id: [u8; 16],
        resolved_at: u64,
    ) -> DirectoryReplicaResolutionCommand {
        DirectoryReplicaResolutionCommand::sign(
            resolver,
            command_id,
            incident_digest,
            tip.producer,
            tip.tip_height,
            tip.tip_hash,
            tip.quarantine_kind.clone().unwrap(),
            tip.last_resolution_digest,
            resolved_at,
        )
        .unwrap()
    }

    fn frame_tip_hash(frame: &[u8]) -> [u8; 32] {
        let DirectorySyncMessage::BlockRangeResponseV1 { tip_hash, .. } =
            decode_directory_sync_message(frame).unwrap()
        else {
            unreachable!()
        };
        tip_hash
    }
}
