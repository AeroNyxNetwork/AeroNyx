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
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.
//! This root keeps the module docs, shared limits and constants, the small
//! crate-internal mode/provenance/scope enums, the audit test observer hooks,
//! the `DirectoryReplicaStore` struct, and the shared validation and numeric
//! helpers. Every public item is re-exported here, so all
//! `services::directory_replica::X` paths are unchanged.
//! - `store_access.rs`, `schema.rs`, `verified_import.rs`, `retry_state.rs`,
//!   `quarantine_resolution.rs`, `observation_*.rs`, `evidence_export.rs`:
//!   the `impl DirectoryReplicaStore` operation groups (earlier split)
//! - `transport.rs` and `../directory_replica_policy.rs`: process-only
//!   transport health and signed policy-history reconciliation (earlier split)
//! - `error.rs`: `DirectoryReplicaStoreError`
//! - `certificate_import.rs`: certificate trust policy, frame verification,
//!   import report, and the signed import-history entry
//! - `store_reports.rs`: audit, snapshot, convergence, checkpoint-append,
//!   page-import, and retry-state values
//! - `witness_outcome.rs`: witness targets, decisions, outcome buckets, and
//!   audited outcome counters/snapshot
//! - `evidence_types.rs`: producer snapshots, incidents, tips, evidence pages,
//!   and outbound gossip announcements
//! - `resolution_command.rs`: the node-signed quarantine resolution command
//! - `witness_telemetry.rs`: process-only witness recovery/carrier telemetry
//! - `sync_runtime.rs`: producer sync observations, mirror runtime snapshots,
//!   and `DirectoryReplicaSyncRuntime`
//! - `storage_rows.rs`: raw `SQLite` row shapes and audit batch size
//! - `signed_epochs.rs`: signed witness/route-domain/attestor policy epochs,
//!   reconcile reports, and policy anchors
//! - `audit_state.rs`: private audit accumulators
//! - `range_evidence.rs`: signed range-response evidence verification
//! - `codec.rs`: persisted payload codecs and bounded blob materialization
//! - Tests: `tests.rs` holds the shared fixtures; topic tests are in `tests/`.
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
// [ARCH-SPLIT 2026-10-10 by Claude] Value, row, codec and verification children.
mod audit_state;
mod certificate_import;
mod codec;
mod error;
mod evidence_export;
mod evidence_types;
mod observation_certificate;
mod observation_checkpoint;
mod observation_convergence;
mod observation_witness;
mod quarantine_resolution;
mod range_evidence;
mod resolution_command;
mod retry_state;
mod schema;
mod signed_epochs;
mod storage_rows;
mod store_access;
mod store_reports;
mod sync_runtime;
mod verified_import;
mod witness_outcome;
mod witness_telemetry;

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

// [ARCH-SPLIT 2026-10-10 by Claude] Explicit re-exports keep every existing path.
use audit_state::{
    AuditedResolutionIndex, ObservationCheckpointTip, ObservationWitnessAudit,
    ObservationWitnessPolicyAudit, RouteDomainAttestorPolicyAudit, RouteDomainPolicyAudit,
    VerifiedObservationWitness, VerifiedObservationWitnessSet,
};
pub use certificate_import::{
    verify_directory_observation_certificate_frame, DirectoryObservationCertificateImportReport,
    DirectoryObservationCertificateTrustPolicy, VerifiedDirectoryObservationCertificate,
};
use certificate_import::{
    DirectoryObservationCertificateImportEntry, ObservationCertificateImportAudit,
};
use codec::{
    decode_block, decode_descriptor_object, decode_observation_checkpoint, encode_block,
    encode_descriptor_object, encode_observation_checkpoint, materialize_admitted_replica_blob,
};
pub use error::DirectoryReplicaStoreError;
pub(crate) use evidence_types::DirectoryReplicaGossipAnnouncement;
use evidence_types::DirectoryReplicaGossipCandidate;
pub use evidence_types::{
    DirectoryReplicaEvidencePage, DirectoryReplicaIncidentEvidence, DirectoryReplicaIncidentPage,
    DirectoryReplicaIncidentSummary, DirectoryReplicaProducerSnapshot, DirectoryReplicaTip,
};
use range_evidence::{
    incident_digest, validate_exact_descriptor_objects, validate_page_tip_contract,
    verify_incident_response_evidence, verify_range_response_evidence,
};
pub use resolution_command::{DirectoryReplicaResolutionCommand, DirectoryReplicaResolutionReport};
pub(crate) use signed_epochs::{
    DirectoryObservationWitnessPolicyAnchor, DirectoryObservationWitnessPolicyAnchorDecision,
    DirectoryObservationWitnessPolicyReconcileReport,
    DirectoryRouteDomainAttestorPolicyReconcileReport, DirectoryRouteDomainPolicyReconcileReport,
};
use signed_epochs::{
    DirectoryObservationWitnessPolicyEpoch, DirectoryRouteDomainAttestorPolicyEpoch,
    DirectoryRouteDomainPolicyEpoch,
};
pub(super) use storage_rows::PendingReplicaCommitment;
use storage_rows::{
    QuarantineIncident, StoredObservationCertificateImportRow, StoredObservationCheckpointRow,
    StoredObservationWitnessOutcomeRow, StoredObservationWitnessPolicyAnchorReceiptRow,
    StoredObservationWitnessPolicyRow, StoredObservationWitnessRow, StoredReplicaBlockRow,
    StoredResolutionRow, StoredRouteDomainAttestorPolicyRow, StoredRouteDomainPolicyRow,
    AUDIT_REPLICA_BLOCK_BATCH,
};
pub use store_reports::{
    DirectoryObservationCheckpointAppendReport, DirectoryReplicaAudit,
    DirectoryReplicaImportReport, DirectoryReplicaObservationConvergenceSnapshot,
    DirectoryReplicaRetryState, DirectoryReplicaStoreSnapshot,
};
pub use sync_runtime::{
    DirectoryFullNodeMirrorRuntimeSnapshot, DirectoryReplicaSyncObservation,
    DirectoryReplicaSyncRuntime,
};
pub use witness_outcome::{
    DirectoryObservationWitnessDecision, DirectoryObservationWitnessOutcome,
    DirectoryObservationWitnessOutcomeCounters, DirectoryObservationWitnessOutcomeSnapshot,
    DirectoryObservationWitnessTarget,
};
pub use witness_telemetry::{
    DirectoryObservationWitnessCarrierHealth, DirectoryObservationWitnessCarrierOutcome,
    DirectoryObservationWitnessCarrierSnapshot, DirectoryObservationWitnessRecoveryHealth,
    DirectoryObservationWitnessRecoveryOutcome, DirectoryObservationWitnessRecoverySnapshot,
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
mod tests;
