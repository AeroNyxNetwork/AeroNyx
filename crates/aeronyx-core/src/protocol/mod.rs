// ============================================
// File: crates/aeronyx-core/src/protocol/mod.rs
// ============================================
//! # Protocol Module
//!
//! ## Creation Reason
//! Defines the wire protocol for AeroNyx privacy network communication,
//! including message types, formats, and serialization.
//!
//! ## Modification Reason
//! - Added `memchain` submodule for MemChain P2P memory synchronisation
//!   messages. These messages travel **inside** existing encrypted DataPackets,
//!   multiplexed by a single magic byte (0xAE) after decryption.
//! - 🌟 v0.5.0: MemChainMessage now includes `BlockAnnounce(BlockHeader)`.
//! - 🌟 v1.1.0-ChatRelay: Added `chat` submodule for zero-knowledge P2P
//!   messaging data structures. `ChatEnvelope`, `ChatContentType`, and
//!   `MediaPointer` live here; they are consumed by `MemChainMessage::ChatRelay`
//!   in `memchain.rs` and by `ChatRelayService` in `aeronyx-server`.
//! - v0.1.0-DiscoveryPhase1: Added `discovery` submodule for signed node
//!   descriptors used by decentralized peer discovery and encrypted relay.
//! - v1.0.0-BlindVaultWire: Added a separate, versioned node-blind durable
//!   object protocol for encrypted contact-vault and optional message-archive
//!   segments. Its outer metadata deliberately carries no account identity,
//!   correspondent, application namespace, or public-chain commitment.
//! - v1.1.0-OnionRoutePurpose: Standardized onion terminal-workload purpose
//!   negotiation for Rust nodes, Apps, SDKs, and autonomous agents.
//! - [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] Re-exported signed,
//!   backward-compatible node wire-feature negotiation.
//! - [ONION-REPLY 2026-08-28 by Codex] Added a workload-neutral, fixed-size
//!   encrypted return carrier for anonymous terminal recovery responses.
//! - [BLIND-VAULT-REPLICA-WORKFLOW 2026-08-28 by Codex] Added source-owned,
//!   client-authorized replica execution with verified evidence and bounded
//!   retries; this local state is never a discovery or ledger payload.
//! - [VERIFIED-ONION-ROUTE 2026-08-29 by Codex] Added source-side route plans
//!   derived only from authentic, current, capability-compatible descriptors.
//! - [BLIND-VAULT-DISPATCH-CONTRACT 2026-08-29 by Codex] Exposed validated
//!   replica work as explicit ordered onion terminal purposes.
//! - [ONION-ROUTE-FAILURE-DISPOSITION 2026-08-29 by Codex] Re-exported the
//!   shared fail-closed recovery decision for source-side route adapters.
//! - [BLIND-VAULT-REPLACEMENT-RETIREMENT-PERMIT 2026-08-29 by Codex] Exposed
//!   the evidence-backed gate that protects old replicas during replacement.
//! - [BLIND-VAULT-SEALED-RESTART-SNAPSHOT 2026-08-29 by Codex] Added bounded,
//!   identity-sealed source workflow persistence; it is not a network frame.
//! - [BLIND-VAULT-RESTART-RECOVERY-PLAN 2026-08-29 by Codex] Exposed typed,
//!   source-only recovery decisions for ambiguous restored terminal attempts.
//! - [BLIND-VAULT-DURABLE-RECOVERY 2026-08-29 by Codex] Re-exported exact
//!   durable generation loading and atomic post-evidence journal resolution.
//! - [BLIND-VAULT-DURABLE-SNAPSHOT 2026-08-29 by Codex] Re-exported the
//!   resolved workflow bootstrap and ordinary-state persistence command.
//! - [BLIND-VAULT-PREPARED-EFFECTS 2026-08-29 by Codex] Re-exported exact,
//!   payload-blind terminal effect bindings for source-side orchestration.
//! - [BLIND-VAULT-BOUND-CONTINUATION 2026-08-29 by Codex] Bound effect order
//!   and one-time reply sessions inside restart-safe private journals.
//! - [BLIND-VAULT-BOUND-DURABLE-DISPATCH 2026-08-29 by Codex] Added a typed
//!   durability pipeline ending in one ordered, payload-verifying transport.
//! - [BLIND-VAULT-RECOVERED-BOUND-ATTEMPT 2026-08-29 by Codex] Added
//!   committed-only restart authority for exact ordered retransmission.
//! - [BLIND-VAULT-VERIFIED-ONION-TRANSPORT 2026-08-29 by Codex] Composed
//!   ordered effects, purpose-bound verified routes, and opaque envelope I/O.
//! - [BLIND-VAULT-DURABLE-TERMINAL-OUTCOMES 2026-08-29 by Codex] Added
//!   bounded request/reply failure disposition and atomic journal resolution.
//! - [BLIND-VAULT-RETIREMENT-TRANSPORT-PERMIT 2026-08-29 by Codex] Bound old
//!   lease retirement to the active workflow and exact verified terminal.
//! - [BLIND-VAULT-DISTILLED-ADMISSION 2026-08-29 by Codex] Added a bounded,
//!   credential-free admission proof for sequential replica verification.
//! - [BLIND-VAULT-REPLACEMENT-REPLY-POLICY 2026-08-29 by Codex] Composed
//!   request-bound replies into a fail-closed replacement lifecycle policy.
//! - [BLIND-VAULT-DURABLE-REPLACEMENT 2026-08-29 by Codex] Added a typed
//!   durable path for complete policy-issued replacement capabilities.
//! - [BLIND-VAULT-REPLACEMENT-PERMIT-COMPOSITION 2026-08-30 by Codex] Bound
//!   policy authorization and runtime retirement to one workflow permit.
//! - [BLIND-VAULT-REPLACEMENT-LEASE-LIFETIME 2026-08-30 by Codex] Prevented
//!   authorization, retry, or completion after replacement lease expiry.
//! - [BLIND-VAULT-AUTHORIZED-RETIREMENT-DISPATCH 2026-08-30 by Codex] Joined
//!   active workflow authority to permit-gated transport and reply validation.
//! - [BLIND-VAULT-PROVISIONING-LEASE-LIFETIME 2026-08-30 by Codex] Prevented
//!   expired new leases from satisfying aggregate replica provisioning.
//! - [BLIND-VAULT-SHARED-VERIFICATION-CLOCK 2026-08-30 by Codex] Located the
//!   source-time trait at the common private reply-policy boundary.
//! - [BLIND-VAULT-PROVISIONING-REPLY-POLICY 2026-08-30 by Codex] Added an
//!   exact-count state machine for aggregate admission, writes, and inventory.
//! - [BLIND-VAULT-DURABLE-PROVISIONING 2026-08-30 by Codex] Added typed atomic
//!   resolution for complete policy-issued aggregate provisioning evidence.
//! - [BLIND-VAULT-REPLACEMENT-WRITE-LIFETIME 2026-08-30 by Codex] Bound each
//!   replacement write receipt to its signed admission lease window.
//! - [BLIND-VAULT-ATTEMPT-RESOLUTION 2026-08-30 by Codex] Re-exported one
//!   closed durable adapter outcome for verified completion or bounded failure.
//! - [BLIND-VAULT-TERMINAL-FAILURE-CLASSIFICATION 2026-08-30 by Codex]
//!   Standardized privacy-safe failure mapping across terminal adapters.
//! - [BLIND-VAULT-TERMINAL-FAILURE-DISTILLATION 2026-08-30 by Codex] Connected
//!   detailed runtime errors to bounded source-private attempt outcomes.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`messages`]: Protocol message structures
//! - [`codec`]: Binary serialization/deserialization
//! - [`version`]: Protocol versioning
//! - [`memchain`]: 🌟 MemChain application-layer messages
//! - [`chat`]: 🌟 Chat Relay E2E envelope and media pointer types
//! - [`discovery`]: Signed node descriptors and public capability hints
//! - [`blind_vault`]: Anonymous durable ciphertext objects and node receipts
//! - [`blind_vault_replica_workflow`]: Source-owned replica execution state
//!
//! ### Message Types
//! - `ClientHello`: Initial handshake from client
//! - `ServerHello`: Server response with session parameters
//! - `DataPacket`: Encrypted tunnel data
//! - `MemChainMessage`: 🌟 AI memory sync + chat relay messages (inside DataPacket)
//! - `ChatEnvelope`: 🌟 E2E-encrypted chat message (carried by ChatRelay variant)
//! - `SignedNodeDescriptor`: Signed node metadata for discovery snapshots/gossip
//!
//! ## ⚠️ Important Note for Next Developer
//! - ANY protocol change requires version bump
//! - Maintain backward compatibility where possible
//! - The `memchain` module does NOT touch outer protocol wire format
//! - The `chat` module does NOT touch outer protocol wire format
//! - `chat` module is intentionally separate from `memchain` to keep
//!   the crypto/signing logic isolated and independently testable
//! - The `discovery` module is control-plane metadata only; do not include
//!   client traffic, payloads, DNS contents, or private keys in descriptors
//! - [BLIND-VAULT-WIRE 2026-07-22 by Codex] The `blind_vault` protocol is
//!   independent from MemChain indexing and legacy chat routing. Never add an
//!   owner/sender/receiver/namespace field to its outer wire structures.
//! - [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Use the core route-purpose
//!   parser for untrusted wire values. Unknown purposes must fail closed.
//!
//! ## Last Modified
//! v1.42.0-BlindVaultTerminalFailureDistillation - Added the standard detailed
//! terminal-error to bounded-attempt-failure conversion
//! [BLIND-VAULT-TERMINAL-FAILURE-DISTILLATION 2026-08-30 by Codex]
//! v1.41.0-BlindVaultTerminalFailureClassification - Added one extensible,
//! bounded runtime error classification boundary
//! [BLIND-VAULT-TERMINAL-FAILURE-CLASSIFICATION 2026-08-30 by Codex]
//! v1.40.0-BlindVaultAttemptResolution - Added one closed durable adapter
//! outcome spanning verified completion and bounded failure
//! [BLIND-VAULT-ATTEMPT-RESOLUTION 2026-08-30 by Codex]
//! v1.39.0-BlindVaultReplyOutcomeConversion - Added standard conversion from
//! action reply outcomes into unified typed completion
//! [BLIND-VAULT-REPLY-OUTCOME-CONVERSION 2026-08-30 by Codex]
//! v1.38.0-BlindVaultCompletionBindingGate - Rejected typed capabilities from
//! any other work id or attempt at the durable boundary
//! [BLIND-VAULT-COMPLETION-BINDING-GATE 2026-08-30 by Codex]
//! v1.37.0-BlindVaultAttemptBoundCompletion - Bound every typed completion
//! capability to its exact work id and runtime attempt
//! [BLIND-VAULT-ATTEMPT-BOUND-COMPLETION 2026-08-30 by Codex]
//! v1.36.0-BlindVaultCompletedAction - Added one closed typed capability enum
//! and generic durable completion boundary
//! [BLIND-VAULT-COMPLETED-ACTION 2026-08-30 by Codex]
//! v1.35.0-BlindVaultReplyPolicyExports - Restored the flat protocol API for
//! every typed completion and reply policy
//! [BLIND-VAULT-REPLY-POLICY-EXPORTS 2026-08-30 by Codex]
//! v1.34.0-BlindVaultSingleEffectContext - Centralized fail-closed context
//! verification for single-effect reply policies
//! [BLIND-VAULT-SINGLE-EFFECT-REPLY-CONTEXT 2026-08-30 by Codex]
//! v1.33.0-BlindVaultDurableRenewal - Added typed atomic resolution for
//! completed exact-generation renewal policies
//! [BLIND-VAULT-DURABLE-RENEWAL-COMPLETION 2026-08-30 by Codex]
//! v1.32.0-BlindVaultRenewalReplyPolicy - Added exact lease-generation
//! compare-and-swap verification for renewal attempts
//! [BLIND-VAULT-RENEWAL-REPLY-POLICY 2026-08-30 by Codex]
//! v1.31.0-BlindVaultDurableObservation - Added typed atomic resolution for
//! completed observation-retry policies
//! [BLIND-VAULT-DURABLE-OBSERVATION-COMPLETION 2026-08-30 by Codex]
//! v1.30.0-BlindVaultObservationReplyPolicy - Added exact single-effect,
//! freshness-bounded observation-retry verification
//! [BLIND-VAULT-OBSERVATION-REPLY-POLICY 2026-08-30 by Codex]
//! v1.29.0-BlindVaultDurableReconciliation - Added typed atomic resolution for
//! completed reconciliation reply policies
//! [BLIND-VAULT-DURABLE-RECONCILIATION-COMPLETION 2026-08-30 by Codex]
//! v1.28.0-BlindVaultReconcileReplyPolicy - Added exact ordered mutation and
//! post-mutation inventory verification for reconciliation attempts
//! [BLIND-VAULT-RECONCILE-REPLY-POLICY 2026-08-30 by Codex]
//! v1.27.0-BlindVaultReplacementPermitComposition - Issued and installed one
//! exact active-workflow permit for policy and runtime consumption
//! v1.26.0-BlindVaultDurableReplacement - Added a typed atomic resolution API
//! for completed replacement-policy evidence
//! v1.25.0-BlindVaultReplacementReplyPolicy - Added a reusable source-private
//! replacement reply state machine and workflow-permit boundary
//! v1.24.0-BlindVaultDistilledAdmission - Removed one-time blind credentials
//! from the state retained between admission and inventory verification
//! v1.23.0-BlindVaultRetirementTransportPermit - Enforced workflow-issued
//! retirement authority before transport and verified terminal selection
//! v1.22.0-BlindVaultDurableTerminalOutcomes - Re-exported request-bound
//! failure disposition and durable terminal-attempt failure resolution
//! v1.21.0-BlindVaultVerifiedOnionTransport - Re-exported route-provider and
//! opaque sender composition for ordered Blind Vault effects
//! v1.20.0-BlindVaultRecoveredBoundAttempt - Re-exported committed-only
//! payload-bound restart and resend authority
//! v1.19.0-BlindVaultBoundDurableDispatch - Re-exported exact effect-bound
//! durability markers and ordered terminal transport capability
//! v1.18.0-BlindVaultBoundContinuation - Re-exported exact effect/session
//! restart composition for private compound replica attempts
//! v1.17.0-BlindVaultPreparedEffects - Re-exported source-local ordered effect
//! commitments that bind durable attempts to exact send-time payloads
//! v1.16.0-BlindVaultDurableSnapshot - Re-exported resolved-state persistence
//! v1.15.0-BlindVaultDurableRecovery - Re-exported authenticated recovery
//! loading plus rollback-safe committed-attempt resolution
//! v1.14.0-BlindVaultAttemptContinuation - Re-exported typed recoverable
//! adapter state and single-use onion reply session ownership
//! v1.13.0-BlindVaultPreparedAttemptJournal - Re-exported the typed
//! persist-before-dispatch private attempt handle
//! v1.12.0-BlindVaultPrivateAttemptJournal - Re-exported identity-sealed,
//! action-bound continuation state for restart-safe replica repair
//! v1.11.0-BlindVaultRestartRecoveryPlan - Re-exported fail-closed restored
//! attempt recovery decisions
//! v1.10.0-BlindVaultSealedRestartSnapshot - Re-exported the local encrypted
//! restart snapshot size contract
//! v1.9.0-BlindVaultReplacementRetirementPermit - Re-exported the active
//! attempt gate for safe old-lease retirement
//! v1.8.0-OnionRouteFailureDisposition - Re-exported shared source-side route
//! recovery semantics without changing the onion wire format
//! v1.7.0-BlindVaultDispatchContract - Re-exported purpose-level compound
//! replica dispatch requirements for App, SDK, and agent adapters
//! v1.6.0-VerifiedOnionRoute - Re-exported descriptor-authenticated route
//! planning with bounded hops and path-derived TTL
//! v1.5.0-BlindVaultReplicaWorkflow - Re-exported source-owned, evidence-gated
//! replica execution domain types without changing the public wire protocol
//! v1.4.0-OnionBlindVaultLifecycle - Re-exported single-use anonymous recovery
//! and deletion sessions with request-bound encrypted reply verification
//! v0.1.0 - Initial protocol definitions
//! v0.2.0 - Added memchain submodule for MemChain P2P memory sync
//! v0.5.0 - 🌟 BlockAnnounce variant added to MemChainMessage
//! v1.1.0-ChatRelay - 🌟 Added chat submodule for ChatEnvelope, ChatContentType,
//!                        MediaPointer and related signing helpers
//! v0.1.0-DiscoveryPhase1 - Added discovery submodule for signed descriptors
//! v0.2.0-DiscoveryPhase2 - Re-exported bootstrap snapshot type
//! v0.3.0-DiscoveryPhase4 - Re-exported discovery gossip message helpers
//! v1.0.0-BlindVaultWire - Added anonymous encrypted durable-object contract
//! v1.1.0-BlindVaultLease - Added anonymous lease and signed deletion contract
//! v1.1.0-OnionRoutePurpose - Added stable onion route-purpose negotiation
//! v1.2.0-SignedProtocolFeatures - Re-exported descriptor-bound feature tokens

pub mod auth;
pub mod blind_vault;
pub mod blind_vault_replica_workflow;
pub mod chat;
pub mod codec;
pub mod discovery;
pub mod memchain;
pub mod messages;
pub mod onion;
pub mod onion_reply;
pub mod version;

// Re-export primary types
// [SESSION-TERMINATION 2026-08-15 by Codex] Keep the close domain on the same
// audited public protocol surface as every other wallet-authenticated frame.
pub use auth::{
    verify_signed_message, AuthError, DOMAIN_CHAT_ACK, DOMAIN_CHAT_PULL, DOMAIN_CHAT_PULL_V2,
    DOMAIN_DEVICE_REGISTER, DOMAIN_SESSION_CLOSE_V1, DOMAIN_WALLET_PRESENCE,
};
pub use blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, is_blind_vault_frame,
    BlindVaultBlindAdmissionToken, BlindVaultBlindLeaseAcceptedReceipt,
    BlindVaultBlindLeaseAdmissionRequest, BlindVaultBlindLeaseRenewalRequest,
    BlindVaultBlindLeaseRenewedReceipt, BlindVaultDeleteRequest, BlindVaultDeletedReceipt,
    BlindVaultError, BlindVaultFrame, BlindVaultInventoryCommitmentBuilder,
    BlindVaultInventoryCommitmentEntry, BlindVaultInventoryCommitmentSummary,
    BlindVaultLeaseCreateRequest, BlindVaultLeaseInventoryReceipt, BlindVaultLeaseInventoryRequest,
    BlindVaultLeaseRenewRequest, BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt,
    BlindVaultLeaseStatusReceipt, BlindVaultLeaseStatusRequest, BlindVaultManifestReplicaPlanner,
    BlindVaultOnionDeleteError, BlindVaultOnionDeleteSession, BlindVaultOnionLeaseAdmissionError,
    BlindVaultOnionLeaseAdmissionSession, BlindVaultOnionLeaseInventoryError,
    BlindVaultOnionLeaseInventorySession, BlindVaultOnionLeaseRenewalError,
    BlindVaultOnionLeaseRenewalSession, BlindVaultOnionLeaseRetireError,
    BlindVaultOnionLeaseRetireSession, BlindVaultOnionLeaseStatusError,
    BlindVaultOnionLeaseStatusSession, BlindVaultOnionPullError, BlindVaultOnionPullSession,
    BlindVaultOnionPutError, BlindVaultOnionPutSession, BlindVaultPullRequest,
    BlindVaultPullResponse, BlindVaultPutRequest, BlindVaultRecoveredObject,
    BlindVaultReplicaAction, BlindVaultReplicaEvidence, BlindVaultReplicaEvidenceError,
    BlindVaultReplicaManifestExpectation, BlindVaultReplicaPlan, BlindVaultReplicaPlanError,
    BlindVaultReplicaPlanHealth, BlindVaultReplicaPlanner, BlindVaultReplicaPolicy,
    BlindVaultReplicaTarget, BlindVaultStoredReceipt, BlindVaultTerminalFailure,
    BlindVaultTerminalFailureCode, BlindVaultTerminalOperation, BlindVaultVerifiedReplicaInventory,
    BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES, BLIND_VAULT_PROTOCOL_VERSION, MAX_BLIND_VAULT_FRAME_BYTES,
    MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS, MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS,
};
pub use blind_vault_replica_workflow::{
    load_blind_vault_replica_recovery, BlindVaultReplacementRetirementPermit,
    BlindVaultReplicaActionEvidence, BlindVaultReplicaAttemptContinuation,
    BlindVaultReplicaAttemptDurabilityPhase, BlindVaultReplicaAttemptFailure,
    BlindVaultReplicaAttemptJournal, BlindVaultReplicaAttemptJournalError,
    BlindVaultReplicaAttemptResolution, BlindVaultReplicaAuthenticatedPreparedAttempt,
    BlindVaultReplicaBoundAttemptContinuation, BlindVaultReplicaBoundContinuationError,
    BlindVaultReplicaBoundRuntimeError, BlindVaultReplicaCommittedAttemptBinding,
    BlindVaultReplicaCommittedAttemptDispatch, BlindVaultReplicaCommittedAttemptRecord,
    BlindVaultReplicaCommittedBoundAttemptDispatch, BlindVaultReplicaCompletedAction,
    BlindVaultReplicaCompletedObservation, BlindVaultReplicaCompletedProvisioning,
    BlindVaultReplicaCompletedReconciliation, BlindVaultReplicaCompletedRenewal,
    BlindVaultReplicaCompletedReplacement, BlindVaultReplicaConvergence,
    BlindVaultReplicaDispatchContract, BlindVaultReplicaDispatchFailure,
    BlindVaultReplicaDispatchReadiness, BlindVaultReplicaDurableAttemptDispatch,
    BlindVaultReplicaDurableBoundAttemptDispatch, BlindVaultReplicaDurableDispatchError,
    BlindVaultReplicaDurableResolution, BlindVaultReplicaDurableResolutionError,
    BlindVaultReplicaDurableSnapshot, BlindVaultReplicaDurableSnapshotError,
    BlindVaultReplicaExecution, BlindVaultReplicaExecutionPhase, BlindVaultReplicaExecutionPolicy,
    BlindVaultReplicaLoadedRecovery, BlindVaultReplicaObservationReplyOutcome,
    BlindVaultReplicaObservationReplyPolicy, BlindVaultReplicaObservationReplyPolicyBuildError,
    BlindVaultReplicaObservationReplyPolicyError, BlindVaultReplicaOnionDispatchPlan,
    BlindVaultReplicaOnionEnvelopeSender, BlindVaultReplicaOnionRouteProvider,
    BlindVaultReplicaPersistedAttemptJournal, BlindVaultReplicaPersistedBoundAttemptJournal,
    BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaPreparedAttemptRecord,
    BlindVaultReplicaPreparedBoundAttemptJournal, BlindVaultReplicaPreparedEffectError,
    BlindVaultReplicaPreparedEffectSet, BlindVaultReplicaPrivateReplyPolicy,
    BlindVaultReplicaProvisioningReplyOutcome, BlindVaultReplicaProvisioningReplyPolicy,
    BlindVaultReplicaProvisioningReplyPolicyBuildError,
    BlindVaultReplicaProvisioningReplyPolicyError, BlindVaultReplicaReconcileReplyOutcome,
    BlindVaultReplicaReconcileReplyPolicy, BlindVaultReplicaReconcileReplyPolicyBuildError,
    BlindVaultReplicaReconcileReplyPolicyError, BlindVaultReplicaRecoveredBoundAttempt,
    BlindVaultReplicaRecoveredBoundAttemptError, BlindVaultReplicaRecoveredSendPermit,
    BlindVaultReplicaRecoveryLoadError, BlindVaultReplicaRecoveryState,
    BlindVaultReplicaRecoveryStore, BlindVaultReplicaRenewalReplyOutcome,
    BlindVaultReplicaRenewalReplyPolicy, BlindVaultReplicaRenewalReplyPolicyBuildError,
    BlindVaultReplicaRenewalReplyPolicyError, BlindVaultReplicaReplacementAuthorizationError,
    BlindVaultReplicaReplacementPermitIssueError, BlindVaultReplicaReplacementReplyOutcome,
    BlindVaultReplicaReplacementReplyPolicy, BlindVaultReplicaReplacementReplyPolicyBuildError,
    BlindVaultReplicaReplacementReplyPolicyError,
    BlindVaultReplicaReplacementRetirementDispatchError, BlindVaultReplicaRequestBoundReply,
    BlindVaultReplicaRequestBoundReplyError, BlindVaultReplicaRequestBoundReplyVerifier,
    BlindVaultReplicaRestartRecoveryKind, BlindVaultReplicaRestartRecoveryTask,
    BlindVaultReplicaRestartRecoveryTiming, BlindVaultReplicaRestoredExecution,
    BlindVaultReplicaSnapshotRecord, BlindVaultReplicaTerminalAttemptError,
    BlindVaultReplicaTerminalAttemptRuntime, BlindVaultReplicaTerminalAttemptRuntimeBuildError,
    BlindVaultReplicaTerminalAttemptState, BlindVaultReplicaTerminalEffect,
    BlindVaultReplicaTerminalEffectTransport, BlindVaultReplicaTerminalReplyVerifier,
    BlindVaultReplicaTerminalSendContext, BlindVaultReplicaTerminalSendError,
    BlindVaultReplicaTerminalSendSequence, BlindVaultReplicaTerminalVerificationFailure,
    BlindVaultReplicaVerificationClock, BlindVaultReplicaVerifiedOnionTransport,
    BlindVaultReplicaVerifiedOnionTransportError, BlindVaultReplicaWorkId,
    BlindVaultReplicaWorkItem, BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
    BlindVaultVerifiedProvisionedReplica, BlindVaultVerifiedRetiredReplica,
    DEFAULT_BLIND_VAULT_REPLICA_MAXIMUM_IN_FLIGHT,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_ADAPTER_STATE_BYTES,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_RETENTION_MS,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_REPLY_SESSIONS, MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES,
    MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECTS, MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECT_BYTES,
    MAX_BLIND_VAULT_REPLICA_WORK_ITEMS,
};
pub use chat::{decode_envelope, encode_envelope, ChatContentType, ChatEnvelope, MediaPointer};
pub use codec::{Codec, ProtocolCodec};
pub use discovery::{
    decode_discovery_message, encode_discovery_message, NodeBootstrapSnapshot, NodeCapability,
    NodeCapacity, NodeDescriptor, NodeDiscoveryMessage, NodePolicy, NodeProtocolFeature,
    SignedNodeDescriptor, NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION, NODE_DESCRIPTOR_SCHEMA_VERSION,
};
pub use memchain::{
    decode_memchain, encode_memchain, MemChainMessage, MAX_CHAT_PULL_CURSOR_V2_BYTES,
    MEMCHAIN_MAGIC,
};
pub use messages::{ClientHello, DataPacket, MessageType, ServerHello};
pub use onion::{
    build_onion_envelope, is_onion_blob, open_onion_layer, OnionHop, OnionPeel,
    OnionRouteFailureDisposition, OnionRoutePlanError, OnionRoutePurpose, VerifiedOnionRoute,
    KEM_ALG_X25519, KEM_ALG_XWING, MAX_VERIFIED_ONION_ROUTE_HOPS,
    ONION_FORWARD_HOP_REQUIRED_CAPABILITIES, ONION_MAGIC, ONION_ROUTE_PURPOSE_VALUES, ONION_SALT,
    ONION_TERMINAL_REQUIRED_CAPABILITIES,
};
pub use onion_reply::{
    decode_onion_reply_request, decode_onion_sealed_response, encode_onion_reply_request,
    encode_onion_sealed_response, is_onion_reply_request, open_onion_reply, seal_onion_reply,
    OnionReplyError, OnionReplyPayload, OnionReplyProofMode, OnionReplyRequest, OnionReplySession,
    OnionSealedResponse, MAX_ONION_REPLY_REQUEST_PAYLOAD_BYTES, MAX_ONION_SEALED_RESPONSE_BYTES,
    ONION_REPLY_RESPONSE_SIZE_CLASSES,
};
pub use version::{ProtocolVersion, CURRENT_PROTOCOL_VERSION};
