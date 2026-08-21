# AeroNyx Node Discovery and Encrypted Relay Development Plan

## File Creation / Modification Notes

Creation Reason: Define the long-term Rust protocol plan for node-to-node discovery, signed node descriptors, encrypted envelope relay, Memory Chain coordination, and a future Directory Chain without smart contracts.

Modification Reason: v1.10.0 - Added bounded custody renewal retry backoff and
made the post-round durable audit authoritative after partial collection error.

Main Functionality:

- Record the product boundary for AeroNyx as an open privacy protocol, not a node operator.
- Define the non-negotiable blind-node invariant for relay nodes and Memory Chain coordinators.
- Describe how nodes discover, verify, and sync with each other.
- Define the first protocol primitives: Node Identity, Signed Node Descriptor, Peer Store, Bootstrap Snapshot, Gossip Sync, and Encrypted Envelope Relay.
- Track the Rust, backend, nodeboard, client, and docs files that are expected to change.
- Provide a phased implementation checklist for future developers.

Dependencies:

- Rust protocol primitives in `crates/aeronyx-core/src/crypto/*`, `crates/aeronyx-core/src/ledger/*`, and `crates/aeronyx-core/src/protocol/*`.
- Rust node runtime in `crates/aeronyx-server/src/server.rs`, `crates/aeronyx-server/src/services/*`, and `crates/aeronyx-server/src/management/*`.
- Existing nodeboard and backend observability contracts for node health, capacity, and privacy protocol status.

Important Note for Next Developer:

- Do not describe AeroNyx as a centralized node operator or public exit provider.
- Do not implement any relay, coordinator, queue, health report, or analytics path that lets a node operator read content, reconstruct who is talking to whom, or correlate user-level traffic.
- Do not add smart contracts to this design. The proposed Directory Chain is a signed, append-only node directory ledger only.
- Do not store or sync packet payloads, DNS contents, destinations, domains, URLs, browsing history, voucher secrets, client public IPs, chat plaintext, private keys, or wallet-level traffic.
- Default routing policy must be no-exit unless an operator explicitly enables a future exit capability.

Last Modified: v1.09.0 - [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Renews an expiring exact-anchor threshold only when explicitly enabled, using authenticated PeerStore pins, bounded transport, durable-before-counting, and immediate post-round fail-closed audit.
Previous: v1.08.0 - [CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex] Bounds explicit witness collection to one concurrent request per distinct configured pin while retaining durable-before-counting and fail-closed adverse evidence.
Previous: v1.07.0 - [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] Emits one warning per expiring quorum horizon and one recovery event after explicit evidence refresh, without new APIs or heartbeat fields.
Previous: v1.06.0 - [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] Derives the threshold set's exact aggregate lifetime and warns locally before expiry without contacting witnesses or changing authority.
Previous: v1.05.0 - [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Reuses exact startup readiness during runtime and triggers controlled process recovery when local durable evidence expires, changes, or fails audit, without contacting witnesses.
Previous: v1.04.0 - [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] Copies a bounded immutable receipt snapshot under SQLite and performs read-only cryptographic verification after releasing the connection lock.
Previous: v1.03.0 - [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Makes startup and operator tools consume one cryptographically audited SQLite snapshot and one typed readiness decision.
Previous: v1.02.0 - [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] Optionally blocks startup unless the exact current custody anchor has enough fresh durable signed receipts and no adverse evidence, without contacting witnesses.
Previous: v1.01.0 - [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] Runs one explicit signed-snapshot-pinned witness round, persists every verified receipt before counting it, and re-audits current-checkpoint readiness without enabling a scheduler.
Previous: v1.00.0 - [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Re-audits the complete local receipt vault against the current custody checkpoint with an optional fail-closed operator readiness exit.
Previous: v0.99.0 - [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Imports operator-carried signed receipts only for the current local checkpoint and preserves their typed admission policy for restart audit.
Previous: v0.98.0 - [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] Persists exact portable witness receipts atomically, revalidates every frame after restart, and reconstructs fresh exact-anchor policy without exposing custody contents.
Previous: v0.97.0 - [CUSTODY-WITNESS-TRANSPORT 2026-08-16 by Codex] Sends an exact producer-signed anchor only when explicitly invoked, verifies portable request-bound receipts, and never lets adverse evidence be outvoted.
Previous: v0.96.0 - [CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex] Validates independent producer witness eligibility locally and authenticates witness requests before consulting private trust pins.
Previous: v0.95.0 - [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Accepts exact producer-signed custody anchors only from independently pinned, currently verified peers and returns portable request-bound witness evidence.
Previous: v0.94.0 - [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Prevents configured cognitive providers or workers from silently disappearing behind healthy node readiness.
Previous: v0.93.0 - [OPERATOR-PATH-PRIVACY 2026-08-14 by Codex] Keeps storage and rollout readiness observable without exporting operator filesystem identity.
Previous: v0.92.0 - [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] Prevents an explicitly enabled Chat Relay from silently disappearing after durable initialization failure.
Previous: v0.91.0 - [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Resolves authority-carrier policy once and prevents a configured follower from disappearing behind healthy process startup.
Previous: v0.90.0 - [AUTHORITY-CARRIER-POLICY 2026-08-14 by Codex] Gives dual-signed authority-proof transport a dedicated follower-only pin set with an explicit legacy witness fallback.
Previous: v0.89.0 - [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Recovers exact dual-signed coordinator proofs through bounded operator-pinned carriers without expanding transport into authority.
Previous: v0.88.0 - [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] Synchronizes one exact-next dual-signed coordinator proof at a time, stops block pages at activation boundaries, and applies the audited authority schedule to follower control traffic.
Previous: v0.87.0 - [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Pins a process-local commitment authority root, audits every historical proposer at startup, and enforces the active coordinator at each appended height.
Previous: v0.86.0 - [VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex] Preserves canonical user-storage paths, rejects orphaning reloads, serializes placement with reload, and removes owner identifiers from volume logs.
Previous: v0.85.0 - [DISCOVERY-RATE-LIMIT-RECOVERY 2026-07-30 by Codex] Keeps node discovery gossip available after a recovered rate-limiter lock-owner panic.
Previous: v0.84.0 - [DIRECTORY-BLOCKING-BOUNDARY 2026-07-30 by Codex] Gives all authenticated Directory peer blocking workers one privacy-safe failure and observability contract.
Previous: v0.83.0 - [BLOCKING-WORKER-RECOVERY 2026-07-30 by Codex] Keeps SystemDb usable after a failed worker and prevents raw JoinError payloads from entering anchor/status diagnostics.
Previous: v0.82.0 - [STORAGE-JOIN-PRIVACY 2026-07-30 by Codex] Prevents Tokio panic payloads and raw join errors from entering shared storage errors, API logs, or scheduler diagnostics.
Previous: v0.81.0 - [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] Prevents Tokio panic payloads and raw join errors from entering process-health status or structured shutdown diagnostics.
Previous: v0.80.0 - [STARTUP-TASK-REGISTRY 2026-07-30 by Codex] Aborts every owned process task when a later startup gate fails instead of detaching its JoinHandle.
Previous: v0.79.0 - [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Prevents configured Directory synchronization or Full-node Mirror from disappearing behind a ready process.
Previous: v0.78.0 - [MANAGEMENT-RUNTIME-OWNERSHIP 2026-07-30 by Codex] Prevents heartbeat, command, or session-reporting workers from disappearing behind a healthy process.
Previous: v0.77.0 - [DNS-STARTUP-READINESS 2026-07-30 by Codex] Prevents readiness before DNS bind and bounds the complete DNS forwarding task lifecycle.
Previous: v0.76.0 - [DATA-PLANE-FAILURE-POLICY 2026-07-30 by Codex] Prevents persistent UDP/TUN receive errors from leaving a hot-looping or falsely healthy node.
Previous: v0.75.0 - [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] Escalates unexpected API supervisor and configured follower exits to process recovery.
Previous: v0.74.0 - [FOLLOWER-READINESS-LIVENESS 2026-07-30 by Codex] Revokes follower readiness on task exit and after three missed signed convergence checks.
Previous: v0.73.0 - [FOLLOWER-EFFECTIVE-READINESS 2026-07-30 by Codex] Prevents block convergence from being reported as complete follower readiness while required proof is unavailable.
Previous: v0.72.0 - [FOLLOWER-CERTIFICATE-RETRY 2026-07-30 by Codex] Retries deferred follower certificate persistence promptly without creating a polling loop.
Previous: v0.71.0 - [CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex] Reports authenticated-but-unpersisted follower certificates without claiming durable recovery.
Previous: v0.70.0 - [STICKY-SECURITY-EVIDENCE 2026-07-29 by Codex] Preserves source-blind security-stop times when later retrievals succeed.
Previous: v0.69.0 - [CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex] Reports coordinator certificate backfill without exposing carrier identity or mixing follower state.
Previous: v0.68.0 - [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Prevents later certificate carriers from masking security failures and cools repeated coordinator-backfill outages.
Previous: v0.67.0 - [TYPED-CARRIER-CIRCUIT 2026-07-29 by Codex] Reuses one circuit algorithm while compile-time domains and runtime ownership isolate block-page and certificate carrier health.
Previous: v0.66.0 - [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Reports only anonymous cooling-slot, skipped-selection, and half-open-probe aggregates.
Previous: v0.65.0 - [BLOCK-CARRIER-CIRCUIT-BREAKER 2026-07-29 by Codex] Cools repeatedly unavailable fixed carrier slots across follower rounds without retaining identity-bearing health history.
Previous: v0.64.0 - [MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex] Avoids repeating earlier carrier availability failures across one bounded multi-page follower round and hands off safely when the preferred carrier disappears.
Previous: v0.63.0 - [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex] Reports typed, source-blind block-page retrieval outcomes and bounded carrier attempts without exposing identities or treating transport as certification.
Previous: v0.62.0 - [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Lets a follower recover coordinator-authored blocks from bounded operator pins while requiring an exact-tip threshold certificate before reporting recovery.
Previous: v0.61.0 - [FOLLOWER-CERTIFICATE-TIP-BINDING 2026-07-29 by Codex] Prevents readiness evaluated for an older audited tip from being reported as current after follower advancement.
Previous: v0.60.0 - [FOLLOWER-CERTIFICATE-READINESS 2026-07-29 by Codex] Reports whether the current audited follower tip satisfies the current local witness policy without exposing identities.
Previous: v0.59.0 - [RUNTIME-IDENTITY-POLICY 2026-07-29 by Codex] Rejects self-referential coordinator and witness trust pins before node startup.
Previous: v0.58.0 - [FOLLOWER-CERTIFICATE-CONFIG 2026-07-29 by Codex] Allows validated followers to configure independent witness pins for current-tip certificate verification and carrier recovery.
Previous: v0.57.0 - [FOLLOWER-CERTIFICATE-TELEMETRY 2026-07-29 by Codex] Reports source-blind aggregate outcomes for current-tip certificate retrieval and bounded carrier recovery.
Previous: v0.56.0 - [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex] Recovers audited current-tip certificates through bounded operator-pinned witness carriers when coordinator transport is unavailable.
Previous: v0.55.0 - [FOLLOWER-CERTIFICATE-SYNC 2026-07-29 by Codex] Replicates audited current-tip checkpoint certificates to converged followers under each follower's current witness policy.
Previous: v0.54.0 - [DIRECTORY-TRANSPORT-LIFECYCLE 2026-07-29 by Codex] Centralizes Directory transport health policy in the service runtime, records aggregate degraded/recovered transitions, and prevents diagnostic timestamps from regressing after wall-clock rollback.
Previous: v0.53.0 - [DIRECTORY-TRANSPORT-WINDOW 2026-07-28 by Codex] Classifies Directory synchronization health from a fixed recent outcome window so one final success cannot conceal meaningful transport churn, without retaining identity-bearing request history.
Previous: v0.52.0 - [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] Classifies every completed coordinator-owned Directory HTTP exchange into one mutually exclusive process-only outcome without peer, endpoint, operation, status-code, frame, or payload dimensions.
Previous: v0.51.0 - [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Restores the Directory replica synchronizer's canonical 10-second failover deadline while retaining a separate 12-second process-lifetime operator diagnostic profile.
Previous: v0.50.0 - [PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex] Built bounded control, shared Directory, MemChain sync, and gossip transports once before mutable services started, then reused their connection pools across runtime tasks.
Previous: v0.49.0 - [DISCOVERY-ENDPOINT-SSRF 2026-07-28 by Codex] Canonicalizes peer targets, rejects unsafe permissionless descriptor endpoints, and disables redirects/proxies across discovery, relay, onion, and MemChain transport.
Previous: v0.48.0 - [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] Uses receiver identity hints only for a unique verified canonical endpoint owner.
Previous: v0.47.0 - [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] Rotated alternate proof gossip across producer namespaces and suppressed known receiver-self anchors.
Previous: v0.46.0 - [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] Prevented valid but too-new exact-block proofs from racing ahead of healthy replica synchronization.
Previous: v0.45.0 - [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] Prevented one slow peer from serially blocking a complete gossip round while preserving deterministic aggregate outcomes.
Previous: v0.44.0 - [GOSSIP-OUTCOME-INTEGRITY 2026-07-28 by Codex] Separated proof and legacy gossip failure domains so a later compatibility-path failure cannot erase an already observed proof outcome.
Previous: v0.43.0 - [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] Added one bounded audited proof fallback plus aggregate convergence and rejection buckets without peer dimensions.
Previous: v0.42.0 - [DIRECTORY-GOSSIP-NEGOTIATION 2026-07-27 by Codex] Added bounded, fail-closed public feature negotiation so optional proof frames are sent only to peers that explicitly advertise support.
Previous: v0.41.0 - [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] Added bounded, rotating, replica-audited outbound proof gossip with mandatory legacy self-announcement fallback.
Previous: v0.40.0 - [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] Added proof-carrying discovery gossip that must match exact audited local replica evidence before PeerStore import.
Previous: v0.39.0 - [DIRECTORY-PEER-ADMISSION 2026-07-27 by Codex] Added locally anchored, proof-matched PeerStore admission with preflight/postflight replica audits.
Previous: v0.38.0 - [REPLICA-PROOF-RECOVERY 2026-07-27 by Codex] Added direct-first requester proof recovery with bounded explicit-carrier failover and independent producer/carrier verification.
Previous: v0.37.0 - [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] Added audited carrier recovery for exact original producer descriptor inclusion proofs without expanding mirror authority.
Previous: v0.36.0 - [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] Added compact exact-block descriptor inclusion proofs plus an audit-gated, pinned-peer-only transport contract for light verifiers.
Previous: v0.35.0 - [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] Added a separate process-only carrier-side status contract so nodes can prove bounded encrypted-evidence transport participation without retaining identities, routes, frames, or payload data.
Previous: v0.34.0 - [WITNESS-CARRIER-LIVE 2026-07-27 by Codex] Proved the bounded checkpoint-witness carrier path across three audited live nodes, including automatic direct-path restoration and privacy-safe runtime evidence.
Previous: v0.33.0 - [WITNESS-CARRIER 2026-07-26 by Codex] Added direct-first, one-hop, bounded checkpoint-witness availability recovery through explicitly advertised operator-pinned carriers without expanding witness authority.
Previous: v0.32.0 - [WITNESS-CATCHUP 2026-07-26 by Codex] Added bounded multi-checkpoint witness catch-up with strict forward progress, same-sequence retry suppression, and additive backlog telemetry.
Previous: v0.31.0 - [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] Added a POST-only pinned-peer certificate route plus an operator pull command that separately verifies transport identity, exact frame bytes, local observer/witness pins, threshold, and checkpoint age before schema-v10 import.
Previous: v0.30.0 - [PORTABLE-CERTIFICATE-IMPORT 2026-07-26 by Codex] Added schema v10 and a host-local import command that binds exact foreign certificate bytes and local trust policy into a node-signed, hash-linked, rollback-audited history.
Previous: v0.29.0 - [PORTABLE-CERTIFICATE-VERIFIER 2026-07-26 by Codex] Added a fail-closed offline verifier command with exact frame SHA-256, canonical-codec, chain, time, pinned observer/witness policy, checkpoint, local threshold, and signature checks.
Previous: v0.28.0 - [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] Added operator-only export of one observer-signed checkpoint plus independently signed current-pin witness receipts.
Previous: v0.27.0 - Distinguished mirror convergence from bounded catch-up progress and terminal failure.
Previous: v0.26.0 - Added capacity-bounded public mirrors that cannot affect checkpoint, witness, or policy authority.
Previous: v0.25.0 - Added monotonic external policy-head anchors without exporting witness membership or claiming consensus.
Previous: v0.24.0 - Distinguished independent multi-node corroboration from one-receipt evidence without claiming consensus.
Previous: v0.23.0 - Kept recurring witness selection cost fixed while preserving complete startup and explicit operator audits.
Previous: v0.22.0 - Prevented asymmetric replica schedules from perpetually witnessing a checkpoint that peers have not received yet.
Previous: v0.21.0 - Prevented unsupported witness endpoints from being misreported as transport faults without changing the descriptor wire schema.
Previous: v0.20.0 - Added durable/runtime witness outcome buckets without retaining peer identity or introducing reputation, quorum, consensus, or finality.
Previous: v0.19.0 - Added independently recomputed external checkpoint witness receipts without introducing votes, quorum, consensus, or finality.
Previous: v0.18.0 - Added signed local observation checkpoints binding exact producer tips and recomputable recent commitment overlap without introducing votes, consensus, or finality.
Previous: v0.17.0 - Added signed host-local quarantine resolution, exact active-incident/tip CAS, linked resolution history, and startup tamper detection.
Previous: v0.16.0 - Added bounded incident pagination and canonical producer-signed evidence export with verification on every read.
Previous: v0.15.0 - Added recent signed-commitment overlap and an operator-only deterministic observation root.
Previous: v0.14.0 - Added atomic replica schema v1-to-v2 migration and restart-durable producer retry scheduling.
Previous: v0.13.0 - Added a 45-second producer deadline, bounded failure backoff, and additive aggregate/operator retry status.
Previous: v0.12.0 - Added dedicated replica coordinator/status modules, bounded producer concurrency, and 5-15 second deterministic startup synchronization.
Previous: v0.11.0 - Added aggregate/public and fingerprinted/operator replica status plus bounded multi-page synchronization.
Previous: v0.10.1 - Verified pinned, signed replica synchronization across US1, Korean1, and Noway1 without mixing producer histories.
Previous: v0.10.0 - Added audited remote replica namespaces, signed page/object verification, atomic import, and durable producer quarantine.
Previous: v0.9.0 - Added the signed tip, block-range, and descriptor-object serving half of Directory Sync V1.
Previous: v0.8.0 - Added producer-pinned SQLite persistence and startup recovery for local Directory Chain blocks.
Previous: v0.7.0 - Added the privacy-bounded Directory Chain V1 protocol core.
Previous: v0.6.0 - Added authenticated external witnessing for verified-client delivery-cache anchors.
Previous: v0.5.0 - Added local signed rollback protection for verified-client delivery evidence.
Previous: v0.4.0 - Added fail-closed verified-client delivery evidence recovery.
Previous: v0.3.0 - Added restart-recovery gate for PeerStore relay foundation readiness.
Previous: v0.2.0 - Added Blind Node Invariant for relay and Memory Chain coordination.
Previous: v0.1.0 - Initial node discovery and encrypted relay architecture plan.

## 1. Background

### v1.09 Strict custody evidence can renew before expiry

[CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex]

- `custody_audit_witness_auto_renewal_enabled` is independently default-off
  and valid only with both strict startup and runtime gates. Existing nodes
  preserve their network-silent local audit behavior after upgrade.
- The supervised runtime task starts only after authenticated PeerStore
  bootstrap. A renewal attempt occurs only when the current threshold enters
  its bounded warning window; healthy evidence creates no witness traffic.
- Each attempt uses the process-lifetime no-proxy, no-redirect control client
  and the existing maximum of three exact operator pins. Permissionless
  discovery can provide current signed descriptors but cannot grant witness
  authority, replace a pin, or increase fan-out.
- The custody maintenance guard remains held across exact-anchor generation,
  concurrent network transport, durable receipt writes, and one final atomic
  readiness audit. A backup rotation cannot change the checkpoint mid-round.
- Temporary transport shortfall is retried only on the existing 30-to-300
  second skipped-tick cadence while prior evidence remains valid. It never
  extends receipt lifetime. Persisted authentic adverse evidence enters the
  existing typed supervised shutdown path immediately.

### v1.10 Renewal failures cannot create synchronized retry pressure

[CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex]

- Strict local custody audits and external witness collection now have
  independent schedules. Every audit tick still verifies the durable vault and
  fails closed on expiry, conflict, stale evidence, or generation gap.
- A failed collection round enters bounded exponential backoff aligned to the
  audit cadence. A locally derived node-identity +/- one-tick spread prevents a
  fleet from retrying all exact pins in one synchronized burst; identities and
  endpoints never enter logs or status payloads.
- Retry delay is capped at the final timer tick strictly before the old quorum
  expires. When no safe retry tick remains, the runtime reports
  `retry_before_expiry=false` and lets the next strict audit stop the process.
- The post-round atomic vault audit is authoritative even if the collector
  returns a partial transport or persistence error. Receipts already persisted
  by completed peer futures can recover the quorum, while authentic adverse
  evidence still takes the existing supervised fail-closed path immediately.
- Logs retain aggregate checkpoint/round/policy fields only. No identity,
  endpoint, signature, hash, message, user, route, payload, memory, address,
  destination, DNS, or social-graph metadata is exported.
- This is bounded independent corroboration for opaque custody continuity. It
  is not validator voting, leader election, fork choice, consensus, finality,
  reputation, proof of storage, or settlement.

### v1.08 Custody witness collection is concurrent but still hard-bounded

[CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex]

- Explicit collection de-duplicates configured witness pins and excludes the
  producer identity before starting any request.
- One future runs per remaining exact pin through
  `buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)`. The existing maximum of
  three pins is therefore both the policy limit and absolute concurrency ceiling.
- A slow or unavailable witness consumes one timeout window for the round; it
  cannot multiply maintenance-lock hold time by the configured witness count.
- Signature, descriptor, endpoint, request binding, and exact-anchor checks are
  unchanged. Each valid receipt is durably written before it can increment the
  verified/accepted aggregate, and any persistence failure fails the round.
- Authentic stale, conflict, or gap receipts remain retained adverse evidence
  and continue to make quorum readiness fail closed. Completion order cannot
  become voting weight, route preference, trust, consensus, or finality.
- This v1.08 primitive introduced no scheduler, retry loop, API, heartbeat
  field, witness identity disclosure, endpoint disclosure, or payload field.
  The later v1.09 runtime composition is separately opt-in.

### v1.07 Custody renewal warnings have a bounded local lifecycle

[CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex]

- The runtime uses the aggregate `quorum_valid_through` value as a local
  incident key. One horizon emits `receipt_renewal_required` once; repeated
  timer observations remain debug-only instead of flooding the journal.
- Explicitly refreshed signed receipts may create a newer horizon. If it leaves
  the warning window, the runtime emits `receipt_renewal_recovered` once. If it
  remains near expiry, the new horizon opens one new warning.
- The state machine is process-local. It adds no endpoint, heartbeat field,
  CMS payload, storage row, witness request, automatic scheduler, trust-policy
  mutation, consensus claim, or finality claim.
- Expiry, adverse evidence, threshold loss, vault failure, and current-anchor
  failure continue through the existing supervised fail-closed shutdown path.

### v1.06 Custody quorum expiry is exact and locally actionable

[CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex]

- Atomic readiness sorts only accepted, non-ambiguous current-anchor receipts
  by signed observation time. The threshold-th newest receipt defines the
  inclusive `quorum_valid_through` boundary. Older surplus receipts cannot
  produce a false early warning, and one unusually fresh receipt cannot hide
  that the remaining threshold is about to expire.
- Startup and runtime logs expose only the aggregate validity timestamp and
  remaining seconds. Operator import, collection, and vault-audit JSON add the
  same fields plus `renewal_recommended` under a shared bounded warning window.
- The warning window is one quarter of configured receipt age, clamped between
  60 and 900 seconds. It emits fixed reason `receipt_renewal_required`; it does
  not contact a witness, export an anchor, alter a pin, mutate evidence, extend
  validity, or defer the existing fail-closed runtime decision.
- This is host-local operational readiness, not consensus, finality, voting,
  reputation, liveness proof, or proof of user content.

### v1.05 Strict custody readiness persists for the process lifetime

[CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex]

- An independently default-off runtime gate reuses the exact startup audit and
  typed readiness contract against the current immutable custody anchor.
- The task is owned by the required runtime supervisor, skips missed timer
  ticks, and requests controlled process shutdown when evidence expires,
  changes, becomes adverse, or fails vault/policy audit.
- It is local-only and never performs automatic witness collection. Existing
  deployments remain unchanged unless both strict startup and runtime policy
  are explicitly enabled.

### v1.04 Custody vault reads use a bounded two-phase audit

[CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex]

- Startup readiness and operator read audits first copy the complete bounded
  raw receipt row set from one deferred SQLite transaction. The transaction is
  committed and the connection mutex released before cryptographic work begins.
- Decode, canonical re-encode, digest verification, signature verification,
  redundant-column checks, receipt classification, and policy reduction operate
  only on those immutable process-owned bytes. A later database mutation cannot
  change the decision for the already captured snapshot.
- The existing receipt-vault capacity remains a memory and CPU bound before any
  rows are copied. The loader also preflights fixed-size index BLOBs and signed
  frame lengths inside the snapshot, preventing a replaced database from
  forcing an unbounded allocation. A corrupted count, oversized frame,
  malformed signature, or inconsistent index still fails closed.
- Receipt persistence deliberately keeps the full before/after audit inside its
  `Immediate` write transaction. This is a separate mutation invariant: no
  malformed pre-state or post-state may be committed merely to shorten a lock.
- No database schema, public JSON field, CLI flag, startup setting, identity,
  network frame, or compatibility contract changes in this milestone.

### v1.03 Custody readiness is one atomic typed contract

[CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex]

- Vault integrity totals and exact-anchor policy evidence now come from one
  SQLite snapshot. Startup can no longer audit one state and make its decision
  from a later state; operator import, collection, and audit commands use the
  same primitive.
- `CustodyAuditWitnessPolicyReadiness` is the only readiness interpretation:
  `Ready`, `EvidenceUnavailable`, `ThresholdUnmet`, or `AdverseEvidence`.
  Runtime reason codes and existing CLI status labels are projections of this
  type, not independent count logic.
- Counter invariants are verified before readiness is returned: configured
  pins and threshold must be possible after self/duplicate exclusion, fresh
  decisions must equal accepted plus adverse, missing must equal configured
  minus fresh, and the compatibility quorum flag must equal the derived rule.
- The prior library path could return an impossible aggregate when direct API
  callers supplied only self pins or duplicate pins with a larger threshold.
  That path now returns typed `PolicyInvalid` and cannot authorize startup.
- The SQLite mutex is released immediately after copying the canonical signed
  snapshot. Signature classification and policy reduction remain CPU-only and
  do not extend storage lock duration.
- Public JSON fields and `ready` / `collecting` / `adverse` labels are preserved.
  A structurally inconsistent synthetic or future result becomes `invalid` and
  always fails readiness. No node identity, hash, signature, endpoint, message,
  route, payload, user, address, destination, or social graph is exposed.

### v1.02 Current custody evidence can become a strict startup invariant

[CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex]

- `discovery.custody_audit_witness_startup_required` is default-off. Existing
  nodes preserve availability until an operator explicitly enables the gate
  after collecting and auditing compatible independent receipts.
- Strict evaluation runs after local MemChain and ChatRelay storage open but
  before PeerStore bootstrap, self-advertisement, any listener, gossip, or
  background task. It performs no HTTP request and cannot derive authority
  from permissionless discovery.
- The node holds the cross-process custody maintenance lock, regenerates the
  exact current producer-signed anchor, audits every canonical receipt in the
  bounded vault, and reconstructs policy from distinct configured witness
  identities. A self witness is rejected before storage or transports open.
- Startup fails closed for an unavailable current anchor, malformed vault,
  invalid policy, absent/expired evidence, threshold shortfall, or authentic
  stale/conflict/gap evidence. Accepted count can never outvote an adverse
  receipt for the exact current anchor.
- `custody_audit_witness_max_age_secs` is bounded to 60 seconds through seven
  days and defaults to two hours. Freshness is one-sided: delayed past evidence
  may use the configured window, but future observations receive at most 60
  seconds of clock-skew tolerance. This closes the prior `abs_diff` behavior
  that could let an imported future timestamp extend apparent readiness.
- Passing the gate proves only that the configured independent nodes signed
  the exact opaque custody checkpoint recently. It does not prove message
  storage, availability, consensus, finality, validator voting, or fork choice,
  and it reveals no message, user, route, payload, destination, or social graph.

### v1.01 Witness collection is explicit, pinned, and durable

[CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex]

- `relay-custody collect-audit-witnesses` is the first operator-invoked online
  composition of the already tested custody witness transport and durable
  receipt vault. Configuration and normal startup still perform no custody
  witness transmission, retry, timer, callback, or scheduler work.
- The caller supplies one bounded signed `NodeBootstrapSnapshot`. The command
  filters it to current `discovery.custody_audit_witness_node_ids` before
  PeerStore import, then independently verifies descriptor signatures,
  validity windows, `EncryptedStorage` capability, and public-safe endpoint
  policy. Unrelated descriptors cannot consume this ephemeral transport view.
- The HTTP client has an operator-bounded 1-60 second complete request timeout,
  disables redirects and environment proxies, and never falls back from an
  unavailable pin to an arbitrary discovered peer. Only exact configured
  identities can receive the current anchor: producer identity, generation,
  coarse archived record/byte totals, opaque digest, and signatures only.
- The current checkpoint maintenance guard remains held across descriptor
  admission, network requests, receipt persistence, and final policy audit.
  A concurrent custody maintenance process therefore cannot change the anchor
  between signing and the command's readiness decision.
- A verified receipt is counted only after durable producer-side persistence.
  After the round, the complete vault is revalidated and current policy is
  reconstructed. The command exits zero only for `ready`; transport shortfall
  and authentic stale/conflict/gap evidence produce aggregate diagnostics then
  fail closed. Adverse receipts remain durable and cannot be outvoted.
- Output contains aggregate snapshot/round/vault/policy counters only. It
  excludes node identities, endpoint strings, hashes, signatures, paths,
  messages, users, routes, payloads, memory, DNS, destinations, IP addresses,
  and social-graph metadata.
- This closes the reviewed online operator workflow. It is independent
  corroboration for an opaque custody checkpoint, not a validator round,
  consensus, fork choice, leader election, transaction settlement, or global
  finality.

### v1.00 Current-checkpoint witness policy is restart-observable

[CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex]

- `relay-custody audit-witness-vault` derives the node identity locally, holds
  the same cross-process maintenance lock used by receipt import, regenerates
  the exact current custody checkpoint, and revalidates every retained signed
  receipt before reconstructing policy.
- The command performs no witness request, anchor transmission, descriptor
  gossip, retry, startup callback, or background scheduling. It cannot widen
  the live transport admission window or create receipt evidence.
- `--max-age-seconds` is parser-bounded from 60 seconds through seven days and
  applies only to current policy freshness. Per-row import/live admission
  evidence remains independently enforced by the complete vault audit.
- Stable aggregate states are `ready`, `collecting`, and `adverse`. Any adverse
  evidence wins over readiness defensively. `--require-ready` is an explicit
  operator health gate; without it, intact-but-incomplete policy is reportable
  without making a diagnostic command indistinguishable from corruption.
- Output includes only evaluation time, current generation, configured/fresh/
  accepted/adverse/missing counts, vault totals, threshold, and readiness. It
  excludes node identities, hashes, signatures, paths, endpoints, messages,
  users, routes, payloads, memory, destinations, DNS, IP addresses, and social
  graph metadata.
- This closes restart observability for the manually transported receipt path.
  It deliberately does not enable a default startup gate, consensus, validator
  voting, fork choice, leader election, settlement, or global finality.

### v0.99 Air-gapped witness receipts can safely rejoin producer state

[CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex]

- An operator can carry one canonical producer anchor to an independent node,
  return its signed receipt, and import that receipt into the producer without
  adding a CMS, HTTP callback, startup transmission, or background scheduler.
- Import requires exact SHA-256 pins for both files, canonical re-encoding, both
  Ed25519 signatures, the locally configured witness pin, and the producer's
  own current identity. The producer regenerates its current immutable custody
  checkpoint; a historical receipt cannot become current readiness.
- Schema v18 records a typed admission policy beside every immutable signed
  receipt. `live_transport` is always fixed to 60 seconds. `operator_import`
  stores the operator-selected bound from 60 seconds through seven days. The
  complete vault audit applies each row's own bound after every restart.
- Upgrading schema-v17 evidence is conservative: every existing row becomes
  `live_transport` with a 60-second bound. Migration never infers an import or
  widens historical evidence.
- Accepted evidence is retained idempotently and exact-anchor policy is
  reconstructed from distinct current pins. Signed adverse evidence is also
  retained, reported in aggregate, and makes the command fail after durable
  preservation so review cannot be bypassed by retrying.
- The import report contains only aggregate vault/policy counts, disposition,
  checkpoint generation, and witness observation time. It excludes identities,
  hashes, paths, signatures, endpoints, messages, routes, payloads, memory,
  destinations, DNS, IP addresses, and social-graph metadata.
- This is independent evidence retention, not consensus, validator voting,
  fork choice, leader election, transaction settlement, or global finality.

### v0.98 Producer receipt evidence is durable and restart-verifiable

[CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex]

- A verified witness receipt contributes to the durable round only after one
  immediate SQLite transaction stores its exact canonical frame and re-audits
  the complete producer-side vault. Local persistence failure aborts the round
  instead of publishing a verified count that cannot survive restart.
- Schema v17 stores a maximum of 256 receipt frames. The signed frame remains
  authoritative; producer id, witness id, generations, digests, outcome, and
  observation time are redundant indexes that must match it byte-for-byte.
- Startup/recovery callers can reconstruct policy for one exact producer,
  generation, and canonical anchor SHA-256. Only fresh receipts from distinct
  non-self operator pins count; duplicate pins never inflate coverage.
- `advanced` and `idempotent` both prove the requested exact anchor. Fresh
  `stale` and `conflict` decisions are sticky for that exact anchor because a
  monotonic witness cannot legitimately reverse either state. A `gap` may be
  resolved only by a later accepted receipt after intermediate generations
  have advanced. Same-time accepted/adverse ambiguity always fails closed.
- At capacity, only the oldest normal accepted receipt may rotate. Adverse
  evidence is never automatically deleted; an all-adverse vault rejects new
  writes and requires operator review.
- Audit/status structures expose only aggregate record, accepted, adverse,
  freshness, missing, and threshold counts. They do not expose identities,
  endpoints, signatures, anchor hashes, archive counts, messages, or users.
- The diagnostic non-persisting round remains available for tests and manual
  inspection. A future scheduler/startup gate must use the durable primitive.
  This milestone deliberately adds no timer, background transmission, voting,
  consensus, finality, leader election, or fork choice.

### v0.97 Custody witness transport is explicit, bounded, and fail-closed

[CUSTODY-WITNESS-TRANSPORT 2026-08-16 by Codex]

- The producer now has an explicit library primitive for sending one canonical
  custody anchor to one exact operator pin. It is not called by startup, a
  timer, or a background task; merely configuring witness ids sends nothing.
- Immediately before the request, the node requires a fresh authenticated
  descriptor, `EncryptedStorage`, a canonical endpoint, and the shared public-
  IP SSRF policy. Self-witnessing remains impossible.
- The request signs the canonical anchor SHA-256, request id, producer, and
  timestamp. The response is capped at 1 KiB and must pass both the witness's
  outer request-bound signature and the nested portable receipt signature.
- A bounded round accepts at most three configured pins and exposes only
  aggregate counters. Duplicate and self pins cannot inflate the threshold.
- `advanced` and `idempotent` receipts count as accepted. Any authentic
  `stale`, `conflict`, or `gap` receipt sets adverse evidence and prevents the
  round from reporting quorum, even if enough other witnesses accepted.
- Producer-side persistence and restart policy are implemented by v0.98. This
  transport milestone still does not schedule network transmission.

### v0.96 Producer custody witness planning is local and non-transmitting

[CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex]

- `custody_audit_witness_node_ids` is a producer-side pin set independent from
  delivery witnesses and from witness-side requester admission. The validated
  list is bounded to three identities; `custody_audit_witness_min_verified`
  must be between one and the configured count.
- Startup performs a read-only dry run against the authenticated local
  PeerStore. A candidate is eligible only when its signed descriptor is fresh,
  advertises `EncryptedStorage`, contains a canonical endpoint, and passes the
  public-IP SSRF policy. Self pins and duplicates never satisfy the threshold.
- The resulting plan contains only aggregate counts: configured, eligible,
  unavailable, excluded, minimum, and ready. It contains no node id, endpoint,
  anchor, hash, archive count, byte count, user, route, or message metadata.
- Planning does not construct, sign, encode, or transmit a custody anchor. The
  network exchange remains disabled until a separate explicit rollout enables
  bounded sends and defines receipt persistence/reconciliation policy.
- Both custody and verified-delivery witness endpoints now require canonical
  frames and authenticate signatures before consulting private operator pins.
  Pin and PeerStore failures share one response, preventing unauthenticated
  callers from enumerating local trust relationships through status codes.

Producer-side dry-run example:

```toml
[memchain.chat_relay]
enabled = true

[discovery]
enabled = true
custody_audit_witness_node_ids = [
  "<reviewed-independent-witness-ed25519-node-id-hex>",
]
custody_audit_witness_min_verified = 1
```

### v0.95 Independent custody evidence has a fail-closed peer endpoint

[CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex]

- `POST /api/memchain/peer/custody-audit-anchor-witness` accepts only canonical
  `MemChainMessage` variant 36. The producer-signed nested anchor and the outer
  request signature are verified independently; variant 37 returns both a
  portable signed receipt and a request-bound outer response signature.
- Admission is bilateral. `custody_audit_witness_requester_node_ids` is a
  separate exact Ed25519 pin set from permissionless discovery and from
  `verified_delivery_witness_requester_node_ids`. Empty is the default and
  rejects every custody write. A current signed PeerStore descriptor is also
  required, but discovery alone never grants write authority.
- A witness cannot witness itself. This check runs before durable mutation.
  Requests also pass canonical encoding, 60-second freshness, signature,
  shared replay, and per-peer rate checks before the independent schema-v16
  monotonic custody table is touched.
- `advanced`, `idempotent`, `stale`, `conflict`, and `gap` are all signed and
  portable. Adverse outcomes prove continuity failure; they are not transport
  errors and never mutate the retained high-water row.
- The witness stores only producer node id, checkpoint generation, canonical
  anchor-frame SHA-256, and observation time. The signed anchor contains only
  aggregate archived-entry count, aggregate archived bytes, private audit-root
  commitment, and producer identity/signature. It contains no archive content,
  record id, owner, user, route, message, endpoint, destination, or plaintext.
- Producer-side automatic outbound exchange remains intentionally disabled in
  this milestone. Upgrading a node exposes a fail-closed receiving endpoint but
  does not transmit custody anchors anywhere. Outbound rollout requires an
  explicit bounded witness list, privacy review, and operator activation.
- This is independent anti-rollback evidence. It is not a vote, quorum,
  consensus, fork choice, finality, proof of storage, or financial blockchain.

Witness-side example:

```toml
[memchain]
mode = "local"

[discovery]
enabled = true
custody_audit_witness_requester_node_ids = [
  "<reviewed-producer-ed25519-node-id-hex>",
]
```

### v0.94 Explicit SuperNode activation is a runtime contract

[SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex]

- `memchain.supernode.enabled = false` remains the backward-compatible
  default and creates neither provider clients nor a cognitive task worker.
- SuperNode requires `memchain.mode` to be `local`, `p2p`, or `saas`; combining
  `mode=off` with `supernode.enabled=true` is rejected by both configuration
  validation and the defensive runtime boundary.
- Once enabled, every configured provider must pass endpoint, environment-key,
  required-secret, and HTTP-client initialization before the node can report
  ready. Partial provider sets are rejected because routing and fallback policy
  may refer to any configured member.
- OpenAI-compatible providers still support intentionally keyless local
  endpoints. However, an explicit `$ENV_VAR` reference that is unavailable is
  configuration drift and now rejects startup rather than becoming keyless.
- Anthropic keeps its default official endpoint while now honoring a configured
  root, `/v1`, or complete `/v1/messages` API base. Embedded URL credentials,
  non-HTTP(S) schemes, query strings, and fragments are rejected.
- Provider clients use a fixed request deadline and do not inherit OS or
  environment proxy state. This keeps cognitive traffic bound to the explicit
  validated endpoint and avoids platform proxy-adapter startup failures.
- Provider startup failures expose only fixed reason codes. API endpoints,
  provider names, environment-variable names, and secrets are excluded from
  process-health diagnostics.
- The spawned SuperNode worker is supervised as required runtime. Unexpected
  return, panic, or cancellation triggers the same process-recovery path as
  other required node services.
- Provider initialization completes before self descriptors, peer cache,
  discovery gossip, or Agent Relay capability publication can start. A node
  cannot briefly advertise cognitive relay participation while initialization
  is still unresolved.
- This changes only node-local cognitive task execution. It does not alter
  relay frames, identities, key derivation, encrypted payloads, client APIs,
  LiveKit meeting grants, or the blind-node privacy boundary.

Verification:

- Disabled configuration still returns no router without error.
- Missing configured secrets and invalid API bases reject initialization with
  stable, non-sensitive reason codes.
- Keyless local OpenAI-compatible endpoints remain valid.
- Custom Anthropic endpoint normalization is covered without network access.
- Worker disappearance is attached to process-level required-task supervision.

### v0.93 Operator telemetry does not export host filesystem identity

[OPERATOR-PATH-PRIVACY 2026-08-14 by Codex]

- Nodeboard and the CMS heartbeat retain aggregate MemChain and Chat Relay
  storage readiness, backend kind, quotas, and service state, but no longer
  receive configured SQLite or append-only-log paths. Historical path keys are
  retained with `null` values so older consumers do not lose their shape.
- Runtime rollout detection still inspects the local executable target for the
  Linux ` (deleted)` replacement marker. The serialized legacy
  `executable_path` field is retained as `null`; only `executable_replaced` and
  `restart_required` leave the node.
- Bounded journal summaries now redact Unix, home-relative, file-URI, and
  Windows drive path tokens before they enter node health or heartbeat JSON.
- These paths are operator infrastructure metadata rather than user content,
  but exporting them can reveal usernames, mount layout, deployment tooling,
  and node placement. The blind-node boundary therefore excludes them.
- No wire frame, storage schema, key derivation, routing decision, client API,
  service enablement, or capacity metric changes.

Verification:

- Canary database, AOF, executable, Unix, and Windows paths do not appear in
  serialized operator telemetry or sanitized journal summaries.
- Executable replacement still sets both aggregate rollout booleans, so
  maintenance and restart guidance remains available to Nodeboard.
- Existing status field shape is preserved: historical `db_path`, `aof_path`,
  and `executable_path` keys remain present with privacy-safe null values.

### v0.92 Explicit Chat Relay activation is fail closed

[CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex]

- `memchain.chat_relay.enabled = false` remains the backward-compatible
  default and continues to start without a Chat Relay service.
- Once an operator explicitly sets `enabled = true`, opening the durable queue,
  applying its schema, rebuilding usage counters, and deriving opaque-cursor
  protection are required startup work. Any failure now rejects node startup;
  the process cannot remain healthy with chat routes silently absent.
- This aligns configured service state, signed capability readiness, health
  reporting, and actual route availability. A node advertises Chat Relay only
  after the same service instance has initialized successfully.
- Startup diagnostics expose only a stable aggregate bucket such as
  `sqlite_error`. Database paths and raw SQLite errors are not copied into
  process-health errors, and successful startup no longer logs the path.
- The relay remains blind: this change does not expose or parse sender,
  receiver, message ID, ciphertext, blob ID, endpoint, session, or plaintext.
  No wire frame, schema, retention policy, or client behavior changes.

Implementation paths:

- `crates/aeronyx-server/src/server.rs`: one fallible service initializer wired
  directly into the startup transaction.
- `crates/aeronyx-server/src/services/chat_relay.rs`: path-free success logging
  and stable error classification retained at the service boundary.

Verification:

- A disabled configuration still returns no service without error.
- An explicitly enabled relay with an unusable database path rejects startup,
  returns `sqlite_error`, and does not include that private path in the error.
- Existing Chat Relay storage, migration, pull, ACK, quota, quarantine, and
  capability tests remain the functional regression gate.

### v0.91 Configured followers cannot silently lose required runtime

[FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex]

- Authority-proof carrier source and identities are now resolved into one
  validated typed policy before the follower task is spawned. Runtime cannot
  independently derive a telemetry label and a `filter_map`-decoded identity
  list, so malformed pins cannot be silently discarded or reported under a
  different policy source.
- The follower task constructor returns `Result<Option<JoinHandle<()>>>`:
  `None` means only that commitment synchronization is disabled. A configured
  follower with a missing coordinator, self-referential coordinator, or invalid
  carrier policy returns a startup error instead of leaving a healthy process
  without its required synchronization worker.
- The existing startup task registry owns rollback. If this gate fails after an
  earlier listener or worker was created, those owned tasks are aborted before
  startup returns. Readiness therefore describes the complete configured node,
  not a partially initialized process.
- Operational state records only fixed privacy-safe error codes such as
  `invalid_authority_carrier_policy`. Logs and status do not expose identities,
  endpoints, proof contents, epochs, hashes, routes, messages, owners, or memory
  data.
- Public compatibility helpers and all old configuration defaults remain
  available. This changes neither block format nor handover proof verification,
  and it does not introduce voting, fork choice, consensus, or finality.

Implementation paths:

- `crates/aeronyx-server/src/config_memchain.rs`: typed, single-parse effective
  carrier policy plus panic-free compatibility helpers.
- `crates/aeronyx-server/src/server.rs`: fallible follower task construction and
  startup transaction propagation.
- `crates/aeronyx-server/src/services/memchain/storage_ops.rs`: stable
  privacy-safe status classification for the new startup rejection.

Verification:

- Unit tests cover disabled, legacy witness-fallback, and dedicated policy
  sources; malformed policies fail the typed runtime boundary.
- Startup tests cover missing coordinator, coordinator self-reference, and
  malformed carrier pins, including the fixed follower status error code.
- Existing handover, authority, block-carrier, certificate, and configuration
  suites remain the cryptographic and backward-compatibility regression gate.

### v0.90 Authority-proof transport is not witness authority

[AUTHORITY-CARRIER-POLICY 2026-08-14 by Codex]

- `commitment_authority_carrier_node_ids` is a dedicated follower-only list of
  at most three operator-pinned identities allowed to transport exact-next,
  dual-signed coordinator handover proofs. A carrier cannot sign a checkpoint
  certificate, satisfy witness policy, produce a commitment block, vote,
  select a fork, or change the immutable authority root.
- Empty preserves every existing deployment by using
  `commitment_witness_node_ids` as the effective proof-transport set. A
  non-empty dedicated list replaces that fallback rather than merging with it,
  so narrowing transport eligibility cannot silently retain old witnesses.
- When MemChain is enabled, static validation rejects malformed, zero,
  duplicate, over-capacity, coordinator-self, and non-follower configurations.
  Runtime identity validation additionally rejects a carrier pin that resolves
  to the local node itself before any listener is opened.
- Witnesses and carriers reuse one internal Ed25519 pin parser only for syntax,
  bounds, and duplicate rejection. Their authorization, runtime collections,
  circuits, and call sites remain separate. This is deliberate type-and-policy
  separation, not a new quorum or consensus role.
- Startup telemetry exposes only the source-blind policy label
  (`dedicated`, `witness_fallback`, or `disabled`) and aggregate pin count. It
  does not log identities, endpoints, proof contents, epochs, hashes, routes,
  messages, owners, or memory data.

Implementation paths:

- `crates/aeronyx-server/src/config_memchain.rs`: dedicated serde-defaulted
  field, shared syntax-only parser, role validation, legacy fallback helper,
  runtime self-pin rejection, and regression tests.
- `crates/aeronyx-server/src/server.rs`: separate effective carrier and witness
  collections; only the carrier collection enters authority-proof recovery.
- `deploy/node/server.example.toml`: operator guidance spelling out the
  transport-only role and exact fallback behavior.

Verification:

- Config tests prove legacy followers retain witness-based transport, explicit
  carriers replace rather than extend it, carrier-only followers are valid,
  and invalid role/identity/bound combinations fail closed.
- Existing handover, block-carrier, certificate, and authority tests remain
  unchanged to demonstrate the configuration split does not alter wire or
  cryptographic verification behavior.

### v0.89 Coordinator handover proofs survive producer downtime

[AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex]

- A follower always requests the exact-next handover proof from its currently
  audited coordinator first. A carrier is considered only after an explicit,
  classified availability failure; protocol, signature, authority, epoch,
  canonical-encoding, endpoint-policy, and storage errors stop the round.
- Recovery is restricted to at most three distinct operator-pinned carrier
  identities. Legacy configurations use witness pins as transport carriers;
  discovery resolves only the exact effective pins and cannot nominate an
  arbitrary peer, widen the set, or grant authority.
- A carrier signs only the response transport envelope. The transition remains
  valid only when the previous and next coordinators independently signed the
  same exact proof, the previous coordinator matches the immutable local root
  schedule, and the activation boundary matches the audited block prefix.
- An empty carrier history cannot prove that no handover exists. It is treated
  as stale availability and the bounded recovery loop may try the next pin.
  One malformed, mismatched, or invalid response fails closed immediately so a
  later carrier cannot conceal security evidence.
- Authority recovery owns a typed cursor and process-lifetime circuit distinct
  from block-page and certificate circuits. Repeated availability faults cool
  only the corresponding anonymous pin slot; half-open probes remain bounded.
- Follower status exposes source-blind aggregate outcomes, actual bounded
  carrier attempt counts, anonymous cooling/skip/half-open counters, monotonic
  latest observations, and sticky security-stop times. It never stores or
  reports a carrier identity, endpoint, proof, epoch, height, hash, signature,
  raw failure, route, user, message, or memory payload.
- Deployments without an authority root retain the complete legacy static-pin
  behavior. This milestone adds availability recovery only; it is not
  consensus, finality, fork choice, validator voting, coordinator election,
  transactions, balances, or smart contracts.

Implementation paths:

- `crates/aeronyx-server/src/api/memchain_peer.rs`: direct-first classified
  recovery, independent responder/proof authority verification, typed circuit,
  bounded pin selection, and source-blind terminal observations.
- `crates/aeronyx-server/src/services/memchain/storage.rs`: additive public
  status fields plus private typed authority-sync dispositions.
- `crates/aeronyx-server/src/services/memchain/storage_ops.rs`: follower-only
  aggregate outcome/circuit accounting and privacy-safe status projection.
- `crates/aeronyx-server/src/server.rs`: process-lifetime circuit ownership and
  the validated effective carrier pins without changing legacy mode.

Verification:

- A real localhost test makes the active coordinator unavailable, receives an
  empty result from one stale pin, obtains the exact proof from a second pin,
  persists the transition, changes active coordinator, and reports only
  aggregate recovery evidence.
- Adversarial coverage proves a carrier may authenticate its own envelope but
  cannot replace the proof predecessor or manufacture coordinator authority.
- Failure classification coverage proves only explicit availability failures
  advance; security failures stop immediately.
- Storage tests prove authority telemetry is monotonic, follower-only, sticky
  for security evidence, and isolated from block/certificate carrier state.

### v0.88 MemChain followers synchronize coordinator handovers

[AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex]

- The peer protocol appends request and response variants for one exact-next
  coordinator handover proof. Existing bincode discriminants are unchanged,
  so older frames remain wire compatible.
- Requests bind chain id, current local authority epoch, random request id,
  requester identity, and timestamp. Responses bind the same request, the
  responder, response time, proof digest, and advertised history head.
- The response contains at most one dual-signed proof. A responder that claims
  a newer history head but omits the exact-next proof is rejected. A proof for
  another predecessor, chain, epoch, or already-passed activation height is
  also rejected before persistence.
- A cold follower asks its currently audited coordinator for the next proof
  before every block page. When the proof activates in the future, the page is
  capped at `activation_height - 1`; after that exact block prefix is audited,
  the follower persists the proof and switches to the next coordinator.
- Block announcements and witness lease grant/release now resolve authority
  from the same exact-height audited schedule. The old static coordinator pin
  remains the complete behavior when no authority root is configured.
- An obsolete producer checks next-height authority before reading uncommitted
  records and stops quietly after rotation. Atomic block append repeats the
  authority check and remains the final decision against races.
- The endpoint is admitted-peer-only, timestamp checked, signed, replay
  protected, rate limited, POST-only, allocation bounded, and SSRF guarded on
  outbound use. It carries no user identity, message, payload, route, memory
  ciphertext, blind index, or social graph data.
- Handover, block-announcement, and coordinator-lease control handlers verify
  cryptographic identity before consulting authority or peer-admission state.
  Forged traffic therefore cannot use error classes as a peer-membership or
  active-authority oracle, and cannot trigger storage-backed authority audits.
- v0.89 adds bounded operator-pinned transport recovery when the active
  coordinator is unavailable; proof authority and activation rules are
  unchanged.
- This remains an append-only privacy-protocol commitment log. It does not add
  consensus, finality, fork choice, transactions, balances, smart contracts,
  or permissionless coordinator election.

Implementation paths:

- `crates/aeronyx-core/src/protocol/memchain.rs`: append-only wire variants and
  canonical request/response signing bytes.
- `crates/aeronyx-server/src/services/memchain/storage_ops.rs`: audit-bound
  authority snapshots, exact-next proof paging, and configured-root persistence.
- `crates/aeronyx-server/src/api/memchain_peer.rs`: authenticated endpoint,
  client verification, dynamic announcement/lease authority, and bounded pages.
- `crates/aeronyx-server/src/server.rs`: interleaves proof synchronization and
  block-prefix catch-up without crossing an activation boundary.
- `crates/aeronyx-server/src/miner/reflection.rs`: preflights next-height
  production authority before selecting sealed record commitments.

Verification:

- Canonical wire round-trip and signature-tamper coverage passes.
- Storage tests prove exact-next paging and height-based authority resolution.
- Adversarial tests reject omitted proofs and wrong predecessor identities.
- Admission-order regression coverage proves forged known and unknown
  requester identities receive the same unauthenticated result.
- A real localhost two-node test catches up block one, persists the exact
  transition, and switches authority for block two.
- Legacy static-pin announcement and witness-lease tests remain green.

### v0.87 MemChain commitment authority is enforced at runtime

[COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex]

- A coordinator or follower now installs one immutable, process-local authority
  root before the commitment-chain startup audit. The root is supplied by the
  operator and is never inferred from mutable SQLite history.
- `commitment_authority_root_node_id` is the explicit genesis trust anchor for
  coordinator rotation. Existing deployments remain compatible while the
  field is empty: followers reuse their configured coordinator pin and a
  coordinator uses its runtime identity. Operators must set the field
  explicitly before the first dual-signed handover and keep it unchanged
  across later rotations.
- Startup verifies the complete dual-signed handover schedule and every stored
  block proposer in one SQLite snapshot before publishing an integrity
  baseline. A legacy block signed by a cryptographically valid but
  height-unauthorised coordinator prevents readiness.
- Live block append resolves the coordinator authorised at each exact height
  and rejects a stale or premature proposer before inserting any row. A batch
  failure rolls back atomically and cannot advance the in-memory integrity tip.
- Full startup and operator audits remain `O(blocks + handovers)`. Normal live
  append audits only the bounded handover schedule, avoiding chain-height cost
  on every block while preserving the stronger independent startup scan.
- The process rejects attempts to replace or disable an installed root.
  Coordinator rotation must use the dual-signed handover schedule; callers
  cannot bypass it by mutating the in-memory trust anchor.
- The root and coordinator identities are not logged or exposed through public
  telemetry. This change does not reveal memory ciphertext, owners, message
  parties, routes, payloads, blind indexes, or any user-level activity.
- This is a signed privacy-protocol commitment log, not a general-purpose
  blockchain, payment ledger, smart-contract platform, permissionless
  consensus protocol, or finality claim. Coordinator-handover submission and
  network gossip. The later v0.88 milestone adds direct authenticated handover
  exchange; carrier recovery and permissionless coordinator election remain
  intentionally outside this milestone.

Implementation paths:

- `crates/aeronyx-server/src/config_memchain.rs`: validates and resolves the
  immutable authority root with backward-compatible role fallbacks.
- `crates/aeronyx-server/src/server.rs`: installs the root before startup audit.
- `crates/aeronyx-server/src/services/memchain/storage.rs`: owns the
  non-persistent process-local trust anchor.
- `crates/aeronyx-server/src/services/memchain/storage_ops.rs`: audits handover
  history and enforces exact-height proposer authority atomically.
- `deploy/node/server.example.toml`: documents the operator configuration.

Verification:

- Authority tests reject an old coordinator after handover activation and
  accept the new coordinator at the same height.
- Upgrade recovery tests reject a historically stored unauthorised proposer
  during startup and leave integrity state `not_verified`.
- Existing handover tamper, epoch, lease, activation, and acceptance-time
  regression tests remain green.

### v0.86 MemChain volume routing fails closed

[VOLUME-ROUTER-INTEGRITY 2026-07-30 by Codex]

- `VolumeRouter` now uses a non-poisoning `parking_lot::RwLock` for its short
  synchronous configuration critical sections. A failed writer no longer
  makes all later storage routing operations panic.
- New-owner placement and volume configuration reload share one async control
  gate. A reload cannot remove a volume after placement selected it but before
  the durable `SystemDb` assignment commits.
- Hot reload now enforces the contract already documented by the source: a
  changed path for an existing volume is warned and ignored, so existing
  owner database and vector-index filenames remain at their canonical path.
- A reload may remove an unassigned volume, but it rejects the complete update
  before mutation if any durable assignment still references that volume.
- Startup likewise fails closed when durable assignments reference an absent
  volume. The node cannot report ready while those users' storage paths are
  unresolved.
- Startup and assignment logs no longer contain full or abbreviated owner
  public keys. Operational evidence is limited to volume identifiers and
  aggregate assignment counts.
- This milestone does not change Memory Chain ciphertext, record formats,
  owner authentication, database filenames, API routes, vector-index formats,
  chain authority, witness policy, consensus, or finality.

Verification:

- `rustfmt +1.97.1 --edition 2021 --check` on `volume_router.rs`.
- `cargo +1.97.1 test -j 1 -p aeronyx-server volume_router`: 20 passed.
- `cargo +1.97.1 test -j 1 -p aeronyx-server storage_pool`: 12 passed.
- `cargo +1.97.1 check -j 1 -p aeronyx-server`.
- `git diff --check`.

### v0.85 Discovery admission survives a recovered panic

[DISCOVERY-RATE-LIMIT-RECOVERY 2026-07-30 by Codex]

- The permissionless gossip API's process-local global rate limiter now uses
  the crate-standard non-poisoning `parking_lot::Mutex`.
- Previously, a panic while the limiter lock was held poisoned
  `std::sync::Mutex`; every later `/api/discovery/gossip` request then panicked
  at `lock().expect()`. The process and health endpoint could remain alive while
  descriptor propagation had become permanently unavailable.
- Lock scope remains one small synchronous counter update. No guard crosses an
  `.await`, and request admission still occurs before signature verification
  and PeerStore mutation.
- A regression test deliberately panics while holding the limiter lock, then
  sends a real snapshot gossip message through the handler and requires the
  established `200 OK` response.
- This milestone does not change rate limits, request body ceilings, API JSON,
  descriptor verification, allow/deny policy, Directory proof admission,
  PeerStore semantics, transport frames, or discovery authority.

### v0.84 Directory peer workers share one failure contract

[DIRECTORY-BLOCKING-BOUNDARY 2026-07-30 by Codex]

- All eleven synchronous Directory peer operations now enter Tokio's blocking
  pool through `run_directory_chain_blocking`. This includes producer tip,
  block-range, object and inclusion-proof audits; replica carrier reads;
  policy-anchor persistence; certificate export; and witness recomputation.
- A worker panic, cancellation, or unexpected join failure remains fail-closed
  with the established `503 audit_task_failed` protocol response. Existing
  authenticated peer clients therefore require no compatibility change.
- Internal diagnostics now retain the static operation role plus only the
  shared fixed `Panicked` / `Cancelled` / `Failed` category. The raw
  `JoinError` and its potentially sensitive panic payload are never formatted.
- The helper accepts only a `'static` operation label defined in source code,
  preventing request data, descriptor material, paths, identities, or payloads
  from becoming diagnostic labels.
- A regression test terminates a real blocking worker with a sensitive marker
  and proves the HTTP status and response body remain exactly the stable,
  privacy-safe protocol bucket.
- This milestone does not change Directory Sync frames, request admission,
  signatures, chain selection, witness authority, mirror policy, rate limits,
  API paths, or consensus semantics.

### v0.83 Blocking worker failure is bounded and recoverable

[BLOCKING-WORKER-RECOVERY 2026-07-30 by Codex]

- `SystemDb` now uses the crate-standard `parking_lot::Mutex` around its
  private SQLite connection. A blocking worker that panics while holding the
  lock can no longer poison the mutex and force every later tenant metadata
  operation through another `lock().unwrap()` panic.
- SQLite work remains inside `spawn_blocking`; no mutex guard crosses an
  `.await`, no database schema changes, and the public `SystemDb` API remains
  unchanged.
- Signed commitment-tip and checkpoint-certificate anchor writes now share one
  `run_blocking_local_anchor_write` boundary. Join failures contain only the
  static anchor role and fixed `Panicked` / `Cancelled` / `Failed` category.
- Directory replica status workers likewise log only a static operation role
  and the fixed shared join category. Public/operator API response reason
  buckets remain unchanged.
- The connection-lock recovery test deliberately terminates one worker while
  it holds the lock, then proves a later volume assignment and lookup still
  complete. The anchor-worker test proves panic text cannot enter returned
  persistence errors.
- This recovery property does not make panics acceptable and does not suppress
  Rust's global panic hook. A worker that panics fails its current operation;
  the invariant is that unrelated later operations do not inherit a permanent
  poisoned-lock failure.
- No wire schema, SQLite schema, anchor format, API JSON shape, trust policy,
  routing decision, synchronization authority, or consensus behavior changes
  in this milestone.

### v0.82 Storage task failures remain blind

[STORAGE-JOIN-PRIVACY 2026-07-30 by Codex]

- `RuntimeTaskJoinFailureKind` now lives in the shared server error module, so
  runtime supervision, Directory startup, `SystemDb`, and `StoragePool` use one
  fixed `Panicked` / `Cancelled` / `Failed` classification.
- The public `SystemDbError::Join(JoinError)` and
  `StoragePoolError::Join(JoinError)` tuple variants remain available for
  source compatibility. Existing callers can still construct them through
  `From<JoinError>` and can still pattern-match the inner Tokio state.
- Their public `Display`, `Debug`, and `Error::source` boundaries no longer
  expose or chain the raw Tokio error. API and scheduler logging therefore
  receives only a fixed task-join category, never a panic payload.
- The inner `JoinError` remains private to explicit Rust pattern matching for
  backward compatibility. New status, telemetry, management, or API code must
  use the fixed classifier and must not add raw join-error accessors.
- This does not suppress Rust's global panic hook. Tasks must still avoid panic
  messages containing payloads, identities, routes, endpoints, message IDs,
  memory contents, or other user-derived data.
- No wire schema, storage schema, database path, routing decision, trust
  policy, synchronization authority, or public response shape changes in this
  milestone.

### v0.81 Task failure diagnostics remain blind

[JOIN-FAILURE-PRIVACY 2026-07-30 by Codex]

- Required workers, required API listeners, Directory startup/reconciliation
  blocking work, and bounded shutdown joins now share one typed
  `RuntimeTaskJoinFailureKind` classification: `Panicked`, `Cancelled`, or
  `Failed`.
- Tokio `JoinError` values are inspected but never retained, formatted into a
  process-health failure, forwarded to systemd status, or copied into
  structured shutdown fields. A panic payload therefore cannot make that
  control-plane hop even if the panic originated in request-handling code.
- Listener-local I/O diagnostics remain local to the failure site. The
  process-level recovery message carries only the fixed listener role and
  local bind address, which is enough for restart policy without forwarding
  arbitrary implementation text.
- This boundary does not replace Rust's process panic hook. Code must still
  avoid panicking on user-derived values and must not put payloads, identities,
  routes, endpoints, or message identifiers into panic messages.
- No wire schema, API response, routing decision, trust policy, synchronization
  authority, storage format, or consensus behavior changes in this milestone.

### v0.80 Process tasks belong to one startup transaction

[STARTUP-TASK-REGISTRY 2026-07-30 by Codex]

- `Server::run` creates one `RuntimeTaskRegistry` before spawning the first
  process-lifetime task. Peer-cache persistence, Directory persistence,
  Directory replica synchronization, discovery gossip, management workers,
  data-plane workers, cleanup jobs, miners, and API supervisors all enter this
  same ownership boundary.
- Early persistence and gossip handles are registered immediately instead of
  remaining in local variables across later fallible UDP, TUN, DNS, management,
  and API initialization.
- Rust/Tokio normally detaches a task when its `JoinHandle` is dropped. The
  registry's RAII `Drop` instead aborts every still-running owned handle when
  startup returns an error, preventing failed embedded starts from leaving
  hidden background work in the caller's Tokio runtime.
- Required-task supervisor handles remain nested owners. Aborting one through
  the registry also aborts its guarded inner task, so supervision does not
  reintroduce detachment during startup unwind.
- After a successful runtime wait, ownership is explicitly transferred to the
  existing concurrent bounded-shutdown implementation. Normal SIGINT, SIGTERM,
  programmatic stop, or critical-task recovery therefore retains cooperative
  broadcast, bounded joins, timeout aborts, and cancellation confirmation.
- Registry diagnostics contain only fixed local task role names. They never
  include peers, endpoints, routes, packet metadata, block contents, message
  identifiers, client identities, payloads, or other user-plane data.
- No wire schema, synchronization authority, mirror policy, routing policy,
  storage format, API behavior, or blind-node privacy invariant changes in
  this milestone.

### v0.79 Configured Directory synchronization is required runtime state

[DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex]

- Directory replica synchronization remains disabled by default. A node with
  no pinned Directory producers and Full-node Mirror disabled still starts
  without a replica store or synchronization task.
- Once an operator configures pinned producer synchronization or enables
  Full-node Mirror, the runtime requires an initialized, audited replica store.
  Programmatic configuration paths can no longer bypass the normal static
  validation and silently omit the configured service.
- Coordinator construction failures now propagate as stable startup errors.
  Store promotion, mirror-capacity audit, retry-state restoration, and policy
  initialization therefore complete before the node can advertise readiness.
- A successfully started `directory-replica-sync` worker is adopted by the
  required-task supervisor. Unexpected normal return, panic, cancellation, or
  join failure initiates bounded graceful recovery and a non-zero process exit
  for the service manager.
- The existing pre-READY failure gate also covers this worker. If it exits
  during later startup work, the node refuses to emit a contradictory
  systemd `READY` transition.
- Startup and runtime failures retain only fixed task names and stable reason
  buckets. They do not expose producer identities, endpoints, paths, blocks,
  descriptors, signatures, request metadata, or user-plane data.
- This changes only local process health semantics. Full-node Mirror remains
  non-authoritative and cannot affect witness policy, fork choice, quorum,
  consensus, finality, routing weight, or producer authority.

### v0.78 Management workers are owned required tasks

[MANAGEMENT-RUNTIME-OWNERSHIP 2026-07-30 by Codex]

- `init_management_reporter` now returns a typed `ManagementRuntime` containing
  the session-event sender and exactly three long-lived task handles: command
  handling, heartbeat/policy reporting, and session-event reporting.
- The main runtime immediately adopts all three handles into the existing
  required-task supervisor. An unexpected normal return, panic, cancellation,
  or join failure now enters graceful process recovery instead of leaving an
  apparently healthy node without policy updates, remote commands, or session
  telemetry.
- Immediately before systemd readiness, the main task consumes any required
  failure already queued during startup and treats a disconnected supervisor
  channel as fatal. A node with a known dead required worker therefore never
  emits a contradictory `READY` transition.
- Ordinary backend timeouts and request errors remain handled inside the
  workers under their existing fail-open and retry behavior. A transient
  management-service outage therefore does not restart the privacy data plane;
  only disappearance of the worker itself is process-fatal.
- Global shutdown still sets the shared shutdown marker before broadcasting.
  Cooperative worker exits are not misclassified as failures, and the main
  task applies its existing bounded join, abort, and cancellation-confirmation
  policy to every management worker.
- The fixed-size ownership boundary makes a future additional management task
  an explicit architecture change rather than allowing another detached
  `tokio::spawn`.
- No management wire schema, command semantics, membership policy, reporting
  interval, user-plane handling, or blind-node privacy boundary changes in
  this milestone.

### v0.77 Required DNS startup and forwarding are lifecycle-owned

[DNS-STARTUP-READINESS 2026-07-30 by Codex]

- An enabled built-in DNS proxy now binds the configured VPN gateway address
  before systemd readiness can be reported. A bind conflict or missing local
  address fails the startup transaction instead of disappearing inside a
  detached task after the node advertises a usable privacy data plane.
- The production runtime uses the new fallible `start_dns_proxy` entry point.
  The original `spawn_dns_proxy` signature remains available for backward
  source compatibility, but it is no longer used by production startup.
- The enabled DNS task is a required process task. Unexpected normal return,
  panic, or cancellation enters the same graceful recovery path as the API,
  UDP ingress, TUN egress, and configured commitment follower.
- At most 256 opaque DNS datagrams may be forwarded concurrently. Additional
  datagrams are dropped before payload cloning, and diagnostics expose only
  aggregate power-of-two drop counts plus the fixed configured limit.
- Every forwarding child is owned by a Tokio `JoinSet`. Shutdown aborts and
  reaps all children before the DNS parent completes, so upstream requests
  cannot become detached work after node shutdown or supervised recovery.
- An operating-system receive error stops the required DNS task instead of
  creating an unbounded hot loop. The required-task supervisor then recovers
  the complete process so health cannot remain green with a failed listener.
- The privacy boundary is unchanged: the proxy treats datagrams as opaque,
  matches responses only by the two-byte DNS transaction identifier, and does
  not parse, store, index, or log query names, answers, client addresses,
  destinations, domains, URLs, or browsing history.

### v0.76 Required data-plane receive failures are bounded

[DATA-PLANE-FAILURE-POLICY 2026-07-30 by Codex]

- The always-required UDP ingress task and Linux TUN egress task now run
  behind the same process-level required-task supervisor as the API and an
  enabled commitment follower.
- A transient receive error retries with source-blind exponential backoff from
  25 milliseconds to a one-second ceiling. Any successful receive resets the
  consecutive failure streak.
- Eight consecutive receive failures stop the affected task. The supervisor
  then enters the existing graceful process-recovery path instead of leaving
  systemd with an API-healthy process whose privacy data plane is unavailable.
- Backoff waits remain interruptible by the global shutdown broadcast. Normal
  operator, signal, and programmatic shutdown therefore do not wait for a
  retry timer and do not create a false critical failure.
- Logs contain the local operating-system error and aggregate consecutive
  count only. The process failure channel retains fixed reasons and never
  receives packet bytes, client addresses, session identities, destinations,
  routes, domains, or other user-plane metadata.
- DNS startup transactionality was intentionally not mixed into v0.76 and is
  completed separately by the v0.77 lifecycle-owned DNS milestone above.

### v0.75 Required background tasks are process-supervised

[REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex]

- The node API listener group and an explicitly enabled commitment follower
  now run behind one shared required-task wrapper. The wrapper owns each inner
  `JoinHandle`, so an unexpected normal return, panic, or cancellation reaches
  the main runtime failure channel immediately.
- A reported required-task failure enters the existing graceful shutdown path.
  A systemd-managed production node can therefore restart the complete process
  instead of remaining half healthy with an API but no follower.
- The wrapper emits fixed failure reasons only. Rust panic payloads, task
  internals, endpoints, identities, routes, records, and client data cannot
  cross this operational boundary.
- Global operator, signal, and programmatic shutdown remain non-errors. The
  process sets the shared shutdown marker before broadcasting, and wrappers
  suppress expected cooperative task exits after that marker.
- Each wrapper owns its inner `JoinHandle` through an abort-on-drop guard.
  Bounded shutdown cancellation therefore propagates to the inner task instead
  of triggering Tokio's default detach-on-drop behavior and leaking work past
  the shutdown report.
- The existing RAII follower guard remains responsible for immediately
  changing local follower readiness to `stopped`; process supervision adds
  recovery orchestration rather than replacing fail-closed status.
- Witness reconciliation, miner, cleanup, discovery gossip, and other
  degradable workers are not silently promoted to required tasks in this
  release. Each needs an explicit product availability decision and local
  fail-closed state before process-fatal supervision is appropriate.

### v0.74 Follower readiness has a bounded lifetime

[FOLLOWER-READINESS-LIVENESS 2026-07-30 by Codex]

- A follower synchronization task now owns an RAII liveness guard. Normal
  return, panic unwinding, and Tokio cancellation all revoke process-local
  readiness by moving the legacy runtime state to `stopped`.
- A signed equal-tip producer checkpoint remains fully ready for at most three
  configured polling windows. At the exact deadline, the additive
  `follower_readiness_state` becomes `stale` and
  `follower_fully_ready` becomes false until another signed convergence round
  succeeds.
- `follower_convergence_confirmed_at` and
  `follower_readiness_stale_after` expose the anonymous operational boundary to
  local status consumers and the signed heartbeat. Existing fields and legacy
  `state=current` remain backward compatible.
- The follower changes state to `syncing` before any remote I/O, so an ordinary
  slow request already fails closed. The freshness deadline covers the
  different failure class where the task disappears between scheduled rounds.
- The deadline derives only from validated local configuration. It contains no
  peer identity, endpoint, witness set, hash, signature, payload, route, owner,
  or client metadata, and it cannot affect chain choice, certificate policy,
  consensus, finality, or authority.

### v0.73 Effective follower readiness is fail-closed

[FOLLOWER-EFFECTIVE-READINESS 2026-07-30 by Codex]

- The legacy synchronization `state` remains backward compatible and continues
  to describe block transport and convergence only.
- Additive `follower_readiness_state` and `follower_fully_ready` fields combine
  that block state with the exact audited-tip certificate policy. A follower is
  fully ready only after an equal-tip producer checkpoint and either a disabled
  witness policy or a durable certificate satisfying the current local pins and
  threshold.
- Missing, unavailable, stale, configuration-invalid, or security-stopped
  certificate evidence therefore cannot coexist with a misleading
  `follower_fully_ready=true`.
- `certified_recovered` remains explicitly degraded: an exact threshold-
  certified recovered prefix does not prove that an unavailable producer has
  no later tip.
- Non-follower roles report `not_applicable`. Unknown follower or certificate
  states fail closed as synchronizing or waiting for certificate.
- The fields expose no node identity, endpoint, witness membership, certificate
  frame, hash, signature, raw error, block, record, payload, route, or client
  metadata. They are operations evidence only and cannot affect authority,
  consensus, finality, reputation, or fork choice.

### v0.72 Bounded retry for deferred follower certificate persistence

[FOLLOWER-CERTIFICATE-RETRY 2026-07-30 by Codex]

- The follower runtime no longer represents one synchronization round as an
  opaque four-element tuple. A typed result now names inserted blocks, remote
  tip height, block backlog, deferred certificate persistence, and certified
  carrier recovery independently.
- When an authentic certificate loses a concurrent local tip or policy
  persistence race, the next round is scheduled promptly instead of waiting the
  normal synchronization interval. Retry delay grows as `1, 2, 4, ...` seconds
  and is capped by the configured normal interval.
- A real block backlog retains the existing one-second continuation priority.
  Certificate deferral has its own process-local streak and cannot increment or
  reset the block transport failure backoff.
- Any subsequent non-deferral result clears this dedicated streak; availability
  and security failures retain their existing independent control paths.
  Persistent local churn therefore converges back to the normal interval rather
  than becoming a hot loop.
- The scheduler stores no source identity, endpoint, witness set, frame, hash,
  signature, raw error, record, payload, route, or client metadata. It changes
  neither chain state nor authority, consensus, finality, reputation, or fork
  choice.

### v0.71 Verified transport is not durable certificate recovery

[CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex]

- Follower certificate synchronization now reports `verified_unpersisted` when
  the response passes identity, signature, canonicalization, membership,
  threshold, and exact-tip checks but a concurrent local tip or policy change
  prevents durable persistence.
- Direct coordinator retrieval counts as `coordinator` only after the exact
  current-policy certificate is durable. Pinned-carrier retrieval counts as
  `carrier_recovered` only under the same durable condition; an unpersisted
  result cannot advance either success counter or the last carrier-recovery
  timestamp.
- One pure source-by-persistence classifier is shared by both branches, so a
  future transport change cannot silently restore the old semantic mismatch.
- Every completed follower certificate round has exactly one terminal outcome:
  `coordinator`, `carrier_recovered`, `verified_unpersisted`,
  `availability_exhausted`, or `security_stopped`. The round counter equals the
  sum of those five mutually exclusive counters.
- Local status and signed management heartbeat expose the new aggregate counter
  only. They still exclude coordinator/carrier identity, endpoint, witness set,
  certificate frame, hash, signature, raw error, record, payload, route, and
  client metadata.
- A verified-but-unpersisted result waits for the next bounded reconciliation.
  It grants no production authority, mutates no commitment block, ranks no peer,
  and establishes neither consensus, finality, nor fork choice.

### v0.70 Sticky, role-isolated security evidence

[STICKY-SECURITY-EVIDENCE 2026-07-29 by Codex]

- Block-page recovery, follower certificate retrieval, and coordinator
  certificate backfill now each retain the timestamp of their most recent
  fail-closed security or protocol-integrity stop.
- A later successful source may update the latest terminal result, but it
  cannot erase that process-lifetime incident time. Operators can therefore
  distinguish “currently recovered after a security stop” from “no security
  stop observed” without receiving source identity or cryptographic material.
- The three timestamps remain role- and domain-isolated, reset only when the
  runtime role is configured for a new process lifecycle, and share one
  monotonic wall-clock clamp so clock rollback cannot regress evidence.
- Local status and signed management heartbeat expose only Unix time. They
  still exclude source identity, endpoint, slot order, witness set, frame,
  hash, signature, raw error, record, payload, route, and client metadata.
- These timestamps are operational evidence only. They do not blacklist or
  rank peers, change source selection, mutate the commitment chain, grant
  authority, establish consensus or finality, or choose a fork.

### v0.69 Privacy-safe coordinator certificate-backfill telemetry

[CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex]

- Coordinator post-startup certificate backfill now has its own local status
  and signed management-heartbeat contract. It is deliberately separate from
  follower certificate synchronization, because the two roles have different
  lifecycle, policy, and operational meanings.
- Every completed coordinator backfill round records exactly one terminal
  disposition: `persisted`, `verified_unpersisted`,
  `availability_exhausted`, or `security_stopped`. The round counter equals
  the sum of those four mutually exclusive outcome counters.
- The same atomic storage update records bounded carrier attempts, the
  anonymous cooling-slot gauge at that observation, cumulative cooldown
  skips, and cumulative half-open attempts. Cancellation or later logging
  cannot leave an outcome published without the matching scheduler evidence.
- The last-observed timestamp is monotonic within the process even if the wall
  clock moves backwards. Counters are saturating and reset with the configured
  runtime role; follower calls and disabled verifier calls are no-ops.
- The API accepts no source identity, endpoint, witness set, certificate frame,
  hash, signature, raw error, block, record, owner, payload, route, or client
  metadata. Heartbeat serialization tests continue to reject those fields.
- These values are operations evidence only. They do not select carriers,
  change certificate policy, mutate the commitment chain, grant production
  authority, rank peers, establish consensus, claim finality, or choose forks.

### v0.68 Fail-closed certificate carrier recovery

[CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex]

- Follower certificate refresh and coordinator post-startup certificate
  backfill now share one bounded carrier primitive for candidate
  normalization, typed circuit decisions, transport attempts, and failure
  classification. This removes two implementations of the same trust rule.
- Only a classified availability fault may advance to the next exact
  operator-pinned carrier. Decode, endpoint-policy, authentication, responder,
  signature, canonicalization, tip, membership, threshold, digest, and durable
  evidence failures stop immediately; a later source cannot hide them by
  returning a valid-looking certificate.
- Any fully verified response ends the carrier round. If certificate
  persistence returns `false` because the local audited tip or policy changed
  concurrently, the runtime reports an anonymous `verified_unpersisted`
  disposition and waits for the next reconciliation instead of requesting
  additional sources.
- Coordinator backfill now retains its own process-lifetime certificate
  circuit. Two consecutive availability failures place the corresponding
  anonymous fixed slot into a 60-second monotonic cooldown, with bounded
  half-open recovery after expiry. This state is separate from follower and
  block-page circuits.
- Runtime logs contain only disposition, checkpoint height, signer/threshold
  counts, attempts, cooldown skips, half-open attempts, and cooling-slot count.
  They never include source identities, endpoints, slot order, errors,
  signatures, certificate bytes, hashes, records, payloads, routes, or client
  metadata.
- The coordinator retains the existing maximum of three carrier attempts per
  reconciliation round. Certificate exchange remains post-startup evidence:
  it cannot satisfy the live startup witness threshold, select a chain, vote,
  establish consensus, or grant finality.
- Regression tests prove that a malformed first carrier prevents contacting a
  later carrier and that repeated unavailable coordinator-backfill sources
  cool across rounds until no transport request is issued.

### v0.67 Typed and isolated carrier circuits

[TYPED-CARRIER-CIRCUIT 2026-07-29 by Codex]

- Block-page and checkpoint-certificate carrier recovery now use one generic
  fixed-slot circuit implementation. Rust domain markers make their circuit
  types distinct at compile time, so a caller cannot accidentally pass mutable
  block availability state into certificate recovery or the reverse.
- The follower owns one process-lifetime instance per domain. Repeated
  classified certificate-carrier availability failures now open the same
  bounded 60-second monotonic cooldown already proven for block pages, while
  the configured coordinator remains mandatory and is attempted every round.
- Certificate and block circuits never share failures, cooldowns, gauges, or
  counters. A block carrier outage therefore cannot suppress retrieval of
  independently signed certificate evidence, and a certificate endpoint outage
  cannot alter block-page scheduling.
- Only the narrow availability allowlist advances or opens a circuit. Decode,
  endpoint-policy, authentication, responder, signature, canonicalization, tip,
  member, threshold, digest, persistence, and other integrity failures remain
  terminal and are never hidden by trying another carrier.
- Local status and signed heartbeat add certificate-specific anonymous cooling
  slots, cumulative cooldown skips, and cumulative half-open attempts. These
  values contain no identities, endpoints, slot order, errors, deadlines,
  timing by source, certificate bytes, hashes, signatures, blocks, records,
  owners, payloads, routes, or client metadata.
- Compatibility remains additive. The existing public certificate sync helper
  creates a fresh circuit exactly as before; only the long-running server
  follower passes retained typed state across rounds.
- Tests prove block/certificate telemetry isolation, follower-only accounting,
  exact heartbeat serialization, and a three-round certificate outage where
  real carrier attempts fall from two to zero after both anonymous slots cool.

### v0.66 Privacy-safe block-carrier circuit telemetry

[BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex]

- Local status and signed management heartbeat now expose three operational
  aggregates: carrier slots cooling at the latest follower observation,
  cumulative selections skipped during cooldown, and cumulative requests
  attempted after an anonymous slot entered half-open state.
- The current gauge is replaced on every terminal page retrieval; skip and
  half-open counters are additive and saturating for the process lifetime.
  They distinguish real outbound attempts from requests intentionally avoided
  by the circuit breaker without inventing per-peer health history.
- Pin normalization and circuit-slot alignment now happen before each direct
  coordinator request. A pin-count change therefore clears positional state
  even when the coordinator succeeds and no carrier fallback is needed.
- Circuit storage still contains only bounded positional failure state and
  monotonic deadlines. Status receives no slot order, coordinator or carrier
  identity, witness set, endpoint, status code, raw error, request id, per-slot
  deadline, per-source timing, block, certificate, signature, payload, owner,
  route, or client metadata.
- The counters remain process-local operations evidence. They cannot influence
  source eligibility, witness policy, certificates, canonical chain state,
  production authority, reputation, consensus, finality, or fork choice.
- Tests cover follower-only/reset semantics, additive accounting, current gauge
  replacement, exact JSON heartbeat fields, cooldown skip accounting, and one
  real half-open request that fails availability and reopens before another
  pinned carrier completes verified delivery.

### v0.65 Block-carrier circuit breaker

[BLOCK-CARRIER-CIRCUIT-BREAKER 2026-07-29 by Codex]

- The follower now keeps one process-only availability circuit for the
  normalized operator-pin positions. Two consecutive classified availability
  failures put that slot into a 60-second monotonic cooldown across sync rounds.
- A cooling slot is skipped before any network request. This prevents one
  repeatedly unavailable carrier from consuming the same connection budget on
  every follower round while other exact pins remain healthy.
- The coordinator is still contacted first for every page. The circuit cannot
  skip the coordinator, import a discovery peer, change the configured pin set,
  or grant proposer, checkpoint, certificate, consensus, finality, or
  fork-choice authority.
- After cooldown, one half-open probe is allowed. A fully verified page closes
  the slot; another availability failure immediately reopens it for the same
  bounded interval.
- Every observed decode, endpoint-policy, responder, signature, proposer,
  continuity, pagination, rollback, size, or storage error remains terminal.
  Security failures are never converted into cooldown events and never fall
  through to a later carrier.
- Circuit slots contain only a failure count and `Instant` deadline. They hold
  no node id, endpoint, status code, error text, payload, route, or wall-clock
  timestamp; they are not persisted, serialized, or logged. Later v0.66 status
  reports only anonymous aggregate counts, never slot contents or order. A
  pin-count change clears all slots to prevent positional state from being
  reassigned silently.
- Compatibility remains additive: the public one-page recovery API creates a
  fresh circuit, so its historical direct-first operator-order behavior is
  unchanged. Only the server follower task retains circuit state across rounds.
- Tests cover threshold opening, monotonic cooldown, failed half-open reopening,
  verified recovery, pin-count reset, and a real three-round HTTP path where
  carrier attempts fall from `2, 2` to `1` without weakening verification.

### v0.64 Multi-page block-carrier handoff

[MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex]

- The follower now owns one carrier cursor for each bounded multi-page sync
  round. A pinned carrier that delivers a fully verified page becomes the first
  carrier attempted for the next page in that same round.
- The coordinator is still attempted first on every page. Carrier preference
  changes transport order only after a classified coordinator availability
  fault; it cannot bypass the producer, change proposer authority, or select a
  permissionless discovery peer.
- Availability failures advance cyclically through the same normalized,
  deduplicated, operator-pinned set. A carrier that disappears between pages
  hands off directly to the next exact pin instead of retrying earlier failed
  pins first.
- Decode, endpoint-policy, identity, signature, proposer, continuity,
  pagination, rollback, size, and storage failures remain terminal. Tests prove
  that a malformed preferred carrier stops with `security_stopped` even when a
  later valid carrier is available.
- The cursor stores only a bounded array index. It is discarded after the sync
  round, never persisted or serialized, and never enters status, heartbeat,
  logs, source selection policy, reputation, certificates, consensus, finality,
  or fork choice.
- The public one-page API preserves its original signature and operator-order
  behavior by creating a fresh default cursor. Only the server's existing
  multi-page follower loop shares cursor state.
- A 17-block loopback integration test proves two-page transfer with three
  exact pins: one unavailable pin, one initially successful carrier that then
  disappears, and a final carrier that completes the original
  coordinator-signed chain. The second page avoids recontacting the earlier
  unavailable pin.

### v0.63 Privacy-safe block-page carrier telemetry

[FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex]

- Every follower commitment-block page retrieval now records exactly one typed
  terminal disposition: `coordinator`, `carrier_recovered`,
  `availability_exhausted`, or `security_stopped`.
- Process-local totals separately count terminal page pulls, direct coordinator
  successes, actual bounded carrier requests, carrier page recoveries, exhausted
  source availability, and fail-closed security stops. Timestamps remain
  monotonic if the wall clock moves backward.
- `carrier_recovered` means only that a pinned carrier delivered an
  authenticated page whose blocks remain signed by the configured coordinator.
  It does not claim checkpoint certification. A terminal carrier page must
  still pass the separate exact-tip certificate gate before runtime may report
  `certified_recovered`.
- A direct-only follower can report `availability_exhausted` with zero carrier
  attempts. This means its configured eligible source budget was exhausted; it
  never implies that another node was contacted.
- Local status and signed heartbeat expose the same additive fields. They
  contain no coordinator or carrier identity, witness set, endpoint, block,
  certificate frame, hash, signature, commitment, raw error, owner, payload,
  route, request identifier, or client metadata.
- These values are process-lifetime operational evidence only. They do not
  persist as ledger state and cannot influence source selection, production
  authority, certificate policy, reputation, consensus, finality, or fork
  choice.
- Tests enforce mutually exclusive accounting, follower-only updates,
  monotonic timestamps, invalid-policy security stops, direct-only exhaustion,
  and a real pinned-carrier page recovery through loopback HTTP.

### v0.62 Certified block-carrier recovery

[CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex]

This milestone proves that a follower can restore an original coordinator-signed
Memory Chain prefix while the producer is temporarily unavailable:

- The configured coordinator is always attempted first.
- Fallback is disabled unless local policy requires at least two independent
  checkpoint witnesses and the configured pin set can satisfy that threshold.
- At most three distinct operator-pinned witnesses may be tried, in operator
  order. Permissionless discovery never chooses a block source.
- Fallback is allowed only for admission/availability failures. Unsafe
  endpoints, malformed frames, wrong responder identity, invalid signatures,
  wrong proposer identity, broken continuity, rollback, pagination mismatch,
  oversized responses, and storage failures stop immediately.
- The carrier signs only the page envelope. Every contained block must still
  verify under the configured coordinator identity and exact local continuity.
- A terminal carrier page is not enough to report `current`. The follower must
  import or already hold an immutable checkpoint certificate for that exact
  local tip satisfying its current witness pins and threshold.
- Successful outage recovery reports `certified_recovered`, not `current`,
  because a carrier can prove a certified prefix but cannot prove that the
  unavailable producer has not authored a later block.
- When the coordinator returns, the ordinary signed equal-tip checkpoint path
  restores the stronger `current` state automatically.

Authority boundary:

```text
carrier envelope signature = authenticated transport
coordinator block signature = block authorship
threshold checkpoint certificate = certified recovered prefix
live coordinator checkpoint = current-tip convergence
```

This remains authenticated replication, not global consensus, longest-chain
selection, economic finality, or permissionless fork choice. Public status and
heartbeat retain only aggregate state and heights; carrier identities,
endpoints, routes, block hashes, signatures, commitments, and user data remain
local or absent.

AeroNyx currently has several important building blocks:

- Rust privacy protocol node runtime.
- Aggregate health, capacity, and event reporting to backend/nodeboard.
- `nodeboard` for node registration, health review, capacity decision, and incident closure.
- Memory Chain primitives and append-only ledger structures.
- Chat relay and wallet route cache concepts.

### v0.61 Follower certificate readiness is tip-bound

[FOLLOWER-CERTIFICATE-TIP-BINDING 2026-07-29 by Codex]

- Every applicable follower certificate-policy result is now bound to the
  exact fully audited local tip height evaluated by the cryptographic verifier.
- Status compares that height with the latest complete process integrity
  baseline. If the chain advances before certificate refresh, an old `ready`
  result is immediately projected as `waiting_for_certificate`.
- If the complete integrity baseline disappears during re-audit or after a
  fail-closed append condition, the old result is projected as
  `waiting_for_convergence`.
- The evaluated height is aggregate chain progress already available in
  integrity status. No witness identity, endpoint, block hash, certificate
  member, signature, payload, owner, route, or client metadata is exposed.
- This closes an observability overclaim window only. It does not change chain
  selection, evidence authority, consensus, finality, or certificate storage.

### v0.60 Follower checkpoint-certificate policy readiness

[FOLLOWER-CERTIFICATE-READINESS 2026-07-29 by Codex]

- A follower now reports whether its current fully audited local commitment tip
  satisfies its current operator-pinned witness set and signature threshold.
- `ready` is set only by the exact local cryptographic policy verifier. A
  successful HTTP response, a signer count, or a previous certificate under
  retired pins cannot manufacture readiness.
- Runtime state distinguishes `disabled`, `waiting_for_convergence`,
  `waiting_for_certificate`, `ready`, `source_unavailable`,
  `security_stopped`, and `configuration_error`.
- Restart and policy reconfiguration reset readiness before reevaluation.
  Existing durable evidence may restore `ready` without a network transfer,
  but only when it matches the current tip, pins, and threshold.
- Public/management status exposes only state, readiness, evaluation time,
  configured witness count, and minimum signer count. Witness identities,
  endpoints, certificate bytes, hashes, signatures, and raw errors remain
  outside the status contract.
- This is process-local operational readiness. It is not consensus, global
  finality, authority, reputation, leader election, or fork choice.

### v0.59 Runtime identity separation

[RUNTIME-IDENTITY-POLICY 2026-07-29 by Codex]

- Static TOML validation cannot know the Ed25519 public identity derived from a
  node's private key. The server now performs an identity-aware trust-policy
  check before initializing transports, storage, listeners, or background
  tasks.
- A follower fails startup when its pinned coordinator is itself.
- Coordinators and followers fail startup when any external checkpoint witness
  pin is the local node identity. This prevents a nominal 2-of-N policy from
  silently counting local authority as independent corroboration.
- Disabled MemChain configurations remain backward compatible and do not parse
  inactive trust fields.
- The existing runtime guard remains as defense in depth, but a production
  process can no longer become API-healthy while follower synchronization was
  silently disabled by a self-reference.
- This is local fail-closed configuration validation. It adds no vote, quorum,
  consensus, finality, fork choice, discovery trust, or user-plane metadata.

### v0.58 Follower certificate policy activation

[FOLLOWER-CERTIFICATE-CONFIG 2026-07-29 by Codex]

- Fixed a role-validation conflict that previously made the implemented
  follower certificate path impossible to enable through a valid production
  configuration: witness pins were accepted only on a coordinator even though
  a follower is forbidden from also being the coordinator.
- `commitment_witness_node_ids` and `commitment_witness_min_verified` now form a
  shared operator-pinned trust-policy primitive. Coordinators use it for
  startup evidence/certification; followers use it only after signed
  convergence to verify and recover an immutable current-tip certificate.
- Coordinator production controls remain coordinator-only:
  `commitment_witness_startup_required` and
  `commitment_coordinator_lease_required` are rejected on followers.
- A follower certificate policy requires at least two distinct witnesses,
  threshold two or greater, and witnesses different from the pinned
  coordinator. Defaults remain certificate-disabled and backward compatible.
- This change adds no automatic trust, discovery-derived membership, vote,
  quorum, consensus, finality, fork choice, chain mutation, or user-plane
  metadata. A carrier still transports evidence bytes only.

### v0.57 Privacy-safe certificate recovery telemetry

[FOLLOWER-CERTIFICATE-TELEMETRY 2026-07-29 by Codex]

- The follower runtime now reports one mutually exclusive terminal result for
  each real checkpoint-certificate transport round: `coordinator`,
  `carrier_recovered`, `availability_exhausted`, or `security_stopped`.
- Process-local totals separately count completed rounds, direct coordinator
  success, bounded carrier attempts, carrier recoveries, exhausted availability,
  and fail-closed security stops. The latest completion and latest carrier
  recovery timestamps are monotonic within one process lifetime.
- Local status and signed heartbeat use the same additive contract. Existing
  consumers remain compatible, while backend/nodeboard can prove that carrier
  recovery is actually exercised instead of inferring it from generic sync
  failures.
- Telemetry retains no coordinator or carrier identity, witness set, endpoint,
  certificate frame, hash, signature, raw error, request identifier, payload,
  owner, route, or client metadata. It cannot reconstruct which source was
  contacted or what certificate was transported.
- These counters are operational evidence only. They do not change canonical
  chain state, witness policy, production authority, startup gates, reputation,
  consensus, finality, or fork choice.
- Unit and integration tests enforce the exact accounting invariant, ignore
  non-follower roles, clamp timestamps against clock rollback, record real
  carrier recovery, and prove that a malformed coordinator response records a
  security stop without trying a valid carrier.

### v0.56 Follower checkpoint-certificate carrier recovery

[FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex]

- A converged follower still requests its fixed coordinator first. Only a
  narrow, explicit availability class permits fallback: missing/expired
  discovery state, missing endpoint, connect/timeout/interrupted-body failure,
  or HTTP `403`, `404`, `408`, `429`, `500`, `502`, `503`, or `504`. On this
  endpoint, `403` means requester admission has not converged; `401` remains a
  terminal authentication failure.
- Carrier candidates are limited to the operator's existing checkpoint-witness
  pins, preserve configured order, exclude the local node and coordinator, and
  are capped by the protocol's three-member certificate bound. Permissionless
  discovery peers can never become evidence carriers through peer count.
- A carrier transports bytes only. The follower independently verifies the
  carrier response signature, exact local tip, canonical frame, certificate
  digest, every embedded historical witness signature, distinct membership,
  current witness pins, local signer threshold, timestamps, and durable
  evidence-vault invariants before accepting the certificate.
- Any decode, canonicalization, size, signature, responder, chain, request,
  timestamp, tip, policy, member, digest, or persistence failure stops the
  round immediately. A later carrier must never hide a security anomaly from
  the coordinator or an earlier pinned source.
- Exhausting all carriers preserves the coordinator's original privacy-safe
  availability code for backward-compatible operations. Certificate failure
  remains additive and cannot mutate the canonical chain, choose a fork,
  authorize production, satisfy startup witness gates, or create consensus.
- Integration tests cover a dead coordinator followed by successful recovery
  from a witness carrier, plus a malformed coordinator response that must fail
  closed even while a valid carrier is online.

### v0.55 Follower checkpoint-certificate replication

[FOLLOWER-CERTIFICATE-SYNC 2026-07-29 by Codex]

- A Block Sync follower now requests the coordinator's current-tip checkpoint
  certificate only after the signed block page and signed checkpoint paths have
  independently established exact chain convergence.
- The coordinator is certificate transport, not certificate authority. Every
  historical member frame is independently verified against the follower's
  current operator-pinned witness allowlist and minimum signer threshold before
  the immutable certificate enters local storage.
- A retained same-height certificate is considered current only after the
  complete bounded evidence vault is re-audited and all stored members still
  satisfy the current local witness policy. Rotating witness pins therefore
  cannot leave a stale policy certificate looking valid.
- Certificate absence on a mixed-version coordinator is additive evidence
  unavailability. It does not undo a signature-verified follower tip, trigger
  fork choice, mutate the canonical chain, or satisfy the live coordinator
  startup witness gate.
- Successful replication lets full-node followers preserve independently
  verifiable confirmation evidence across coordinator outages and restarts.
  It remains operator-pinned corroboration, not permissionless consensus,
  global finality, voting, leader election, or financial state.
- Logs and status remain aggregate-only. No witness identities, endpoints,
  hashes, signatures, request ids, frames, commitments, owners, ciphertext,
  payloads, client addresses, routes, DNS contents, or social graph data are
  exported.

### v0.54 Directory transport degradation/recovery lifecycle

[DIRECTORY-TRANSPORT-LIFECYCLE 2026-07-29 by Codex]

- Directory transport health policy now belongs to the service runtime rather than the HTTP presentation layer. The API serializes the canonical service-owned classification, thresholds, invariants, and transition evidence instead of independently reimplementing them.
- The process runtime records aggregate transitions into `degraded` and back into `healthy`, the current degradation age, and the latest degraded/recovered ages. Repeated failures while already degraded and repeated successes while healthy do not create duplicate transitions.
- Lifecycle invariants fail closed: degraded transitions may exceed recovery transitions by at most one, an open transition must agree with current bounded-window health, transition timestamps must exist exactly when their counters require them, and their ordering must agree with the current state.
- Transport timestamps are monotonic maxima. If wall-clock time moves backward because of NTP or operator correction, public ages and lifecycle ordering do not regress even though the outcome is still counted.
- This lifecycle is process-only availability telemetry. It never inserts or resolves a durable Directory security incident, never changes producer quarantine, and never affects authority, peer reputation, routing, fork choice, consensus, or finality.
- Public status remains aggregate-only: no outcome sequence, per-request timestamp, peer, producer, carrier, endpoint, URL, operation, status code, request id, frame, payload, or user-plane data is retained or serialized.

### v0.53 Bounded recent Directory transport health

[DIRECTORY-TRANSPORT-WINDOW 2026-07-28 by Codex]

- A single successful request must not immediately erase evidence of a recently unstable synchronization transport. The prior latest-outcome classification could report `healthy` after one success even when half of the observed process requests had just failed.
- The Directory synchronization runtime now retains at most 32 terminal outcome classes in a private `VecDeque`. It continues to expose lifetime counters, but derives current health from recent success/failure totals plus the bounded trailing failure count.
- `degraded` means that at least 20% of the current recent window failed or at least three consecutive requests failed. `healthy` therefore means the bounded recent evidence is below both published thresholds, not merely that the last request succeeded.
- Runtime invariants require recent successes plus recent failures to equal recent requests, the window never to exceed its declared capacity or lifetime total, and idle lifetime/recent state to agree. Any violation is reported as `inconsistent`.
- The public status remains aggregate-only and process-local. It does not expose the outcome sequence, peer, producer, carrier, endpoint, URL, operation, status code, request id, frame, payload, or user-plane metadata.
- Status strings, protocol frames, retry/failover behavior, persistence schema, authority boundaries, and lifetime telemetry remain backward compatible; the new fields are additive.

### v0.52 Privacy-safe Directory transport outcomes

[DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex]

- The Directory synchronization coordinator now scopes one aggregate transport recorder around each complete synchronization round. Deep range, object, witness, policy-anchor, and carrier helpers remain observable without adding metrics parameters to protocol-verification interfaces.
- Tokio task-local ownership keeps the profile boundary exact: production Directory synchronization is counted, while local/VPN operator smokes and standalone protocol helpers are not misclassified as synchronization traffic.
- Every completed coordinator-owned HTTP exchange contributes to exactly one terminal bucket: success, connect timeout, request timeout, connect failure, other request failure, non-success HTTP status, oversized success response, or interrupted success-body stream.
- `/api/discovery/directory/status` exposes only aggregate counts, coarse ages, latest outcome class, and a `terminal_outcomes_consistent` invariant. It never exposes peer, producer, carrier, endpoint, URL, operation, status code, request id, frame, response body, payload, or user-plane metadata.
- Existing stable `directory_*_transport_failed` reasons, retry/failover behavior, protocol frames, status contract version, authority policy, persistence schema, and privacy boundaries remain backward compatible.
- Transport health is process diagnostics only. It is not durable evidence, peer reputation, routing rank, authority, voting, fork choice, consensus, or finality.

### v0.51 Role-specific Directory transport budgets

[PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex]

- Directory replica synchronization and operator-triggered carrier diagnostics are separate availability roles even though both speak Directory Sync V1.
- The replica synchronizer now uses the same canonical 3-second connect and 10-second request deadlines in production, standalone constructors, tests, and bounded recovery arithmetic.
- The local/VPN-only operator carrier and cold-bootstrap smokes retain their historical 12-second request budget without stretching normal synchronization failover.
- Five redirect-free, proxy-free clients are built once at startup: control, Directory sync, Directory operator, MemChain sync, and gossip. No client is built inside a request handler or periodic synchronization round.
- Startup logs publish only static timeout budgets and profile count. They do not expose endpoints, identities, request contents, selected carriers, routes, payloads, or user metadata.
- Protocol frames, API routes, signed evidence, authority policy, persistence schema, and privacy boundaries remain unchanged.

### v0.50 Process-lifetime peer transport runtime

[PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex]

- The server constructs redirect-free, proxy-free HTTP profiles before any mutable peer service starts.
- Each role retains its connect timeout, request deadline, idle-pool budget, and discovery-configured fetch timeout. The change centralizes ownership without flattening distinct availability budgets.
- Relay requests, external delivery-cache witnesses, checkpoint witnesses, coordinator leases, follower sync, Directory Replica pulls, operator carrier smoke, and discovery gossip now reuse process-lifetime connection pools.
- Peer-cache persistence no longer constructs up to two fresh HTTP clients per save. This is important because verified client-delivery events may trigger a debounced save every 250 milliseconds under load.
- A transport-profile initialization failure now fails startup before listeners, UDP/TUN, coordinator authority, or background synchronization are exposed. A configured protocol capability can no longer disappear later because its task-local HTTP client failed to build.
- Existing endpoint validation, no-proxy/no-redirect guarantees, timeouts, protocol frames, API routes, status contracts, trust policy, and privacy boundaries remain unchanged.

### v0.49 Permissionless endpoint security boundary

[DISCOVERY-ENDPOINT-SSRF 2026-07-28 by Codex]

- A valid node signature proves which identity advertised an endpoint; it does not make that endpoint safe for another host to contact.
- Operator-configured bootstrap seeds remain trusted configuration and may use DNS/private endpoints for closed deployments.
- Endpoints learned from signed permissionless descriptors must be public IPv4/IPv6 literals. DNS names are rejected to prevent DNS rebinding, and loopback, private, link-local, CGNAT, benchmark, documentation, multicast, and reserved ranges are rejected. This boundary covers discovery gossip, Chat Relay, Blind Relay probes/forwarding, onion next hops, and MemChain peer transport.
- All outbound peer URLs use one canonical parser that rejects credentials and unsupported schemes, removes untrusted paths/query/fragment, normalizes equivalent URL forms, and pins the protocol route.
- Permissionless peer clients do not inherit host proxy settings and never follow HTTP redirects. A public peer therefore cannot redirect discovery, chat, blind relay, onion, or MemChain requests into cloud metadata or an internal control-plane service.
- The wire schema, signed descriptor format, configured seed behavior, public API routes, and PeerStore signature/sequence/expiry admission rules are unchanged.
- Client-side encrypted communication and privacy connection direction.

The missing protocol foundation is node-to-node autonomy:

- Nodes should discover other compatible nodes.
- Nodes should verify other nodes by signature, not by blind trust in a central service.
- Nodes should sync signed descriptors and relay encrypted envelopes.
- Clients should eventually select routes from verified descriptors.

The immediate goal is not to build a financial blockchain. The immediate goal is to build the protocol substrate that lets independent nodes form a verifiable AeroNyx network.

## 2. Product Boundary

AeroNyx provides:

- Open privacy protocol specifications.
- Rust reference node implementation.
- `nodeboard` as an operator tool.
- Public documentation and aggregate network transparency.
- Protocol formats for node descriptors, discovery, relay, capacity reporting, and future Directory Chain snapshots.

AeroNyx does not provide:

- Centralized operation of all nodes.
- A public exit service by default.
- A guarantee that every independent operator follows the same jurisdiction or policy.
- Smart contracts or a general-purpose execution chain.
- Custody of user content, private keys, packet payloads, DNS contents, domains, URLs, or browsing history.

## 3. Design Goals

The base layer should provide:

1. Node identity.
2. Signed node descriptors.
3. Bootstrap discovery.
4. Local peer store.
5. Descriptor gossip.
6. Encrypted envelope relay.
7. Short-lived store-and-forward queues.
8. Future append-only Directory Chain for descriptor history.

The architecture should support these future products:

- Privacy relay.
- Encrypted chat relay.
- Encrypted storage.
- Memory Chain sync.
- Agent-to-agent encrypted service relay.
- No-exit onion relay.
- Limited exit only as a separate high-risk, opt-in operator capability.

## 4. Trust Model

Bootstrap services may distribute data, but they should not be the root of trust.

The intended trust model is:

```text
Node signs descriptor -> directory distributes descriptor -> clients and peers verify signature
```

This means:

- Backend/nodeboard can help discovery.
- Rust nodes and clients must verify descriptor signatures.
- Expired descriptors must be rejected.
- Revoked descriptors must be removed or marked unsafe.
- Directory snapshots can be signed by witnesses later, but node descriptor self-signature remains required.

## 5. Blind Node Invariant

The first invariant of AeroNyx privacy protocol design is:

```text
Relay nodes and Memory Chain coordinators must be blind.
```

This invariant is more important than any individual feature. If a commercial
node operator can read user content, reconstruct the social graph, or correlate
user-level traffic, the protocol has failed its privacy promise.

### 5.1 Relay node blindness

An AeroNyx relay node may process only the minimum control-plane data needed to
move an encrypted object to the next hop or local delivery queue.

Allowed relay-visible data:

- encrypted blob bytes
- bounded next-hop or delivery class
- expiry / TTL
- coarse capability class
- anti-replay or deduplication token that is not globally linkable
- aggregate counters needed for abuse control and health

Forbidden relay-visible data:

- chat plaintext
- Memory Chain plaintext
- encrypted storage plaintext
- packet payload contents
- DNS contents
- destination domains or URLs
- client public IPs
- long-lived sender-to-recipient route identifiers
- wallet-level traffic records
- stable social graph edges such as "user A talks to user B"

Relay operators must not be able to answer:

```text
Who is talking to whom?
What did they say?
Which destinations, domains, or URLs did they access?
Which wallet generated which traffic stream?
```

The first relay implementation may still have narrower metadata than a full
onion design, but every step must move toward less operator visibility, not
more. Future onion routing, cover traffic, batching, padding, and timing
defense work must be treated as privacy requirements, not decorative features.

### 5.2 Memory Chain coordinator blindness

The centralized-first Memory Chain coordinator is allowed to be a dumb
append-only ordering service only.

Allowed coordinator-visible data:

- encrypted object bytes
- object hash / content address
- append sequence or version vector
- timestamp or logical clock
- owner-controlled authorization proof that does not reveal plaintext
- coarse storage pressure and replication health

Forbidden coordinator-visible data:

- decrypted memory records
- chat plaintext
- social graph contents
- raw user identity mappings
- private keys
- recovery secrets
- plaintext file names
- semantic tags derived from plaintext
- wallet-level traffic analysis

The coordinator may order, timestamp, store, replicate, and return encrypted
objects. It must not interpret them. The correct mental model is closer to a
Git object store for ciphertext plus version vectors than to an application
database that understands user data.

### 5.3 Engineering gates

Before any new discovery, relay, Memory Chain, or onion-routing feature ships,
the implementation must answer these questions:

1. What exact fields can the node operator see?
2. Can those fields reveal content?
3. Can those fields reveal who communicates with whom?
4. Can logs, metrics, health reports, or nodeboard views leak more than the
   protocol payload itself?
5. Can timing, replay IDs, route IDs, or queue IDs become stable cross-session
   correlators?
6. What gets deleted, rotated, padded, batched, or aggregated to reduce linkage?

No feature should be considered production-ready until the privacy answer is
explicit in code comments, docs, API contracts, and nodeboard copy.

### 5.4 Design consequence for node discovery

Peer discovery may reveal node descriptors and aggregate capability metadata.
It must not reveal user routes. A `PeerStore` entry is about node capability,
not user relationships.

Discovery status may report:

- total peers
- valid peers
- public peers
- gossip freshness
- restart recovery readiness
- rejected or stale descriptor counters
- seed endpoint count

Discovery status must not report:

- user route choices
- per-user next hops
- sender-recipient pairs
- client public IPs
- destination IPs, domains, URLs, or DNS contents
- plaintext or ciphertext samples

This is the boundary between a privacy protocol and a network of readable
middleboxes.

### 5.5 Restart recovery gate for relay foundation

A fresh in-memory peer view is not enough for a commercial relay foundation.
Rust nodes restart during upgrades, host maintenance, kernel work, and incident
recovery. If the node loses all verified peers after restart and has no seed
recovery path, later relay or multihop features will fail unpredictably.

`PeerStoreStabilityStatus.restart_recovery_configured` is therefore part of
the discovery readiness contract.

The field is true when at least one restart recovery path is configured:

- discovery seed endpoints can rehydrate peers through signed gossip; or
- peer-cache persistence can restore the last verified snapshot locally.

Relay foundation readiness should require:

1. discovery enabled
2. at least two valid signed peers
3. fresh outbound gossip when gossip is enabled
4. no repeated gossip failure threshold breach
5. restart recovery configured through seed endpoints or peer cache

This gate is intentionally privacy-safe. It reports only whether recovery is
configured; it does not expose seed endpoint values, peer URLs, full public
keys, user routes, packet payloads, DNS contents, destinations, Memory Chain
plaintext, or wallet-level traffic.

## 6. Node Identity

Each Rust node needs a long-lived identity:

```text
node_id = hash(node_signing_public_key)
node_signing_key
node_transport_key
operator_public_key
created_at
key_rotation_state
```

Requirements:

- `node_id` must be deterministic from the public signing key.
- Node signing keys must be stored locally and protected by filesystem permissions.
- Transport keys may rotate more frequently than signing keys.
- Operator key binds the node to an operator account or wallet without making AeroNyx the operator.

Potential file changes:

```text
crates/aeronyx-core/src/crypto/keys.rs
crates/aeronyx-core/src/protocol/node_descriptor.rs
crates/aeronyx-server/src/config.rs
crates/aeronyx-server/src/config_discovery.rs
crates/aeronyx-server/src/services/discovery/identity.rs
deploy/node/aeronyx-node.sh
```

## 7. Signed Node Descriptor

`NodeDescriptor` is the minimum unit of discovery.

Example shape:

```text
node_id
node_signing_public_key
node_transport_public_key
operator_public_key
protocol_version
software_version
region
endpoint
supported_transports
capabilities
policy
capacity
health_summary
epoch
issued_at
expires_at
signature
```

Capabilities should be explicit:

```text
relay
chat_relay
storage
memory_chain
directory
onion_entry
onion_middle
onion_exit_optional
agent_relay
```

Policy should be explicit:

```text
no_exit
exit_limited
max_sessions
bandwidth_limit_mbps
allowed_transports
operator_abuse_contact_hash
jurisdiction_hint
```

Privacy boundary:

- Descriptor must not include client IPs, user identities, DNS contents, payloads, domains, URLs, chat plaintext, private keys, or wallet-level traffic.

Potential file changes:

```text
crates/aeronyx-core/src/protocol/node_descriptor.rs
crates/aeronyx-core/src/protocol/mod.rs
crates/aeronyx-core/src/protocol/messages.rs
crates/aeronyx-server/src/api/vpn_health.rs
crates/aeronyx-server/src/management/reporter.rs
crates/aeronyx-server/src/services/discovery/descriptor.rs
```

## 8. Bootstrap Directory

The first version may use backend/nodeboard as a bootstrap directory.

Important distinction:

- Backend distributes signed descriptors.
- Backend does not make unsigned descriptors trustworthy.
- Clients and Rust peers verify signatures locally.

Bootstrap API proposal:

```text
GET /api/directory/snapshot
GET /api/directory/nodes/{node_id}
POST /api/directory/announce
POST /api/directory/revoke
```

Rust node API proposal:

```text
GET /api/discovery/descriptor
GET /api/discovery/peers
POST /api/discovery/announce
POST /api/discovery/gossip
```

Potential backend file changes:

```text
privacy_network/models.py
privacy_network/serializers.py
privacy_network/urls.py
privacy_network/api/directory.py
privacy_network/services/directory_service.py
```

Potential Rust file changes:

```text
crates/aeronyx-server/src/api/discovery.rs
crates/aeronyx-server/src/services/discovery/bootstrap.rs
crates/aeronyx-server/src/services/discovery/snapshot.rs
crates/aeronyx-server/src/server.rs
```

## 9. Peer Store

Every Rust node should keep a local verified peer store.

Peer entry:

```text
node_id
descriptor
source
last_verified_at
last_seen_at
score
failure_count
revoke_state
expires_at
```

Sources:

```text
bootstrap_directory
peer_gossip
manual_seed
nodeboard_registration
future_directory_chain
```

Peer store responsibilities:

- Verify descriptor signatures.
- Reject expired descriptors.
- Prefer newer epochs.
- Mark stale nodes.
- Persist known good descriptors.
- Feed route selection and encrypted relay.

Potential file changes:

```text
crates/aeronyx-server/src/services/discovery/peer_store.rs
crates/aeronyx-server/src/services/discovery/mod.rs
crates/aeronyx-server/src/config_discovery.rs
crates/aeronyx-server/src/server.rs
```

## 10. Gossip Sync

Gossip should start simple.

First version:

```text
Node A asks Node B for descriptor inventory.
Node B returns node_id + epoch + descriptor_hash.
Node A requests missing descriptors.
Node A verifies signatures and stores valid descriptors.
```

Later version:

```text
Merkle inventory by epoch.
Delta sync by descriptor hash.
Signed directory snapshot by witness set.
Revoke event propagation.
```

Gossip message types:

```text
PeerInventory
DescriptorRequest
DescriptorBatch
NodeRevoke
PeerPing
PeerPong
```

Potential file changes:

```text
crates/aeronyx-core/src/protocol/discovery.rs
crates/aeronyx-server/src/services/discovery/gossip.rs
crates/aeronyx-server/src/api/discovery.rs
```

## 11. Encrypted Envelope Relay

Nodes should relay encrypted envelopes, not plaintext.

Current implemented Phase 9 bridge:

```text
Client ChatEnvelope
  -> local Rust node verifies sender signature
  -> local online delivery if receiver is connected
  -> discovered ChatRelay peers receive the same signed encrypted envelope through:
     POST /api/chat/peer/relay
  -> receiving peer verifies signature
  -> receiving peer delivers to local receiver sessions or stores pending
```

This bridge intentionally reuses the existing `ChatEnvelope` wire contract
instead of inventing a new generic relay envelope first. That keeps the client
protocol backward compatible while proving that discovery can move encrypted
messages across nodes.

Envelope shape:

```text
message_id
from_node_id
next_hop_node_id
target_hint
ttl
created_at
expires_at
payload_ciphertext
route_hint
signature
```

The generic relay envelope above remains the future onion/agent relay shape.
The current Phase 9 implementation is narrower and safer:

- Payload type: `ChatEnvelope`
- Inbound endpoint: `POST /api/chat/peer/relay`
- Sender proof: existing `ChatEnvelope.signature`
- Deduplication: existing `ChatRelayService` online-path message id LRU
- Offline fallback: existing `pending_messages` SQLite queue
- Peer selection: `PeerStore::peers_with_capability(NodeCapability::ChatRelay)`

Relay responsibilities:

- Verify envelope signature if required by message class.
- Drop expired envelopes.
- Deduplicate by `message_id`.
- Enforce rate limits.
- Forward to next hop or store briefly for offline target.
- Never inspect plaintext payload.

Potential file changes:

```text
crates/aeronyx-core/src/protocol/envelope.rs
crates/aeronyx-core/src/protocol/messages.rs
crates/aeronyx-server/src/services/relay/mod.rs
crates/aeronyx-server/src/services/relay/envelope_queue.rs
crates/aeronyx-server/src/services/relay/forwarder.rs
crates/aeronyx-server/src/api/relay.rs
```

Existing files to reuse:

```text
crates/aeronyx-server/src/api/chat_peer.rs
crates/aeronyx-server/src/services/chat_relay.rs
crates/aeronyx-server/src/services/wallet_routes.rs
crates/aeronyx-server/src/services/routing.rs
crates/aeronyx-server/src/services/peer_store.rs
crates/aeronyx-server/src/server.rs
```

## 12. Store-and-Forward Queue

For offline chat, agent messages, or delayed relay, each node may keep a bounded pending queue.

Queue item:

```text
message_id
target_hint
next_hop_node_id
ciphertext
expires_at
attempt_count
next_retry_at
last_error
```

Rules:

- Queue must be size limited.
- TTL must be short by default.
- Payload remains ciphertext.
- Queue metadata must avoid user browsing or wallet-level traffic history.
- Operators may disable store-and-forward capability.

Potential file changes:

```text
crates/aeronyx-server/src/services/relay/envelope_queue.rs
crates/aeronyx-server/src/config_discovery.rs
crates/aeronyx-server/src/config_chat_relay.rs
```

## 13. Directory Chain Without Smart Contracts

Directory Chain is an append-only descriptor-attestation ledger foundation.
V1 now has a protocol core, a local producer journal, authenticated bounded
serving, and producer-isolated remote replicas. It is not global consensus or
finality: every remote chain remains explicitly scoped to its signing producer.

It is not:

- A smart contract platform.
- A token execution layer.
- A financial settlement chain.

It is:

- A signed history of node descriptor events.
- A way to audit node announce/update/revoke events.
- A basis for clients and nodes to verify directory snapshots.

Event types:

```text
NodeAnnounce
NodeUpdate
NodeRevoke
CapabilityUpdate
PolicyUpdate
WitnessSignature
DirectorySnapshot
```

Existing primitives to reuse:

```text
crates/aeronyx-core/src/ledger/block.rs
crates/aeronyx-core/src/ledger/fact.rs
crates/aeronyx-core/src/ledger/merkle.rs
crates/aeronyx-core/src/ledger/record.rs
crates/aeronyx-server/src/services/memchain/aof.rs
crates/aeronyx-server/src/services/memchain/mempool.rs
```

Current implementation files:

```text
crates/aeronyx-core/src/protocol/discovery.rs
crates/aeronyx-server/src/services/directory_chain.rs
crates/aeronyx-server/src/services/directory_replica.rs
crates/aeronyx-server/src/api/directory_chain_peer.rs
crates/aeronyx-server/src/api/directory_replica_sync.rs
crates/aeronyx-server/src/api/directory_replica_status.rs
```

## 14. Onion Routing Relationship

Onion routing should come after discovery and encrypted relay.

Minimum prerequisites:

- Signed descriptors.
- Peer store.
- Capability filtering.
- Relay-only policy.
- Encrypted envelope forwarding.
- Path selection.
- Circuit state.

Default policy:

```text
onion_entry: optional
onion_middle: optional
onion_exit: disabled by default
```

Future files:

```text
crates/aeronyx-core/src/protocol/onion.rs
crates/aeronyx-server/src/services/onion/circuit.rs
crates/aeronyx-server/src/services/onion/packet.rs
crates/aeronyx-server/src/services/onion/path_selection.rs
```

## 15. Client Product Implications

The client should be feature-oriented, not server-list-oriented.

Examples:

- Encrypted chat uses chat relay capable nodes.
- Privacy connection uses relay or onion relay capable nodes.
- Encrypted backup uses storage capable nodes.
- Memory sync uses memory_chain capable nodes.
- Agent-to-agent service uses agent_relay plus storage or memory_chain as needed.

Client route selection should use:

```text
descriptor signature
capabilities
policy
health
capacity
latency
region
freshness
operator risk flags
```

Potential client files to inspect later:

```text
lib/services/
lib/network/
lib/features/vpn/
lib/features/chat/
lib/features/backup/
rust/
```

Exact client paths should be filled in when the client implementation work starts.

## 16. nodeboard Product Implications

nodeboard should show descriptor and discovery health without becoming the operator.

Future UI surfaces:

- Descriptor status.
- Node ID and public key fingerprint.
- Descriptor epoch and expiry.
- Discovery source.
- Capabilities.
- Policy: no-exit, relay-only, storage-enabled, chat-relay-enabled.
- Peer count.
- Gossip status.
- Directory snapshot health.
- Revoke state.

Potential nodeboard files:

```text
types/index.ts
app/dashboard/nodes/[id]/page.tsx
app/dashboard/services/page.tsx
app/dashboard/events/page.tsx
lib/i18n/index.ts
```

## 17. Backend Product Implications

Backend should act as a bootstrap and observability service, not the source of cryptographic trust.

Backend responsibilities:

- Receive signed descriptors.
- Store descriptor history.
- Serve bootstrap snapshots.
- Expose node descriptor status to nodeboard.
- Reject invalid signatures.
- Mark expired descriptors.
- Provide aggregate stats only.

Potential backend files:

```text
privacy_network/models.py
privacy_network/serializers.py
privacy_network/urls.py
privacy_network/api/directory.py
privacy_network/services/directory_service.py
privacy_network/api/vpn_observability.py
```

## 18. Development Phases

### Phase 0 - Current State Audit

Status: Planned.

Goals:

- Confirm existing key material, node identity, and registration flow.
- Confirm current chat relay and wallet route behavior.
- Confirm current Memory Chain ledger primitives.
- Confirm client routing assumptions.

Expected output:

- Update this document with exact current-state evidence.
- No production behavior change.

### Phase 1 - NodeDescriptor MVP

Status: Planned.

Goals:

- Add `NodeDescriptor` protocol type.
- Add canonical serialization.
- Add descriptor signing and verification.
- Expose local Rust descriptor endpoint.
- Include privacy protocol health, capacity summary, capabilities, and policy.

Files likely changed:

```text
crates/aeronyx-core/src/protocol/node_descriptor.rs
crates/aeronyx-core/src/protocol/mod.rs
crates/aeronyx-server/src/services/discovery/descriptor.rs
crates/aeronyx-server/src/api/discovery.rs
crates/aeronyx-server/src/server.rs
crates/aeronyx-server/src/config.rs
```

Verification:

- Unit tests for canonical serialization and signature verification.
- `cargo fmt --check`
- `cargo check -p aeronyx-server`
- Local endpoint returns descriptor without private data.

### Phase 2 - Backend Bootstrap Directory

Status: Planned.

Goals:

- Backend accepts signed descriptors.
- Backend stores descriptor history.
- Backend serves bootstrap snapshot.
- Backend rejects invalid or expired descriptors.
- nodeboard displays descriptor status.

Files likely changed:

```text
privacy_network/models.py
privacy_network/services/directory_service.py
privacy_network/api/directory.py
privacy_network/urls.py
privacy_network/api/vpn_observability.py
types/index.ts
app/dashboard/nodes/[id]/page.tsx
```

Verification:

- Django model migration.
- Signature verification tests.
- API returns signed snapshot.
- nodeboard build passes.

### Phase 3 - Rust Peer Store

Status: Planned.

Goals:

- Rust node pulls bootstrap snapshot.
- Rust node verifies descriptors.
- Rust node persists peer store.
- Rust node exposes peer store health.

Files likely changed:

```text
crates/aeronyx-server/src/services/discovery/peer_store.rs
crates/aeronyx-server/src/services/discovery/bootstrap.rs
crates/aeronyx-server/src/config_discovery.rs
crates/aeronyx-server/src/server.rs
crates/aeronyx-server/src/api/discovery.rs
```

Verification:

- Peer store unit tests.
- Expired descriptor rejection tests.
- Snapshot pull integration test or example.

### Phase 4 - Descriptor Gossip

Status: Planned.

Goals:

- Nodes exchange descriptor inventory.
- Nodes request missing descriptors.
- Nodes verify and store descriptors.
- Nodes evict stale peers.

Files likely changed:

```text
crates/aeronyx-core/src/protocol/discovery.rs
crates/aeronyx-server/src/services/discovery/gossip.rs
crates/aeronyx-server/src/api/discovery.rs
```

Verification:

- Two-node local test.
- Descriptor sync by epoch/hash.
- Invalid descriptor ignored.

### Phase 5 - Encrypted Envelope Relay

Status: Planned.

Goals:

- Add encrypted envelope protocol type.
- Add dedup and TTL enforcement.
- Add bounded pending queue.
- Add relay forwarder.
- Integrate with chat relay or agent relay path.

Files likely changed:

```text
crates/aeronyx-core/src/protocol/envelope.rs
crates/aeronyx-server/src/services/relay/envelope_queue.rs
crates/aeronyx-server/src/services/relay/forwarder.rs
crates/aeronyx-server/src/api/relay.rs
crates/aeronyx-server/src/services/chat_relay.rs
```

Verification:

- Envelope serialization tests.
- TTL/drop tests.
- Dedup tests.
- Store-and-forward queue limit tests.
- No plaintext payload logging.

### Phase 6 - Directory Chain

Status: Protocol core, local producer persistence, authenticated serving,
producer-isolated replica import, and bounded pinned-peer pull implemented.

Goals:

- Pack authenticated descriptor commitments into hash-linked blocks.
- Add a deterministic Merkle root for canonically sorted commitments.
- Add witness signatures.
- Add snapshot validation.

Implemented in the V1 protocol core:

- A fixed production chain identifier prevents replay between independent
  directory networks.
- Each leaf commits to one already-authenticated signed node descriptor using
  a domain-separated digest; endpoints and capabilities are not copied into
  the block payload.
- Canonical ordering, exact-duplicate rejection, a 256-commitment bound, and a
  stable Merkle root make independent implementations deterministic.
- Ed25519 producer signatures bind height, timestamp, previous block hash,
  commitment root, count, and producer identity.
- Verification enforces genesis/non-genesis continuity, monotonic timestamps,
  bounded future clock skew, payload integrity, producer identity, and
  signature authenticity.
- Same-node/same-sequence conflicting commitments remain visible as evidence
  instead of being silently collapsed.

Implemented in the local Rust runtime:

- Optional `discovery.directory_chain_path` enables a dedicated SQLite journal;
  omission preserves backward-compatible disabled behavior.
- The database pins schema version, production chain id, and this node's exact
  producer identity. Identity or metadata mismatch fails startup closed.
- WAL, `synchronous=FULL`, foreign keys, and one immediate transaction keep
  signed blocks, commitment indexes, and content-addressed signed descriptor
  objects on the same atomic tip.
- Startup scans every persisted block and verifies height, previous hash,
  timestamp, Merkle payload, producer signature, stored columns, and every
  commitment index field before network listeners start. Every referenced
  descriptor object is independently signature-verified and rehashed.
- Authenticated PeerStore records are reconciled at startup, periodically, and
  during graceful shutdown. Exact commitments are skipped; new or conflicting
  authenticated descriptor observations become bounded signed blocks.
- A 64 KiB pre-deserialization limit and the protocol's 256-commitment block
  limit bound recovery memory and block construction.

Implemented in Directory Sync V1 serving:

- Domain-separated Ed25519 request and response signatures bind request ids,
  timestamps, ordered block hashes/object hashes, and the audited tip.
- `/api/discovery/peer/directory/tip`, `block-range`, and
  `descriptor-objects` use bounded binary frames and exact content addressing.
- Authority-sensitive carrier, checkpoint-witness, and policy-anchor routes
  require `discovery.directory_chain_sync_peer_node_ids` plus a current valid
  signed PeerStore descriptor.
- When Full-node Mirror Mode is enabled, valid public discovery peers may read
  only this node's own signed tip, block range, and committed descriptor objects.
  Disabling the mode preserves the original pinned-only admission behavior.
- Requests enforce timestamp freshness, replay rejection, per-peer and global
  rate limits, strict body/page/object limits, canonical decoding, and chain id.
- Every response is gated by a complete persisted-chain audit. Object batches
  are all-or-nothing and preserve the requested hash order.

Implemented in Directory Descriptor Inclusion Proof V1:

- `DirectoryDescriptorInclusionProofV1` packages one exact signed descriptor,
  its canonical commitment, zero-based block position, at most eight sibling
  hashes, the signed Directory block header, and the producer signature.
- Generic Merkle construction and verification support odd-leaf duplication,
  reject malformed tree positions and path lengths, and are exhaustively tested
  for every leaf in tree sizes 1 through 33. The signed Directory header binds
  the exact commitment count; the block contract separately rejects duplicate
  commitments.
- Verification requires three independent trust inputs: production chain id,
  pinned producer identity, and exact selected block hash. A proof cannot choose
  those inputs for its verifier.
- `DirectoryChainStore::audited_descriptor_inclusion_proof` performs a complete
  local chain audit, resolves the exact committed descriptor and block, builds
  the compact path, then re-verifies the finished proof before returning it.
  A descriptor committed by another block is never silently substituted.
- `DescriptorInclusionProofRequestV1` and
  `DescriptorInclusionProofResponseV1` are appended after all existing
  `DirectorySyncMessage` variants, preserving every older bincode discriminant.
- `POST /api/discovery/peer/directory/descriptor-inclusion-proof` requires a
  current signed peer descriptor, fresh Ed25519 request, unique request id,
  existing rate budgets, and `PinnedAuthority` admission. Enabling public
  Full-node Mirror reads does not make this route public.
- The transport response signature binds the exact request, responder, time,
  descriptor hash, block hash, and compact proof digest. Receivers must still
  call `verify_at` with their independently pinned producer and block hash.
- This proves only that one producer signed one exact Directory block containing
  one authenticated public node descriptor commitment. It is not canonical
  chain selection, voting, quorum, consensus, finality, a financial
  transaction, user activity, message delivery, or Memory Chain content proof.
- No client identity, IP, selected route, message id, payload, ciphertext,
  Memory Chain record, DNS content, destination, private key, or wallet traffic
  enters the proof or API.

Implemented in Replica Descriptor Inclusion Proof V1:

- `DirectoryReplicaStore` audits the complete requested producer namespace and
  its mirror registry membership when required, then loads the exact block and
  descriptor from the same SQLite read transaction.
- A descriptor absent from the replica, or committed in a different selected
  block, returns not found. The carrier never substitutes another block.
- `ReplicaDescriptorInclusionProofRequestV1` and
  `ReplicaDescriptorInclusionProofResponseV1` are append-only wire variants, so
  every previous bincode discriminant remains stable for rolling upgrades.
- `POST /api/discovery/peer/directory/replica-descriptor-inclusion-proof` uses
  the existing `VerifiedPublicRecovery` admission and rate/replay controls.
  Configured producer namespaces use the general audited reader; non-configured
  producers require both public mirror reads and durable retained-mirror status.
- The response has two separately verifiable layers: the original producer
  signature inside the inclusion proof and a carrier signature binding the
  exact request, producer, carrier, time, hashes, and proof digest.
- Carrier or mirror status grants availability only. It grants no producer,
  checkpoint, witness, policy, voting, fork-choice, consensus, or finality
  authority and cannot choose the verifier's trusted producer or block hash.
- The route serves only public signed node-directory metadata. It contains no
  user identity, IP, route, payload, ciphertext, message, traffic, DNS,
  destination, Memory Chain, private-key, or wallet-level data.

Implemented in Requester Replica Proof Recovery V1:

- `fetch_directory_descriptor_inclusion_proof_with_recovery` requires the
  caller to supply the original producer, exact selected producer block hash,
  and exact descriptor hash. The network cannot choose these trust anchors.
- The requester contacts the original producer route first. Only typed
  transport, optional-route, overload, or admission unavailability may enter
  carrier recovery.
- A semantic producer `proof_not_found`, noncanonical response, contract
  mismatch, invalid producer signature, wrong block/descriptor binding, or
  invalid Merkle path stops closed and never falls through to a carrier.
- Recovery selects at most two current public descriptors that explicitly
  advertise `DirectoryMirrorCarrier`; endpoint derivation is bound to the exact
  signed descriptor sequence selected for that attempt.
- Each carrier receives a fresh signed request id. Carrier route absence or a
  retained-mirror miss may try the next bounded candidate; malformed or
  cryptographically invalid carrier evidence stops immediately.
- Successful carrier recovery independently verifies the carrier envelope and
  the original producer proof. The returned transport class contains no
  carrier identity or endpoint metadata.
- This is availability recovery only. It does not create consensus, select a
  canonical producer chain, grant mirror authority, or prove user activity,
  message delivery, payload contents, traffic, or Memory Chain state.

Implemented in Directory-authenticated PeerStore Admission V1:

- `fetch_and_admit_directory_authenticated_descriptor` accepts only a producer,
  exact block hash, and exact descriptor hash selected by the local caller.
- Before any outbound request, the node fully audits its retained producer
  replica and requires that exact tuple to resolve to a valid locally rebuilt
  inclusion proof. Unknown network-supplied anchors are never probed.
- After direct or bounded carrier recovery, the node verifies the producer
  proof again, re-audits the local replica, and requires the deterministic local
  proof to equal the recovered proof exactly. This closes the public-wrapper
  construction and concurrent quarantine/change boundaries.
- Only then does the normal PeerStore path verify the descriptor validity
  window and signature, enforce capacity, reject sequence rollback/conflict,
  invalidate changed route-surface evidence, and record the coarse
  `directory_proof` source bucket.
- The admission result exposes only inserted/unchanged, direct/carrier class,
  direct-attempted, and bounded carrier-attempt count. It omits all identities,
  endpoints, hashes, requests, proof bytes, routes, and user data.
- A directory-authenticated descriptor means only that one locally selected
  producer block committed to that exact signed public descriptor. It does not
  make the descriptor an authority, grant relay permission, prove reachability,
  defeat Sybil operators, or create voting, fork-choice, consensus, or finality.

Implemented in Directory-authenticated Gossip Admission V1:

- `NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1` is appended after the
  three established discovery variants. Existing bincode discriminants
  `0/1/2` remain unchanged; the new proof announcement uses discriminant `3`.
- The frame carries an original producer, exact producer-signed block hash,
  exact descriptor commitment hash, and compact inclusion proof. The gossip
  sender is deliberately absent from the trust decision and gains no producer,
  mirror, witness, checkpoint, policy, voting, fork-choice, consensus, or
  finality authority.
- Both local-operator and public discovery routers receive the already audited
  `DirectoryReplicaStore`. A node without that store returns service
  unavailable for the stronger proof contract and never downgrades it to
  signature-only admission.
- The receiver re-verifies the producer signature, descriptor binding, Merkle
  path, chain id, exact block hash, and exact descriptor hash, then reconstructs
  the same proof from its local audited replica. The network proof must equal
  local evidence exactly before normal PeerStore processing begins.
- PeerStore then applies its existing descriptor validity, signature, capacity,
  sequence anti-rollback, and route-surface invalidation rules. Results retain
  the existing aggregate `inserted / unchanged / stale / rejected` contract.
- Invalid proofs, missing anchors, and unavailable replica trust state retain no
  producer, sender, endpoint, block, hash, proof, or route detail. Only one
  aggregate rejection is recorded.
- Legacy `DescriptorAnnounce` remains available for rolling compatibility. It
  is explicitly signature-only and must not be described as Directory-backed.
- The frame contains only public signed node-directory evidence. It contains no
  user identity, client IP, selected route, message id, payload, ciphertext,
  traffic, DNS content, destination, Memory Chain record, private key, or wallet
  activity.

Implemented in Directory-authenticated Outbound Gossip V1:

- Each gossip round reads at most 64 recent descriptor commitments from
  non-quarantined, non-local replica namespaces. This cost is bounded
  independently of retained history and runs on Tokio's blocking pool rather
  than the async network executor.
- Candidate objects must decode, reproduce their committed descriptor hash, and
  retain a valid node signature. Authentic expired descriptors remain durable
  history but are never re-announced as current routeable state. Within the
  bounded window, only the highest live sequence per producer and public node
  is eligible, preventing old-but-valid revisions from crowding out diversity.
- The cadence epoch rotates across the live candidate set. It deliberately does
  not use raw Unix seconds modulo the candidate count, which could repeatedly
  select one descriptor when a fixed gossip interval shares that divisor.
- The candidate query is only a selector, never a trust shortcut. Before
  publication, the node fully audits the selected producer namespace and
  rebuilds/re-verifies the exact compact inclusion proof against the selected
  producer, block hash, and descriptor hash.
- A proof-aware exchange sends one optional
  `DirectoryDescriptorAnnounceV1` first. Only an HTTP `422` exact-evidence miss
  may advance to one different, independently audited candidate. The sender
  stops after success or after two proof frames, then always sends the current
  legacy `DescriptorAnnounce` followed by `SnapshotRequest`. Replica
  unavailability, rate limiting, other protocol status, transport failure, or
  an unsupported new enum variant cannot suppress legacy liveness.
- Proof transport failure is best effort in this mixed-version phase. The
  established legacy exchange still determines whether the peer gossip round
  succeeded. Mandatory proof-only admission requires a future explicit
  capability/version negotiation and is not inferred from HTTP `422`.
- Process logs and `PeerStoreStatus.bootstrap` expose only aggregate capability
  checks, capable/attempted peers, proof/fallback frames, accepted peers,
  acceptance percentage, exact-evidence misses, replica unavailability, rate
  limiting, other protocol rejection, transport failure, and consecutive
  zero-acceptance rounds. They never add producer, descriptor, block, proof,
  peer, endpoint, route, message, payload, client, user, or traffic dimensions.
- Stable convergence buckets are `idle`, `legacy_only`, `converged`, `partial`,
  `evidence_diverged`, and `degraded`. These are local transport observations,
  not peer reputation, authority, quorum, voting, fork choice, consensus, or
  finality.
- The carrier still gains no producer, witness, mirror, checkpoint, policy,
  routing, voting, fork-choice, consensus, or finality authority. The proof
  authenticates one public descriptor commitment only.

Implemented in Directory Sync V1 replica pull:

- Remote blocks never enter the local producer tables. Every producer has an
  independent replica tip, block namespace, descriptor object namespace, and
  quarantine state in `directory_replica_*` SQLite tables.
- Startup audits every accepted producer prefix, every block signature/link,
  exact commitment index, content-addressed descriptor object, replica tip,
  and durable incident digest before listeners start.
- Outbound sync requires the same operator pin plus a current signed PeerStore
  descriptor. Endpoints must be public IP literals; redirects, DNS endpoints,
  loopback, private, CGNAT, documentation, and reserved ranges are rejected.
- Each low-frequency round requests one block, then hydrates exact descriptor
  objects in batches of 16. This bounds memory, request amplification, and use
  of the peer API rate budget.
- The client verifies canonical encoding, request binding, chain id, producer,
  freshness, response signature, block producer identity, exact object order,
  and every descriptor signature/hash. The replica store independently decodes
  and verifies the signed range evidence again before its atomic transaction.

Implemented in Full-node Mirror Mode V1:

- `discovery.directory_full_node_mirror_enabled` is an explicit default-off
  opt-in; `directory_full_node_mirror_max_producers` sets a hard 1-64 durable
  namespace ceiling (default 32).
- Candidates come only from fresh, signature-verified, publicly discoverable
  descriptors with safe public endpoints. Self and operator-pinned producers
  are excluded.
- Each round rotates through at most eight candidates, imports at most one
  direct page per candidate, and never uses replica carrier fallback.
- SQLite schema v9 reserves a mirror slot only in the same transaction as an
  accepted, fully verified first page. Capacity rejection rolls back the new
  producer row, so descriptor churn cannot create unbounded namespaces.
- Mirror membership is durable and intentionally has no automatic eviction in
  V1. A configured operator pin atomically promotes an existing mirror out of
  mirror classification; authority producers cannot be silently demoted.
- Lowering `directory_full_node_mirror_max_producers` below the durable retained
  count fails mirror coordinator startup. The node never deletes signed history
  merely to satisfy a changed capacity setting; operators must raise the limit
  or perform an explicit audited store migration.
- Mirror producers never enter observation convergence, observation
  checkpoints, witness thresholds, policy anchors, fork choice, voting,
  consensus, finality, or financial state.
- Public status exposes only aggregate capacity, round counts, successes,
  failures, and freshness. It never exposes mirror identities or endpoints.
- Exact repeated pages are idempotent. A producer-signed rollback, same-height
  tip fork, block fork, or contradictory empty range persists signed evidence
  and permanently quarantines only that producer; no automatic rewind, delete,
  or fork selection occurs.
- Same-node/same-sequence descriptor conflicts are retained as authenticated
  incidents without automatically quarantining an honest producer that merely
  recorded third-party equivocation.
- The status API computes exact commitment-hash overlap across each configured,
  non-quarantined producer's most recent 32 blocks. At most 16 validated pins
  participate, so work is bounded independently of retained chain history.
- A deterministic observation root binds the eligible producer identities,
  their independently signed tips, and commitments present in every eligible
  recent window. The root is operator-only and locally recomputable; it is not
  signed by the local node and grants no voting weight, fork choice, consensus,
  or finality.
- `directory_chain_sync_interval_secs` defaults to 120 seconds and accepts
  60 seconds through 24 hours. Empty peer pins disable the outbound task.
- The local/VPN operator listener exposes bounded, digest-ordered incident
  summaries at `GET /api/discovery/directory/incidents`. The default page is 20
  records, the hard maximum is 50, and the exclusive cursor is the previous
  page's final 32-byte incident digest.
- `GET /api/discovery/directory/incident?digest=<hex32>` exports one exact
  canonical `BlockRangeResponseV1` frame as base64. Before returning bytes, the
  store rechecks metadata, evidence size, chain id, canonical re-encoding,
  producer identity/signature, incident digest, and evidence SHA-256.
- Public listeners do not mount either incident route and return `404`.
  Summary output uses 12-character producer/subject fingerprints; full keys
  appear only inside the single operator evidence package because independent
  signature verification requires the producer identity.
- Incident export is deliberately read-only. No endpoint can clear quarantine,
  rewind a prefix, choose a fork, or mark evidence resolved.

Implemented in Portable Observation Certificate V1:

- `DirectoryObservationCertificateV1` packages one observer-signed checkpoint
  with independently signed `accepted` witness receipts. It is a stable
  transport contract rather than an open-ended runtime enum.
- The certificate uses a fixed version and chain id, a 64 KiB allocation bound,
  canonical witness ordering, unique responder identities, an explicit
  1-16 witness threshold, exact checkpoint hashes, request ids, timestamps, and
  Ed25519 signatures.
- The threshold records the exporting operator's current evidence policy. It
  does not create validator membership, voting weight, a quorum certificate,
  fork choice, consensus, global finality, transaction inclusion, or proof of
  user content.
- `DirectoryReplicaStore::latest_observation_certificate_for_pins` audits the
  metadata, latest retained receipt set, exact checkpoint, canonical transport
  frames, signatures, and current configured pin membership on every export.
  Receipts from retired pins remain durable history but cannot satisfy the
  current threshold.
- `GET /api/discovery/directory/observation-certificate` is mounted only on the
  local/VPN operator listener. Public listeners return `404` because the
  certificate necessarily contains complete observer and witness public keys.
- The response includes checkpoint sequence/hash, receipt count, stable
  certificate id, exact frame SHA-256, and one base64 bounded canonical frame.
  It contains no endpoints, descriptors, routes, selected hops, client IPs,
  user message ids, payloads, ciphertext, Memory Chain records, DNS contents,
  destinations, private keys, wallet traffic, or social graph metadata.
- An offline verifier must first check the frame SHA-256, decode the bounded
  V1 contract, and call `verify_at` with the production chain id and its own
  current time. Merely decoding or seeing `status=verified` is not a trust
  decision.
- `aeronyx-server directory-replica verify-observation-certificate` provides
  that exact offline path in the Rust node binary. It requires the canonical
  binary frame and the expected SHA-256 published by the transport that
  supplied it:

  ```bash
  aeronyx-server directory-replica verify-observation-certificate \
    --input /var/lib/aeronyx/evidence/observation-certificate.bin \
    --expected-sha256 <64-lower-or-upper-case-hex-characters> \
    --expected-observer <trusted-observer-node-id-hex> \
    --allowed-witness <trusted-witness-a-node-id-hex> \
    --allowed-witness <trusted-witness-b-node-id-hex> \
    --minimum-witnesses 2 \
    --json
  ```

- The command opens one regular file, enforces the protocol-owned 64 KiB plus
  magic-byte complete-frame bound before and during the read, checks the exact
  SHA-256 before decoding, rejects non-canonical re-encoding, uses the
  production chain id and the verifier's current clock, and verifies the
  observer checkpoint plus every witness receipt.
- Cryptographic validity is not treated as identity trust. The command also
  requires one independently pinned observer, a repeatable 1-16 member witness
  allowlist, and the verifier's own minimum witness threshold. It rejects an
  otherwise valid certificate if the observer differs, any included witness
  falls outside the allowlist, the allowlist is ambiguous, or the local
  threshold is not met. The certificate's self-declared threshold is reported
  but never substitutes for the verifier's local policy.
- Stable JSON output contains only aggregate verification metadata, hashes,
  checkpoint sequence/time/age, a 12-character observer fingerprint, local
  trust-policy status, and witness count/threshold. Complete observer and
  witness identities remain in the
  caller-supplied certificate because offline Ed25519 verification requires
  them; the command does not duplicate those identities into logs.
- A valid command result proves only that the included independent signatures
  bind one exact observation checkpoint under the certificate's stated
  threshold. The operator still decides whether the observer and witness set
  belong to its trust policy. The result is not consensus, finality, fork
  choice, transaction inclusion, or proof of user content.

Implemented in Durable Third-Party Certificate Import V1:

- `aeronyx-server directory-replica import-observation-certificate` reuses the
  same service-layer exact-SHA, canonical-codec, production-chain, current-time,
  signature, pinned-observer, witness-allowlist, and local-threshold verifier:

  ```bash
  aeronyx-server directory-replica import-observation-certificate \
    --input /var/lib/aeronyx/evidence/peer-observation-certificate.bin \
    --expected-sha256 <exact-frame-sha256> \
    --expected-observer <trusted-external-observer-node-id-hex> \
    --allowed-witness <trusted-witness-a-node-id-hex> \
    --allowed-witness <trusted-witness-b-node-id-hex> \
    --minimum-witnesses 2 \
    --config /etc/aeronyx/server.toml \
    --json
  ```

- The command is host-local and requires the node configuration, node identity
  key, exact binary frame, external digest, and explicit pins. There is no
  network mutation endpoint, so an unauthenticated peer cannot fill or alter
  the import store.
- SQLite schema v10 adds a hard 4,096-certificate capacity and one append-only
  import row per accepted frame. Each row binds the exact frame SHA-256,
  certificate id, observer checkpoint sequence/hash/time, canonical local trust
  policy digest, verification time, previous import digest, and importer node
  identity under the local node's Ed25519 signature.
- Metadata stores the import sequence and head digest. The row insert and head
  compare-and-swap share one immediate transaction. Exact frame plus exact
  policy re-import is idempotent; a policy substitution, same-observer
  same-sequence conflict, older checkpoint, or capacity overflow fails closed.
- Startup and explicit operator audit walk the complete bounded history,
  reconstruct every pinned policy, decode and canonically re-encode every
  frame, verify every observer/witness/importer signature, check the hash links,
  enforce per-observer monotonically increasing checkpoints, and compare the
  final row to the metadata head. Deletion, reordering, content mutation,
  signature mutation, metadata rollback, or observer rollback prevents startup.
- Aggregate status exposes only retained count, local import sequence, and local
  head digest. It does not publish observer/witness identities, certificate
  frames, endpoints, routes, or user-plane data.
- Imported evidence cannot enter producer admission, route selection,
  observation convergence, witness-policy membership, voting, fork choice,
  consensus, global finality, financial state, or proof of user content.
- Existing Directory Sync wire frames and older node behavior remain
  compatible. Schema v9 stores migrate atomically to v10 with an empty import
  history; no signed block, mirror, witness, incident, or policy evidence is
  rewritten.

Implemented in Authenticated Certificate Exchange V1:

- `ObservationCertificateRequestV1` and
  `ObservationCertificateResponseV1` are appended to `DirectorySyncMessage`;
  no existing bincode discriminant is reordered, so mixed-version peers keep
  their existing Directory Sync behavior.
- `POST /api/discovery/peer/directory/observation-certificate` is mounted only
  when the replica store passed startup audit. It uses the existing 60-second
  request freshness, request-id replay cache, per-peer/global budgets, current
  signed descriptor check, Ed25519 authentication, and `PinnedAuthority`
  admission. There is deliberately no public GET alias.
- The responder re-reads its current signed witness policy and rebuilds the
  latest certificate from audited retained evidence. Missing policy, an
  unsatisfied current threshold, invalid persistence, or absent evidence returns
  a fixed fail-closed error; the handler never exports a weaker certificate.
- The response signature binds chain id, request id, requester, responder,
  response timestamp, SHA-256, and exact certificate-frame length. The pull
  side independently enforces canonical outer encoding, expected source
  identity, response freshness, exact digest, frame bound, and signature.
- `aeronyx-server directory-replica pull-observation-certificate` performs one
  explicit, redirect-free, proxy-bypassed pull from a public node endpoint:

  ```bash
  aeronyx-server directory-replica pull-observation-certificate \
    --source-endpoint https://<pinned-node-host>:8422 \
    --expected-observer <same-pinned-source-node-id-hex> \
    --allowed-witness <trusted-witness-a-node-id-hex> \
    --allowed-witness <trusted-witness-b-node-id-hex> \
    --minimum-witnesses 2 \
    --max-age-seconds 900 \
    --config /etc/aeronyx/server.toml \
    --json
  ```

- Authenticated transport is not treated as certificate trust. After response
  verification, the command independently decodes and verifies the observer
  checkpoint, every witness receipt, the operator's complete witness allowlist,
  local threshold, production chain id, canonical bytes, and checkpoint age.
  The network age gate defaults to 900 seconds and cannot exceed 3,600 seconds.
- Only after all three gates pass (source authentication, local certificate
  policy, freshness) does the node append the exact frame to the existing
  node-signed schema-v10 import history. Source endpoints are neither logged nor
  persisted. Existing imported-certificate status remains aggregate-only.
- This exchange carries public control-plane observation evidence. It does not
  carry user messages, ciphertext, routes, selected hops, Memory Chain records,
  traffic metadata, DNS contents, destinations, private keys, wallet traffic,
  financial state, votes, fork choice, consensus, or global finality.

Still pending before Directory Chain can be described as live:

- Policy-driven multi-source certificate scheduling, independently specified
  co-signature policy, and deterministic fork selection.
- Independent implementation verification of the convergence root contract.

Files likely changed:

```text
crates/aeronyx-core/src/protocol/discovery.rs (V1 protocol core implemented)
crates/aeronyx-server/src/services/directory_chain.rs (local persistence implemented)
crates/aeronyx-server/src/services/directory_replica.rs (remote replicas implemented)
crates/aeronyx-server/src/api/directory_chain_peer.rs (serve and pull implemented)
crates/aeronyx-server/src/config.rs
crates/aeronyx-server/src/server.rs
```

Verification:

- Deterministic block hash tests.
- Snapshot root verification.
- Fork/epoch selection tests.

### Phase 7 - No-Exit Onion Relay

Status: Future.

Goals:

- Build path selection from verified descriptors.
- Add entry/middle circuit state.
- Add layered encryption packet format.
- Keep exit disabled by default.

Files likely changed:

```text
crates/aeronyx-core/src/protocol/onion.rs
crates/aeronyx-server/src/services/onion/circuit.rs
crates/aeronyx-server/src/services/onion/packet.rs
crates/aeronyx-server/src/services/onion/path_selection.rs
```

Verification:

- Three-node local circuit test.
- Each hop only sees previous and next hop metadata.
- No public exit by default.

## 19. Open Questions

- Should node signing keys be generated during registration or first local startup?
- Should operator key be wallet-based, nodeboard-account-based, or both?
- What is the minimum descriptor expiry window for reliable mobile clients?
- Should bootstrap snapshots include only public nodes or also private invite-only nodes?
- What descriptor fields must be visible to clients versus only to operators?
- How should a node rotate keys without losing reputation/history?
- What is the first client use case: chat relay, privacy relay, storage, or agent relay?

## 20. Maintenance Log

Use this section to record implementation progress.

Format:

```text
YYYY-MM-DD - Change summary
- Files changed:
- Verification:
- Notes:
```

Latest entry:

```text
<!-- [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] -->
2026-07-28 - Failed closed on duplicate gossip endpoint identity claims.
- Files changed:
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Root cause:
  - The receiver-self proof optimization used first-writer-wins when multiple
    verified descriptors normalized to one gossip URL.
  - A conflicting descriptor could not gain admission authority, but it could
    make the sender suppress the wrong producer proof and delay optional
    Directory evidence convergence.
  - The round-limited selection snapshot could also omit a conflicting
    descriptor just beyond the fan-out boundary.
- Architecture:
  - `PeerStore::valid_public_endpoint_identities` returns a lightweight,
    side-effect-free view of all currently valid public endpoint identities;
    its maximum size remains bounded by PeerStore capacity.
  - `DiscoveryPeerIdentityHints` records a node id only while one canonical URL
    maps to exactly one verified identity. A conflicting observation makes the
    URL ambiguous for the complete hint snapshot.
  - Ambiguous or unknown receivers use the existing bounded producer-diverse
    fallback without guessing identity.
- Security and compatibility:
  - No wire frame, API schema, configuration, persistence format, admission
    rule, receiver proof check, or legacy gossip behavior changed.
  - Identity hints remain local selection inputs only and cannot grant trust.
  - No endpoint, peer, producer, descriptor, proof, route, payload, client,
    traffic, wallet, or social-graph telemetry was added.

<!-- [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] -->
2026-07-28 - Made optional Directory proof fallbacks producer-diverse.
- Files changed:
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production evidence:
  - The maturity window removed the original too-new-anchor race, but later
    rounds showed that two descriptor-level alternates could still belong to
    one producer namespace.
  - A receiver correctly rejects evidence produced by itself when that local
    chain is not represented as an independently audited replica namespace.
- Architecture:
  - Selection now rotates the producer dimension before choosing a live
    descriptor inside that producer's bounded candidate group.
  - One outbound round carries at most three distinct producer proofs.
  - A URL-to-node-id hint derived only from verified PeerStore descriptors
    suppresses a known receiver's own producer proof. Unknown seed identities
    continue through the same bounded fallback path.
- Security and privacy:
  - The hint is process-local, never serialized, logged, exposed through
    status, or used for admission.
  - Receiver exact-anchor verification remains unchanged and fail-closed.
  - No public node-id, endpoint, producer, descriptor, block, proof, route,
    payload, client, traffic, wallet, or social-graph telemetry was added.

<!-- [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] -->
2026-07-28 - Aligned Directory proof publication with replica convergence.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Root cause:
  - Production selected valid proofs from blocks only 46-52 seconds old while
    healthy replicas normally pulled exact anchors every 120 seconds.
  - Receivers correctly returned evidence-rejected until the same producer
    block/hash was present in their independently audited local replica.
- Architecture:
  - Missing `directory_gossip_proof_min_age_secs` derives two configured
    Directory sync intervals, preserving backward-compatible configuration.
  - An explicit override may only increase or preserve that convergence floor.
  - SQL candidate selection excludes newer blocks before applying its existing
    bounded window, live-descriptor checks, rotation, and full producer audit.
  - Runtime logs expose only the effective aggregate maturity policy.
- Security:
  - Receiver proof verification, exact block/hash matching, producer signature,
    Merkle inclusion, local anchor audit, quarantine, and PeerStore rollback
    checks are unchanged and remain fail-closed.
  - Legacy descriptor and snapshot gossip stay immediate; maturity applies only
    to optional stronger proof publication.
- Privacy:
  - No peer identity, endpoint, selected producer, block hash, descriptor hash,
    proof bytes, route, payload, client, traffic, or social graph field was
    added to logs or public status.

<!-- [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] -->
2026-07-28 - Added bounded concurrent gossip and per-peer failure isolation.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - A configurable `gossip_concurrency_limit` defaults to eight and validates
    within `1..=64`; runtime also caps it by selected peers.
  - Peer exchanges run through `buffer_unordered`, then return to original
    selection order before aggregation so status remains deterministic.
  - One total peer deadline is shared across all work. Optional Directory proof
    negotiation receives at most the first third; unused time carries forward,
    while the remainder is reserved for mandatory descriptor/snapshot gossip.
  - Mandatory failures are typed phase/kind values internally and render into
    the existing stable privacy-safe string buckets only at status boundaries.
- Bugs fixed:
  - Default `32`-peer fan-out with serial multi-request timeouts could exceed
    several 60-second scheduling periods when public peers stalled.
  - Invalid or unavailable proof-negotiation responses were previously
    indistinguishable from valid legacy-only peers.
- Compatibility:
  - Missing `gossip_concurrency_limit` uses the backward-compatible default.
  - No endpoint, discovery frame, signed descriptor, persistence schema,
    admission authority, proof retry ceiling, or public status field changed.
- Privacy boundary:
  - Concurrency reports contain aggregate duration and bounded reason buckets
    only. They never retain peer ids, URLs, proof bytes, routes, payloads,
    clients, traffic, or social graph dimensions.

<!-- [GOSSIP-OUTCOME-INTEGRITY 2026-07-28 by Codex] -->
2026-07-28 - Added outbound gossip outcome integrity and responsibility split.
- Files changed:
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - Optional Directory proof negotiation/transmission is one isolated exchange.
  - Mandatory legacy descriptor/snapshot synchronization is a second exchange.
  - A per-peer report preserves both outcomes without changing either wire
    contract, while a round accumulator owns all aggregate counter fan-in.
  - Negotiation progress is a typed `NotChecked / LegacyOnly / Attempted`
    state, preventing contradictory boolean combinations in runtime telemetry.
- Bug fixed:
  - A successful or rejected proof was previously returned only when the later
    legacy exchange also completed. Descriptor/snapshot failure therefore
    erased valid proof telemetry and understated convergence or rejection.
  - Proof results are now retained even when the legacy exchange fails; the
    legacy failure still controls the ordinary gossip success/backoff state.
- Compatibility:
  - No endpoint, discovery message, signed descriptor, Directory block,
    inclusion proof, persistence schema, feature flag, or retry policy changed.
- Privacy boundary:
  - Reports and accumulators retain only aggregate counters and bounded phase
    reason buckets. They contain no peer ids, URLs, producers, hashes, proofs,
    routes, messages, payloads, clients, traffic, or social graph metadata.

<!-- [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] -->
2026-07-28 - Added Directory Proof Gossip Reliability V2.
- Files changed:
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Runtime flow:
  - Builds at most two distinct live candidates through the existing complete
    producer audit and exact inclusion-proof reconstruction path.
  - Sends the first proof only after explicit V1 feature negotiation.
  - Retries exactly once, and only when the receiver returns HTTP `422` for an
    exact-evidence miss. Success, replica unavailability, rate limiting, other
    protocol status, and transport failure stop optional proof transmission.
  - Always executes the legacy descriptor and snapshot exchange afterward.
  - Records aggregate per-round convergence, fallback count, rejection buckets,
    and consecutive zero-acceptance rounds in PeerStore status and audit events.
- Compatibility:
  - No signed descriptor, discovery enum, bincode discriminant, Directory
    block, inclusion-proof, SQLite, or public feature schema changed.
  - Mixed-version peers continue through legacy-only gossip.
- Privacy boundary:
  - Status and logs contain counts and stable reason buckets only.
  - They contain no peer ids, endpoints, producers, descriptors, block hashes,
    proof bytes, routes, messages, payloads, clients, traffic, DNS contents,
    destinations, Memory Chain plaintext, private keys, wallet-level activity,
    or social graph metadata.

<!-- [DIRECTORY-GOSSIP-NEGOTIATION 2026-07-27 by Codex] -->
2026-07-27 - Added Directory Gossip Capability Negotiation V1.
- Files changed:
  - crates/aeronyx-server/src/api/discovery.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Runtime flow:
  - Publishes additive, non-authoritative transport feature booleans through
    the existing compact `/api/discovery/summary` contract.
  - Reads the peer summary through the shared bounded HTTP decoder before
    deciding whether to send `DirectoryDescriptorAnnounceV1`.
  - Missing routes, non-success responses, oversized bodies, malformed JSON,
    and absent feature fields all resolve to legacy-only gossip.
  - A positive feature hint never grants trust. The receiver still verifies
    the proof against exact audited local replica evidence before PeerStore
    admission.
  - The signed `NodeDescriptor` schema and bincode capability discriminants
    remain unchanged, preserving mixed-version descriptor compatibility.
  - Aggregate logs distinguish capability checks, capable peers, proof
    attempts, and accepted proofs without retaining peer or route dimensions.
- US1 rollout evidence for the preceding publication milestone:
  - Exact GitHub main commit `f95509c2213367c4dd210733db340a44f644946e`
    was built and deployed with an atomic binary replacement and timestamped
    rollback backup.
  - Startup audits passed for 7,367 persisted local Directory blocks, 18,452
    replica blocks across three producers, and a two-signature checkpoint
    certificate at height 33.
  - Korean1 and Noway1 each reached US1's public discovery API with HTTP 200.
  - Two consecutive outbound rounds contacted all three selected peers and
    preserved 3/3 legacy gossip success while old peers rejected the optional
    proof frame. This proves rolling-upgrade availability, not cross-node proof
    acceptance; acceptance requires one additional upgraded audited node.
- Privacy boundary:
  - Capability negotiation reveals only protocol support booleans.
  - It never exports node ids, peer endpoints in public status, producer ids,
    descriptor hashes, block hashes, route ids, message metadata, payloads,
    client IPs, destinations, DNS contents, Memory Chain plaintext, private
    keys, wallet-level traffic, or social graph data.

<!-- [DIRECTORY-PEER-ADMISSION 2026-07-27 by Codex] -->
2026-07-27 - Added Directory-authenticated PeerStore Admission V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Runtime flow:
  - Audits the exact local replica anchor before any proof request.
  - Uses the existing direct-first, at-most-two-carrier recovery path.
  - Re-audits after recovery and requires exact deterministic proof equality.
  - Admits through the existing PeerStore verification and anti-rollback path.
- Verification:
  - Covers inserted and idempotent admission, unknown local block rejection,
    public-wrapper proof re-verification, impossible transport summaries, and
    preservation of a newer PeerStore descriptor.
- Privacy and authority:
  - Returns only coarse transport/admission state and retains no carrier,
    endpoint, hash, proof, route, message, traffic, or user metadata.
  - Local replica acceptance is a caller-selected verification anchor, not
    network consensus, voting weight, fork choice, authority, or finality.

<!-- [REPLICA-PROOF-RECOVERY 2026-07-27 by Codex] -->
2026-07-27 - Added Requester Replica Proof Recovery V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Runtime flow:
  - Requests one exact proof from the original producer first.
  - Falls back to at most two current explicit mirror carriers only for typed
    availability or route-admission failure.
  - Uses fresh signed requests and descriptor-sequence-bound carrier endpoints.
- Verification:
  - Direct responses require the producer transport signature and complete
    producer proof verification.
  - Carrier responses separately require the carrier envelope signature and
    original producer proof.
  - Wrong hashes, noncanonical frames, bad signatures, and invalid proof paths
    stop closed without trying another source.
- Privacy and authority:
  - Returned source telemetry is only `direct_producer` or `replica_carrier`;
    no carrier identity, endpoint, route, or request data is retained.
  - Recovery changes availability only and grants no producer, checkpoint,
    witness, policy, consensus, fork-choice, or finality authority.

<!-- [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] -->
2026-07-27 - Added Replica Descriptor Inclusion Proof V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - Added a transactionally audited exact-block proof reader for producer
    replica namespaces.
  - Added append-only replica proof request/response frames and a dedicated
    carrier route.
- Admission and authority:
  - Pinned producer namespaces remain available through the general audited
    reader.
  - Permissionless recovery requires public mirror mode plus durable retained
    mirror membership.
  - Carrier signatures authenticate transport only; original producer proof
    verification remains mandatory.
- Compatibility and privacy:
  - Existing enum indexes, routes, block/object recovery, and authority policy
    remain unchanged.
  - No user, message, route, payload, traffic, DNS, Memory Chain, private-key,
    or wallet-level data enters the proof path.

<!-- [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] -->
2026-07-27 - Added Directory Descriptor Inclusion Proof V1.
- Files changed:
  - crates/aeronyx-core/src/ledger/merkle.rs
  - crates/aeronyx-core/src/ledger/mod.rs
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/services/directory_chain.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - Added bounded canonical Merkle path construction and verification.
  - Added one compact proof contract requiring an independently pinned
    producer and exact selected Directory block hash.
  - Added audit-gated SQLite proof lookup and an append-only signed peer wire
    request/response pair.
- Admission and compatibility:
  - Proof export is POST-only and pinned-peer-only even when public Full-node
    Mirror reads are enabled.
  - Existing `DirectorySyncMessage` discriminants remain unchanged; older
    nodes continue their existing Directory behavior.
- Security and privacy boundary:
  - Producer-signed inclusion is not canonical-chain selection, consensus,
    finality, financial state, user activity, message delivery, or Memory Chain
    content proof.
  - The contract contains public node-directory data only and introduces no
    user, payload, route, DNS, destination, private-key, or wallet data.
- Verification:
  - Generic Merkle proof tests cover every leaf for tree sizes 1 through 33,
    odd-leaf duplication, malformed positions, proof lengths, and tampering.
  - Core proof tests cover wrong chain, producer, block hash, descriptor
    substitution, signature/path tampering, absence, and maximum depth.
  - Store and Axum route tests cover exact-block lookup, signed response
    verification, complete proof verification, and pinned-only admission.

<!-- [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] -->
2026-07-27 - Added privacy-safe witness carrier service telemetry.
- Files changed:
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Runtime contract:
  - Observer-side recovery counters continue to answer whether this node used
    another carrier after a direct pinned-witness availability failure.
  - The new `observation_witness_carrier` status answers whether this process
    itself authenticated and transported a bounded witness request.
  - Every request that passes pinned-requester authentication is reduced to
    exactly one terminal bucket: forwarded, policy rejected, invalid request,
    target unavailable, target capability unavailable, target rejected, target
    invalid response, or local transport initialization failure.
  - The peer and status routers share the same runtime instance. Route
    availability comes from the actual mount prerequisites, not from storage
    inference or historical counters.
- Privacy and authority boundary:
  - Only process-lifetime counts and relative ages are reported.
  - Requester, target witness, endpoint, route, descriptor, request id,
    checkpoint hash, frame, digest, signature, and user-plane data never cross
    the telemetry API boundary.
  - Carrier activity cannot affect witness pins, authority, route ranking,
    reputation, voting, quorum, fork choice, consensus, or finality.
- Pre-deployment verification:
  - `cargo check` passed.
  - Runtime, peer-handler, and public-status focused tests passed.
  - All 1,202 `aeronyx-server` library tests passed.
  - Strict `clippy::all` completed successfully and the release binary built.
- Live validation:
  - Deployed the same release to US1 and Noway1 through the rollback-protected
    binary promoter after both reported zero active privacy-network sessions.
  - Rejected only US1-to-Korean1 TCP/8422 control-plane traffic. On the next
    bounded witness round, US1 advanced recovery to `attempts=1/succeeded=1`
    and Noway1 advanced carrier service to `requests=1/forwarded=1`; every
    carrier failure bucket remained zero and the round remained `2/2 accepted`.
  - Removed the fault and observed three further `2/2 accepted` rounds.
    Recovery and carrier counters remained exactly one, proving automatic
    direct-first restoration.
  - Korean1 direct access returned HTTP 200 after cleanup. All three services
    remained healthy, and no temporary firewall marker remained on any node.
  - Carrier request-scoped success logging was removed before the final smoke.
    Service journals contained no carrier event carrying checkpoint sequence,
    identity, route, endpoint, frame, digest, signature, or payload metadata.

<!-- [WITNESS-CARRIER-LIVE 2026-07-27 by Codex] -->
2026-07-27 - Validated Bounded Observation Witness Carrier Recovery V1 across
three audited live nodes.
- Deployment:
  - The observer and one explicit `DirectoryMirrorCarrier` ran the same
    `main` commit while the target witness intentionally remained on the
    compatible prior release.
  - Both upgraded nodes passed rollback-protected release deployment, local
    health, zero-active-session restart gates, Directory status, and signed
    discovery propagation.
  - A second carrier upgrade was deferred whenever a real privacy-network
    session became active; no client session was interrupted for rollout.
- Controlled fault:
  - Before injection, the observer directly reached both pinned witnesses and
    its signed snapshot contained four valid peers and two explicit carriers.
  - The test rejected only observer-to-target TCP/8422 control-plane traffic.
    Privacy transport, chat relay, user payloads, and the target service were
    not stopped or modified.
  - The temporary firewall rule used an exit/signal cleanup guard, was removed
    immediately after evidence arrived, and direct target access returned 200.
- Result:
  - Direct witness transport failed by construction.
  - The observer selected the only eligible non-target pinned carrier.
  - The carrier forwarded one fresh exact observer-signed inner request and
    returned the target witness's independently signed response.
  - Runtime evidence reported `selections=1`, `attempts=1`, `succeeded=1`,
    `failed_closed=0`, and no capability, transport, or exhaustion failure.
  - The checkpoint round remained `2/2 accepted` with zero reported transport
    failures, proving the carrier transported evidence without becoming the
    witness authority.
- Automatic restoration:
  - After direct connectivity was restored, three subsequent checkpoint rounds
    remained `2/2 accepted`.
  - Carrier selections, attempts, and successes remained unchanged at one,
    proving the scheduler returned to direct-first operation instead of
    sticking to the recovery path.
- Privacy verification:
  - Observer, carrier, and witness journals contained no carrier-event lines
    with payloads, request ids, signatures, endpoints, frames, checkpoint
    hashes, node identities, public keys, or user-plane data.
  - Public status exposed only aggregate process counters and bounded ages.
- Security boundary:
  - This validates authenticated availability recovery only. It does not
    establish validator voting, witness reputation, quorum, fork choice,
    consensus, governance, transaction inclusion, or global finality.

<!-- [WITNESS-CARRIER 2026-07-26 by Codex] -->
2026-07-26 - Added Bounded Observation Witness Carrier Recovery V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Problem solved:
  - A pinned witness may remain healthy and independently hold all producer
    evidence while its direct observer-to-witness route is temporarily
    unavailable. Direct-only transport delayed otherwise valid external
    evidence and could grow a mature checkpoint backlog after network faults.
- Trust architecture:
  - The observer signs one exact inner witness request.
  - The target witness independently recomputes the checkpoint and signs the
    exact witness response under the existing policy.
  - A carrier verifies and forwards those exact frames once, then signs only a
    transport envelope binding both frame digests and the exact identities.
  - The carrier cannot create, replace, approve, aggregate, or finalize witness
    evidence. Only the configured target witness signature can satisfy policy.
- Admission and bounds:
  - Direct witness transport always runs first.
  - Fallback is permitted only after endpoint absence or bounded availability
    failure, never after contract, canonical-codec, signature, or target
    rejection failures.
  - The observer selects at most two carriers that are both local operator pins
    and current signed descriptors explicitly advertising
    `DirectoryMirrorCarrier`.
  - The carrier accepts only pinned requesters and pinned target witnesses,
    resolves the target by node identity from the validated Peer Store, allows
    only public IP literals, disables environment proxies and redirects, caps
    the inner response at 16 KiB, uses a 10-second deadline, and never recurses.
  - The observer applies a separate 32 KiB ceiling to the complete signed
    carrier envelope instead of the general 512 KiB Directory page ceiling.
  - Each carrier attempt uses a fresh inner request id so a lost direct response
    cannot turn the witness replay guard into a false protocol failure.
- Failure semantics:
  - Carrier admission or availability failure may try the next bounded carrier.
  - If direct transport failed and no carrier is eligible, the existing
    `transport_failure` outcome is preserved rather than relabeled unavailable.
  - Target capability absence is counted as capability unavailable, exhausts
    that recovery without trying equivalent routes, and remains peer unavailable.
  - Target rejection, invalid target response, wrong binding, noncanonical
    frame, digest mismatch, or signature failure stops closed immediately.
  - The existing exact witness receipt persistence and restart audit remain the
    only durable evidence path.
- Observability and privacy:
  - Status adds process-only aggregate selections, candidates, attempts,
    successes, capability misses, transport failures, exhaustion, and
    fail-closed counts.
  - No carrier identity, witness identity, endpoint, route, request id,
    checkpoint hash, frame, signature, client metadata, or user-plane content
    is retained or exposed.
  - Recovery counters are transport diagnostics, never peer reputation, votes,
    quorum, consensus, fork choice, governance, or finality.
- Compatibility:
  - New bincode variants are appended, preserving every existing discriminant.
  - Unsupported carrier routes are cached only for the exact signed descriptor
    sequence and are probed again after the peer publishes a newer descriptor.
  - Direct witness behavior and all existing status fields remain compatible.

<!-- [WITNESS-CATCHUP 2026-07-26 by Codex] -->
2026-07-26 - Added Bounded Observation Witness Catch-up.
- Production evidence:
  - A strict 900-second certificate pull from Korean1 correctly rejected US1's
    latest fully witnessed checkpoint as stale.
  - The same pinned observer and two-witness policy succeeded inside the
    explicit 3600-second recovery ceiling at checkpoint age 1792 seconds.
  - Runtime outcomes showed fresh successful `2/2` witness rounds, proving the
    delay was a persistent sequence backlog rather than transport or signature
    failure.
- Root cause:
  - Each 120-second synchronization round appended one new checkpoint but
    witnessed only one older checkpoint. A restart backlog therefore remained
    constant even while every witness request succeeded.
- Behavior:
  - One synchronized round may now advance at most four distinct mature
    checkpoint sequences.
  - Missing witnesses for one sequence remain concurrent, but checkpoint
    sequences are processed strictly in order.
  - If the audited selector returns the same or an older sequence, the batch
    stops immediately instead of retrying that checkpoint in the same round.
  - The existing one-interval maturity delay, current-pin threshold, canonical
    signature verification, capability cache, and durable outcome telemetry are
    unchanged.
  - Status adds `catch_up_checkpoint_budget_per_round` and
    `pending_to_head_sequence_gap`; all existing fields and policy labels remain
    backward compatible.
- Security and privacy boundary:
  - Catch-up remains bounded to four checkpoints and at most the configured
    pinned witnesses for each checkpoint.
  - No public identity, endpoint, checkpoint hash, signature, route, payload,
    client metadata, or user-plane data is added to status or logs.
  - Witness receipts remain independent recomputation evidence, never votes,
    quorum, consensus, fork choice, or finality.

<!-- [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] -->
2026-07-26 - Added Authenticated Certificate Exchange V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/main.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - Appended domain-separated signed request/response frames without changing
    any existing bincode enum index.
  - Reused pinned-peer admission, live descriptor authentication, replay and
    rate guards, and the hardened redirect/proxy-free Directory HTTP client.
  - Kept transport authentication, certificate trust, and freshness as
    independent fail-closed checks.
- Operator flow:
  - Pull one exact frame from an expected observer, verify local witness pins
    and threshold, reject checkpoints older than the bounded policy, then append
    through the existing node-signed schema-v10 import path.
- Security boundary:
  - This is authenticated exchange of public node-directory observation
    evidence, not a vote, consensus, finality, financial ledger, user-message
    proof, or Memory Chain content proof.

<!-- [PORTABLE-CERTIFICATE-IMPORT 2026-07-26 by Codex] -->
2026-07-26 - Added Durable Third-Party Certificate Import V1.
- Files changed:
  - crates/aeronyx-server/src/main.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - Moved exact-frame and pinned trust-policy verification from CLI-private
    code into the reusable Directory Replica service boundary.
  - Added a host-local import command; no public or peer mutation route exists.
- Persistence:
  - Schema v10 retains at most 4,096 foreign certificates in a local-node-signed
    hash chain with metadata-head compare-and-swap.
  - Every row commits to exact bytes, certificate/checkpoint identity, local
    trust policy, verification time, previous row, and importer identity.
- Verification:
  - Migration, canonical validation, idempotent re-import, restart recovery,
    observer rollback, same-sequence conflict, policy substitution, content
    tamper, and row deletion are covered by Rust tests.
- Security boundary:
  - This is durable local observation evidence only. It adds no validator set,
    vote, quorum, fork choice, consensus, finality, financial transaction, user
    message, or Memory Chain content proof.

<!-- [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] -->
2026-07-26 - Added Portable Directory Observation Certificate V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Contract:
  - One bounded canonical frame packages an observer-signed checkpoint and
    independently signed accepted receipts from distinct current pins.
  - Offline verification checks version, production chain id, checkpoint,
    threshold, exact receipt bindings, canonical order, timestamps, identities,
    and every Ed25519 signature.
- Runtime:
  - The replica store rebuilds certificates from audited retained evidence and
    excludes retired pins on every read.
  - `GET /api/discovery/directory/observation-certificate` exists only on the
    local/VPN operator listener and fails closed until the current threshold is
    satisfied.
- Compatibility and privacy:
  - Existing Directory Sync frames, SQLite schema, status fields, public
    listener, and synchronization behavior remain unchanged.
  - The certificate exposes only public control-plane identities required for
    signature verification; user-plane and route data remain absent.
- Security boundary:
  - This is independently signed observation evidence, not a validator set,
    vote, quorum certificate, fork choice, consensus, finality, transaction
    inclusion proof, financial chain, or proof of user content.

<!-- [MIRROR-CATCHUP 2026-07-24 by Codex] -->
2026-07-24 - Added Bounded Full-node Mirror Catch-up V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production problem:
  - A selected public mirror previously imported at most one page per round.
  - One accepted page was counted as success even when the signed producer tip
    proved that the mirror still had additional authenticated history to pull.
- Runtime behavior:
  - Each selected producer may advance through at most four pages and twenty-four
    successful HTTP requests inside the existing 45-second producer deadline.
    A new page starts only when the remaining budget can contain its complete
    worst-case direct/carrier/object request sequence.
  - Direct producer reads remain first; only availability/admission failures
    may use at most two independently authenticated public carriers per page.
  - Producer outcomes are classified as converged, catching up, or failed.
    Accepted page/request totals remain aggregate-only.
- Compatibility:
  - Existing status fields remain present. New convergence, progress, page,
    and request fields are additive.
  - Existing one-result runtime recording remains available as a compatibility
    wrapper for internal callers.
- Security boundary:
  - Multi-page catch-up cannot add checkpoint/witness/policy authority.
  - Every block and object is still verified against the original producer.
  - This improves availability and recovery speed only; it is not fork choice,
    voting, quorum, consensus, global finality, or a financial blockchain.

2026-07-20 - Added Full-node Mirror Mode V1.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - crates/aeronyx-server/src/server.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Security boundary:
  - Verified public discovery permits only local signed producer-history reads.
  - Carrier export, witness recomputation, and policy-head anchors remain pinned.
  - Mirrors are capacity-bounded untrusted evidence and never authority members.
- Verification:
  - Configuration default/validation, public/private/disabled admission, global
    rate limiting, read-only peer selection, schema v8-to-v9 migration, capacity
    rollback, operator promotion, aggregate status, and runtime telemetry tests.
- Notes:
  - Mirror Mode does not turn Directory Chain into global consensus or a
    financial blockchain. It improves independent data availability only.

2026-07-19 - Added Directory Witness Policy Head Anchor V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production problem:
  - Schema v7 made each node's local witness-policy history signed and
    hash-linked, but a whole-host rollback could still restore both SQLite and
    the local policy head to an older internally valid snapshot.
  - External witnesses therefore need an opaque monotonic observation of the
    policy head without learning the policy member list or becoming validators.
- Protocol:
  - `ObservationWitnessPolicyAnchorRequestV1` binds chain id, request id,
    observer, request time, policy epoch, previous policy digest, opaque policy
    digest, and observer signature.
  - `ObservationWitnessPolicyAnchorResponseV1` binds the exact observed epoch
    and digest, responder, response time, outcome, and responder signature.
  - The peer route is
    `POST /api/discovery/peer/directory/observation-policy-anchor` and uses the
    existing pinned-peer descriptor, replay, timestamp, request-size, and rate
    admission controls.
  - The wire contract never sends the policy member list, threshold history,
    peer endpoints, routes, payloads, or user data.
- Persistence and verification:
  - Schema v8 adds append-only remote policy-head observations and accepted
    local receipt evidence, plus an atomic v7-to-v8 migration.
  - A witness accepts its first valid head for an observer as signed TOFU.
    Exact retry is idempotent; lower epoch is rollback; a different digest at
    the same epoch is conflict; a non-contiguous forward epoch or wrong
    predecessor is a history gap. Rejected frames never replace accepted state.
  - Local receipts count only when signed by a member of that exact local policy
    epoch and binding the exact epoch/digest. Startup audit replays canonical
    encoding, signatures, continuity, membership, and receipt contracts before
    any listener opens.
  - Outbound anchoring runs after every bounded replica sync round, independently
    of complete replica convergence, so producer unavailability cannot suppress
    rollback detection. It requests only missing current pins with bounded
    concurrency; unsupported mixed-version peers remain retryable capability
    misses.
- Observability and privacy:
  - Public status exposes only current receipt count, threshold-met boolean,
    and retained remote-head count. It omits policy digests, witness identities,
    signatures, endpoints, and member lists.
  - Runtime status and outbound scheduling verify only the signed current policy
    head and a bounded current-epoch receipt set (at most 16 members plus one
    overflow sentinel). Startup and explicit operator audit still verify all
    historical policy epochs and receipts.
  - Startup logs add only audited aggregate anchor/remote-head counts.
  - These anchors are rollback/conflict evidence only. They are not votes,
    validators, quorum, fork choice, consensus, or finality.
- Verification:
  - Policy-anchor protocol round-trip and signature-domain test.
  - Monotonic/idempotent/restart-durable remote-head test.
  - Exact-pin signed receipt and tamper-evident startup-audit test.
  - Stale in-flight receipt rejection after a signed policy rotation.
  - Complete response binding, rollback classification, and signature-tamper test.
  - Public aggregate redaction and schema v7-to-v8 atomic migration tests.
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D clippy::correctness`
  - `cargo build -p aeronyx-server --release`

2026-07-19 - Added Directory Witness Policy Epoch V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production problem:
  - Witness receipts were already signature-audited, but a later operator pin
    rotation or threshold change existed only in mutable runtime configuration.
    A restart could prove the receipts yet could not prove which local policy
    epoch made those receipts sufficient at that time.
- Architecture:
  - Schema v7 adds an append-only `directory_observation_witness_policies`
    history. Each row binds the sorted witness pins, threshold, activation time,
    local signer, previous policy digest, and Ed25519 signature.
  - `directory_replica_meta` anchors the current epoch and digest. Appending a
    policy and advancing that head use one immediate SQLite transaction and a
    compare-and-swap. This detects partial deletion or replacement of the policy
    history while the metadata head remains, plus torn or inconsistent local
    state.
  - Startup first audits the complete prior policy chain, then canonicalizes the
    validated runtime pins. Reordering is idempotent; only a pin-set or threshold
    change appends an epoch. A second complete audit runs before synchronization
    or any network listener starts.
  - Public/operator status exposes only aggregate epoch, historical change
    count, configured witness count, threshold, activation age, and runtime
    match. Member identities, endpoints, signatures, and policy digests remain
    host-local.
- Security boundary:
  - A policy epoch records one node operator's external evidence target only.
    It is not a validator set, voting weight, quorum, governance, fork choice,
    consensus, or finality.
  - The SQLite metadata head is not an external anti-rollback anchor. A
    coordinated whole-database or whole-host snapshot rollback can replace the
    metadata and policy table together. Detecting that class of rollback needs
    an independently retained opaque policy-head anchor or external witness and
    remains future work.
- Compatibility:
  - Existing v1 status fields remain unchanged; `observation_witness_policy` is
    additive. Schema v1-v6 databases migrate in one transaction, and an empty
    pin list retains the backward-compatible disabled policy with threshold one.
- Verification:
  - Full workspace tests passed: 17 common, 198 core, 1,118 server,
    2 server CLI, 22 transport, and all enabled doctests; zero failures.
  - `cargo clippy --workspace --all-targets -- -D clippy::correctness`
    passed. Existing non-correctness warnings remain outside this milestone.
  - `cargo build -p aeronyx-server --release` passed.
  - Focused tests passed for canonical pin ordering, idempotent restart,
    threshold change, pin rotation, v6-to-v7 migration, signature tamper,
    metadata-head tamper, whole-policy-table deletion, and public identity
    redaction.
  - `git diff --check` passed for the four changed files. Repository-wide
    `cargo fmt --all -- --check` remains blocked by pre-existing formatting
    differences in untouched files; the only new formatting finding was fixed
    before this verification record was written.
- US1 rollout evidence:
  - Code commit `54f3bbc` was deployed to US1 only after aggregate VPN health
    reported zero active sessions. Korean1 and Noway1 were intentionally left
    unchanged for this milestone.
  - First startup migrated the replica namespace to schema v7 and appended
    policy epoch 1 for two configured witnesses with threshold 2. The public
    aggregate status reported `healthy`, policy `active`, and exact runtime
    configuration match without exposing policy identities or digests.
  - A second controlled zero-session restart reported `appended=false`; epoch
    and row count remained 1, proving restart idempotency. The service remained
    active with systemd `NRestarts=0`.

2026-07-19 - Added Directory Witness Failure Drills V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production finding:
  - Historical `latest_sequence_witnesses` includes valid receipts from retired
    pins, while `latest_checkpoint_current_pinned_witnesses` describes only the
    newest observer checkpoint. After a pin rotation or restart, neither field
    alone proves that the latest witnessed checkpoint satisfies the current
    operator-pinned threshold.
- Architecture:
  - Added aggregate current-pin receipt count and target-met status for the
    latest witnessed checkpoint. The handler reuses one bounded cryptographic
    receipt-set audit and derives the head count only when head and witnessed
    sequence match, avoiding a second signature-verification pass.
  - No witness identities, endpoints, signatures, checkpoint hashes, routes, or
    user-plane metadata are exposed by the additive status fields.
  - Added a deterministic failure drill covering one accepted receipt plus an
    offline peer, durable restart recovery, repeated fail-closed retry at the
    same forward floor, threshold completion, and pin rotation that invalidates
    a retired receipt until a new current pin signs the same checkpoint.
- Compatibility:
  - Existing status fields and persistence schema remain unchanged. The new
    response fields are additive and old clients continue to parse the v1
    contract without modification.
- Verification:
  - `cargo check -p aeronyx-server` passed.
  - Focused status and replica-store modules passed 8 and 32 tests,
    respectively, including the new deterministic failure drill.
  - Full server regression passed: 1,114 library tests, 2 binary tests, and 1
    documentation test passed with 9 documentation tests intentionally ignored.
  - `cargo clippy --all-targets -- -D clippy::correctness` and the release build
    passed.
  - Commit `95f1f87` was deployed by rolling restart to Korean1, Noway1, and
    US1. Korean1 initially had one active session, so its restart was deferred
    until the session drained; all three services then reported
    `active/running`, `NRestarts=0`, and healthy Directory Replica status.
  - Every node returned the additive latest-witnessed current-pin fields with a
    satisfied target. US1 reported checkpoint 324 with two raw receipts, two
    current-pin receipts, threshold two satisfied, and zero latest-round
    verification or persistence failures.
  - A malformed witness frame sent to Noway1's peer route returned HTTP 400;
    the service remained healthy and no valid receipt was produced from it.
  - Production witnesses were not stopped to recreate unit-only outage faults.

2026-07-19 - Added Mature Witness Pipeline Status and Bounded Cold Catch-Up V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production finding:
  - The newest observation checkpoint intentionally remains unwitnessed for one
    complete synchronization interval. Treating that unmatured head as the
    witness-health result made a healthy forward pipeline look permanently
    unavailable.
  - Korean1 was running an older binary whose authenticated observation witness
    route returned 404. Noway1 was already returning accepted signed receipts;
    US1 had durably retained 64 before the fleet compatibility rollout.
- Architecture:
  - Added additive `observation_witness_pipeline` status computed with the same
    bounded, audited mature-checkpoint selector used by the scheduler.
  - The status separates `head_maturity_status` from the current mature forward
    floor and reports a pending checkpoint only when an eligible target is
    actually below the current-pin threshold.
  - Full witness identities stay private. Public and operator responses contain
    only aggregate counts, checkpoint sequence numbers, and coarse state.
  - Increased the sparse cold-catch-up page cap from four to eight. The existing
    30-request budget, dense-page worst-case estimate, 45-second timeout, frame
    bounds, signatures, admission, and fail-closed audits remain unchanged.
  - Incomplete rounds use a bounded 60-second catch-up cadence; fully converged
    nodes return to the operator-configured normal interval. Durable producer
    backoff remains authoritative, so accelerated scheduling never bypasses a
    persisted retry deadline.
- Compatibility and deployment:
  - The status contract is additive; existing checkpoint and outcome fields are
    retained unchanged.
  - US1, Korean1, and Noway1 source and binaries were aligned to `c59cdae` before
    this follow-up. Korean1's witness route changed from 404 to an authenticated
    200 surface and began bounded cold replica catch-up.
- Verification:
  - `cargo check -p aeronyx-server` passed.
  - Mature witness status, catch-up cadence, and request-budget focused tests
    passed.
  - Directory Replica regression passed: 52 tests across library and binary
    targets.
  - Full server regression passed: 1,113 library tests, 2 binary tests, and 1
    documentation test passed with 9 documentation tests intentionally ignored.
  - `cargo clippy --all-targets -- -D clippy::correctness` and the release build
    passed.
  - Commit `d575757` was deployed by rolling restart to US1, Korean1, and
    Noway1 with zero active sessions. Every service returned `active/running`,
    `NRestarts=0`, and successful startup health after the rollout.
  - Korean1 recovered from approximately 900 missing blocks to zero lag under
    the bounded catch-up policy without entering backoff or quarantine.
  - Before raising policy, US1 checkpoint 297 received two distinct current-pin
    signed receipts in one round. US1 was then validated and restarted with
    `directory_observation_witness_min_verified = 2`.
  - After restart, the threshold-2 runtime accepted two receipts for checkpoint
    301 with zero evidence-unavailable, verification, or persistence failures.
  - `observation_witness_pipeline` is a forward-work view: after one checkpoint
    reaches its target, it may immediately expose the next newly mature
    checkpoint awaiting the next bounded round. Health assessment therefore
    combines monotonic checkpoint progress, `last_round_accepted`, and explicit
    failure counters rather than requiring the queue to remain continuously
    empty between normal rounds.

2026-07-19 - Added Directory Witness Threshold V1.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - Added `discovery.directory_observation_witness_min_verified` with a
    backward-compatible default of one and a hard range of 1-16.
  - Configuration fails closed when the target exceeds the distinct pinned
    Directory Sync peer set or is enabled without any pinned peers.
  - The scheduler now finishes the next mature checkpoint at its forward floor
    before advancing, instead of stopping after one receipt or perpetually
    chasing a newer head while an earlier target remains incomplete.
  - Already retained receipts from current pins are returned by the same
    bounded audited selection transaction and excluded from duplicate network
    requests. Receipts from removed pins remain historical evidence but no
    longer satisfy the current operator target.
  - Candidate selection remains history-bounded and verifies the candidate,
    signed predecessor, latest receipt set, durable outcome, exact producer
    tips, and recomputed overlap root before any request is sent.
- Status contract:
  - Additive public/operator fields report the configured target, remaining
    receipts, all historical latest-sequence receipts, the current-pinned
    subset, `awaiting_external_receipt` / `below_target` / `target_met` state,
    and whether the latest checkpoint satisfies the target.
  - Status re-verifies the bounded latest receipt set and computes threshold
    state from current pins, preventing removed-pair evidence from producing a
    false `target_met` result after operator pin rotation.
  - Full witness identities remain private store/coordinator data used only to
    prevent duplicate requests; public status exposes aggregate counts only.
- Compatibility and security semantics:
  - Existing configurations retain one-receipt behavior.
  - Directory Sync frames, signed checkpoint format, SQLite schema, admission,
    cadence, response limits, and privacy boundaries are unchanged.
  - The threshold is independent recomputation corroboration, not voting
    weight, quorum consensus, fork choice, financial-chain security, or
    finality. Portable certificates and fork policy remain future work.
- Verification:
  - Threshold, current-pin rotation, retired-pin exclusion, forward-floor
    scheduling, duplicate/invalid configuration, tampered receipt, and restart
    persistence coverage passed.
  - Directory Replica coverage passed: 50 tests across the library and binary
    targets.
  - Full `aeronyx-server` regression passed: 1,111 library tests, 2 binary
    tests, and 1 documentation test passed with 9 intentionally ignored.
  - `cargo check`, targeted `rustfmt --check`, Clippy correctness, and the
    optimized release build passed.

2026-07-19 - Added Bounded Witness Selection Audit V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production motivation:
  - US1 retained 162 signed observation checkpoints. Re-verifying every
    historical checkpoint and receipt every 120 seconds would make recurring
    scheduler cost grow linearly for the lifetime of the node.
- Architecture:
  - Startup and explicit operator `audit()` retain complete checkpoint,
    receipt, outcome, producer-prefix, incident, resolution, and retry audits.
  - Recurring selection verifies only evidence that can move or satisfy the
    forward floor: the latest bounded receipt set, its checkpoint, the durable
    outcome checkpoint, and the selected candidate plus signed predecessor.
  - Candidate verification still checks canonical encoding, local observer,
    row duplication, sequence/predecessor/timestamp continuity, Ed25519
    signature, exact producer-tip availability, and recomputed overlap root.
  - A latest receipt set is capped at the protocol's 16-producer bound plus one
    detection row; an over-bound set fails closed instead of increasing work.
  - The same 16-receipt ceiling is enforced transactionally on insertion and
    by complete startup audit, so persistence and recurring reads share one
    explicit resource contract.
- Compatibility and scope:
  - Directory Sync frames, descriptor schema, capability enum, SQLite schema,
    peer admission, synchronization cadence, response limits, and privacy
    boundaries are unchanged.
  - This remains local observer and independent recomputation evidence, not a
    vote, quorum, fork choice, consensus, financial chain, or finality claim.
- Verification:
  - Targeted normal-path, candidate/predecessor/latest-receipt/outcome tamper,
    restart, public-status, and receipt-boundary tests passed.
  - The receipt resource contract accepted 16 independently signed receipts,
    rejected the 17th transactionally, and remained consistent across status,
    recurring selection, and complete startup audit.
  - Directory Replica tests passed: 48 total (47 library + 1 binary).
  - Full `aeronyx-server` library tests passed: 1109/1109.
  - Binary tests passed: 2/2. Documentation tests passed: 1 passed, 9 ignored.
  - Modified-file rustfmt, `git diff --check`, Clippy correctness with all
    targets, and the release build all passed.

2026-07-19 - Added Mature Checkpoint Witness Scheduling V1.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production evidence:
  - US1 and Noway1 both reported healthy, fully synchronized producer replicas,
    zero lag, zero quarantine, and no persistence or verification failures.
  - Noway1 had accepted external receipts for all 44 local checkpoints, while
    US1 had 0 receipts across 151 checkpoints and one `evidence_unavailable`
    outcome every round. The asymmetry matched coordinator schedule ordering:
    US1 requested evidence for a checkpoint created from a producer tip that
    Noway1 would only import on its next interval.
- Architecture:
  - The coordinator now waits one complete configured Directory sync interval
    before a checkpoint becomes eligible for external recomputation.
  - The store selects the newest eligible checkpoint without any accepted
    receipt. It audits the complete checkpoint chain, every receipt, and the
    durable outcome aggregate before running the indexed selection query.
  - Selection is forward-only. Its minimum sequence is the newer of the latest
    authenticated receipt and latest durable outcome sequence, preventing
    restart recovery from walking backwards through historical gaps.
  - A selected checkpoint still passes the unchanged exact-prefix, overlap-root,
    canonical frame, request binding, Ed25519, and durable receipt checks.
- Compatibility and scope:
  - Directory Sync frames, descriptor schema, capability enum, SQLite schema,
    peer admission, response limits, and public privacy boundaries are unchanged.
  - Status adds only static aggregate scheduling semantics. This remains
    observer evidence, not voting, quorum, fork choice, consensus, or finality.
- Verification:
  - Mature-time boundary, restart-monotonicity, accepted-receipt, and public
    status serialization tests passed.
  - Directory Replica suite: 47 passed.
  - `aeronyx-server` library: 1,108 passed; binaries: 2 passed; docs: 1 passed,
    9 ignored by their existing annotations.
  - Modified Rust files pass direct `rustfmt --check`; repository diff check,
    Clippy correctness for all server targets, and the optimized release build
    passed. Existing repository-wide pedantic/deprecation warnings remain
    outside this milestone and were not broadened into unrelated refactoring.

2026-07-19 - Added descriptor-scoped witness capability negotiation.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Problem solved:
  - During rolling upgrades, nodes without the optional observation-witness
    route returned HTTP 404. The coordinator counted that explicit lack of a
    feature as a transport fault and retried the same unchanged descriptor on
    every completed checkpoint round.
- Architecture:
  - Peer HTTP results now retain a typed boundary for transport, status code,
    and bounded-response failures until operation-specific policy is applied.
    Existing range and object callers still receive their stable reason strings.
  - Only HTTP 404, 405, and 501 mean that the witness service is unavailable.
    Authentication, admission, conflict, throttling, and server faults remain
    ordinary failures and are never silently downgraded.
  - A process-local negative cache binds the unavailable observation to the
    exact sequence of the peer's already verified signed descriptor. Publishing
    a newer descriptor sequence automatically re-enables probing, so an upgrade
    does not depend on a timer, semantic version parsing, or operator action.
  - The cache cannot make a witness trusted. Every successful response still
    requires canonical frame equality, exact contract binding, an accepted
    evidence result, Ed25519 verification, and durable receipt persistence.
- Compatibility:
  - `NodeDescriptor` remains schema v2 and `NodeCapability` is unchanged. This
    avoids introducing a new bincode enum discriminant or causing old nodes to
    reject newer signed descriptors during a rolling deployment.
  - Unsupported service outcomes use the existing `peer_unavailable` aggregate
    bucket; durable SQLite schema v6 and all status consumers remain compatible.
- Privacy:
  - The cache and logs do not expose endpoint, node id, request id, signature,
    checkpoint hash, response body, routes, or any user-plane metadata.
- Verification:
  - Capability unit tests passed: 2/2.
  - Directory Replica tests passed: 47/47 across library and binary targets.
  - Full server tests passed: 1108/1108 library and 2/2 binary tests.
  - Documentation tests passed their enabled case; 9 examples remain
    intentionally ignored by the existing suite.
  - Modified-file rustfmt, `git diff --check`, Clippy correctness across all
    server targets, and the optimized `aeronyx-server` release build passed.
  - Repository-wide rustfmt still reports pre-existing formatting differences
    outside this milestone; those unrelated files were deliberately not changed.

2026-07-19 - Added privacy-safe Directory witness outcome telemetry.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Problem solved:
  - The coordinator previously reduced every outbound witness result to one
    `accepted` or `failed` count. A node could not distinguish honest evidence
    propagation lag from peer admission, transport, canonical verification, or
    signed-receipt persistence faults.
- Architecture:
  - Every attempt now terminates in one closed enum bucket: `accepted`,
    `evidence_unavailable`, `evidence_conflict`, `peer_unavailable`,
    `transport_failure`, `verification_failure`, or `persistence_failure`.
  - SQLite schema v6 stores one singleton aggregate containing cumulative and
    latest-round counters plus timestamps and the local checkpoint sequence.
    A foreign key binds that sequence to an audited local checkpoint.
  - Process runtime separately tracks this-start counters and failures to
    persist the telemetry aggregate itself. Signed accepted receipts continue
    to use their existing append-only table and independent restart audit.
- Privacy and semantics:
  - The aggregate never stores witness identity, endpoint, request id,
    signature, checkpoint hash, response body, route, or user-plane metadata.
  - Counters are diagnostic evidence only. They are not peer reputation,
    voting weight, quorum, fork choice, consensus, financial blocks, or finality.
- Compatibility:
  - The wire protocol is unchanged. Older peers still return their existing
    witness responses or ordinary HTTP failure; classification is local only.
  - Schema v1-v5 migrations remain transactional. Existing receipt, checkpoint,
    producer, incident, resolution, and retry evidence is preserved.
- Verification:
  - Targeted Directory Replica suite passed with durable round, restart,
    migration-v5, runtime bucket, status separation, and tamper rejection tests.
  - Server full suite passed: 1106/1106 library tests and 2/2 binary tests;
    documentation tests passed their enabled case with 9 intentionally ignored.
  - Modified-file rustfmt, `git diff --check`, Clippy correctness, and the
    optimized `aeronyx-server` release build passed.

2026-07-19 - Added commitment-bounded multi-block Directory catch-up.
- Files changed:
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Production finding:
  - The first live carrier round recovered Korean producer evidence correctly,
    but one-block pages advanced only three blocks per 120-second round. A cold
    1,000-block replica would require hours before checkpoint witnessing.
- Bounded optimization:
  - Coordinators now request at most the existing protocol maximum of eight
    contiguous blocks per range page.
  - Direct and carrier responders stop the page before aggregate commitments
    exceed 256, the existing single-block maximum. They also stop before a
    descriptor hash repeats across blocks, preserving exact object hydration.
  - The response body cap, object chunk size, per-peer rate limit, producer
    signatures, carrier signatures, fork quarantine, and SQLite audit rules
    are unchanged. The request budget is bounded at 18 requests in the worst
    case per page and 30 requests per producer round, matching but never
    exceeding the existing inbound identity limit.
  - Older peers remain compatible because `limit=8` was already valid in the
    Directory Sync V1 contract; only page utilization changes.
- Verification:
  - Unique eight-block and repeated-descriptor boundary tests passed.
  - Peer API tests: 6/6 passed; coordinator tests: 9/9 passed.
  - Clippy correctness gate passed.
  - Server full suite: 1102/1102 library and 2/2 binary tests passed; the
    auxiliary integration target passed its enabled test (9 remain ignored).
  - Modified-file rustfmt, `git diff --check`, and release build passed.

2026-07-19 - Added Directory Signed Evidence Carrier V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Problem solved:
  - Direct producer synchronization required every configured node to pin every
    other node bilaterally. That N-by-N operational dependency could prevent an
    otherwise honest observer from obtaining all producer evidence needed to
    witness a checkpoint during a rolling upgrade.
  - The carrier layer allows one already pinned, audited node to transport its
    retained copy of another configured producer's public signed evidence. It
    does not grant the carrier authority over the producer namespace.
- Protocol contract:
  - Appended producer-bound replica block-range and descriptor-object request /
    response variants after all existing Directory Sync bincode variants.
  - Requests bind chain id, producer, range or ordered hashes, requester,
    request id, and timestamp. Responses additionally bind carrier identity,
    exact producer block hashes or descriptor hashes, audited tip, and time.
  - The outer Ed25519 signature authenticates the carrier transport. Every
    inner block remains signed by the producer and every descriptor remains
    signed by its subject node; receivers verify all layers before import.
- Storage and admission:
  - Replica export performs the complete metadata, checkpoint, witness,
    incident, resolution, retry, producer-prefix, commitment-index, and object
    audit inside the same SQLite read transaction as the bounded export.
  - Only a configured producer with a retained non-quarantined prefix may be
    exported. Requesters still require a bilateral operator pin, a current
    signed PeerStore descriptor, timestamp freshness, signature, replay id,
    body cap, and per-identity rate budget.
- Coordinator behavior:
  - Producer direct pull remains first choice and preserves the old wire path.
  - Carrier fallback is allowed only before a trusted range is obtained and
    only for unavailable endpoint, transport, HTTP 403/404/408/429, or 5xx.
  - Noncanonical frames, wrong producer/carrier, invalid signatures, wrong
    descriptor hashes, and other contract failures stop closed without fallback.
  - The conservative per-page budget is 18 requests, including one failed
    direct range request plus the worst bounded carrier hydration page.
- Security boundary:
  - A carrier cannot forge, rewrite, finalize, vote on, or choose producer
    history. Conflicting producer-signed evidence still enters the existing
    durable quarantine and incident path, including after restart audit.
  - Carrier transport is not an independent network-path claim and does not
    create consensus, fork choice, quorum, financial blocks, or finality.
- Deployment gate:
  - Deploy first to US1 and Noway1 while Korean1 remains on its active-session
    binary. Configure Noway1 for US1 + Korean1, verify direct Korean admission
    fails, carrier recovery through US1 succeeds, and then obtain the first
    independently recomputed external checkpoint witness receipt.
- Verification:
  - Modified-file rustfmt check and `git diff --check` passed. Workspace-wide
    rustfmt still reports pre-existing formatting drift in unrelated files.
  - Core and server Clippy correctness gates passed with dependency warnings
    suppressed; no new correctness diagnostic remains.
  - Core full suite: 198/198 passed.
  - Server full suite: 1101/1101 library tests and 2/2 binary tests passed;
    the auxiliary integration target passed its enabled test (9 remain ignored).
  - Release build for `aeronyx-server` passed.

2026-07-19 - Added cross-node Directory observation checkpoint witness V1.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Protocol contract:
  - Appended `ObservationCheckpointWitnessRequestV1` and
    `ObservationCheckpointWitnessResponseV1` after every existing bincode enum
    variant, preserving old Directory Sync discriminants.
  - The request carries one bounded canonical checkpoint and signs its exact
    hash, request id, observer identity, chain id, and timestamp.
  - The response signs the exact checkpoint hash and observer sequence,
    request id, witness identity, timestamp, and one stable outcome:
    `accepted`, `evidence_unavailable`, or `evidence_conflict`.
- Independent verification invariant:
  - The witness first verifies checkpoint structure, chain id, observer
    identity, timestamp, and Ed25519 signature.
  - Signature validity alone is never acceptance. The witness then reads its
    own audited local producer chain plus producer-isolated SQLite replicas,
    requires every exact producer block hash at every referenced height, and
    recomputes the overlap root locally. This mixed-source rule is required
    because a node does not redundantly mirror its own producer chain into its
    remote-replica namespace.
  - A missing prefix returns signed `evidence_unavailable`; a retained hash or
    recomputed-root mismatch returns signed `evidence_conflict`. Neither is
    persisted by the observer as accepted evidence.
- Admission and transport:
  - `POST /api/discovery/peer/directory/observation-checkpoint-witness` uses the
    existing bilateral `directory_chain_sync_peer_node_ids` pins, current
    signed PeerStore descriptor, request timestamp window, Ed25519 request
    signature, replay id, body cap, and per-peer rate limit.
  - The pre-witness router builder remains available for compatibility. The
    witness route is mounted only when a startup-audited replica store exists.
  - After a complete synchronized producer round, the coordinator retries a
    bounded witness round for the latest audited checkpoint. Older peers may
    return 404 without blocking producer synchronization or checkpoint append.
- Persistence and restart audit:
  - SQLite schema v5 adds
    `directory_observation_checkpoint_witnesses`; schema v1-v4 migrations are
    transactional and preserve all prior producer, incident, resolution,
    retry, and checkpoint evidence.
  - One witness may retain only one checkpoint hash for an observer sequence;
    exact repeated receipts are idempotent and conflicting hashes fail closed.
  - Startup streams every receipt, canonicalizes its frame, verifies row/object
    equality, local checkpoint linkage, timestamps, accepted outcome, witness
    identity, and Ed25519 signature before the node may start.
- Observability and privacy:
  - Public/operator status adds only aggregate receipt count, latest witnessed
    sequence, witness count for that sequence, and whether the current local
    checkpoint has external evidence. Witness identities, request ids,
    signatures, checkpoint hashes, endpoints, and producer identities remain
    absent.
  - A receipt proves one external node independently recomputed one exact
    checkpoint. It is not a vote, quorum certificate, fork choice, consensus,
    financial block, or finality claim.
- Verification:
  - Core canonical/signature witness tests passed.
  - Independent-evidence, unavailable/conflict, idempotency, tamper/restart,
    schema-v4 migration, authenticated route, outbound response, and public
    redaction tests passed.

2026-07-19 - Added signed Directory observation checkpoint continuity.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Protocol and persistence:
  - A canonical Ed25519-signed checkpoint binds the local observer identity,
    sequence, predecessor hash, timestamp, exact configured producer tips, and
    deterministic recent commitment-overlap root.
  - The coordinator appends only after every pinned producer reaches its exact
    authenticated remote tip during the same synchronization round. Backoff,
    timeout, partial catch-up, quarantine, or any producer failure suppresses
    checkpoint creation for that round.
  - Unchanged observation roots are idempotent and do not grow the ledger.
  - SQLite schema v4 stores checkpoints as a hash-linked append-only sequence;
    v1, v2, and v3 migrate transactionally without changing accepted blocks,
    commitments, incidents, resolutions, or retry state.
- Startup verification:
  - Startup decodes and re-encodes each bounded canonical blob, verifies local
    observer identity, Ed25519 signature, sequence, predecessor, timestamps,
    and duplicated row metadata.
  - Every referenced producer tip hash must exist at the exact historical
    height, and the observation root is recomputed from retained commitment
    windows before startup may pass.
  - Startup streams the append-only sequence through a SQLite cursor instead
    of materializing all checkpoint rows, keeping audit memory bounded as the
    local evidence history grows.
- Observability and privacy:
  - Status exposes only aggregate checkpoint availability, count, latest
    sequence, and age. It never exposes checkpoint hashes or full identities.
  - Checkpoints contain public signed directory control-plane evidence only;
    no endpoints, routes, selected hops, client metadata, message identifiers,
    payloads, ciphertext, DNS contents, destinations, Memory Chain records,
    private keys, wallet traffic, or social graph metadata are added.
  - This is local observer evidence, not a vote, witness quorum, fork choice,
    consensus, financial chain, or global finality claim.
- Verification:
  - Canonical checkpoint protocol tests: 2/2 passed.
  - Directory Replica store/coordinator/status tests: 32/32 passed.
  - aeronyx-core regression suite: 196/196 passed.
  - aeronyx-server regression suite: 1,091/1,091 passed.
  - Binary target tests: 2/2 passed; doctests: 1 passed, 9 intentionally
    ignored.
  - `cargo check -p aeronyx-server --tests --locked` passed.
  - Targeted Clippy inspection completed; the new checkpoint protocol,
    persistence, coordinator, and status paths add no lint.
  - Final optimized release build completed in 5m 56s, and the release binary
    accepted the existing US1 `/etc/aeronyx/server.toml` configuration.

2026-07-18 - Added authenticated Directory Replica quarantine resolution.
- Files changed:
  - crates/aeronyx-server/src/main.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Security boundary:
  - Resolution is available only through the host-local `aeronyx-server
    directory-replica` CLI. No public, peer, operator HTTP, management, gossip,
    or backend mutation route was added.
  - `inspect-incident` re-verifies the canonical producer-signed evidence and
    prints the exact active incident, accepted tip, quarantine kind, and prior
    resolution head required for an explicit command.
  - `resolve-quarantine` requires the incident digest to be repeated with
    `--confirm-incident`, loads the configured node identity private key, and
    signs every compare-and-swap field plus the fixed
    `resume_existing_prefix` action.
- Persistence and invariants:
  - Atomically migrates Directory Replica SQLite metadata from schema v1 or v2
    to v3 and adds `active_incident_digest`, `last_resolution_digest`, and the
    append-only `directory_replica_resolutions` table.
  - A resolution may only retain the already accepted height/hash. It cannot
    delete evidence, rewind blocks, choose a fork, import remote content, or
    change another producer namespace.
  - Resolution records form one producer-local hash-addressed linked history.
    Startup audit verifies local-node identity, Ed25519 signature, content
    digest, incident binding, retained block, predecessor ownership/order, and
    the absence of missing, cyclic, branched, or orphaned records.
  - The write transaction rejects a resolution timestamp that predates either
    its incident or its linked predecessor, so a successful write satisfies
    the same temporal ordering enforced again during startup audit.
  - Every producer-authored quarantine incident must be either the exact active
    incident or covered by a signed resolution. Directly clearing SQLite flags
    without a signed audit record therefore fails startup closed.
  - Repeated hostile evidence can quarantine the producer again; the next
    resolution must CAS against the previous resolution head. Incidents remain
    immutable and exact repeated evidence stays content-addressed/idempotent.
- Compatibility and privacy:
  - Directory Sync V1 frames, accepted block format, peer routes, discovery,
    configuration, automatic retry policy, and public mutation surface remain
    unchanged.
  - Status adds only aggregate/fingerprint-scoped resolution counts. No client,
    route, endpoint, payload, ciphertext, Memory Chain, DNS, destination,
    private-key, wallet-traffic, or social-graph data is persisted or exposed.
- Operator flow:
  - Review: `aeronyx-server directory-replica inspect-incident --digest <HEX>`
  - Resolve only after independent evidence review by running the exact command
    printed by inspection. A stale tip, active incident, kind, or history head
    is rejected without changing SQLite.

2026-07-18 - Added auditable Directory Replica incident evidence export.
- Files changed:
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Evidence and API contract:
  - Adds deterministic exclusive-cursor incident summary pages with a default
    limit of 20 and a hard maximum of 50.
  - Separates low-cost summaries from the bounded signed response frame, which
    can be as large as 512 KiB.
  - Re-verifies canonical encoding, production chain id, producer identity and
    signature, incident digest, evidence size, and evidence SHA-256 immediately
    before a single evidence package is returned.
  - Uses stable `directory_replica_incident_list.v1` and
    `directory_replica_incident_evidence.v1` response contracts.
- Safety and privacy:
  - Incident routes are registered only for LocalOperator scope; the public
    listener receives 404 and cannot infer whether evidence exists.
  - Summary pages expose truncated producer/subject fingerprints. Full producer
    identity is present only in the single proof needed for signature checks.
  - Evidence contains signed Directory Sync control-plane bytes only. No peer
    endpoints, descriptors, client identifiers, routes, selected hops, message
    ids, payloads, ciphertext, Memory Chain records, DNS contents, destinations,
    private keys, wallet traffic, or social graph data are added.
  - Automatic quarantine recovery remains disabled. A later recovery command
    must have strong operator authentication, command audit, evidence binding,
    and compare-and-swap protection before it can be considered.
- Compatibility:
  - No SQLite migration, Directory Sync frame change, config field, public API,
    producer namespace, retry policy, or accepted-prefix behavior changed.
- Verification:
  - Directory Replica focused tests: 24 passed, including evidence corruption
    rejection, cursor/limit bounds, local-only mounting, public 404 behavior,
    and stable invalid/not-found responses.
  - aeronyx-server regression suite: 1,083/1,083 passed.
  - Package integration group: 1 passed, 9 intentionally ignored.
  - cargo check -p aeronyx-server --tests --locked passed.
  - cargo clippy -p aeronyx-server --lib --no-deps --locked completed; the new
    incident storage/API paths added no lint after narrowing the SQLite mutex
    lifetime before cryptographic verification.
  - Optimized release build completed; the existing US1
    `/etc/aeronyx/server.toml` passed the release binary `validate` command.

2026-07-18 - Added bounded Directory Replica observation convergence.
- Files changed:
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Evidence model:
  - Compares exact commitment hashes from each configured, non-empty,
    non-quarantined producer replica's most recent 32 accepted blocks.
  - Supports at most the existing 16 validated Directory Sync producer pins;
    work and memory are bounded independently of total retained history.
  - Reports distinct, multi-source, and all-eligible-source recent commitment
    counts without assigning producer weight or selecting a preferred chain.
  - Derives a deterministic observation root over eligible producer identities,
    their signed tip heights/hashes, and all-eligible commitment intersection.
- Safety and privacy:
  - Quarantined producers are excluded rather than automatically rewound,
    deleted, trusted, or included in a fork decision.
  - A single eligible producer cannot generate a multi-source root.
  - Duplicate, zero, local, or over-limit producer inputs fail closed.
  - Public status exposes aggregate overlap counts only. The root remains on
    the local/VPN operator listener and neither scope exposes full producer
    identities, descriptors, endpoints, routes, selected hops, payloads,
    client metadata, private keys, wallet traffic, or social graph data.
  - API labels explicitly define this as local recomputable observation
    evidence, not voting, quorum, fork choice, consensus, or finality.
- Compatibility:
  - Directory Sync V1 frames, signatures, endpoints, SQLite schema v2, retry
    persistence, configuration, and the `directory_replica_status.v1` contract
    remain unchanged. New status data is additive.
- Verification:
  - Directory Replica focused tests: 23 passed, including deterministic input
    ordering, duplicate-pin rejection, signed-fork quarantine exclusion,
    public-root redaction, and an explicit 33-block/32-block-window bound.
  - aeronyx-server regression suite: 1,082/1,082 passed.
  - Package integration group: 1 passed, 9 intentionally ignored.
  - cargo check -p aeronyx-server --tests --locked passed.
  - cargo clippy -p aeronyx-server --lib --no-deps --locked completed; the new
    convergence and status paths added no lint.
  - Optimized release build completed, and the existing US1
    `/etc/aeronyx/server.toml` passed the release binary's `validate` command.

2026-07-18 - Made Directory Replica retry scheduling restart-durable.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Persistence and recovery:
  - Directory Replica metadata migrates atomically from schema v1 to v2 in one
    SQLite IMMEDIATE transaction before startup audit can pass.
  - Schema v2 stores only producer id, bounded consecutive failure count,
    stable internal reason bucket, retry boundary, failure/update timestamps,
    and a saturated skipped-round counter.
  - The coordinator restores retry rows only for currently pinned producers
    before its first request, preventing restart loops from bypassing backoff.
  - Failure and skipped-round writes run on blocking workers so synchronous
    SQLite access cannot stall the async transport runtime.
  - An authenticated empty or non-empty successful page clears its producer's
    retry row inside the same transaction as the accepted import.
- Safety bounds:
  - Failure streaks saturate at 64 in memory and SQLite.
  - Retry delay remains capped at 30 minutes.
  - Failure reasons accept only 1-96 lowercase ASCII letters, digits, and
    underscores; peer-controlled endpoints, bodies, and error strings fail
    validation before a producer row is created.
  - Skip counters saturate at SQLite's signed integer maximum, and update time
    never moves backward if the system clock is corrected.
- Observability and privacy:
  - Startup audit reports the aggregate number of validated retry rows.
  - Status policy adds `retry_state_persistence = audited_sqlite` and
    `successful_import_clears_retry_atomically = true` without changing the v1
    response contract or exposing additional producer identity.
  - Retry persistence never stores endpoints, response bodies, descriptors,
    routes, selected hops, message ids, payloads, ciphertext, Memory Chain
    records, client metadata, private keys, wallet traffic, or social graphs.
- Compatibility and rollback:
  - Directory Sync V1 frames, endpoints, signatures, request budgets, config,
    producer isolation, and public/operator privacy tiers remain unchanged.
  - Existing schema v1 databases upgrade automatically and retain all replica
    chain, object, commitment, and incident rows.
  - Schema v2 is intentionally strict. Rolling back to a pre-v2 binary requires
    restoring the matching pre-upgrade SQLite backup as well as the binary.
- Verification:
  - Directory Replica focused tests: 19 passed, including v1-to-v2 migration,
    reopen recovery, bounded-field rejection, runtime restoration, and atomic
    success cleanup.
  - aeronyx-server regression suite: 1,078/1,078 passed.
  - Package integration group: 1 passed, 9 intentionally ignored.
  - cargo check -p aeronyx-server --tests --locked passed.
  - cargo clippy -p aeronyx-server --lib --no-deps --locked completed; the
    changed coordinator/status paths and new persistence methods added no lint.
  - Optimized release build completed, and the existing US1
    `/etc/aeronyx/server.toml` passed the release binary's `validate` command.

2026-07-18 - Added producer-local Directory Replica failure containment.
- Files changed:
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Runtime behavior:
  - Each producer has a 45-second wall-clock deadline in addition to the
    existing five-second per-request timeout and producer-local request budget.
  - The first failure retries on the next ordinary synchronization tick.
  - Repeated consecutive failures defer approximately 1, 3, 7, then at most
    15 nominal intervals, capped at 30 minutes.
  - Any authenticated successful page immediately clears active backoff while
    preserving process-lifetime failure and skipped-round counters.
  - One producer's timeout or backoff never changes another producer's budget,
    accepted prefix, quarantine state, or retry schedule.
- Observability and privacy:
  - Public status adds only aggregate backoff producer count and next-retry
    timing; it continues to omit the `producers` collection entirely.
  - Local/VPN status adds only the existing truncated producer fingerprint,
    backoff state, retry timing, and skipped-round count.
  - Retry logs contain stable reason buckets and bounded counters only; they do
    not include endpoints, full identities, response bodies, descriptor hashes,
    routes, clients, payloads, or social graph metadata.
- Compatibility:
  - Directory Sync V1 frames, endpoints, authentication, config fields, SQLite
    schema, and persisted producer-isolated chain data are unchanged.
- Verification:
  - Directory Replica focused tests: 12 passed.
  - aeronyx-server regression suite: 1,071/1,071 passed.
  - Integration group: 1 passed, 9 intentionally ignored.
  - cargo check -p aeronyx-server --tests --locked passed.
  - cargo clippy -p aeronyx-server --tests --no-deps --locked passed; the new
    coordinator/status code and new runtime methods introduced no warnings.
  - Release build and existing US1 production configuration validation passed.

2026-07-18 - Split Directory Replica architecture and removed serial producer blocking.
- Files changed:
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/directory_replica_sync.rs
  - crates/aeronyx-server/src/api/directory_replica_status.rs
  - crates/aeronyx-server/src/api/mod.rs
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Architecture:
  - directory_chain_peer.rs now owns only authenticated inbound serving,
    replay/rate admission, audit-gated reads, and signed responses.
  - directory_replica_sync.rs owns outbound request creation, response
    verification, exact object hydration, atomic imports, catch-up policy, and
    lifecycle scheduling.
  - directory_replica_status.rs owns listener-fixed privacy scopes and status
    response serialization; public callers cannot request operator scope.
  - server.rs now constructs and starts one coordinator instead of embedding
    producer page loops in the main server lifecycle.
- Scheduling behavior:
  - Up to four independent pinned producers synchronize concurrently, while
    pages for one producer remain ordered and producer-local.
  - Each producer retains the four-page and 24-request round limits and reserves
    the 17-request worst-case next-page cost before continuing.
  - The first round starts after a deterministic identity-derived 5-15 second
    delay instead of waiting the full 120-second interval.
  - Later rounds use MissedTickBehavior::Skip and cannot overlap; shutdown
    cancels the complete in-flight round through the coordinator select.
- Privacy and compatibility:
  - Directory Sync V1 wire frames, endpoints, config fields, SQLite schema, and
    status JSON contract are unchanged.
  - Concurrency never broadens trust: only operator-pinned identities with a
    current signed PeerStore descriptor are contacted.
  - Logs and telemetry remain bounded reason/counter fields with no endpoint,
    full producer identity, response body, descriptor hash, route, client, or
    user payload data.
- Verification:
  - Directory Replica store/coordinator/status tests: 9 passed.
  - Authenticated inbound Directory Sync API tests: 3 passed.
  - aeronyx-server regression suite: 1,068/1,068 passed.
  - Integration group: 1 passed, 9 intentionally ignored.
  - cargo check -p aeronyx-server --tests --locked passed.
  - cargo clippy -p aeronyx-server --tests --no-deps --locked passed with zero
    warnings attributed to either new Directory Replica module.
  - Release build passed; the resulting binary accepted the existing US1
    production configuration without compatibility changes.

2026-07-18 - Added Directory Replica status and request-budgeted multi-page catch-up.
- Files changed:
  - crates/aeronyx-server/src/services/directory_replica.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- API:
  - GET /api/discovery/directory/status
  - The public listener returns aggregate producer, block, commitment, incident,
    lag, quarantine, and synchronization-health fields only.
  - The local/VPN operator listener additionally returns twelve-hex-character
    producer fingerprints and per-producer operational counters.
  - Neither scope returns endpoints, full producer identities, signed
    descriptors, routes, selected hops, user traffic, payloads, or wallet data.
- Catch-up behavior:
  - A producer may advance by at most four one-block pages per 120-second round.
  - The round has a hard 24-request budget and reserves the worst-case
    17-request cost before requesting another page.
  - The policy remains below the existing 30 requests/minute per-peer inbound
    limit, including a six-request safety margin.
  - Every page is independently authenticated, hydrated, and atomically
    imported; a later page failure cannot roll back an earlier accepted page.
- Verification:
  - Directory Replica store/runtime tests: 4 passed.
  - Directory peer API/status/request-budget tests: 5 passed.
  - aeronyx-server regression suite: 1,065/1,065 passed.
  - Integration group: 1 passed, 9 intentionally ignored.
  - cargo clippy -p aeronyx-server --tests --no-deps passed.
  - Release build and live US1 configuration validation passed.
- Notes:
  - Synchronization observations are process-lifetime control-plane telemetry.
    Accepted blocks, commitments, incidents, and quarantine remain durable.
  - A restart therefore reports pending synchronization until the next
    authenticated round, while persisted accepted prefixes remain available.

2026-07-17 - Completed the first audited three-node Directory Replica Sync deployment.
- Deployment:
  - US1, Korean1, and Noway1 run commit d324b98.
  - US1 pins Korean1 and Noway1 as independent signed producers.
  - Korean1 and Noway1 each pin only US1.
  - Gossip discovery still grants no Directory Chain import permission.
- Live verification:
  - All three production configs passed the Rust binary's built-in validator.
  - All services returned healthy local and public discovery API responses with
    zero systemd restart loops.
  - After two bounded rounds, US1 retained two producer-isolated tips at height
    2: four remote blocks, 16 commitments, and zero incidents or quarantines.
  - Korean1 and Noway1 independently retained US1's signed tip at height 2 with
    no incidents or quarantine.
  - US1 restart recovery re-audited two producers, four blocks, and 16
    commitments before serving traffic.
  - Noway1 restart recovery re-audited one producer, two blocks, and eight
    commitments before serving traffic.
  - SQLite integrity checks passed. Korean1 used Python's standard sqlite3
    library because the host intentionally has no sqlite3 CLI package.
- Operational safety:
  - Every restart was gated on active VPN sessions. Korean1 was not restarted
    during the recovery test because one real session became active.
  - Online backups were created before persistence recovery tests.
  - No deliberate fork was injected into production. Signed fork quarantine,
    retained-prefix behavior, and durable incident evidence remain covered by
    isolated automated tests.

2026-07-17 - Added producer-isolated Directory Chain replicas and bounded pull.
- Files changed:
  - crates/aeronyx-server/src/services/directory_replica.rs (new)
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/memchain_peer.rs
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Replica open/reopen, exact idempotence, signed block-fork quarantine,
    retained accepted prefix, and unrelated-object rejection tests.
  - Signed outbound range/object response verification and tamper rejection.
  - `aeronyx-server`: 1,062/1,062 unit tests passed.
  - `cargo clippy -p aeronyx-server --tests --no-deps` passed.
  - Release build and the existing production config validation passed.
- Notes:
  - Local and remote producer chains use separate tables in the same durable
    SQLite file; remote data cannot advance the local producer tip.
  - US1 remains fail-closed with no outbound sync while its pin list is empty.
  - Quarantine is persistent and intentionally has no automatic recovery path.

2026-07-17 - Added Directory Sync V1 authenticated serving transport.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-server/src/api/directory_chain_peer.rs
  - crates/aeronyx-server/src/api/mod.rs
  - crates/aeronyx-server/src/services/directory_chain.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Core canonical frame/signing-domain tests.
  - Store audit-gated bounded page, exact object ordering, and invalid-bound tests.
  - API pin/live-peer/signature/replay/range/object integration tests.
- Notes:
  - Permissionless discovery does not grant Directory Chain history access.
  - This transport proves what one producer signed; it does not establish
    consensus, finality, quorum, longest-chain selection, or financial state.
  - Replica persistence and fork quarantine are the next reviewed layer.

2026-07-17 - Added transactional local Directory Chain persistence.
- Files changed:
  - crates/aeronyx-server/src/services/directory_chain.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - SQLite create/open/reopen, exact deduplication, new-sequence append,
    same-sequence equivocation preservation, 257-commitment atomic batching,
    producer mismatch, block-blob tamper, commitment-index tamper, descriptor
    resolution, and signed descriptor-object tamper tests.
  - Directory-path backward compatibility, disabled-mode, and database-path
    isolation tests.
  - `cargo clippy -p aeronyx-server --tests --no-deps` completed successfully;
    no new production warning remains in the Directory Chain store.
  - `aeronyx-server`: 1,055/1,055 unit tests passed; one doctest passed and
    nine existing examples remained explicitly ignored.
  - `cargo build -p aeronyx-server --release` completed successfully in 5m39s
    on the reviewed US1 host.
- Notes:
  - Setting `discovery.directory_chain_path` is an explicit fail-closed opt-in.
    A corrupt, wrong-chain, or wrong-producer database prevents listeners from
    starting; history is never silently deleted or rebuilt.
  - Runtime reconciliation stores authenticated public signed node descriptor
    objects separately from opaque block commitments so historical commitments
    remain resolvable. Public node endpoints/capabilities may therefore exist
    in the local object table; they are never embedded in block payloads.
  - The journal contains no client identity/IP, sender/receiver pair, route,
    message id, payload, ciphertext, memory content, DNS content, destination,
    domain, URL, browsing history, private key, or wallet-level traffic.
  - This is one producer's durable signed observation chain. It is not peer
    synchronization, witness quorum, fork choice, consensus, finality, token
    accounting, smart-contract execution, or a financial blockchain.

2026-07-17 - Added Directory Chain V1 protocol core.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Deterministic descriptor commitment, block construction, canonical
    ordering, binary round-trip, tamper, bounds, chain continuity, clock skew,
    equivocation preservation, and fixed cross-implementation vector tests.
  - `cargo clippy -p aeronyx-core --tests --no-deps` completed successfully;
    the repository's existing warning backlog remains outside this change.
  - `aeronyx-core`: 192/192 unit tests passed; one doctest passed and three
    existing examples remained explicitly ignored.
  - `aeronyx-server`: 1,046/1,046 tests passed; one doctest passed and nine
    existing examples remained explicitly ignored.
  - `cargo build -p aeronyx-server --release` completed successfully in 5m34s
    on the reviewed US1 host.
- Notes:
  - V1 blocks contain public node identity, descriptor sequence, and opaque
    descriptor digests only. They contain no client identity, IP address,
    sender/receiver pair, route, message ID, payload, ciphertext, Memory Chain
    content, DNS content, destination, domain, URL, or browsing history.
  - This change defines deterministic protocol primitives only. It does not
    start block production, storage, synchronization, witness voting, fork
    choice, consensus, finality, token accounting, or financial execution.
  - A signed descriptor may be committed after expiry because the block is an
    immutable observation record; descriptor authenticity and schema are still
    verified before commitment.

2026-07-17 - Added optional external witnesses for delivery-cache anchors.
- Files changed:
  - crates/aeronyx-core/src/protocol/memchain.rs
  - crates/aeronyx-server/src/api/memchain_peer.rs
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/memchain/storage.rs
  - crates/aeronyx-server/src/services/memchain/storage_ops.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - deploy/node/server.example.toml
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Canonical request/response signing and protocol round-trip test.
  - Real HTTP witness exchange with pinned identity, contiguous generation,
    bounded response, forged outcome, sentinel, and restart-durability tests.
  - Configuration rejection tests for duplicates, impossible thresholds,
    strict mode without pins, and disabled local storage.
  - PeerStore aggregate status precedence and scoped evidence-clear tests.
  - `aeronyx-core`: 185/185 unit tests passed; one doctest passed and three
    existing examples remained explicitly ignored.
  - `aeronyx-server`: 1,046/1,046 tests passed on the reviewed US1 host.
  - `cargo build -p aeronyx-server --release` completed successfully in 5m45s.
- Notes:
  - Requesters may pin at most three distinct witness node identities and set
    a minimum verified threshold. Each witness must separately pin the exact
    requester identity it agrees to protect; the witness-side default is an
    empty, fail-closed list. This bilateral policy is independent of ordinary
    permissionless discovery, so it does not restrict the wider relay network.
  - Each witness stores one high-water row per requester: requester node id,
    positive contiguous generation, opaque anchor digest, and observed time.
    It never receives the embedded delivery count, route, endpoint, peer pair,
    sender, receiver, message id, payload commitment, receipt, or ciphertext.
  - First contact is trust-on-first-use for the configured requester. Later
    generations must be exactly contiguous; stale, conflicting, and gapped
    updates are signed adverse outcomes and never count toward the threshold.
  - Requests and responses are signed with domain-separated canonical bytes.
    The caller verifies the pinned responder identity, exact request echo,
    returned state/outcome relationship, response bound, and signature.
  - Startup clears only restored aggregate delivery evidence when established
    witnesses prove rollback/conflict/gap. Descriptor, routeability, proof, and
    relay counters retain their independent authentication and recovery paths.
  - `verified_delivery_witness_required_for_restore = true` also fails closed
    when the configured witnesses are unavailable or below threshold. Enable
    it only after every pinned witness has accepted a live anchor generation.
  - This is an anti-rollback checkpoint for an aggregate local cache. It is
    separate from Memory Chain commitment witnesses and does not claim
    consensus, finality, a financial blockchain, or lifetime network totals.

2026-07-17 - Added monotonic rollback protection for signed delivery evidence.
- Files changed:
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Legacy-v1 compatibility, anchored-v2, signed rollback, missing-anchor,
    cache-ahead crash-window, tamper, expiry, and independent-section tests.
  - Full aeronyx-server tests, release build, US1 restart, and live encrypted
    two-hop delivery followed by restart recovery.
- Notes:
  - Schema v2 signs a positive monotonic cache generation together with the
    aggregate delivery count and latest verification timestamp.
  - The independent anchor is derived from discovery.peer_cache_path, so no
    new operator configuration is required. Cache is fsynced first and anchor
    second; a cache one generation ahead is an accepted repairable crash
    window, while a cache behind the signed anchor is rejected as rollback.
  - Rollback rejection applies only to aggregate delivery evidence. Signed
    descriptors, descriptor-bound routeability, and two-hop proof history keep
    their independent authentication and recovery paths.
  - The anchor contains no route, endpoint, peer pair, sender, receiver,
    message ID, payload commitment, receipt, or ciphertext.
  - This is local single-file rollback protection. It does not claim to detect
    a whole-host snapshot rollback that replaces both cache and anchor, and it
    is not consensus, quorum, finality, or a lifetime network counter.

2026-07-17 - Added signed aggregate verified-client delivery restart continuity.
- Files changed:
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Targeted cache compatibility, tamper, expiry, readiness, and debounce tests.
  - Full aeronyx-server tests and release build on the reviewed US1 node.
- Notes:
  - The local peer cache stores only a cumulative verified-delivery count and
    the latest verification timestamp under an independent Ed25519 signature.
  - Route IDs, selected paths, peer pairs, sender/receiver identifiers,
    message IDs, payload commitments, receipt bytes, and ciphertext are never
    written into this evidence section.
  - Legacy caches remain readable. A missing section is treated as empty, and
    an invalid or expired section is rejected without discarding independently
    verified descriptor, routeability, or synthetic proof sections.
  - Restored history cannot by itself report real relay readiness. At least two
    current peers must independently demonstrate fresh signed terminal receipt
    capability after restart.
  - Verified delivery events trigger a debounced atomic cache flush so the
    evidence does not depend on the ordinary low-frequency cache interval.

2026-06-19 - Added Blind Node Invariant as protocol gate.
- Files changed:
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Documentation-only specification update.
- Notes:
  - Relay nodes and Memory Chain coordinators must be blind by design.
  - Nodes may move encrypted blobs and aggregate counters, but must not read
    content, reconstruct social graphs, or correlate user-level traffic.
  - Future discovery, relay, Memory Chain, and onion routing work must document
    visible fields and correlation risks before shipping.

2026-06-18 - Created architecture and development plan.
- Files changed:
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - Documentation-only change.
- Notes:
  - Plan intentionally excludes smart contracts.
  - Plan treats AeroNyx as protocol provider, not node operator.
  - Default policy is no-exit.

2026-06-18 - Reviewed current Rust discovery and relay foundations.
- Files inspected:
  - crates/aeronyx-core/src/ledger/mod.rs
  - crates/aeronyx-core/src/ledger/block.rs
  - crates/aeronyx-core/src/ledger/fact.rs
  - crates/aeronyx-server/src/services/wallet_routes.rs
  - crates/aeronyx-server/src/services/routing.rs
  - crates/aeronyx-server/src/services/chat_relay.rs
- Files changed:
  - crates/aeronyx-server/src/services/chat_relay.rs
- Verification:
  - cargo test -p aeronyx-core ledger -- --nocapture
  - cargo test -p aeronyx-server wallet_routes -- --nocapture
  - cargo test -p aeronyx-server routing -- --nocapture
- Notes:
  - Ledger primitives exist and can be reused conceptually for signed directory snapshots.
  - Wallet route cache is session-local and not yet a cross-node discovery layer.
  - Chat relay stores encrypted envelopes/blobs, but inter-node forwarding is not yet implemented.
  - Maintenance-only import cleanup was applied to chat_relay.rs; runtime behavior unchanged.

2026-06-18 - Removed production-inappropriate auth verification debug logging.
- Files changed:
  - crates/aeronyx-core/src/protocol/auth.rs
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-core auth
  - cargo test -q -p aeronyx-server chat_relay
- Notes:
  - Removed signature verification log output containing sign input, digest, and public key hex.
  - Updated a brittle public-key rejection test so it checks the stable security contract: non-matching keys must never verify another wallet's signature.
  - Updated touched comments to use "AeroNyx clients" wording.

2026-06-18 - Implemented Phase 1 signed node descriptor and verified peer store skeleton.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-core/src/protocol/mod.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - crates/aeronyx-server/src/services/mod.rs
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-core discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added signed NodeDescriptor primitives with capability, capacity, policy, expiry, and sequence fields.
  - Added in-memory PeerStore that verifies descriptors before storage, rejects stale sequences, supports capability queries, and cleans expired peers.
  - This does not yet connect network bootstrap, gossip, or encrypted inter-node forwarding.
  - Default descriptor policy remains no public exit.

2026-06-18 - Implemented Phase 2 bounded bootstrap snapshot loading primitives.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-core/src/protocol/mod.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - crates/aeronyx-server/src/services/mod.rs
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-core discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added NodeBootstrapSnapshot with bounded JSON parsing, schema version validation, pretty JSON output, and verified descriptor counting.
  - Added PeerStore::load_bootstrap_snapshot() with inserted / unchanged / stale / rejected reporting.
  - Bad or expired descriptors no longer poison an entire bootstrap import; healthy descriptors can still hydrate the store.
  - This still does not wire snapshot loading into node startup, persistence, or network gossip.

2026-06-18 - Wired bootstrap snapshot loading into Rust node startup config.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/main.rs
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-server config::tests::test_discovery
  - cargo test -q -p aeronyx-core discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added [discovery] config with enabled, bootstrap_snapshot_path, bootstrap_snapshot_url, and fetch_timeout_secs.
  - Discovery bootstrap is disabled by default for backward compatibility.
  - Server startup now creates a PeerStore and hydrates it from configured local/remote bootstrap snapshots when enabled.
  - Snapshot source failures warn but do not block the node from starting.
  - The validate command now shows discovery bootstrap settings.
  - This still does not start gossip, persistent peer storage, or encrypted inter-node forwarding.

2026-06-18 - Added Phase 4 discovery gossip protocol primitives and peer-store merge helpers.
- Files changed:
  - crates/aeronyx-core/src/protocol/discovery.rs
  - crates/aeronyx-core/src/protocol/mod.rs
  - crates/aeronyx-server/src/services/peer_store.rs
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-core discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added NodeDiscoveryMessage with SnapshotRequest, SnapshotResponse, and DescriptorAnnounce variants.
  - Added bounded bincode encode/decode helpers for discovery gossip messages.
  - Added PeerStore snapshot export, snapshot response generation, and gossip message application.
  - All incoming gossip data still flows through descriptor signature verification, expiry checks, and sequence anti-rollback checks.
  - This does not yet start a periodic network gossip task or expose an HTTP/WebSocket endpoint.

2026-06-18 - Added Phase 5 HTTP discovery snapshot and gossip endpoints.
- Files changed:
  - crates/aeronyx-server/src/api/discovery.rs
  - crates/aeronyx-server/src/api/mod.rs
  - crates/aeronyx-server/src/server.rs
  - crates/aeronyx-server/src/services/peer_store.rs
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-core discovery
  - cargo test -q -p aeronyx-server peer_store
  - cargo test -q -p aeronyx-server api::discovery
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added GET /api/discovery/snapshot for JSON bootstrap snapshots generated from verified PeerStore data.
  - Added POST /api/discovery/gossip for JSON NodeDiscoveryMessage exchange.
  - SnapshotRequest returns a SnapshotResponse; DescriptorAnnounce and SnapshotResponse merge through PeerStore verification.
  - API responses expose signed node descriptors and aggregate import counts only.
  - The route is merged into the existing combined API server, so it currently follows the same API lifecycle.
  - This still does not start a periodic outbound gossip task or encrypted inter-node message relay.

2026-06-18 - Implemented Phase 8 self signed node descriptor generation.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-server config::tests::test_discovery
  - cargo test -q -p aeronyx-server server::tests::self_discovery
  - cargo test -q -p aeronyx-core discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added discovery self-advertisement config: advertise_self, public_endpoint, region, descriptor_ttl_secs, and public_discovery.
  - Server startup now signs this node's descriptor and inserts it into PeerStore after bootstrap import when discovery is enabled.
  - Descriptor sequence uses Unix seconds so restarts do not normally roll back peer-visible metadata.
  - Descriptor capacity currently reports max_sessions; bps/pps remain optional until runtime counters and policy limits are wired.
  - The descriptor exposes protocol metadata only: node id, endpoint, capability, capacity, region, visibility, and software version.
  - Public exit remains hard-disabled in the descriptor policy.
  - This still does not start a periodic outbound gossip task, persist peers to disk, or forward encrypted envelopes across nodes.

2026-06-18 - Implemented Phase 7 local verified peer cache persistence.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-server config::tests::test_discovery
  - cargo test -q -p aeronyx-server server::tests::peer_store_cache
  - cargo test -q -p aeronyx-server server::tests::self_discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added optional discovery.peer_cache_path and discovery.peer_cache_write_interval_secs config.
  - Peer cache uses the same JSON schema as NodeBootstrapSnapshot.
  - Startup imports the local cache before configured bootstrap snapshots so newer bootstrap descriptors can still upgrade stale local data.
  - Runtime writeback exports verified descriptors only and writes atomically through a temporary file followed by rename.
  - Loading cache data still verifies signatures, expiry windows, and sequence anti-rollback through PeerStore.
  - This still does not start outbound gossip or cross-node encrypted envelope forwarding.

2026-06-18 - Implemented Phase 6 optional outbound discovery gossip task.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/api/discovery.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-server config::tests::test_discovery
  - cargo test -q -p aeronyx-server server::tests::discovery_gossip_url
  - cargo test -q -p aeronyx-server server::tests::peer_store_cache
  - cargo test -q -p aeronyx-server server::tests::self_discovery
  - cargo test -q -p aeronyx-server api::discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added optional discovery.gossip_enabled, discovery.gossip_interval_secs, and discovery.gossip_peer_limit config.
  - Outbound gossip is disabled by default to avoid unexpected network traffic when only bootstrap is enabled.
  - When enabled, each round announces this node's signed descriptor to known public peers, then requests a bounded verified snapshot.
  - Peer responses are merged through PeerStore, so signature, expiry, and sequence checks remain mandatory.
  - Gossip URLs are derived from descriptor public_endpoint and normalized to /api/discovery/gossip.
  - This still does not implement an independent listener separate from the combined API lifecycle, nor cross-node encrypted envelope forwarding.

2026-06-18 - Added Phase 10/11 Rust discovery status and safety policy foundation.
- Files changed:
  - crates/aeronyx-server/src/config.rs
  - crates/aeronyx-server/src/api/discovery.rs
  - crates/aeronyx-server/src/services/peer_store.rs
  - crates/aeronyx-server/src/services/mod.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - cargo fmt --check
  - cargo test -q -p aeronyx-server config::tests::test_discovery
  - cargo test -q -p aeronyx-server api::discovery
  - cargo test -q -p aeronyx-server peer_store
  - RUSTFLAGS=-Awarnings cargo check -q -p aeronyx-server
- Notes:
  - Added GET /api/discovery/status for nodeboard-facing peer counts, runtime counters, policy status, and timestamps.
  - PeerStore now tracks cumulative inserted / unchanged / stale / rejected / policy_rejected / rate_limited counters.
  - Added discovery.max_peers, discovery.max_snapshot_limit, discovery.gossip_rate_limit_per_minute, discovery.allowed_peer_ids, and discovery.denied_peer_ids.
  - /api/discovery/snapshot now caps requested snapshot size by configured max_snapshot_limit.
  - /api/discovery/gossip now applies global per-minute rate limiting and allow/deny descriptor policy before import.
  - Server config applies max_peers to PeerStore at startup.
  - This is the Rust API foundation for nodeboard display; the nodeboard UI still needs to consume it.
  - This still does not implement cross-node encrypted envelope forwarding.

2026-06-19 - Implemented Phase 9 first bridge for cross-node encrypted chat envelope relay.
- Files changed:
  - crates/aeronyx-server/src/api/chat_peer.rs
  - crates/aeronyx-server/src/api/mod.rs
  - crates/aeronyx-server/src/server.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- New endpoint:
  - POST /api/chat/peer/relay
- Behavior:
  - Accepts JSON PeerChatRelayRequest containing a sender-signed ChatEnvelope.
  - Verifies ChatEnvelope signature and caps encoded envelope size at 128 KB.
  - Uses ChatRelayService dedup before local delivery.
  - Delivers to locally online receiver sessions through the existing encrypted client transport.
  - Stores in the existing pending SQLite queue when the receiver is offline or all local routes fail.
  - The original sender node keeps local pending fallback even when peer fanout succeeds.
- Outbound peer selection:
  - server.rs selects valid discovered peers advertising NodeCapability::ChatRelay from PeerStore.
  - Self node id is skipped.
  - public_endpoint is normalized to /api/chat/peer/relay.
  - Fanout is capped by CHAT_PEER_RELAY_FANOUT_LIMIT to avoid broad message flooding.
- Privacy boundary:
  - This path relays only encrypted ChatEnvelope content.
  - It does not decrypt ciphertext, inspect plaintext, log packet payloads, DNS contents, domains, URLs, browsing history, voucher secrets, private keys, or client public IPs.
- Remaining work:
  - Add nodeboard UI for discovery status and peer relay counters.
  - Add dedicated peer relay counters/audit entries beyond debug logs.
  - Add route scoring and receiver affinity instead of simple bounded fanout.
  - Add future generic relay envelope for agent/onion relay once the narrower ChatEnvelope bridge is stable.

2026-06-19 - Added PeerStore discovery stability summary for Rust/nodeboard health gates.
- Files changed:
  - crates/aeronyx-server/src/services/peer_store.rs
  - docs/node-discovery-and-encrypted-relay-plan.md
- Verification:
  - cargo fmt --check
  - cargo test -p aeronyx-server peer_store -- --nocapture
  - cargo test -p aeronyx-server vpn_health -- --nocapture
  - cargo build -p aeronyx-server --release
- Behavior:
  - PeerStoreStatus now includes a `stability` block derived from existing verified peer counts, gossip success freshness, consecutive gossip failures, and seed recovery configuration.
  - Stability health buckets are `disabled`, `pending`, `healthy`, `degraded`, `stale`, and `failed`.
  - `relay_foundation_ready` is true only when multiple valid signed peers exist and outbound gossip freshness is acceptable for future relay foundation checks.
  - `last_gossip_success_age_seconds`, `last_gossip_round_age_seconds`, `seed_recovery_configured`, and `stale_after_seconds` are exposed as aggregate operator metadata.
- Privacy boundary:
  - The stability summary is aggregate control-plane telemetry only.
  - It does not expose peer URLs, full peer public keys, client IPs, destinations, DNS contents, packet payloads, chat plaintext, ciphertext, Memory Chain plaintext, voucher secrets, private keys, wallet-level traffic, or per-user traffic.
- Remaining work:
  - Let backend and nodeboard prioritize this `stability` block directly instead of inferring readiness from raw gossip counters.
  - Use the same health gate before future generic blind relay or multi-hop path tests.
```
