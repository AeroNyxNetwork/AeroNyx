// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay.rs
// ============================================================================
// Version: 3.73.0-CleanupFacade
//
// Modification Reason:
//   [CHAT-CLEANUP-FACADE-DOMAIN 2026-08-28 by Codex] Moved bounded cleanup
//   orchestration, aggregate telemetry, and stable failure propagation into a
//   nested facade while preserving deterministic sibling-test seams.
//   [CHAT-EXPIRED-FACADE-DOMAIN 2026-08-28 by Codex] Moved expiry-control
//   delivery, poison-row isolation, compatibility reads, and pushed-state ACK
//   APIs into a nested facade without changing durable semantics.
//   [CHAT-BLOB-FACADE-DOMAIN 2026-08-28 by Codex] Moved opaque encrypted-blob
//   storage, retrieval, and sender-authorized deletion APIs into a nested
//   facade without changing quotas, identifiers, or durable representation.
//   [CHAT-PENDING-FACADE-DOMAIN 2026-08-28 by Codex] Moved pending-message
//   custody, snapshot delivery, quarantine telemetry, and acknowledgement APIs
//   into a nested facade without changing wire or storage semantics.
//   [CHAT-BACKUP-FACADE-DOMAIN 2026-08-28 by Codex] Moved host-local backup,
//   retention, audit, anchor, and restore-plan APIs into a nested facade
//   without widening service field visibility or changing inherent methods.
//   [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex] Composed audit
//   verification, crash recovery, bounded rotation, checkpoint publication,
//   and durable append behind one maintenance coordinator.
//   [CHAT-BACKUP-AUDIT-ANCHOR-DOMAIN 2026-08-28 by Codex] Moved conversion of
//   authenticated private checkpoints into public opaque anchor digests behind
//   a pure domain function and closed failure vocabulary.
//   [CHAT-EXPIRED-CONTRACT-DOMAIN 2026-08-28 by Codex] Re-exported the expiry
//   notification model and bounded decoding invariants from a focused contract.
//   [BLIND-ROUTE-COORDINATOR-DOMAIN 2026-08-28 by Codex] Composed private
//   route identity, durable ownership, effect arming, and exact ACK replay as
//   one use case while retaining aggregate recovery telemetry in this service.
//   [VERIFIED-SUBMIT-COORDINATOR-DOMAIN 2026-08-28 by Codex] Composed private
//   replay, durable ownership, recovery, and response completion as one use case.
//   [CHAT-NODE-SECRET-DOMAIN 2026-08-28 by Codex] Re-exported versioned HKDF
//   node-secret derivation from a focused cryptographic boundary.
//   [CHAT-BACKUP-SQLITE-DOMAIN 2026-08-28 by Codex] Moved SQLite online copy,
//   retry mapping, durability activation, and private file mode into an adapter.
//   [CHAT-CUSTODY-ANCHOR-GUARD-DOMAIN 2026-08-28 by Codex] Moved the signed
//   anchor and maintenance-lock RAII contract into a focused resource type.
//   [CHAT-MAINTENANCE-TELEMETRY-DOMAIN 2026-08-28 by Codex] Moved the
//   maintenance status contract, lock, and all mutation rules into one domain.
//   [CHAT-PENDING-CONTRACT-DOMAIN 2026-08-28 by Codex] Re-exported pending
//   delivery contracts from a dependency-neutral module.
//   [CHAT-STORAGE-USAGE-DOMAIN 2026-08-28 by Codex] Moved aggregate usage
//   snapshot, SQLite reads, and fail-closed counter decoding into a repository.
//   [CHAT-ONLINE-DEDUP-DOMAIN 2026-08-28 by Codex] Moved concurrent bounded
//   online message-id admission behind a focused deduplication capability.
//   [CHAT-CLEANUP-EXECUTION-DOMAIN 2026-08-28 by Codex] Extracted bounded
//   cleanup batching, transaction commits, lock release, partial failure, and
//   deferred-backlog policy behind a replaceable execution capability.
//   [CHAT-PENDING-DELIVERY-DOMAIN 2026-08-28 by Codex] Composed legacy and v2
//   pending pulls, cursor protection, bounded lock scope, quarantine, and final
//   pagination behind one delivery use-case domain.
//   [BLIND-ROUTE-DURABLE-STORE-DOMAIN 2026-08-27 by Codex] Extracted route
//   replay admission, effect arming, lease takeover, release, and atomic
//   completion behind a composed SQLite repository capability.
//   [VERIFIED-SUBMIT-DURABLE-STORE-DOMAIN 2026-08-27 by Codex] Extracted
//   completed-response lookup, reservation capacity, lease takeover, and
//   atomic completion behind a composed SQLite repository capability.
//   [CHAT-RELAY-PENDING-SCHEMA-DOMAIN 2026-08-27 by Codex] Extracted the
//   pending-message schema and atomic legacy queue-sequence migration behind
//   a composed SQLite capability.
//   [CHAT-RELAY-STORAGE-SCHEMA-DOMAIN 2026-08-27 by Codex] Extracted blob,
//   expiry-notification, usage-trigger, and startup accounting reconciliation
//   schema work behind a composed SQLite capability.
//   [CHAT-RELAY-REPLAY-SCHEMA-DOMAIN 2026-08-27 by Codex] Extracted
//   verified-submit and blind-route replay installation, legacy migration,
//   row validation, marker advancement, and startup retention behind a
//   composed SQLite schema migration capability.
//   [CHAT-RELAY-BACKUP-CERTIFICATION-DOMAIN 2026-08-27 by Codex] Extracted
//   SQLite recovery-image journal normalization, physical verification,
//   schema/replay checks, accounting reconciliation, and queue continuity
//   behind a composed certification capability.
//   [CHAT-RELAY-BACKUP-CREATE-DOMAIN 2026-08-27 by Codex] Extracted verified
//   backup certification, idempotent replay verification, no-replace
//   publication, and owned-artifact rollback behind a composed command.
//   [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Extracted restore
//   readiness, authenticated-plan issuance, current-state verification, and
//   stable failure mapping behind a composed command capability.
//   [CHAT-RELAY-BACKUP-PRUNE-DOMAIN 2026-08-27 by Codex] Extracted prune
//   admission, checked planning, candidate deletion, recovery audit, and
//   post-mutation verification behind a composed command executor.
//   [CHAT-RELAY-BACKUP-INVENTORY-DOMAIN 2026-08-27 by Codex] Extracted
//   private backup inventory, artifact revalidation, checked accounting, and
//   active restore-boundary inspection behind a composed capability.
//   [CHAT-RELAY-BACKUP-AUDIT-CHAIN-DOMAIN 2026-08-27 by Codex] Extracted
//   authenticated multi-segment verification, checkpoint admission, and
//   active-tail recovery classification behind a composed verifier.
//   [CHAT-RELAY-BACKUP-AUDIT-IO-DOMAIN 2026-08-27 by Codex] Extracted bounded
//   audit artifact discovery, stable reads/hashing, crash recovery, and atomic
//   checkpoint publication behind a composed host-I/O capability.
//   [CHAT-RELAY-BACKUP-FILESYSTEM-DOMAIN 2026-08-27 by Codex] Extracted the
//   private backup directory, control-file, fsync, and cross-process lock
//   capability while preserving service-owned compatibility wrappers.
//   [CHAT-RELAY-BACKUP-CONTRACT-DOMAIN 2026-08-27 by Codex] Extracted the
//   aggregate backup command/receipt contracts and fail-closed prune admission
//   state while preserving every public path through explicit re-exports.
//   [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Extracted the typed relay
//   failure contract and stable diagnostics buckets into a dependency-light
//   domain while preserving the public `chat_relay` error paths by re-export.
//   [CHAT-RELAY-STATUS-CONTRACT-DOMAIN 2026-08-27 by Codex] Extracted the
//   privacy-safe serialized relay status contracts and shared SLO/circuit
//   policy defaults into a focused domain module while preserving every
//   existing public type path through explicit re-exports.
//   [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Added typed,
//   allowlisted relay-health failure reasons so arbitrary runtime strings can
//   never cross into heartbeat-visible status. Legacy public record methods
//   remain source-compatible and sanitize unknown input to `unknown`.
//   [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] Added aggregate-only
//   verified-submit result counters so nodeboard can distinguish terminal
//   proof success, entry custody fallback, and rejection without exposing
//   message, wallet, route, receipt, endpoint, or payload metadata.
//   [CHAT-VERIFIED-SUBMIT-RESULT-LABELS 2026-08-23 by Codex] Reused the core
//   protocol helper for verified-submit status labels so every consumer shares
//   one closed vocabulary.
//   [CHAT-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-23 by Codex] Added a bounded,
//   node-secret-indexed response cache and fixed-lane single-flight guard so
//   retries return the first verified result without repeating onion custody.
//   [DURABLE-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-24 by Codex] Persisted only
//   HMAC-derived request/envelope fingerprints plus an AEAD-sealed response so
//   exact retries remain idempotent across a clean or crash restart without
//   storing raw request ids, message ids, sender keys, routes, or receipts.
//   [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] Added a durable
//   private reservation before route/custody side effects. Unexpired replay
//   evidence is never evicted to admit new work; saturation and interrupted
//   submissions fail closed without repeating encrypted delivery.
//   [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Added process-owner
//   fencing and lease takeover to verified-submit reservations. A replacement
//   process may recover exact idempotent entry custody without reselecting an
//   uncertain onion path or inventing a lost terminal receipt.
//   [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] Added aggregate-only
//   attempted/completed/failed/deferred recovery transitions so operators can
//   verify restart behavior without receiving request-derived dimensions.
//   [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Reused the node-private
//   custody database for HMAC-indexed blind-route reservations and AEAD-sealed
//   ACK replay. Advertised relay nodes now retain their idempotency boundary
//   across restart without storing route ids, peers, endpoints, or payloads.
//   [BLIND-ROUTE-REPLAY-DOMAIN 2026-08-25 by Codex] Extracted private route
//   identity and exact ACK protection behind a composed trait capability while
//   preserving service-owned SQLite admission and completion transactions.
//   [CHAT-PULL-CURSOR-DOMAIN 2026-08-25 by Codex] Extracted the opaque v2 pull
//   cursor model, HKDF/AEAD mechanism, and binding rules behind a composed
//   trait capability while preserving SQLite snapshot paging and wire bytes.
//   [CHAT-PENDING-PULL-DOMAIN 2026-08-25 by Codex] Extracted bounded SQLite
//   reads and signed durable-row validation behind a composed repository trait
//   while retaining service-owned cursor, quarantine, and telemetry policy.
//   [CHAT-PENDING-CUSTODY-DOMAIN 2026-08-25 by Codex] Extracted pending-message
//   idempotence, quotas, sequence allocation, durable writes, and receiver-bound
//   acknowledgements behind a composed repository capability.
//   [CHAT-EXPIRED-DELIVERY-DOMAIN 2026-08-25 by Codex] Extracted expiry-control
//   reads, durable-row validation, pagination, and pushed-state writes behind a
//   composed repository capability while retaining quarantine and telemetry.
//   [CHAT-BLOB-CUSTODY-DOMAIN 2026-08-25 by Codex] Extracted opaque encrypted
//   blob identity, quotas, persistence, retrieval, and sender-bound deletion
//   behind a composed repository capability.
//   [CHAT-DURABLE-QUARANTINE-DOMAIN 2026-08-25 by Codex] Extracted atomic
//   poison-row replacement, de-identified evidence, retention, and backlog
//   checks behind a typed, composed repository capability.
//   [CHAT-RELAY-CLEANUP-DOMAIN 2026-08-25 by Codex] Extracted immutable TTL
//   cutoffs, expired-row validation, typed replay retention, and bounded SQLite
//   cleanup behind a composed repository capability.
//   [CHAT-DIRECT-PEER-CIRCUIT-DOMAIN 2026-08-25 by Codex] Extracted the
//   generation-safe circuit state machine and anonymous durable checkpoint
//   behind a composed repository capability.
//   [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] Extracted privacy-safe
//   relay health classification and atomic process telemetry composition.
//   [CHAT-BACKUP-RETENTION-DOMAIN 2026-08-26 by Codex] Extracted path-blind
//   recovery-image retention planning behind a trait. Complete and interrupted
//   deletion candidates are now deterministic oldest-first so a partial I/O
//   failure preserves the strongest remaining recovery history.
//   [CHAT-RELAY-AUDIT-CHECKPOINT-DOMAIN 2026-08-26 by Codex] Extracted the
//   immutable maintenance checkpoint model and v1 HMAC policy behind a trait,
//   while keeping sequence continuity, filesystem publication, and recovery
//   transitions in the service-owned I/O boundary.
//   [CHAT-RELAY-AUDIT-ROTATION-DOMAIN 2026-08-26 by Codex] Extracted bounded
//   segment rotation planning and active-tail crash classification behind a
//   pure trait while retaining locks, fsync, links, and deletion in this I/O
//   owner.
//   [CHAT-RELAY-AUDIT-CATALOG-DOMAIN 2026-08-26 by Codex] Extracted canonical
//   artifact naming, parsing, pairing, ordering, duplicate rejection, and
//   catalog bounds behind a path-free composed capability.
//   [CHAT-RELAY-AUDIT-VERIFICATION-DOMAIN 2026-08-27 by Codex] Extracted
//   bounded record/checkpoint admissions and cumulative verification state
//   transitions behind a pure fail-closed policy.
//   [CHAT-RELAY-BACKUP-NAMESPACE-DOMAIN 2026-08-27 by Codex] Extracted
//   canonical recovery-image naming, opaque operation-key derivation, and
//   fail-closed private-directory entry classification.
//   [CHAT-RELAY-BACKUP-ARTIFACT-DOMAIN 2026-08-27 by Codex] Extracted
//   immutable artifact snapshots, exact identity states, and checked byte
//   accounting while retaining all filesystem and SQLite I/O in this service.
//   [CHAT-RELAY-BACKUP-COPY-RETRY-DOMAIN 2026-08-27 by Codex] Extracted the
//   consecutive Busy/Locked timeout state machine and explicit copy actions
//   while retaining SQLite steps and sleeping in this service.
//   [CHAT-RELAY-TEST-MODULE-SPLIT 2026-08-27 by Codex] Moved the complete
//   in-crate test module into `chat_relay_tests.rs` without changing module
//   identity, private access, fixtures, or runtime behavior.
//   [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Added an RAII
//   current-anchor guard so producer receipt import cannot race checkpoint
//   publication after validating the exact signed anchor.
//   v1.3.0-Sovereign — Added WalletRouteCache field to ChatRelayService.
//   The route cache decouples wallet identity from session key, enabling
//   per-message signature-based authentication for all chat operations.
//   Also added dedup_cache as an Arc<Mutex<LruCache>> for the online-path
//   deduplication that will be used by the new handler in server.rs.
//   v1.3.1-Maintenance — Removed stale imports after the chat relay schema and
//   wallet-route integration stabilized. No database schema or API behavior changed.
//   v1.4.0-PeerRelayHealth — Added privacy-safe node-to-node relay health
//   counters for heartbeat/nodeboard diagnostics.
//   v1.5.0-GlobalStorageQuotas — Added transactionally maintained node-wide
//   message/blob usage and hard count/byte ceilings.
//   v1.6.0-OfflineControlReliability — Made ACK and notification batches
//   bounded and atomic, surfaced corrupt durable rows instead of skipping or
//   panicking, and split expiry notifications into transport-safe chunks.
//   v1.7.0-MaintenanceRuntime — Added aggregate cleanup execution evidence
//   and aligned durable pull ordering with the existing message-id cursor.
//   v1.8.0-BoundedMaintenance — Split retention cleanup into bounded SQLite
//   transactions and exposed deferred-backlog evidence.
//   v1.9.0-DurableQuarantine — Added privacy-minimised corrupt-row tombstones,
//   poison-row isolation, complete durable-envelope consistency checks, and
//   atomic concurrent online deduplication.
//   v2.0.0-SnapshotPull — Added a durable monotonic queue sequence and an
//   authenticated opaque cursor for stable ChatPullV2 snapshot pagination.
//   v2.0.1-StartupIntegrity — Removed the configured database path from
//   successful startup logs; server.rs now owns fail-closed activation.
//   v2.0.2-DirectRetryTelemetry — Added aggregate-only direct peer retry
//   recovery, exhaustion, and deterministic-failure telemetry.
//   v2.1.0-DirectRetrySlo — Added a fixed-memory five-minute delivery SLO
//   window with deterministic degradation and outage thresholds.
//   v2.2.0-DirectRelayCircuit — Added a source-blind, generation-safe circuit
//   breaker for target-bound direct relay delivery.
//   v2.3.0-DurableDirectRelayCircuit — Persisted the anonymous circuit safety
//   state so process restarts cannot silently bypass an active outage gate.
//   v2.4.0-DurableSchemaSentinel — Added an atomic installation marker so a
//   deleted circuit checkpoint table cannot be mistaken for a first upgrade.
//   v2.5.0-PowerLossDurability — Requires SQLite FULL durability before the
//   node may acknowledge encrypted custody or persist relay safety state.
//   v2.6.0-CustodyDurabilityStatus — Publishes only the verified aggregate
//   durability mode so operators can audit custody readiness.
//   v2.7.0-PrivateCustodyFile — Restricts the SQLite custody database and its
//   WAL sidecars to the node service account on Unix hosts.
//   v2.8.0-StartupPhysicalIntegrity — Rejects relay activation when SQLite's
//   bounded startup quick-check cannot prove the custody file is intact.
//   v2.9.0-VerifiedCustodyBackup — Added a private, transactionally consistent
//   SQLite backup artifact with pre-publication physical and logical checks.
//   v3.0.0-IdempotentCustodyBackup — Bound audited backup commands to a stable,
//   node-secret HMAC artifact key so restart replay reuses one verified image.
//   v3.1.0-CustodyBackupRetention — Added serialized, oldest-first count/byte
//   retention inspection and an aggregate-only audit operation.
//   v3.2.0-CustodyBackupPrune — Added explicit host-local dry-run/prune with
//   cross-process exclusion, grace-gated partial cleanup, and HMAC audit.
//
// Main Functionality:
//   - ChatRelayService: Central service managing all chat relay state
//   - Message store/pull/ack: SQLite-backed pending message queue
//   - Blob store/get/delete: SQLite-backed encrypted media cache
//   - TTL cleanup: Expires pending messages and blobs, queues notifications
//   - Expired notifications: Queued ChatExpired delivery for offline senders
//   - Online deduplication: LRU cache prevents duplicate delivery (online path)
//   - WalletRouteCache: In-memory wallet → session routing (v1.3.0-Sovereign)
//   - Peer relay health: aggregate outbound/inbound node-to-node relay status
//   - Durable queue quotas: per-receiver and node-wide count/byte ceilings
//   - Maintenance telemetry: aggregate TTL cleanup, batch, and backlog evidence
//   - Durable quarantine: bounded de-identified evidence for corrupt relay rows
//   - Snapshot pull: restart-stable pagination that excludes concurrent inserts
//   - Direct retry telemetry: aggregate ACK-loss recovery evidence without IDs
//   - Direct retry SLO: five fixed minute buckets, no event log or timer task
//   - Direct relay circuit: bounded open/half-open recovery without downgrade
//   - Durable circuit checkpoint: fixed-size anonymous restart protection
//   - Durable schema sentinel: rejects post-install checkpoint table loss
//   - Power-loss durability: WAL + FULL is verified before relay activation
//   - Custody durability status: anonymous verified mode in relay health
//   - Private custody file: Unix database/WAL material is owner-only
//   - Startup physical integrity: corruption fails closed before migrations
//   - Verified custody backup: WAL-aware, private, no-overwrite recovery image
//   - Idempotent custody backup: restart-safe command replay without raw IDs
//   - Custody backup retention: bounded, verified, serialized local audit
//   - Relay health reason boundary: typed ingress with compatibility sanitizer
//   - Verified submit telemetry: four closed aggregate result counters
//   - Verified submit result labels: core-owned canonical status vocabulary
//   - Verified submit idempotency: bounded restart-safe replay and conflict guard
//   - Verified submit admission: crash-safe reservation before relay/custody
//   - Verified submit recovery: owner-fenced, entry-custody-only restart repair
//   - Blind relay replay: private durable reservation and sealed exact ACK
//   - Blind route replay domain: composed private identity and ACK protector
//
// Dependencies:
//   - aeronyx-core/src/protocol/chat.rs: ChatEnvelope, encode_envelope, decode_envelope
//   - aeronyx-server/src/config_chat_relay.rs: ChatRelayConfig
//   - crates/aeronyx-server/src/services/chat_relay/wallet_routes.rs: WalletRouteCache
//
// Main Logical Flow:
//   ChatRelayService::new():
//     1. Open/create SQLite database
//     2. Restrict the durable file to the node account on Unix
//     3. Verify bounded SQLite physical integrity without exposing findings
//     4. Set and verify WAL + FULL pragmas
//     5. init_schema() creates tables if missing
//     6. Initialise MessageDedup (online-path LRU)
//     7. Initialise bounded verified-submit response cache and lock lanes; exact
//        misses consult the private durable cache before routing
//     8. Initialise WalletRouteCache (in-memory, empty on startup)
//
// ⚠️ Important Notes for Next Developer:
//   - wallet_routes is Arc<WalletRouteCache> so server.rs can hold a separate
//     Arc clone for the cleanup background task without borrowing ChatRelayService.
//   - All SQLite operations use parking_lot::Mutex<Connection>. Do NOT call
//     SQLite methods while holding another lock.
//   - ack_messages deletes only WHERE receiver = receiver_wallet.
//   - run_cleanup is synchronous — call from spawn_blocking or a sync task.
//   - node_secret is HKDF-derived from Ed25519 private key; stable across restarts.
//   - `relay_storage_usage` is rebuilt from canonical rows at startup, then
//     maintained only by SQLite triggers in the same write transaction.
//   - Logs must remain aggregate-only. Do not log message IDs, wallet prefixes,
//     blob IDs, sender/receiver keys, payload bytes, or endpoint/session IDs.
//   - Offline control batches are protocol-bounded. Do not remove their limits.
//     Malformed rows must be atomically replaced by de-identified quarantine
//     evidence; never silently skip them or copy raw routing metadata.
//   - Pending-message pages are ordered by message_id because the v1 wire
//     cursor contains only message_id. Chronological display belongs client-side.
//   - ChatPullV2 queue_sequence is an internal ordering primitive. Never expose
//     it on the wire or in logs; only issue the AEAD-protected opaque cursor.
//   - Retention cleanup is batch-bounded. Do not replace it with an unbounded
//     SELECT/DELETE or hold the SQLite connection across multiple batches.
//   - [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] Do not log the relay
//     database path or raw initialization errors. server.rs exposes only the
//     stable `reason_bucket()` when explicit activation fails.
//   - [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Retry telemetry must
//     remain aggregate-only. Never add peer IDs, message IDs, request
//     commitments, endpoints, wallet keys, or payload-derived dimensions.
//   - [DIRECT-RELAY-SLO 2026-08-15 by Codex] The recent window is process-local
//     and fixed-size. Do not replace it with per-delivery events, labels, or a
//     background timer; snapshots prune stale minute buckets on read.
//   - [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Circuit state is source-blind;
//     permit generations remain process-local while the minimal safety state
//     survives restart. Never add peer, route, message, endpoint, wallet, or
//     payload identifiers to permits, state, telemetry, or health snapshots.
//   - [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Only the anonymous
//     circuit state and aggregate counters may cross a restart. A corrupt
//     checkpoint must reject relay activation; an unavailable runtime write
//     must open the in-memory circuit, deny new admission, and stop fanout.
//     Runtime transitions lock circuit -> SQLite; never acquire the circuit
//     while holding `conn`.
//   - [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] The fixed feature
//     marker and checkpoint singleton are one atomic migration. Once installed,
//     a missing checkpoint table is corruption and must never reset to closed.
//   - [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] Signed custody ACKs and
//     direct-relay safety checkpoints share one SQLite connection. It must
//     remain FULL-or-stronger; NORMAL can lose acknowledged writes on power loss.
//   - [CHAT-RELAY-DURABILITY-STATUS 2026-08-16 by Codex] Durability telemetry
//     contains only a fixed state, protection boolean, and SQLite mode number.
//     Never add database paths, row counts, message IDs, or owner dimensions.
//   - [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] Restrict the primary DB
//     before enabling WAL so SQLite derives owner-only permissions for `-wal`
//     and `-shm`. Permission failures reject activation without logging paths.
//   - [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] Run the physical
//     integrity gate before WAL changes and schema migrations. Never log raw
//     SQLite findings; they may disclose schema and storage-layout details.
//   - [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] Never copy the live DB
//     file directly: committed custody can still reside in its WAL. Backups
//     must remain owner-only, validate physical integrity, usage counters and
//     anonymous circuit safety state, then publish without replacing a prior
//     artifact. Backup paths and raw SQLite findings are operator-private.
//   - [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] CMS command IDs are
//     untrusted routing metadata. Derive operation artifact names with a
//     domain-separated node-secret HMAC; never persist or expose the raw ID.
//     Existing operation artifacts must be re-verified and reused, never
//     overwritten, including after process restart or concurrent replay.
//   - [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Backup creation and
//     retention inspection share `backup_operations`; never inspect outside
//     that lock. Verify every managed artifact, model the protected/newest
//     image as retained, and report aggregate counts/bytes only.
//   - [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Deletion is never timed or
//     remote. Require the host-local confirmation contract, cross-process lock,
//     pre-delete identity/integrity recheck, durable directory sync, and the
//     private HMAC-chained aggregate audit around every explicit prune.
//   - [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] Recovery readiness is
//     non-destructive: verify every managed image, select the newest valid
//     image, and inspect only aggregate active-file/sidecar state. It must not
//     copy, rename, remove, open active storage, or imply restore execution.
//   - [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] A restore plan is a
//     short-lived, node-secret HMAC commitment to one verified image and the
//     active custody boundary observed at issuance. It is not authorization to
//     replace data and must remain host-local.
//   - [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Maintenance audit reads
//     enforce per-record and whole-file byte ceilings while checking that the
//     opened file did not grow, shrink, or truncate during verification.
//   - [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] Full audit segments are
//     published immutably behind HMAC-authenticated SHA-256 checkpoints; the
//     global v1 sequence/MAC chain continues across crash-safe rotations.
//   - [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] A verified immutable
//     checkpoint may be compressed into one opaque digest and signed by the
//     node identity for external retention. No private MAC or audit path crosses
//     that boundary, and interrupted rotations cannot be exported.
//   - [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Heartbeat-visible
//     relay failure reasons must enter through the typed allowlist below.
//     Unknown, typoed, URL-bearing, identifier-bearing, or payload-derived
//     strings are reduced to `unknown`; never widen this into a pass-through.
//   - [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] Verified-submit
//     telemetry is an aggregate delivery-mode counter only. Never add message
//     ids, request ids, receipts, routes, wallet keys, endpoints, payload
//     commitments, or per-user dimensions.
//   - [CHAT-VERIFIED-SUBMIT-RESULT-LABELS 2026-08-23 by Codex] Status labels
//     must continue to come from aeronyx-core protocol helpers; do not fork a
//     dashboard-only vocabulary in the relay implementation.
//   - [DURABLE-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-24 by Codex] Keep request
//     and envelope keys as domain-separated node-secret HMACs. Durable response
//     bytes must remain AEAD-sealed and bound to both fingerprints. Never store
//     or export sender keys, request ids, message ids, response receipts, routes,
//     or envelope commitments as plaintext metadata.
//   - [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] A new request
//     must own one durable anonymous reservation before any route, wallet-route,
//     or custody mutation. Never evict an unexpired completed response or
//     reservation to admit new work; reject saturation before side effects.
//   - [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] A foreign-process
//     reservation may be taken over only after its owner grace and exact CAS.
//     Recovery may repeat local exact-idempotent custody only; never announce a
//     wallet route, choose another onion path, or manufacture terminal proof.
//   - [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] One persistent custody
//     database may be owned by only one live service process. The owner-only,
//     no-follow OS lock is acquired before SQLite opens and is released by
//     RAII, so restart recovery proves predecessor exit instead of guessing
//     from wall-clock age.
//   - [REPLAY-COORDINATOR-DOMAINS 2026-08-28 by Codex] Verified-submit and
//     blind-route coordinators compose private identity, response protection,
//     and owner-fenced SQLite repositories. The service supplies only its
//     custody connection, process epoch, time, and aggregate telemetry.
//   - [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Blind-route claims
//     have an anonymous process epoch and durable effect boundary. After the
//     owner grace, an unarmed claim resumes as normal reservation while an
//     armed claim becomes explicit idempotent recovery work. Corrupt or
//     ownership-conflicting state remains fail-closed.
//   - Quarantine events must remain de-identified. Never persist message IDs,
//     sender/receiver keys, ciphertext, endpoints, or raw durable rows there.
//
// Last Modified:
//   v3.73.0-CleanupFacade - Extracted bounded-cleanup API facade
//   v3.72.0-ExpiredNotificationFacade - Extracted expiry-control API facade
//   v3.71.0-EncryptedBlobFacade - Extracted encrypted-blob API facade
//   v3.70.0-PendingMessageFacade - Extracted pending-message API facade
//   v3.69.0-BackupManagementFacade - Extracted host-local backup API facade
//   v3.68.0-BackupAuditMaintenanceCoordinator - Composed audit maintenance
//   v3.67.0-BackupAuditAnchorDomain - Extracted public anchor digest derivation
//   v3.66.0-ExpiredNotificationContract - Extracted expiry notification model
//   v3.65.0-BlindRouteCoordinator - Composed blind-route replay use cases
//   v3.64.0-VerifiedSubmitCoordinator - Composed verified-submit use cases
//   v3.63.0-NodeSecretDomain - Decoupled node-secret HKDF derivation
//   v3.62.0-BackupSqliteDomain - Composed SQLite backup adapter
//   v3.61.0-CustodyAnchorGuardDomain - Composed custody anchor RAII contract
//   v3.60.0-MaintenanceTelemetryDomain - Composed maintenance state machine
//   v3.59.0-PendingContractDomain - Decoupled pending delivery models
//   v3.58.0-StorageUsageDomain - Composed aggregate usage repository
//   v3.57.0-OnlineDedupDomain - Composed process-local duplicate admission
//   v3.56.0-CleanupExecutionDomain - Composed bounded cleanup execution
//   v3.55.0-PendingDeliveryDomain - Composed pending pull use cases
//   v3.54.0-BlindRouteDurableStoreDomain - Composed route replay repository
//   v3.53.0-VerifiedSubmitDurableStoreDomain - Composed replay repository
//   v3.52.0-CustodySchemaDomains - Composed custody schema migrations
//   v3.51.0-ReplaySchemaMigrationDomain - Composed replay schema migrations
//   v3.50.0-RecoveryImageCertificationDomain - Composed SQLite certification
//   v3.49.0-VerifiedBackupCreationDomain - Composed backup creation command
//   v3.48.0-ComposedRestoreCommandDomain - Composed restore plan commands
//   v3.47.0-AuditedBackupPruneDomain - Composed backup prune command
//   v3.46.0-VerifiedBackupInventoryDomain - Composed private backup inventory
//   v3.45.0-BackupAuditChainDomain - Composed audit-chain verification
//   v3.44.0-BackupAuditIoDomain - Trait-based audit artifact host I/O
//   v3.43.0-BackupFilesystemDomain - Trait-based private backup host I/O
//   v3.39.0-TestModuleSplit - Moved the complete test module out of production
//   v3.38.0-BackupCopyRetryDomain - Typed bounded SQLite copy retries
//   v3.37.0-BackupArtifactDomain - Typed artifact identity and accounting
//   v3.36.0-BackupNamespaceDomain — Trait-based private artifact namespace
//   v3.35.0-AuditVerificationDomain — Trait-based verification state machine
//   v3.34.0-AuditCatalogDomain — Trait-based path-free segment catalog
//   v3.33.0-AuditRotationDomain — Trait-based segment rotation planning
//   v3.32.0-AuditCheckpointDomain — Trait-based authenticated checkpoints
//   v3.31.0-BackupAuditRecordDomain — Typed authenticated audit records
//   v3.30.0-RestorePlanDomain — Trait-based authenticated restore planning
//   v3.29.0-BackupRetentionDomain — Trait-based oldest-first retention planning
//   v3.28.0-PeerRelayTelemetryDomain — Trait-based aggregate telemetry
//   v3.27.0-DirectPeerCircuitDomain — Trait-based durable circuit composition
//   v3.25.0-DurableQuarantineDomain — Trait-based quarantine composition
//   v3.24.0-BlobCustodyDomain — Trait-based encrypted-blob custody composition
//   v3.23.0-ExpiredDeliveryDomain — Trait-based expiry delivery composition
//   v3.22.0-PendingCustodyDomain — Trait-based custody write composition
//   v3.21.0-PendingPullDomain — Trait-based pull repository composition
//   v3.20.0-PullCursorDomain — Trait-based opaque cursor composition
//   v3.19.0-BlindRouteReplayDomain — Trait-based route replay composition
//   v3.18.0-VerifiedSubmitReplayDomain — Trait-based replay-domain composition
//   v3.17.0-ChatRelayRuntimeFence — OS-owned single-process custody fencing
//   v3.16.0-VerifiedSubmitRecoveryStatus — Aggregate restart-recovery evidence
//   v3.15.0-VerifiedSubmitEntryRecovery — Owner-fenced custody-only takeover
//   v3.14.0-RecoverableBlindRelayClaim — Restart-safe pre-effect claim takeover
//   v3.13.0-CrashSafeVerifiedSubmitAdmission — Durable pre-side-effect reservation
//   v3.12.0-DurableVerifiedSubmitIdempotency — Restart-safe private response replay
//   v3.11.0-VerifiedSubmitIdempotency — Bounded single-flight response replay
//   v3.10.0-VerifiedSubmitResultLabels — Core-owned verified-submit labels
//   v3.9.0-VerifiedSubmitTelemetry — Aggregate verified-submit result counters
//   v3.8.0-RelayHealthReasonBoundary — Typed privacy-safe failure allowlist
//   v3.7.0-CustodyAuditAnchor — Portable node-signed checkpoint commitment
//   v3.6.0-CustodyAuditRotation — Crash-safe segmented maintenance audit
//   v3.5.0-CustodyAuditVerify — Bounded public maintenance-chain verification
//   v3.4.0-CustodyRestorePlan — Authenticated, state-bound recovery planning
//   v3.3.0-CustodyRestoreReadiness — Read-only latest-image recovery preflight
//   v3.2.0-CustodyBackupPrune — Confirmation-gated local recovery maintenance
//   v3.1.0-CustodyBackupRetention — Bounded verified recovery-image retention
//   v3.0.0-IdempotentCustodyBackup — Restart-safe audited backup replay
//   v2.9.0-VerifiedCustodyBackup — Atomic validated SQLite recovery artifact
//   v2.8.0-StartupPhysicalIntegrity — Pre-migration SQLite quick-check gate
//   v2.7.0-PrivateCustodyFile — Owner-only SQLite custody files on Unix
//   v2.6.0-CustodyDurabilityStatus — Aggregate verified durability evidence
//   v2.5.0-PowerLossDurability — Verified FULL durability for custody writes
//   v2.4.0-DurableSchemaSentinel — Fail-closed checkpoint installation marker
//   v2.3.0-DurableDirectRelayCircuit — Anonymous restart-safe circuit checkpoint
//   v2.2.0-DirectRelayCircuit — Generation-safe fail-closed delivery circuit
//   v2.1.0-DirectRetrySlo — Fixed-memory five-minute delivery health window
//   v2.0.2-DirectRetryTelemetry — Privacy-safe ACK retry outcome counters
//   v2.0.1-StartupIntegrity — Privacy-safe, fail-closed service activation
//   v2.0.0-SnapshotPull — Monotonic queue sequence, atomic legacy backfill,
//     and wallet-bound XChaCha20-Poly1305 snapshot cursors
//   v1.9.0-DurableQuarantine — Poison-row isolation, private tombstones,
//     durable metadata/signature validation, and atomic live-path deduplication
//   v1.8.0-BoundedMaintenance — Bounded transactions and backlog observability
//   v1.7.0-MaintenanceRuntime — Runtime cleanup evidence and cursor-safe paging
//   v1.6.0-OfflineControlReliability — Atomic bounded ACK/expiry control flow
//   v1.5.0-GlobalStorageQuotas — Durable global quotas, enforced message size,
//     and route-safe logging
//   v1.4.0-PeerRelayHealth — Added node-to-node relay health status snapshot
//   v1.1.0-ChatRelay — Initial implementation
//   v1.3.0-Sovereign — Added wallet_routes: Arc<WalletRouteCache> field
//   v1.3.1-Maintenance — Removed stale imports; behavior unchanged
// ============================================================================

#[cfg(test)]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use rusqlite::{params, Connection};

use tracing::{debug, info, warn};

use aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS;
#[cfg(test)]
use aeronyx_core::protocol::chat::encode_envelope;
#[cfg(test)]
use aeronyx_core::protocol::chat::ChatEnvelope;
use aeronyx_core::protocol::memchain::{
    ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1,
};
#[cfg(test)]
use aeronyx_core::protocol::memchain::{
    CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1,
    CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1, CHAT_VERIFIED_SUBMIT_REJECTED_V1,
};

use crate::config::ChatRelayConfig;
use crate::services::chat_relay_backup_audit::{
    BackupAuditPhase, ChatRelayBackupMaintenanceAuditCounts,
};
use crate::services::chat_relay_backup_audit_anchor::{
    derive_backup_audit_anchor_digest, BackupAuditAnchorDigestError,
};
use crate::services::chat_relay_backup_audit_chain::ChatRelayBackupAuditChainVerification;
#[cfg(test)]
use crate::services::chat_relay_backup_audit_io::BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX
    as CHAT_RELAY_BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX;
use crate::services::chat_relay_backup_audit_io::{
    BACKUP_AUDIT_MAX_BYTES as CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
    BACKUP_AUDIT_MAX_SEGMENTS as CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS,
};
#[cfg(test)]
use crate::services::chat_relay_backup_audit_io::BACKUP_AUDIT_FILE_NAME
    as CHAT_RELAY_BACKUP_AUDIT_FILE_NAME;
use crate::services::chat_relay_backup_audit_maintenance::{
    BackupAuditMaintenance, BackupAuditMaintenanceLimits,
};
#[cfg(test)]
use crate::services::chat_relay_backup_audit_rotation::ChatRelayBackupAuditSegmentRange;
pub use crate::services::chat_relay_backup_audit_verification::ChatRelayBackupAuditVerificationReceipt;
use crate::services::chat_relay_backup_audit_verification::ChatRelayBackupAuditVerificationState;
use crate::services::chat_relay_backup_certification::{
    verify_sqlite_physical_integrity, BackupRecoveryImageCertification,
    RecoveryImageSchemaRequirement, SqliteBackupRecoveryImageCertifier,
};
use crate::services::chat_relay_backup_contract::ChatRelayBackupReceipt;
pub use crate::services::chat_relay_backup_contract::{
    ChatRelayBackupPruneReceipt, ChatRelayBackupPruneRequest, ChatRelayBackupRetentionReceipt,
    ChatRelayRestoreReadinessReceipt, CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
};
use crate::services::chat_relay_backup_create::{
    verify_existing_backup_artifact as verify_existing_backup_creation_artifact,
    VerifiedBackupCreationCommand, VerifiedBackupCreationRequest,
};
use crate::services::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, LocalBackupFilesystem,
};
#[cfg(test)]
use crate::services::chat_relay_backup_io::PrivateBackupControlFileMode;
use crate::services::chat_relay_backup_inventory::{
    BackupInventory, BackupInventoryLimits, ChatRelayBackupRetentionInspection,
    VerifiedBackupInventory,
};
use crate::services::chat_relay_backup_namespace::{
    BackupArtifactNamespace, BackupNamespaceError, HmacBackupArtifactNamespace,
};
use crate::services::chat_relay_backup_retention::{
    BackupRetentionLimits, BoundedBackupRetentionPlanner,
};
use crate::services::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions, SqliteRelayBackupDatabase,
};
use crate::services::chat_relay_backup_prune::{
    admit_backup_prune_request, AuditedBackupPruneExecutor, BackupPruneExecutor,
    LocalBackupArtifactRemoval,
};
pub(crate) use crate::services::chat_relay_blind_route::BlindRelayRouteAdmission;
use crate::services::chat_relay_blind_route_coordinator::BlindRouteCoordinator;
#[cfg(test)]
use crate::services::chat_relay_blind_route::RESPONSE_NONCE_BYTES
    as BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES;
use crate::services::chat_relay_blob_custody::EncryptedBlobCustodyDomain;
use crate::services::chat_relay_cleanup_execution::BoundedRelayCleanupExecutor;
pub use crate::services::chat_relay_custody_anchor_guard::ChatRelayCustodyAuditAnchorGuard;
pub(crate) use crate::services::chat_relay_direct_peer_circuit::ChatRelayDirectPeerPermit;
use crate::services::chat_relay_direct_peer_circuit::DirectPeerCircuitDomain;
#[cfg(test)]
use crate::services::chat_relay_direct_peer_circuit::{
    DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION, DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE,
};
#[cfg(test)]
use crate::services::chat_relay_cleanup::CLEANUP_MESSAGE_BATCH_SIZE;
pub use crate::services::chat_relay_expired_contract::ExpiredNotification;
pub(crate) use crate::services::chat_relay_expired_contract::{
    MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION, MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES,
};
use crate::services::chat_relay_expired_delivery::ExpiredNotificationDelivery;
use crate::services::chat_relay_message_dedup::{
    BoundedOnlineMessageDedup as MessageDedup, OnlineMessageDeduplication,
};
pub use crate::services::chat_relay_node_secret::derive_node_secret;
pub use crate::services::chat_relay_maintenance_telemetry::ChatRelayMaintenanceStatus;
use crate::services::chat_relay_maintenance_telemetry::RelayMaintenanceTelemetry;
use crate::services::chat_relay_peer_telemetry::{
    BlindRouteRecoveryEvent, OutboundRouteClass, PeerRelayTelemetryDomain, PeerRelayTelemetrySink,
    VerifiedSubmitEvent,
};
pub(crate) use crate::services::chat_relay_peer_telemetry::{
    ChatRelayInboundFailureReason, ChatRelayOutboundFailureReason, VerifiedSubmitRecoveryOutcome,
};
#[cfg(test)]
use crate::services::chat_relay_pending_custody::allocate_queue_sequence;
use crate::services::chat_relay_pending_custody::PendingMessageCustodyDomain;
pub use crate::services::chat_relay_pending_contract::{PendingMessage, PendingMessagePageV2};
use crate::services::chat_relay_pending_delivery::PendingMessageDeliveryDomain;
use crate::services::chat_relay_pending_schema::{
    ChatRelayPendingSchemaMigration, SqliteChatRelayPendingSchemaMigrator,
};
#[cfg(test)]
use crate::services::chat_relay_pull_cursor::ENCODED_CURSOR_BYTES as CHAT_PULL_CURSOR_V2_BYTES;
#[cfg(test)]
use crate::services::chat_relay_pull_cursor::PullCursorV2;
use crate::services::chat_relay_quarantine::DurableQuarantineDomain;
#[cfg(test)]
use crate::services::chat_relay_quarantine::{
    MAX_QUARANTINE_EVENTS, QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
    QUARANTINE_SOURCE_PENDING_MESSAGE,
};
use crate::services::chat_relay_restore_command::{local_restore_plan_command, RestorePlanCommand};
pub use crate::services::chat_relay_restore_plan::ChatRelayRestorePlanReceipt;
#[cfg(test)]
use crate::services::chat_relay_restore_plan::{RESTORE_PLAN_NONCE_BYTES, RESTORE_PLAN_VERSION};
use crate::services::chat_relay_restore_plan::RESTORE_PLAN_VALIDITY_SECS;
use crate::services::chat_relay_replay_schema::{
    ChatRelayReplaySchemaMigration, ReplaySchemaContract, ReplaySchemaVersion,
    SqliteChatRelayReplaySchemaMigrator,
};
#[cfg(unix)]
use crate::services::chat_relay_runtime_fence::ChatRelayRuntimeFence;
pub(crate) use crate::services::chat_relay_verified_submit::{
    VerifiedSubmitAdmission, VerifiedSubmitCacheLookup,
};
use crate::services::chat_relay_verified_submit_coordinator::VerifiedSubmitCoordinator;
use crate::services::chat_relay_storage_schema::{
    ChatRelayStorageSchemaMigration, SqliteChatRelayStorageSchemaMigrator,
};
pub use crate::services::chat_relay_storage_usage::ChatRelayStorageUsage;
use crate::services::chat_relay_storage_usage::{
    RelayStorageUsageRepository, SqliteRelayStorageUsageRepository,
};
use crate::services::wallet_routes::WalletRouteCache;

// ============================================
// Constants
// ============================================
/// Maximum IDs accepted in one authenticated `ChatAck` frame.
pub const MAX_CHAT_ACK_MESSAGE_IDS: usize = 100;
/// Maximum notification rows offered during one authenticated pull.
pub(crate) const MAX_EXPIRED_NOTIFICATIONS_PER_PULL: usize = 16;
/// Durable verified-submit row format guarded by `relay_schema_features`.
const VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION: i64 = 3;
const VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION: i64 = 2;
const VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION: i64 = 1;
const VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE: &str = "verified_submit_response_cache";
/// A future-dated request may remain authentic for two timestamp windows.
const VERIFIED_SUBMIT_RESPONSE_TTL_SECS: u64 = TIMESTAMP_WINDOW_SECS * 2 + 1;
/// Fixed capacity for live durable blind-route reservations and responses.
pub(crate) const BLIND_RELAY_ROUTE_REPLAY_CAPACITY: usize = 8192;
/// Route evidence outlives the signed envelope acceptance window by one second.
pub(crate) const BLIND_RELAY_ROUTE_REPLAY_TTL_SECS: u64 = 10 * 60;
const BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION: i64 = 3;
const BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION: i64 = 2;
const BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION: i64 = 1;
const BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE: &str = "blind_relay_route_replay";
/// Shared process-owner token for restart-safe replay reservations.
const REPLAY_PROCESS_EPOCH_BYTES: usize = 16;
const REPLAY_OWNER_TAKEOVER_GRACE_SECS: u64 = 5;
/// Grace period before a replacement process may recover verified entry custody.
pub(crate) const VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS: u64 =
    REPLAY_OWNER_TAKEOVER_GRACE_SECS;
/// Grace period before another process may own and reconcile an exact claim.
pub(crate) const BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS: u64 =
    REPLAY_OWNER_TAKEOVER_GRACE_SECS;
/// Minimum SQLite synchronous level permitted for acknowledged relay custody.
const CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
/// Pages copied per online-backup step before SQLite releases its read lock.
const CHAT_RELAY_BACKUP_PAGES_PER_STEP: i32 = 256;
/// Maximum consecutive SQLite busy time before a backup fails closed.
const CHAT_RELAY_BACKUP_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Delay between bounded retries while another process holds a SQLite lock.
const CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY: Duration = Duration::from_millis(10);
/// Maximum UTF-8 bytes accepted from one management-plane operation ID.
const CHAT_RELAY_BACKUP_OPERATION_ID_MAX_BYTES: usize = 128;
/// Fixed validity window for a host-local authenticated restore plan.
pub const CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS: u64 = RESTORE_PLAN_VALIDITY_SECS;
/// Current restore-plan wire contract.
#[cfg(test)]
const CHAT_RELAY_RESTORE_PLAN_VERSION: u8 = RESTORE_PLAN_VERSION;
/// Random nonce bytes encoded into each restore plan.
#[cfg(test)]
const CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES: usize = RESTORE_PLAN_NONCE_BYTES;
/// Defensive ceiling for one verified backup-directory maintenance scan.
const CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT: usize = 1024;
/// Private sibling file serializing maintenance across server processes.
const CHAT_RELAY_BACKUP_LOCK_FILE_NAME: &str = ".aeronyx-relay-backup-maintenance.lock";
/// Hard ceiling for one audit record, including its trailing newline.
const CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES: usize = 4096;
/// Hard ceiling for audit records verified before one append.
const CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS: usize = 65_536;
/// Maximum bytes authenticated across archived and active audit segments.
const CHAT_RELAY_BACKUP_AUDIT_TOTAL_MAX_BYTES: u64 =
    CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES * CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS as u64;

// ============================================
// Peer relay health status
// ============================================

pub use crate::services::chat_relay_status::{
    ChatRelayBlindRouteRecoveryStatus, ChatRelayCustodyDurabilityStatus,
    ChatRelayDirectPeerCircuitStatus, ChatRelayDirectPeerRetryStatus,
    ChatRelayDirectPeerSloStatus, ChatRelayOutboundRouteStatus, ChatRelayPeerStatus,
    ChatRelayVerifiedSubmitRecoveryStatus, ChatRelayVerifiedSubmitStatus,
};
#[cfg(test)]
use crate::services::chat_relay_status::{
    DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS, DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS,
};

// ============================================
// Error type
// ============================================

pub use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};

// ============================================
// ChatRelayService
// ============================================

/// Central service for zero-knowledge P2P chat relay.
///
/// ## v1.3.0-Sovereign additions
/// - `wallet_routes`: Arc-wrapped WalletRouteCache for wallet→session routing.
///   Exposed as a public Arc so `server.rs` can hold an independent clone for
///   the background cleanup task and for passing into message handlers.
pub struct ChatRelayService {
    config: ChatRelayConfig,
    conn: Mutex<Connection>,
    /// Kernel-owned lifetime guard for the persistent custody database.
    ///
    /// [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] Keep this handle alive
    /// until after `conn` drops. It contains no serialized or reported state.
    #[cfg(unix)]
    _runtime_fence: Option<ChatRelayRuntimeFence>,
    node_secret: [u8; 32],
    /// Complete legacy and snapshot pending-message delivery use cases.
    ///
    /// [CHAT-PENDING-DELIVERY-DOMAIN 2026-08-28 by Codex] Ordered reads,
    /// authenticated cursors, bounded lock scope, quarantine, and pagination
    /// are composed while public API telemetry remains service-owned.
    pending_delivery: PendingMessageDeliveryDomain,
    /// Offline-message idempotence, quota, sequence, and ACK capability.
    ///
    /// [CHAT-PENDING-CUSTODY-DOMAIN 2026-08-25 by Codex] The service retains
    /// the public API and lock; the composed domain owns durable write policy.
    pending_custody: PendingMessageCustodyDomain,
    /// Expiry-control read, validation, pagination, and delivery ACK capability.
    ///
    /// [CHAT-EXPIRED-DELIVERY-DOMAIN 2026-08-25 by Codex] Quarantine and
    /// telemetry remain service-owned; durable delivery mechanics are composed.
    expired_notification_delivery: ExpiredNotificationDelivery,
    /// Opaque identity, quota, persistence, retrieval, and deletion capability.
    ///
    /// [CHAT-BLOB-CUSTODY-DOMAIN 2026-08-25 by Codex] The service retains API,
    /// connection locking, and telemetry; the composed domain owns mechanics.
    blob_custody: EncryptedBlobCustodyDomain,
    /// Typed poison-row isolation and de-identified durable evidence.
    ///
    /// [CHAT-DURABLE-QUARANTINE-DOMAIN 2026-08-25 by Codex] Pull and cleanup
    /// flows retain telemetry; this domain owns atomic replacement mechanics.
    durable_quarantine: DurableQuarantineDomain,
    /// Bounded multi-transaction cleanup execution capability.
    ///
    /// [CHAT-CLEANUP-EXECUTION-DOMAIN 2026-08-28 by Codex] The executor owns
    /// cutoffs, lock scope, commits, partial failure, and batch budget while
    /// the service retains scheduling, logs, and aggregate health status.
    cleanup_execution: BoundedRelayCleanupExecutor,
    /// Private verified-submit replay and durable ownership coordinator.
    ///
    /// [VERIFIED-SUBMIT-COORDINATOR-DOMAIN 2026-08-28 by Codex] The service
    /// retains aggregate telemetry while this capability owns request binding,
    /// replay lookup, owner-fenced reservation, recovery, and completion.
    verified_submit: VerifiedSubmitCoordinator,
    /// Private blind-route replay and durable ownership coordinator.
    ///
    /// [BLIND-ROUTE-COORDINATOR-DOMAIN 2026-08-28 by Codex] The service keeps
    /// aggregate recovery telemetry while this capability owns private route
    /// binding, owner-fenced admission, effect arming, and exact ACK replay.
    blind_route: BlindRouteCoordinator,
    /// Random process epoch fencing every restart-recoverable replay claim.
    ///
    /// [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Verified submit
    /// and blind-route reservations share ownership mechanics but retain
    /// independent HMAC namespaces and tables.
    replay_process_epoch: [u8; REPLAY_PROCESS_EPOCH_BYTES],
    dedup: MessageDedup,
    /// Read-only privacy-safe aggregate storage accounting capability.
    ///
    /// [CHAT-STORAGE-USAGE-DOMAIN 2026-08-28 by Codex] Counter SQL and
    /// fail-closed signed conversion stay outside orchestration code.
    storage_usage_repository: SqliteRelayStorageUsageRepository,
    /// Privacy-safe process telemetry and bounded SLO classification.
    ///
    /// [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] One composed state
    /// lock prevents partial SLO/lifetime snapshots without widening labels.
    peer_telemetry: PeerRelayTelemetryDomain,
    /// Source-blind admission state and durable restart checkpoint capability.
    ///
    /// [CHAT-DIRECT-PEER-CIRCUIT-DOMAIN 2026-08-25 by Codex] The service
    /// composes public telemetry separately; this domain owns transitions.
    direct_peer_relay_circuit: DirectPeerCircuitDomain,
    /// Privacy-safe maintenance snapshot and atomic transition capability.
    maintenance_telemetry: RelayMaintenanceTelemetry,
    /// Serializes backup publication, replay verification, and retention.
    backup_operations: Mutex<()>,
    /// In-memory wallet → session routing table.
    ///
    /// Arc so the cleanup task and each handler can hold independent references
    /// without borrowing the whole ChatRelayService.
    pub wallet_routes: Arc<WalletRouteCache>,
}

impl ChatRelayService {
    #[cfg(test)]
    fn restrict_sqlite_file_permissions(path: &Path) -> ChatRelayResult<()> {
        // [CHAT-BACKUP-SQLITE-DOMAIN 2026-08-28 by Codex] Preserve the
        // established in-crate fixture entry point while production ownership
        // remains entirely inside the SQLite backup adapter.
        restrict_private_sqlite_permissions(path)
    }

    fn backup_io_error(code: i32, message: &'static str) -> ChatRelayError {
        backup_io_error(code, message)
    }

    fn backup_audit_maintenance() -> BackupAuditMaintenance {
        // [CHAT-BACKUP-AUDIT-MAINTENANCE-DOMAIN 2026-08-28 by Codex] Build
        // verification, rotation, and append from one immutable limit set so
        // no service wrapper can compose mismatched safety policies.
        BackupAuditMaintenance::new(BackupAuditMaintenanceLimits {
            max_record_bytes: CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES,
            max_records_per_segment: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS)
                .unwrap_or(u64::MAX),
            max_segment_bytes: CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
            max_segments: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS)
                .unwrap_or(u64::MAX),
            max_total_bytes: CHAT_RELAY_BACKUP_AUDIT_TOTAL_MAX_BYTES,
        })
    }

    #[cfg(test)]
    fn reserve_private_backup_file(path: &Path) -> ChatRelayResult<()> {
        // Test fixtures use the production no-follow/private reservation
        // boundary when simulating an interrupted maintenance artifact.
        LocalBackupFilesystem.reserve_private_file(path)
    }

    fn private_backup_directory_for_config(config: &ChatRelayConfig) -> ChatRelayResult<PathBuf> {
        LocalBackupFilesystem.private_directory_for_database(&config.db_path)
    }

    fn private_backup_directory(&self) -> ChatRelayResult<PathBuf> {
        Self::private_backup_directory_for_config(&self.config)
    }

    fn backup_artifact_namespace() -> HmacBackupArtifactNamespace {
        HmacBackupArtifactNamespace::new(CHAT_RELAY_BACKUP_OPERATION_ID_MAX_BYTES)
    }

    fn backup_recovery_image_certifier() -> SqliteBackupRecoveryImageCertifier {
        // [CHAT-RELAY-BACKUP-CERTIFICATION-DOMAIN 2026-08-27 by Codex] Keep
        // runtime schema installation and recovery certification bound to the
        // same exact versions without moving service migration ownership.
        SqliteBackupRecoveryImageCertifier::new(
            RecoveryImageSchemaRequirement::new(
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            ),
            RecoveryImageSchemaRequirement::new(
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
            ),
        )
    }

    fn verify_existing_backup_artifact(path: &Path) -> ChatRelayResult<u64> {
        verify_existing_backup_creation_artifact(
            &LocalBackupFilesystem,
            path,
            Self::verify_sqlite_backup,
        )
    }

    fn backup_inventory() -> VerifiedBackupInventory<
        HmacBackupArtifactNamespace,
        BoundedBackupRetentionPlanner,
        fn(&Path) -> ChatRelayResult<u64>,
    > {
        // [CHAT-RELAY-BACKUP-INVENTORY-DOMAIN 2026-08-27 by Codex] Compose
        // trusted namespace and retention policy around the full SQLite
        // verifier; no private path or mutable state enters the policies.
        VerifiedBackupInventory::new(
            Self::backup_artifact_namespace(),
            BoundedBackupRetentionPlanner,
            Self::verify_existing_backup_artifact,
        )
    }

    fn backup_inventory_limits(config: &ChatRelayConfig) -> BackupInventoryLimits {
        BackupInventoryLimits::new(
            CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT,
            BackupRetentionLimits::new(
                config.custody_backup_retention_target_artifacts,
                config.custody_backup_retention_target_bytes,
                config.custody_backup_partial_grace_secs,
            ),
        )
    }

    fn restore_plan_command() -> impl RestorePlanCommand {
        // [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Compose the
        // verified private inventory with metadata-only active inspection and
        // the v1 authenticator at the service edge.
        local_restore_plan_command(Self::backup_inventory())
    }

    fn inspect_verified_backup_retention(
        config: &ChatRelayConfig,
        backup_directory: &Path,
        now_unix_secs: u64,
    ) -> ChatRelayResult<ChatRelayBackupRetentionInspection> {
        Self::backup_inventory().inspect(
            backup_directory,
            now_unix_secs,
            Self::backup_inventory_limits(config),
        )
    }

    fn map_backup_namespace_error(error: BackupNamespaceError) -> ChatRelayError {
        match error {
            BackupNamespaceError::EmptyOperationId
            | BackupNamespaceError::OperationIdTooLarge
            | BackupNamespaceError::OperationIdLengthOverflow => Self::backup_io_error(
                rusqlite::ffi::SQLITE_MISUSE,
                "invalid relay backup operation identifier",
            ),
            BackupNamespaceError::SecretRejected => Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to derive private relay backup artifact identity",
            ),
        }
    }

    #[cfg(test)]
    fn open_private_backup_control_file(path: &Path, append: bool) -> ChatRelayResult<File> {
        let mode = if append {
            PrivateBackupControlFileMode::Append
        } else {
            PrivateBackupControlFileMode::ReadWrite
        };
        LocalBackupFilesystem.open_control_file(path, mode)
    }

    fn acquire_backup_filesystem_lock(backup_directory: &Path) -> ChatRelayResult<Connection> {
        LocalBackupFilesystem.acquire_maintenance_lock(
            backup_directory,
            CHAT_RELAY_BACKUP_LOCK_FILE_NAME,
        )
    }

    #[cfg(test)]
    fn backup_audit_segment_file_name(range: ChatRelayBackupAuditSegmentRange) -> String {
        Self::backup_audit_maintenance().segment_file_name(range)
    }

    #[cfg(test)]
    fn backup_audit_checkpoint_file_name(range: ChatRelayBackupAuditSegmentRange) -> String {
        Self::backup_audit_maintenance().checkpoint_file_name(range)
    }

    fn backup_audit_anchor_digest(
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<[u8; 32]> {
        derive_backup_audit_anchor_digest(state).map_err(|error| match error {
            BackupAuditAnchorDigestError::MissingImmutableCheckpoint => Self::backup_io_error(
                rusqlite::ffi::SQLITE_NOTFOUND,
                "relay backup maintenance audit has no immutable checkpoint to anchor",
            ),
            BackupAuditAnchorDigestError::InvalidCheckpointAuthenticator => Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint anchor is invalid",
            ),
        })
    }

    #[cfg(test)]
    fn verify_backup_audit_log(
        file: &mut File,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationState> {
        Self::backup_audit_maintenance().verify_log(file, node_secret)
    }

    fn verify_backup_audit_chain(
        parent: &Path,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditChainVerification> {
        Self::backup_audit_maintenance().verify_chain(parent, node_secret)
    }

    #[cfg(test)]
    fn rotate_backup_audit_segment(
        parent: &Path,
        node_secret: &[u8; 32],
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<()> {
        Self::backup_audit_maintenance().rotate_segment(parent, node_secret, state)
    }

    #[cfg(test)]
    fn backup_audit_segment_needs_rotation(
        active_record_count: u64,
        active_bytes: u64,
        next_record_bytes: usize,
    ) -> ChatRelayResult<bool> {
        Self::backup_audit_maintenance().segment_needs_rotation(
            active_record_count,
            active_bytes,
            next_record_bytes,
        )
    }

    fn append_backup_maintenance_audit(
        backup_directory: &Path,
        node_secret: &[u8; 32],
        phase: BackupAuditPhase,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()> {
        Self::backup_audit_maintenance().append(
            backup_directory,
            node_secret,
            phase,
            timestamp,
            counts,
        )
    }

    #[cfg(test)]
    fn is_lower_hex(value: &str, expected_len: usize) -> bool {
        // [CHAT-RELAY-BACKUP-AUDIT-IO-DOMAIN 2026-08-27 by Codex] Preserve
        // the shared private test helper without reintroducing production I/O.
        value.len() == expected_len
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn prune_verified_backup_retention_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        request: &ChatRelayBackupPruneRequest,
        now_unix_secs: u64,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        // [CHAT-RELAY-BACKUP-PRUNE-DOMAIN 2026-08-27 by Codex] Admission
        // remains before path resolution so an invalid command has no storage
        // side effect. The service keeps lock ownership for the full command.
        let admission = admit_backup_prune_request(request)?;
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let audit = |phase, counts| {
            Self::append_backup_maintenance_audit(
                &backup_directory,
                node_secret,
                phase,
                now_unix_secs,
                counts,
            )
        };
        AuditedBackupPruneExecutor::new(
            Self::backup_inventory(),
            LocalBackupArtifactRemoval,
            audit,
        )
        .execute(
            &backup_directory,
            admission,
            now_unix_secs,
            Self::backup_inventory_limits(config),
        )
    }

    fn create_verified_backup_artifact(
        &self,
        backup_directory: &Path,
        destination: &Path,
        reuse_existing: bool,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        let temporary_nonce = rand::random::<u64>();
        let temporary_name = Self::backup_artifact_namespace()
            .temporary_recovery_image_name(now_secs(), temporary_nonce);
        let temporary = backup_directory.join(temporary_name.as_str());
        VerifiedBackupCreationCommand::new(
            LocalBackupFilesystem,
            SqliteRelayBackupDatabase::new(
                &self.conn,
                Self::backup_recovery_image_certifier(),
                CHAT_RELAY_BACKUP_PAGES_PER_STEP,
                CHAT_RELAY_BACKUP_BUSY_TIMEOUT,
                CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY,
            ),
        )
        .execute(VerifiedBackupCreationRequest {
            backup_directory,
            destination,
            temporary: &temporary,
            reuse_existing,
        })
    }

    fn verify_sqlite_backup(conn: &Connection) -> ChatRelayResult<()> {
        Self::backup_recovery_image_certifier().verify(conn, now_secs())
    }

    /// Creates a new `ChatRelayService`, opening (or creating) the SQLite database.
    pub fn new(config: ChatRelayConfig, node_secret: [u8; 32]) -> ChatRelayResult<Self> {
        if let Some(parent) = std::path::Path::new(&config.db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] A raw IO
                // error can include an operator path. Preserve one stable,
                // path-free failure at this storage trust boundary.
                std::fs::create_dir_all(parent).map_err(|_| {
                    rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                        Some("unable to create relay database directory".to_string()),
                    )
                })?;
            }
        }

        // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] Acquire before
        // opening or migrating SQLite. A replacement process can recover an
        // aged reservation only after the kernel has released this guard,
        // proving that its predecessor no longer owns the custody store.
        #[cfg(unix)]
        let runtime_fence = if config.db_path == ":memory:" {
            None
        } else {
            Some(
                ChatRelayRuntimeFence::acquire(Path::new(&config.db_path)).map_err(|error| {
                    ChatRelayError::RuntimeFenceUnavailable {
                        reason: error.as_str(),
                        public_reason_bucket: error.public_reason_bucket(),
                    }
                })?,
            )
        };
        let conn = Connection::open(&config.db_path)?;
        if config.db_path != ":memory:" {
            restrict_private_sqlite_permissions(Path::new(&config.db_path))?;
        }
        // A short bounded wait absorbs transient locks from an operator backup
        // or diagnostic reader without allowing relay requests to hang forever.
        conn.busy_timeout(Duration::from_secs(5))?;
        verify_sqlite_physical_integrity(&conn, "sqlite_startup_integrity")?;
        let synchronous_level =
            configure_full_durability(&conn, CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL)?;

        let dedup_capacity = config.dedup_lru_capacity;
        let relay_enabled = config.enabled;
        let pending_delivery = PendingMessageDeliveryDomain::new(&node_secret)?;
        let pending_custody = PendingMessageCustodyDomain::new(&config);
        let expired_notification_delivery = ExpiredNotificationDelivery::new();
        let blob_custody = EncryptedBlobCustodyDomain::new(node_secret, &config);
        let durable_quarantine = DurableQuarantineDomain::new(&config);
        let cleanup_execution = BoundedRelayCleanupExecutor::new(
            &config,
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
        );
        let verified_submit = VerifiedSubmitCoordinator::new(
            node_secret,
            dedup_capacity,
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS,
        )?;
        let blind_route = BlindRouteCoordinator::new(
            node_secret,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_CAPACITY,
            BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS,
        )?;
        let mut replay_process_epoch = [0_u8; REPLAY_PROCESS_EPOCH_BYTES];
        OsRng.fill_bytes(&mut replay_process_epoch);
        let mut peer_status = ChatRelayPeerStatus::new(relay_enabled);
        peer_status.custody_durability =
            ChatRelayCustodyDurabilityStatus::verified_full(synchronous_level);
        let svc = Self {
            config,
            conn: Mutex::new(conn),
            #[cfg(unix)]
            _runtime_fence: runtime_fence,
            node_secret,
            pending_delivery,
            pending_custody,
            expired_notification_delivery,
            blob_custody,
            durable_quarantine,
            cleanup_execution,
            verified_submit,
            blind_route,
            replay_process_epoch,
            dedup: MessageDedup::new(dedup_capacity),
            storage_usage_repository: SqliteRelayStorageUsageRepository,
            peer_telemetry: PeerRelayTelemetryDomain::new(peer_status),
            direct_peer_relay_circuit: DirectPeerCircuitDomain::default(),
            maintenance_telemetry: RelayMaintenanceTelemetry::default(),
            backup_operations: Mutex::new(()),
            // v1.3.0-Sovereign: initialise empty route cache
            wallet_routes: Arc::new(WalletRouteCache::new()),
        };

        svc.init_schema()?;
        svc.direct_peer_relay_circuit
            .restore(&svc.conn, now_secs())?;
        // [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] The filesystem
        // path is operator-local state and may contain deployment identities.
        // Keep successful activation observable without publishing that path.
        info!("[CHAT_RELAY] Durable service initialized");
        Ok(svc)
    }

    // ============================================
    // Schema initialisation
    // ============================================

    fn init_schema(&self) -> ChatRelayResult<()> {
        let mut conn = self.conn.lock();
        let pending_schema = SqliteChatRelayPendingSchemaMigrator::new();
        pending_schema.migrate(&mut conn)?;
        let storage_schema = SqliteChatRelayStorageSchemaMigrator::new();
        storage_schema.install_custody_tables(&conn)?;
        self.durable_quarantine.init_schema(&conn)?;
        storage_schema.install_usage_accounting(&conn)?;
        self.direct_peer_relay_circuit
            .init_schema(&mut conn, now_secs())?;
        let replay_schema = SqliteChatRelayReplaySchemaMigrator::new(
            ReplaySchemaContract::new(
                ReplaySchemaVersion::new(
                    VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
                    VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
                    VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION,
                    VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
                ),
                ReplaySchemaVersion::new(
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
                ),
                VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
                BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
                REPLAY_PROCESS_EPOCH_BYTES,
            ),
        );
        replay_schema.migrate_verified_submit(&mut conn, now_secs())?;
        replay_schema.migrate_blind_route(&mut conn, now_secs())?;
        storage_schema.reconcile_usage(&conn)?;
        let retained_quarantine_events = self.durable_quarantine.retained_count(&conn)?;
        drop(conn);
        self.maintenance_telemetry
            .set_retained_quarantine_events(retained_quarantine_events);
        Ok(())
    }

    // ============================================
    // Opaque ChatPullV2 cursor protection
    // ============================================

    #[cfg(test)]
    fn decode_pull_cursor_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        self.pending_delivery
            .decode_cursor(receiver, after_timestamp, encoded)
    }

    // ============================================
    // Blob ID derivation
    // ============================================

    pub fn compute_blob_id(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        file_hash: &[u8; 32],
    ) -> String {
        self.blob_custody
            .compute_blob_id(sender, receiver, file_hash)
    }

    // ============================================
    // Online-path deduplication
    // ============================================

    /// Returns `true` if this `message_id` has already been forwarded on the
    /// online path (duplicate detection for live sessions).
    pub fn is_online_duplicate(&self, message_id: &[u8; 16]) -> bool {
        self.dedup.check_and_insert(message_id)
    }

    /// Serializes requests sharing one private sender/request-id cache key.
    ///
    /// Unrelated submissions remain concurrent across fixed lock lanes. The
    /// caller must hold the returned guard through lookup, relay/custody, and
    /// response insertion so duplicate requests cannot both become leaders.
    pub(crate) async fn lock_verified_submit(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> tokio::sync::MutexGuard<'_, ()> {
        self.verified_submit.lock(request).await
    }

    /// Looks up a completed response after request authentication.
    pub(crate) fn verified_submit_cache_lookup(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayResult<VerifiedSubmitCacheLookup> {
        self.verified_submit.lookup(&self.conn, request, now_secs())
    }

    /// Atomically reserves one private replay slot before any external effect.
    pub(crate) fn reserve_verified_submit(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayResult<VerifiedSubmitAdmission> {
        let now = now_secs();
        let outcome = self.verified_submit.reserve(
            &self.conn,
            request,
            self.replay_process_epoch.as_slice(),
            now,
        )?;
        if matches!(
            outcome,
            VerifiedSubmitAdmission::ReservedForEntryRecovery
        ) {
            // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex]
            // Admission remains the authoritative attempted transition.
            self.record_verified_submit_recovery_attempted(now);
        }
        Ok(outcome)
    }

    /// Retains one completed response for exact retry replay across restarts.
    pub(crate) fn remember_verified_submit_response(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
        response: &ChatRelayVerifiedSubmitResponseV1,
    ) -> ChatRelayResult<()> {
        self.verified_submit.remember_response(
            &self.conn,
            request,
            response,
            self.replay_process_epoch.as_slice(),
            now_secs(),
        )
    }

    /// Reserves one authenticated blind route before peel, forward, or store.
    pub(crate) fn reserve_blind_relay_route(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
    ) -> ChatRelayResult<BlindRelayRouteAdmission> {
        let now = now_secs();
        let admission = self.blind_route.reserve(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
            now,
        )?;
        if matches!(admission, BlindRelayRouteAdmission::ReservedForRecovery) {
            self.peer_telemetry
                .record_blind_route_recovery(now, BlindRouteRecoveryEvent::Attempted);
        }
        Ok(admission)
    }

    /// Arms an owned route claim immediately before its first external effect.
    pub(crate) fn arm_blind_relay_route_effect(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        started_at: u64,
    ) -> ChatRelayResult<()> {
        self.blind_route.arm_effect(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
            started_at,
        )
    }

    /// Releases only this process's claim when no external effect was armed.
    pub(crate) fn release_unarmed_blind_relay_route(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
    ) -> ChatRelayResult<bool> {
        self.blind_route.release_unarmed(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
        )
    }

    /// Atomically replaces one route reservation with its sealed exact ACK.
    pub(crate) fn remember_blind_relay_route_response(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        response: &[u8],
        completed_at: u64,
    ) -> ChatRelayResult<()> {
        self.blind_route.remember_response(
            &self.conn,
            route_id,
            request_commitment,
            self.replay_process_epoch.as_slice(),
            response,
            completed_at,
        )
    }

    // ============================================
    // Peer relay health
    // ============================================

    /// Records a compatibility direct node-to-node encrypted relay round.
    ///
    /// The backward-compatible aggregate fields are also advanced.
    /// The failure reason must be a stable bucket such as
    /// `peer_relay_request_timeout` or `peer_relay_http_503`; do not pass peer
    /// URLs, message IDs, wallet IDs, client IPs, or payload-derived data.
    /// Unrecognized legacy input is intentionally reported as `unknown`.
    pub fn record_peer_relay_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        self.record_peer_relay_outbound_typed(
            now,
            attempted,
            accepted,
            failure_reason
                .as_deref()
                .map(ChatRelayOutboundFailureReason::from_bucket),
        );
    }

    /// Records direct relay health through the compiler-checked reason type.
    pub(crate) fn record_peer_relay_outbound_typed(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
    ) {
        self.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            OutboundRouteClass::DirectPeer,
        );
    }

    /// Begins one target-bound direct relay delivery when the circuit permits.
    ///
    /// A returned permit must be completed after a network outcome or cancelled
    /// if local request construction fails before network I/O. `None` means the
    /// caller must fail closed without falling back to an older relay protocol.
    pub(crate) fn begin_direct_peer_delivery(&self, now: u64) -> Option<ChatRelayDirectPeerPermit> {
        // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Admission is intentionally
        // process-global and source-blind; per-peer quarantine remains owned by
        // PeerStore and must not be duplicated with identity-labelled state here.
        self.direct_peer_relay_circuit.begin(&self.conn, now)
    }

    /// Releases an unused half-open permit after a local preflight failure.
    pub(crate) fn cancel_direct_peer_delivery(&self, now: u64, permit: ChatRelayDirectPeerPermit) {
        self.direct_peer_relay_circuit
            .cancel(&self.conn, now, permit);
    }

    /// Completes one aggregate target-bound direct relay delivery observation.
    ///
    /// `retry_triggered` means a second exact request was sent. A triggered
    /// retry is classified as either recovered or exhausted. Independently,
    /// `final_failure_deterministic` records that the final typed failure was
    /// not eligible for another ambiguity retry. Every delivery updates the
    /// recent SLO window, while successful first attempts leave the lifetime
    /// exception counters unchanged.
    ///
    /// [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Parameters are
    /// deliberately aggregate booleans. Do not extend this API with peer,
    /// message, commitment, endpoint, wallet, or payload identifiers.
    pub(crate) fn complete_direct_peer_delivery(
        &self,
        now: u64,
        permit: ChatRelayDirectPeerPermit,
        retry_triggered: bool,
        delivery_succeeded: bool,
        final_failure_deterministic: bool,
    ) -> bool {
        debug_assert!(!(delivery_succeeded && final_failure_deterministic));
        // [DIRECT-RELAY-SLO 2026-08-15 by Codex] Every v3 delivery contributes
        // exactly one aggregate sample, including successful first attempts.
        // The fixed ring retains no event identity or peer dimension.
        let observe_slo_failed = || {
            self.peer_telemetry.record_direct_peer_delivery(
                now,
                retry_triggered,
                delivery_succeeded,
                final_failure_deterministic,
            )
        };
        let circuit_allows_more = self.direct_peer_relay_circuit.complete(
            &self.conn,
            now,
            permit,
            delivery_succeeded,
            observe_slo_failed,
        );
        circuit_allows_more
    }

    /// Records one authenticated receipt-verified onion relay round.
    ///
    /// [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] This advances both the
    /// backward-compatible aggregate and the authenticated-onion snapshot.
    /// A later compatibility fallback may update the aggregate but cannot
    /// erase the authenticated path's latest evidence.
    pub fn record_authenticated_onion_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        self.record_authenticated_onion_outbound_typed(
            now,
            attempted,
            accepted,
            failure_reason
                .as_deref()
                .map(ChatRelayOutboundFailureReason::from_bucket),
        );
    }

    /// Records authenticated onion health through the validated reason type.
    pub(crate) fn record_authenticated_onion_outbound_typed(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
    ) {
        self.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            OutboundRouteClass::AuthenticatedOnion,
        );
    }

    fn record_outbound_round(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<ChatRelayOutboundFailureReason>,
        route_class: OutboundRouteClass,
    ) {
        self.peer_telemetry.record_outbound_round(
            now,
            attempted,
            accepted,
            failure_reason,
            route_class,
        );
    }

    /// Records the closed aggregate result of one explicit verified submit.
    pub(crate) fn record_verified_submit_result(&self, now: u64, result: u8) {
        // [CHAT-VERIFIED-SUBMIT-TELEMETRY 2026-08-23 by Codex] This is a
        // single node-wide counter update. Do not attach request, message,
        // route, endpoint, wallet, receipt, or payload dimensions here.
        self.peer_telemetry
            .record_verified_submit(now, result, VerifiedSubmitEvent::Closed);
    }

    /// Records one exact retry served without repeating route or custody work.
    pub(crate) fn record_verified_submit_replay(&self, now: u64, result: u8) {
        self.peer_telemetry
            .record_verified_submit(now, result, VerifiedSubmitEvent::Replay);
    }

    /// Records fail-closed reuse of a request id for a different envelope.
    pub(crate) fn record_verified_submit_conflict(&self, now: u64, result: u8) {
        self.peer_telemetry
            .record_verified_submit(now, result, VerifiedSubmitEvent::Conflict);
    }

    /// Records a crash-left exact request rejected before repeating effects.
    pub(crate) fn record_verified_submit_pending_rejection(&self, now: u64, result: u8) {
        self.peer_telemetry.record_verified_submit(
            now,
            result,
            VerifiedSubmitEvent::PendingRejection,
        );
    }

    /// Records admission saturation without exposing retained request metadata.
    pub(crate) fn record_verified_submit_capacity_rejection(&self, now: u64, result: u8) {
        self.peer_telemetry.record_verified_submit(
            now,
            result,
            VerifiedSubmitEvent::CapacityRejection,
        );
    }

    /// Records one foreign-owner takeover after durable admission commits.
    pub(crate) fn record_verified_submit_recovery_attempted(&self, now: u64) {
        self.peer_telemetry
            .record_verified_submit_recovery_attempted(now);
    }

    /// Records one closed restart-recovery transition without request data.
    pub(crate) fn record_verified_submit_recovery_outcome(
        &self,
        now: u64,
        outcome: VerifiedSubmitRecoveryOutcome,
    ) {
        // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] Keep this
        // process-local aggregate independent from the normal protocol result
        // counters; one recovered request must not be counted as two submits.
        self.peer_telemetry
            .record_verified_submit_recovery_outcome(now, outcome);
    }

    /// Records an accepted inbound peer relay request.
    pub fn record_peer_relay_inbound_accepted(
        &self,
        now: u64,
        duplicate: bool,
        delivered_online: usize,
        stored_pending: bool,
    ) {
        self.peer_telemetry.record_inbound_accepted(
            now,
            duplicate,
            delivered_online,
            stored_pending,
        );
    }

    /// Records a rejected inbound peer relay request with a stable reason bucket.
    ///
    /// This compatibility entry point sanitizes unrecognized text to `unknown`.
    pub fn record_peer_relay_inbound_rejected(&self, now: u64, reason: impl Into<String>) {
        let reason = reason.into();
        self.record_peer_relay_inbound_rejected_typed(
            now,
            ChatRelayInboundFailureReason::from_bucket(&reason),
        );
    }

    /// Records an inbound rejection through the validated reason type.
    pub(crate) fn record_peer_relay_inbound_rejected_typed(
        &self,
        now: u64,
        reason: ChatRelayInboundFailureReason,
    ) {
        self.peer_telemetry.record_inbound_rejected(now, reason);
    }

    // ============================================
    // Accessors
    // ============================================

    #[must_use]
    pub fn config(&self) -> &ChatRelayConfig {
        &self.config
    }

    /// Returns a privacy-safe node-to-node relay health snapshot.
    #[must_use]
    pub fn peer_status(&self) -> ChatRelayPeerStatus {
        // [CHAT-PEER-TELEMETRY-DOMAIN 2026-08-26 by Codex] Circuit state stays
        // independently durable; all process telemetry is copied atomically.
        let now = now_secs();
        let circuit = self.direct_peer_relay_circuit.snapshot(now);
        self.peer_telemetry.snapshot(now, circuit)
    }

    /// Records a successfully sealed ACK for an armed route takeover.
    pub(crate) fn record_blind_route_recovery_completed(&self, now: u64) {
        self.peer_telemetry
            .record_blind_route_recovery(now, BlindRouteRecoveryEvent::Completed);
    }

    /// Records an armed takeover retained for a later exact retry.
    pub(crate) fn record_blind_route_recovery_deferred(&self, now: u64) {
        self.peer_telemetry
            .record_blind_route_recovery(now, BlindRouteRecoveryEvent::Deferred);
    }

    /// Returns aggregate TTL cleanup execution evidence.
    #[must_use]
    pub fn maintenance_status(&self) -> ChatRelayMaintenanceStatus {
        self.maintenance_telemetry.snapshot()
    }

    /// Records a blocking-worker failure that occurred outside `run_cleanup`.
    ///
    /// Tokio join failures are deliberately converted to stable buckets so a
    /// heartbeat never exposes panic payloads or other runtime internals.
    pub(crate) fn record_maintenance_worker_failure(&self, reason: &'static str) {
        self.maintenance_telemetry
            .record_worker_failure(now_secs(), reason);
    }

    /// Returns aggregate durable queue usage maintained by `SQLite` triggers.
    ///
    /// The result contains no message, wallet, sender, receiver, route, or
    /// payload identifiers and is safe for operator-capacity telemetry.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error if the reconciled singleton usage row cannot be
    /// read. Callers must treat that as unavailable telemetry, not zero usage.
    pub fn storage_usage(&self) -> ChatRelayResult<ChatRelayStorageUsage> {
        let conn = self.conn.lock();
        self.storage_usage_repository.read(&conn)
    }

}

#[path = "chat_relay_backup_facade.rs"]
mod backup_facade;

#[path = "chat_relay_blob_facade.rs"]
mod blob_facade;

#[path = "chat_relay_cleanup_facade.rs"]
mod cleanup_facade;

#[path = "chat_relay_expired_facade.rs"]
mod expired_facade;

#[path = "chat_relay_pending_facade.rs"]
mod pending_facade;

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

// ============================================
// Helpers
// ============================================

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
#[path = "chat_relay_tests.rs"]
mod tests;
