// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay.rs
// ============================================================================
// Version: 3.47.0-AuditedBackupPruneDomain
//
// Modification Reason:
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
//   - [VERIFIED-SUBMIT-REPLAY-DOMAIN 2026-08-25 by Codex] Verified-submit
//     cache policy, lock striping, private key derivation, and AEAD response
//     protection live in a composed domain capability. Keep SQLite admission
//     and completion transactions service-owned until a repository can share
//     the same custody connection without weakening atomicity.
//   - [BLIND-ROUTE-REPLAY-DOMAIN 2026-08-25 by Codex] Blind-route HMAC
//     identities and sealed ACK cryptography live in a composed capability.
//     Keep durable reservation, takeover, and completion transactions on the
//     service-owned SQLite connection until repository extraction can retain
//     the exact same single-transaction safety boundary.
//   - [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Blind-route claims
//     have an anonymous process epoch and durable effect boundary. Only an
//     expired, unarmed claim may move to a new process; armed and legacy claims
//     remain fail-closed for the complete replay horizon.
//   - Quarantine events must remain de-identified. Never persist message IDs,
//     sender/receiver keys, ciphertext, endpoints, or raw durable rows there.
//
// Last Modified:
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

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::{Mutex, RwLock};
use rand::{rngs::OsRng, RngCore};
use rusqlite::{
    backup::{Backup, StepResult},
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use tracing::{debug, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS;
#[cfg(test)]
use aeronyx_core::protocol::chat::encode_envelope;
use aeronyx_core::protocol::chat::{ChatEnvelope, CustodyAuditAnchorV1};
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
    BackupAuditPhase, BackupAuditRecordAuthenticator, ChatRelayBackupMaintenanceAuditCounts,
    HmacBackupAuditRecordAuthenticator,
};
use crate::services::chat_relay_backup_audit_chain::{
    map_backup_audit_checkpoint_error, map_backup_audit_record_error,
    map_backup_audit_verification_error, AuthenticatedBackupAuditChainVerifier,
    BackupAuditChainLimits, BackupAuditChainVerifier, ChatRelayBackupAuditChainVerification,
    LocalBackupAuditChainVerifier,
};
use crate::services::chat_relay_backup_audit_checkpoint::{
    BackupAuditCheckpointAuthenticator, BackupAuditCheckpointState,
    ChatRelayBackupAuditCheckpoint, HmacBackupAuditCheckpointAuthenticator,
};
#[cfg(test)]
use crate::services::chat_relay_backup_audit_io::BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX
    as CHAT_RELAY_BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX;
use crate::services::chat_relay_backup_audit_io::{
    BackupAuditIo, ChatRelayBackupAuditPendingRotation, LocalBackupAuditIo,
    BACKUP_AUDIT_FILE_NAME as CHAT_RELAY_BACKUP_AUDIT_FILE_NAME,
    BACKUP_AUDIT_MAX_BYTES as CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
    BACKUP_AUDIT_MAX_SEGMENTS as CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS,
};
use crate::services::chat_relay_backup_audit_rotation::{
    BackupAuditRotationError, BackupAuditRotationLimits, BackupAuditRotationPolicy,
    BackupAuditRotationState, BoundedBackupAuditRotationPolicy,
    ChatRelayBackupAuditSegmentRange,
};
pub use crate::services::chat_relay_backup_audit_verification::ChatRelayBackupAuditVerificationReceipt;
use crate::services::chat_relay_backup_audit_verification::{
    BackupAuditVerificationLimits, BackupAuditVerificationPolicy,
    BoundedBackupAuditVerificationPolicy, ChatRelayBackupAuditVerificationState,
};
use crate::services::chat_relay_backup_artifact::BackupArtifactSnapshot;
use crate::services::chat_relay_backup_contract::ChatRelayBackupReceipt;
pub use crate::services::chat_relay_backup_contract::{
    ChatRelayBackupPruneReceipt, ChatRelayBackupPruneRequest, ChatRelayBackupRetentionReceipt,
    ChatRelayRestoreReadinessReceipt, CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
};
use crate::services::chat_relay_backup_copy::{
    BackupCopyAction, BackupCopyPolicyError, BackupCopyProgress, BackupCopyRetryPolicy,
    BackupCopyRetryState, BoundedBackupCopyRetryPolicy,
};
use crate::services::chat_relay_backup_io::{
    backup_io_error, BackupFilesystem, LocalBackupFilesystem, PrivateBackupControlFileMode,
};
use crate::services::chat_relay_backup_inventory::{
    inspect_active_restore_boundary, verified_restore_backup_count, BackupInventory,
    BackupInventoryLimits, ChatRelayActiveRestoreBoundary, ChatRelayBackupRetentionInspection,
    VerifiedBackupInventory,
};
use crate::services::chat_relay_backup_namespace::{
    BackupArtifactNamespace, BackupNamespaceError, HmacBackupArtifactNamespace,
};
use crate::services::chat_relay_backup_retention::{
    BackupRetentionLimits, BoundedBackupRetentionPlanner,
};
use crate::services::chat_relay_backup_prune::{
    admit_backup_prune_request, AuditedBackupPruneExecutor, BackupPruneExecutor,
    LocalBackupArtifactRemoval,
};
pub(crate) use crate::services::chat_relay_blind_route::BlindRelayRouteAdmission;
use crate::services::chat_relay_blind_route::BlindRouteReplay;
#[cfg(test)]
use crate::services::chat_relay_blind_route::RESPONSE_NONCE_BYTES
    as BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES;
use crate::services::chat_relay_blob_custody::{
    EncryptedBlobCustodyDomain, EncryptedBlobStoreOutcome,
};
use crate::services::chat_relay_cleanup::{
    CleanupBatchOutcome, CleanupRunSummary, RelayCleanupCutoffs, RelayCleanupDomain,
    CLEANUP_MAX_BATCHES_PER_RUN,
};
pub(crate) use crate::services::chat_relay_direct_peer_circuit::ChatRelayDirectPeerPermit;
use crate::services::chat_relay_direct_peer_circuit::{
    DirectPeerCircuitDomain, SqliteDirectPeerCircuitRepository,
    DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION, DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE,
};
#[cfg(test)]
use crate::services::chat_relay_cleanup::CLEANUP_MESSAGE_BATCH_SIZE;
use crate::services::chat_relay_expired_delivery::ExpiredNotificationDelivery;
use crate::services::chat_relay_peer_telemetry::{
    BlindRouteRecoveryEvent, OutboundRouteClass, PeerRelayTelemetryDomain, PeerRelayTelemetrySink,
    VerifiedSubmitEvent,
};
pub(crate) use crate::services::chat_relay_peer_telemetry::{
    ChatRelayInboundFailureReason, ChatRelayOutboundFailureReason, VerifiedSubmitRecoveryOutcome,
};
#[cfg(test)]
use crate::services::chat_relay_pending_custody::allocate_queue_sequence;
use crate::services::chat_relay_pending_custody::{
    PendingMessageCustodyDomain, PendingMessageStoreOutcome,
};
use crate::services::chat_relay_pending_pull::PendingMessagePullDomain;
#[cfg(test)]
use crate::services::chat_relay_pull_cursor::ENCODED_CURSOR_BYTES as CHAT_PULL_CURSOR_V2_BYTES;
use crate::services::chat_relay_pull_cursor::{ChatPullCursorCodec, PullCursorV2};
use crate::services::chat_relay_quarantine::{
    CorruptDurableRow, DurableQuarantineDomain, QuarantineRowTarget,
};
#[cfg(test)]
use crate::services::chat_relay_quarantine::{
    MAX_QUARANTINE_EVENTS, QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
    QUARANTINE_SOURCE_PENDING_MESSAGE,
};
pub use crate::services::chat_relay_restore_plan::ChatRelayRestorePlanReceipt;
use crate::services::chat_relay_restore_plan::{
    HmacRestorePlanAuthenticator, RestorePlanAggregate, RestorePlanAuthenticator,
    RestorePlanError, RestorePlanPrivateBoundary, RESTORE_PLAN_NONCE_BYTES,
    RESTORE_PLAN_VALIDITY_SECS, RESTORE_PLAN_VERSION,
};
#[cfg(unix)]
use crate::services::chat_relay_runtime_fence::ChatRelayRuntimeFence;
pub(crate) use crate::services::chat_relay_verified_submit::{
    VerifiedSubmitAdmission, VerifiedSubmitCacheLookup,
};
use crate::services::chat_relay_verified_submit::VerifiedSubmitReplay;
use crate::services::wallet_routes::WalletRouteCache;

// ============================================
// Constants
// ============================================
/// Maximum IDs accepted in one authenticated `ChatAck` frame.
pub const MAX_CHAT_ACK_MESSAGE_IDS: usize = 100;
/// Maximum IDs encoded into one `ChatExpired` frame.
pub(crate) const MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION: usize = 32;
/// Maximum notification rows offered during one authenticated pull.
pub(crate) const MAX_EXPIRED_NOTIFICATIONS_PER_PULL: usize = 16;
/// Defensive ceiling for one persisted bincode notification payload.
pub(crate) const MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES: usize = 1024;
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
const CHAT_RELAY_RESTORE_PLAN_VERSION: u8 = RESTORE_PLAN_VERSION;
/// Random nonce bytes encoded into each restore plan.
const CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES: usize = RESTORE_PLAN_NONCE_BYTES;
/// Defensive ceiling for one verified backup-directory maintenance scan.
const CHAT_RELAY_BACKUP_DIRECTORY_ENTRY_HARD_LIMIT: usize = 1024;
/// Domain separation for the public opaque digest of one private checkpoint.
const CHAT_RELAY_BACKUP_AUDIT_ANCHOR_DIGEST_DOMAIN: &[u8] =
    b"AeroNyx-RelayCustodyBackup-MaintenanceAuditAnchorDigest-v1";
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

/// Aggregate durable relay usage with no user or routing identifiers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayStorageUsage {
    /// Active pending message rows.
    pub pending_messages: u64,
    /// Encoded bytes held by active pending messages.
    pub pending_message_bytes: u64,
    /// Pending encrypted blob rows.
    pub pending_blobs: u64,
    /// Encrypted blob bytes retained by the node.
    pub pending_blob_bytes: u64,
}

/// Aggregate TTL maintenance evidence safe for heartbeat and node health APIs.
///
/// This snapshot intentionally excludes message IDs, wallet keys, routes,
/// endpoints, payloads, and per-user counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayMaintenanceStatus {
    /// Total cleanup attempts, including failed transactions.
    pub cleanup_runs_total: u64,
    /// Cleanup attempts that returned an error.
    pub cleanup_failures_total: u64,
    /// Successfully committed bounded cleanup transactions.
    pub cleanup_batches_total: u64,
    /// Runs that reached their transaction budget with work still pending.
    pub cleanup_backlog_deferred_total: u64,
    /// Pending message rows removed by successfully committed batches.
    pub expired_messages_total: u64,
    /// Encrypted blob rows removed by successfully committed batches.
    pub expired_blobs_total: u64,
    /// Delivered or stale expiry-notification rows removed by committed batches.
    pub expired_notifications_removed_total: u64,
    /// Corrupt pending-message rows atomically isolated from active delivery.
    pub quarantined_pending_messages_total: u64,
    /// Corrupt expiry-notification rows atomically isolated from delivery.
    pub quarantined_expired_notifications_total: u64,
    /// De-identified quarantine event rows removed by bounded retention.
    pub quarantine_events_removed_total: u64,
    /// Current durable de-identified quarantine event rows.
    pub quarantine_events_retained: u64,
    /// Unix timestamp of the most recent poison-row isolation.
    pub last_quarantine_at: Option<u64>,
    /// Unix timestamp of the most recent cleanup attempt.
    pub last_cleanup_at: Option<u64>,
    /// Stable state bucket: `succeeded` or `failed`.
    pub last_cleanup_status: Option<String>,
    /// Stable aggregate failure bucket from [`ChatRelayError::reason_bucket`].
    pub last_cleanup_failure_reason: Option<String>,
    /// Number of successfully committed transactions in the latest run.
    pub last_cleanup_batches: u64,
    /// Whether the latest run deferred remaining work to the next timer tick.
    pub last_cleanup_backlog_deferred: bool,
    /// Corrupt pending-message rows isolated by the latest cleanup run.
    pub last_cleanup_quarantined_pending_messages: u64,
}

// ============================================
// Pending message row (returned from pull)
// ============================================

/// A pending offline message retrieved from the store.
#[derive(Debug)]
pub struct PendingMessage {
    /// Opaque client-generated message identifier used for ACK pagination.
    pub message_id: [u8; 16],
    /// Signed end-to-end encrypted envelope; relay code must not inspect its ciphertext.
    pub envelope: ChatEnvelope,
}

/// One stable ChatPullV2 page and the opaque continuation state for it.
#[derive(Debug)]
pub struct PendingMessagePageV2 {
    /// Valid signed envelopes returned to the authenticated receiver.
    pub messages: Vec<PendingMessage>,
    /// AEAD-protected continuation cursor. An empty request cursor starts a new snapshot.
    pub next_cursor: Vec<u8>,
    /// Whether the caller should continue the current snapshot with `next_cursor`.
    pub has_more: bool,
}

// ============================================
// Expired notification row
// ============================================

/// A queued `ChatExpired` notification for an offline sender.
#[derive(Debug)]
pub struct ExpiredNotification {
    /// Local notification row identifier.
    pub id: i64,
    /// Original sender public key used only for authenticated delivery lookup.
    pub sender: [u8; 32],
    /// Original receiver public key returned inside the encrypted client flow.
    pub receiver: [u8; 32],
    /// bincode-serialised `Vec<[u8; 16]>`
    pub message_ids_raw: Vec<u8>,
}

/// Cross-process guard binding one signed custody anchor to the current
/// immutable maintenance checkpoint for the complete lifetime of the value.
///
/// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] The private `SQLite`
/// connection owns an exclusive maintenance transaction and is released by
/// RAII. Callers may inspect only the signed aggregate anchor; no lock path,
/// private audit state, HMAC, or custody metadata crosses this boundary.
pub struct ChatRelayCustodyAuditAnchorGuard {
    _filesystem_lock: Connection,
    anchor: CustodyAuditAnchorV1,
}

impl ChatRelayCustodyAuditAnchorGuard {
    /// Returns the exact current producer-signed anchor protected by the guard.
    #[must_use]
    pub const fn anchor(&self) -> &CustodyAuditAnchorV1 {
        &self.anchor
    }

    fn into_anchor(self) -> CustodyAuditAnchorV1 {
        self.anchor
    }
}

impl ExpiredNotification {
    /// Deserialise the stored message IDs.
    pub fn message_ids(&self) -> ChatRelayResult<Vec<[u8; 16]>> {
        if self.message_ids_raw.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_payload_size",
            });
        }
        let message_ids: Vec<[u8; 16]> = bincode::deserialize(&self.message_ids_raw)?;
        if message_ids.is_empty() || message_ids.len() > MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_message_count",
            });
        }
        Ok(message_ids)
    }
}

// ============================================
// Minimal LRU for online-path deduplication
// ============================================

/// Fixed-capacity LRU cache for `message_id` deduplication on the online path.
struct MessageDedup {
    map: DashMap<[u8; 16], u64>,
    capacity: usize,
    seq: AtomicU64,
}

impl MessageDedup {
    fn new(capacity: usize) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
            capacity,
            seq: AtomicU64::new(0),
        }
    }

    /// Returns `true` if the message_id was already seen (duplicate).
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        match self.map.entry(*message_id) {
            Entry::Occupied(_) => return true,
            Entry::Vacant(entry) => {
                entry.insert(seq);
            }
        }

        if self.map.len() > self.capacity {
            let oldest_key = self.map.iter().min_by_key(|e| *e.value()).map(|e| *e.key());
            if let Some(k) = oldest_key {
                self.map.remove(&k);
            }
        }
        false
    }
}

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
    /// Receiver/filter-bound opaque snapshot cursor capability.
    ///
    /// [CHAT-PULL-CURSOR-DOMAIN 2026-08-25 by Codex] The service owns paging;
    /// this composed object owns the stable wire and cryptographic mechanism.
    pull_cursor_codec: ChatPullCursorCodec,
    /// Bounded pending-message reads and durable-row authentication.
    ///
    /// [CHAT-PENDING-PULL-DOMAIN 2026-08-25 by Codex] The repository is
    /// replaceable; the service retains connection locking and side effects.
    pending_pull: PendingMessagePullDomain,
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
    /// Immutable retention policy and bounded durable cleanup capability.
    ///
    /// [CHAT-RELAY-CLEANUP-DOMAIN 2026-08-25 by Codex] The service retains
    /// connection locking, transaction commits, telemetry, and scheduling.
    cleanup: RelayCleanupDomain,
    /// Private verified-submit replay domain capability.
    ///
    /// [VERIFIED-SUBMIT-REPLAY-DOMAIN 2026-08-25 by Codex] Cache policy,
    /// lock striping, key derivation, and response protection are composed
    /// behind one domain object; SQLite transactions remain service-owned.
    verified_submit_replay: VerifiedSubmitReplay,
    /// Private blind-route replay identity and exact-response protector.
    ///
    /// [BLIND-ROUTE-REPLAY-DOMAIN 2026-08-25 by Codex] Durable ownership and
    /// SQLite transactions stay service-owned; cryptography is composed.
    blind_route_replay: BlindRouteReplay,
    /// Random process epoch fencing every restart-recoverable replay claim.
    ///
    /// [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Verified submit
    /// and blind-route reservations share ownership mechanics but retain
    /// independent HMAC namespaces and tables.
    replay_process_epoch: [u8; REPLAY_PROCESS_EPOCH_BYTES],
    dedup: MessageDedup,
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
    maintenance_status: RwLock<ChatRelayMaintenanceStatus>,
    /// Serializes backup publication, replay verification, and retention.
    backup_operations: Mutex<()>,
    /// In-memory wallet → session routing table.
    ///
    /// Arc so the cleanup task and each handler can hold independent references
    /// without borrowing the whole ChatRelayService.
    pub wallet_routes: Arc<WalletRouteCache>,
}

impl ChatRelayService {
    #[cfg(unix)]
    fn restrict_sqlite_file_permissions(path: &Path) -> ChatRelayResult<()> {
        use std::os::unix::fs::PermissionsExt;

        // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] SQLite creates WAL
        // sidecars using the primary database mode. Tighten the primary file
        // before enabling WAL, and keep the configured path out of the error.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|_| {
            ChatRelayError::Sqlite(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_PERM),
                Some("unable to restrict relay database permissions".to_string()),
            ))
        })
    }

    #[cfg(not(unix))]
    fn restrict_sqlite_file_permissions(_path: &Path) -> ChatRelayResult<()> {
        Ok(())
    }

    fn verify_sqlite_integrity(
        conn: &Connection,
        failure_field: &'static str,
    ) -> ChatRelayResult<()> {
        // [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] `quick_check(1)`
        // bounds returned findings while still traversing the database. Both a
        // non-`ok` finding and an SQLite error collapse to one path-free bucket.
        let outcome = conn
            .query_row("PRAGMA quick_check(1)", [], |row| row.get::<_, String>(0))
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: failure_field,
            })?;
        if outcome != "ok" {
            return Err(ChatRelayError::CorruptStoredData {
                field: failure_field,
            });
        }
        Ok(())
    }

    fn configure_sqlite_durability(conn: &Connection) -> ChatRelayResult<u8> {
        // [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] NORMAL protects
        // SQLite consistency across process failure but may lose a recently
        // acknowledged transaction after host power loss. The relay signs
        // custody receipts and persists its outage circuit from this database,
        // so activation requires FULL-or-stronger durability.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
        let synchronous_level =
            conn.query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))?;
        if synchronous_level < CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_synchronous_level",
            });
        }
        u8::try_from(synchronous_level).map_err(|_| ChatRelayError::CorruptStoredData {
            field: "sqlite_synchronous_level",
        })
    }

    fn backup_io_error(code: i32, message: &'static str) -> ChatRelayError {
        backup_io_error(code, message)
    }

    fn backup_audit_io() -> LocalBackupAuditIo<LocalBackupFilesystem> {
        // [CHAT-RELAY-BACKUP-AUDIT-IO-DOMAIN 2026-08-27 by Codex] Compose the
        // audit-artifact capability over the same private filesystem used by
        // the surrounding compatibility wrappers.
        LocalBackupAuditIo::new(LocalBackupFilesystem)
    }

    fn backup_audit_chain_verifier() -> LocalBackupAuditChainVerifier {
        // [CHAT-RELAY-BACKUP-AUDIT-CHAIN-DOMAIN 2026-08-27 by Codex] Compose
        // host I/O with pure verification and rotation policies at one edge.
        AuthenticatedBackupAuditChainVerifier::new(
            LocalBackupFilesystem,
            Self::backup_audit_io(),
            Self::backup_audit_verification_policy(),
            Self::backup_audit_rotation_policy(),
            BackupAuditChainLimits {
                max_record_bytes: CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES,
                max_segment_bytes: CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
            },
        )
    }

    fn reserve_private_backup_file(path: &Path) -> ChatRelayResult<()> {
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

    fn open_private_backup_control_file(path: &Path, append: bool) -> ChatRelayResult<File> {
        let mode = if append {
            PrivateBackupControlFileMode::Append
        } else {
            PrivateBackupControlFileMode::ReadWrite
        };
        LocalBackupFilesystem.open_control_file(path, mode)
    }

    fn open_existing_private_backup_control_file(path: &Path) -> ChatRelayResult<Option<File>> {
        LocalBackupFilesystem.open_existing_control_file(path)
    }

    fn acquire_backup_filesystem_lock(backup_directory: &Path) -> ChatRelayResult<Connection> {
        LocalBackupFilesystem.acquire_maintenance_lock(
            backup_directory,
            CHAT_RELAY_BACKUP_LOCK_FILE_NAME,
        )
    }

    fn backup_audit_segment_file_name(range: ChatRelayBackupAuditSegmentRange) -> String {
        Self::backup_audit_io().segment_file_name(range)
    }

    #[cfg(test)]
    fn backup_audit_checkpoint_file_name(range: ChatRelayBackupAuditSegmentRange) -> String {
        Self::backup_audit_io().checkpoint_file_name(range)
    }

    fn cleanup_backup_audit_checkpoint_temporaries(parent: &Path) -> ChatRelayResult<()> {
        Self::backup_audit_io().cleanup_checkpoint_temporaries(parent)
    }

    fn map_backup_audit_rotation_error(error: BackupAuditRotationError) -> ChatRelayError {
        match error {
            BackupAuditRotationError::EmptyActiveSegment => Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "empty relay backup maintenance audit segment cannot be rotated",
            ),
            BackupAuditRotationError::SegmentLimitReached => Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit segment limit reached",
            ),
            BackupAuditRotationError::SequenceOverflow => Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit sequence overflow",
            ),
            BackupAuditRotationError::InvalidSegmentRange => Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit segment range is invalid",
            ),
            BackupAuditRotationError::CheckpointIndexOverflow => Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit checkpoint index overflow",
            ),
            BackupAuditRotationError::RecordSizeOverflow => Self::backup_io_error(
                rusqlite::ffi::SQLITE_TOOBIG,
                "relay backup maintenance audit record size exceeds platform limits",
            ),
            BackupAuditRotationError::ByteCountOverflow => Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit byte count overflow",
            ),
        }
    }

    fn backup_audit_verification_policy() -> BoundedBackupAuditVerificationPolicy {
        BoundedBackupAuditVerificationPolicy::new(BackupAuditVerificationLimits {
            max_records_per_segment: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS)
                .unwrap_or(u64::MAX),
            max_bytes_per_segment: CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
            max_total_bytes: CHAT_RELAY_BACKUP_AUDIT_TOTAL_MAX_BYTES,
        })
    }

    fn backup_audit_rotation_policy() -> BoundedBackupAuditRotationPolicy {
        BoundedBackupAuditRotationPolicy::new(BackupAuditRotationLimits {
            max_records_per_segment: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS)
                .unwrap_or(u64::MAX),
            max_bytes_per_segment: CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES,
            max_segments: u64::try_from(CHAT_RELAY_BACKUP_AUDIT_MAX_SEGMENTS)
                .unwrap_or(u64::MAX),
        })
    }

    fn backup_audit_anchor_digest(
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<[u8; 32]> {
        let receipt = state.receipt();
        if receipt.checkpoint_count == 0
            || receipt.archived_record_count == 0
            || receipt.archived_bytes == 0
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_NOTFOUND,
                "relay backup maintenance audit has no immutable checkpoint to anchor",
            ));
        }
        let checkpoint_mac = hex::decode(state.checkpoint_head_mac()).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint anchor is invalid",
            )
        })?;
        let checkpoint_mac: [u8; 32] = checkpoint_mac.try_into().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "relay backup maintenance audit checkpoint anchor is invalid",
            )
        })?;
        // [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] The private checkpoint
        // MAC authenticates the full cumulative chain. Hash it into a separate
        // public domain so exporting an anchor cannot turn the HMAC itself into
        // a reusable capability or reveal the private signing frame.
        let mut hasher = Sha256::new();
        hasher.update(CHAT_RELAY_BACKUP_AUDIT_ANCHOR_DIGEST_DOMAIN);
        hasher.update(receipt.checkpoint_count.to_le_bytes());
        hasher.update(receipt.archived_record_count.to_le_bytes());
        hasher.update(receipt.archived_bytes.to_le_bytes());
        hasher.update(checkpoint_mac);
        Ok(hasher.finalize().into())
    }

    fn hash_backup_audit_segment(file: &mut File) -> ChatRelayResult<(u64, String)> {
        Self::backup_audit_io().hash_segment(file)
    }

    #[cfg(test)]
    fn verify_backup_audit_log(
        file: &mut File,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationState> {
        Self::backup_audit_chain_verifier().verify_log(file, node_secret)
    }

    fn verify_backup_audit_chain(
        parent: &Path,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditChainVerification> {
        Self::backup_audit_chain_verifier().verify_chain(parent, node_secret)
    }

    fn complete_pending_backup_audit_rotation(
        parent: &Path,
        pending: ChatRelayBackupAuditPendingRotation,
    ) -> ChatRelayResult<()> {
        Self::backup_audit_io().complete_pending_rotation(parent, pending)
    }

    fn publish_backup_audit_checkpoint(
        parent: &Path,
        range: ChatRelayBackupAuditSegmentRange,
        checkpoint: &ChatRelayBackupAuditCheckpoint,
    ) -> ChatRelayResult<()> {
        Self::backup_audit_io().publish_checkpoint(parent, range, checkpoint)
    }

    fn rotate_backup_audit_segment(
        parent: &Path,
        node_secret: &[u8; 32],
        state: &ChatRelayBackupAuditVerificationState,
    ) -> ChatRelayResult<()> {
        let rotation_policy = Self::backup_audit_rotation_policy();
        let receipt = state.receipt();
        let rotation_state = BackupAuditRotationState {
            active_record_count: receipt.active_record_count,
            archived_record_count: receipt.archived_record_count,
            record_count: receipt.record_count,
            checkpoint_count: receipt.checkpoint_count,
        };
        rotation_policy
            .validate_admission(rotation_state)
            .map_err(Self::map_backup_audit_rotation_error)?;
        let active_path = parent.join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
        let Some(mut active) = Self::open_existing_private_backup_control_file(&active_path)?
        else {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "active relay backup maintenance audit segment is missing",
            ));
        };
        active.sync_all().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_FSYNC,
                "unable to sync active relay backup maintenance audit segment",
            )
        })?;
        let rotation_plan = rotation_policy
            .plan_rotation(rotation_state)
            .map_err(Self::map_backup_audit_rotation_error)?;
        let range = rotation_plan.range;
        let (segment_bytes, segment_sha256) = Self::hash_backup_audit_segment(&mut active)?;
        let checkpoint = HmacBackupAuditCheckpointAuthenticator
            .build(
                node_secret,
                BackupAuditCheckpointState {
                    checkpoint_index: rotation_plan.checkpoint_index,
                    segment_first_sequence: range.first_sequence,
                    segment_last_sequence: range.last_sequence,
                    segment_bytes,
                    segment_sha256,
                    cumulative_verified_bytes: receipt.verified_bytes,
                    cumulative_last_recorded_at: receipt.last_recorded_at,
                    cumulative_dry_run_count: receipt.dry_run_count,
                    cumulative_planned_count: receipt.planned_count,
                    cumulative_completed_count: receipt.completed_count,
                    cumulative_failed_count: receipt.failed_count,
                    head_mac: state.head_mac().to_string(),
                    previous_checkpoint_mac: state.checkpoint_head_mac().to_string(),
                },
            )
            .map_err(map_backup_audit_checkpoint_error)?;
        drop(active);
        Self::publish_backup_audit_checkpoint(parent, range, &checkpoint)?;
        Self::complete_pending_backup_audit_rotation(
            parent,
            ChatRelayBackupAuditPendingRotation::PublishSegment {
                active_path,
                segment_path: parent.join(Self::backup_audit_segment_file_name(range)),
            },
        )
    }

    fn backup_audit_segment_needs_rotation(
        active_record_count: u64,
        active_bytes: u64,
        next_record_bytes: usize,
    ) -> ChatRelayResult<bool> {
        Self::backup_audit_rotation_policy()
            .should_rotate(active_record_count, active_bytes, next_record_bytes)
            .map_err(Self::map_backup_audit_rotation_error)
    }

    fn append_backup_maintenance_audit(
        backup_directory: &Path,
        node_secret: &[u8; 32],
        phase: BackupAuditPhase,
        timestamp: u64,
        counts: ChatRelayBackupMaintenanceAuditCounts,
    ) -> ChatRelayResult<()> {
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        let audit_path = parent.join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
        Self::cleanup_backup_audit_checkpoint_temporaries(parent)?;
        let mut chain = Self::verify_backup_audit_chain(parent, node_secret)?;
        let verification_policy = Self::backup_audit_verification_policy();
        if let Some(pending) = chain.pending_rotation.take() {
            // [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] A checkpoint is
            // published before its segment name. Completing that publication
            // under the cross-process maintenance lock makes power-loss
            // recovery deterministic without rewriting authenticated bytes.
            Self::complete_pending_backup_audit_rotation(parent, pending)?;
            verification_policy.mark_rotation_recovered(&mut chain.state);
        }
        let verification = chain.state;
        let next_sequence = verification
            .next_sequence()
            .map_err(map_backup_audit_verification_error)?;
        let record = HmacBackupAuditRecordAuthenticator
            .build(
                node_secret,
                next_sequence,
                verification.head_mac().to_string(),
                phase,
                timestamp,
                counts,
            )
            .map_err(map_backup_audit_record_error)?;
        let mut encoded = serde_json::to_vec(&record).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay backup maintenance audit",
            )
        })?;
        encoded.push(b'\n');
        if encoded.len() > CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit capacity exhausted",
            ));
        }
        if verification
            .receipt()
            .verified_bytes
            .checked_add(encoded.len() as u64)
            .map_or(true, |bytes| {
                bytes > CHAT_RELAY_BACKUP_AUDIT_TOTAL_MAX_BYTES
            })
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_FULL,
                "relay backup maintenance audit chain capacity exhausted",
            ));
        }
        let mut file = Self::open_private_backup_control_file(&audit_path, true)?;
        let current_len = file
            .metadata()
            .map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect active relay backup maintenance audit",
                )
            })?
            .len();
        if Self::backup_audit_segment_needs_rotation(
            verification.receipt().active_record_count,
            current_len,
            encoded.len(),
        )? {
            drop(file);
            Self::rotate_backup_audit_segment(parent, node_secret, &verification)?;
            file = Self::open_private_backup_control_file(&audit_path, true)?;
        }
        file.write_all(&encoded).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_WRITE,
                "unable to append relay backup maintenance audit",
            )
        })?;
        file.sync_all().map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR_FSYNC,
                "unable to durably sync relay backup maintenance audit",
            )
        })
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

    fn inspect_active_restore_boundary(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayActiveRestoreBoundary> {
        inspect_active_restore_boundary(&config.db_path)
    }

    fn verified_restore_backup_count(
        inspection: &ChatRelayBackupRetentionInspection,
    ) -> ChatRelayResult<usize> {
        verified_restore_backup_count(inspection)
    }

    // [CHAT-RELAY-RESTORE-PLAN-DOMAIN 2026-08-26 by Codex] The service owns
    // private filesystem identity collection; the composed authenticator owns
    // only canonical policy and cryptography.
    fn restore_plan_private_boundary<'a>(
        config: &'a ChatRelayConfig,
        backup: &'a BackupArtifactSnapshot,
        active: &ChatRelayActiveRestoreBoundary,
    ) -> RestorePlanPrivateBoundary<'a> {
        RestorePlanPrivateBoundary {
            configured_database_path: &config.db_path,
            selected_backup_name: backup.file_name(),
            selected_backup_modified_at: backup.modified_at(),
            active_database_modified_at: active.modified_at,
            selected_backup_device_id: backup.device_id(),
            selected_backup_inode: backup.inode(),
            active_database_device_id: active.device_id,
            active_database_inode: active.inode,
        }
    }

    fn map_restore_plan_error(error: RestorePlanError) -> ChatRelayError {
        match error {
            RestorePlanError::ExpiryOutOfRange => Self::backup_io_error(
                rusqlite::ffi::SQLITE_RANGE,
                "relay restore-plan expiry is out of range",
            ),
            RestorePlanError::FilesystemTimeOutOfRange => Self::backup_io_error(
                rusqlite::ffi::SQLITE_RANGE,
                "relay restore-plan filesystem time is out of range",
            ),
            RestorePlanError::EncodingFailed => Self::backup_io_error(
                rusqlite::ffi::SQLITE_FORMAT,
                "unable to encode relay restore plan",
            ),
            RestorePlanError::AuthenticatorInitFailed => Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to initialize relay restore plan",
            ),
            RestorePlanError::InvalidOrStale => Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "relay restore plan is invalid, expired, or stale",
            ),
        }
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

    fn verify_existing_backup_artifact(path: &Path) -> ChatRelayResult<u64> {
        let metadata = std::fs::symlink_metadata(path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to inspect existing relay backup artifact",
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "existing relay backup artifact is not a private regular file",
            ));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_PERM,
                    "existing relay backup artifact is not owner-private",
                ));
            }
        }

        for suffix in ["-journal", "-wal", "-shm"] {
            let mut sidecar = path.as_os_str().to_os_string();
            sidecar.push(suffix);
            match std::fs::symlink_metadata(PathBuf::from(sidecar)) {
                Ok(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CORRUPT,
                        "existing relay backup artifact has mutable sidecar state",
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CANTOPEN,
                        "unable to inspect existing relay backup sidecar state",
                    ));
                }
            }
        }

        // Resolve only the already-verified private directory. macOS temp paths
        // commonly contain a system-managed ancestor symlink (`/var`), while
        // SQLite NOFOLLOW rejects any symlink component. Keeping the filename
        // separate preserves the final-component race defense.
        let parent = path.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "existing relay backup artifact has no private parent",
            )
        })?;
        let file_name = path.file_name().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "existing relay backup artifact has no private name",
            )
        })?;
        let canonical_parent = std::fs::canonicalize(parent).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to resolve private relay backup directory",
            )
        })?;
        let nofollow_path = canonical_parent.join(file_name);

        // NOFOLLOW closes the final-component symlink race between metadata
        // inspection and SQLite open. Read-only verification must never create
        // journal/WAL sidecars next to an immutable recovery image.
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let backup_conn = Connection::open_with_flags(&nofollow_path, flags).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "unable to open existing relay backup artifact",
            )
        })?;
        Self::verify_sqlite_backup(&backup_conn)?;
        drop(backup_conn);

        let verified_metadata = std::fs::symlink_metadata(&nofollow_path).map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_IOERR,
                "verified relay backup artifact became unavailable",
            )
        })?;
        if verified_metadata.file_type().is_symlink()
            || !verified_metadata.is_file()
            || verified_metadata.len() != metadata.len()
        {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_CORRUPT,
                "verified relay backup artifact changed during inspection",
            ));
        }

        // In a concurrent hard-link publication, this verifier may run before
        // the creator reaches its directory fsync. Syncing here makes either
        // successful receipt sufficient to durably publish the shared name.
        Self::sync_backup_parent(&canonical_parent)?;
        Ok(verified_metadata.len())
    }

    fn create_verified_backup_artifact(
        &self,
        backup_directory: &Path,
        destination: &Path,
        reuse_existing: bool,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        match std::fs::symlink_metadata(destination) {
            Ok(_) if reuse_existing => {
                return Ok(ChatRelayBackupReceipt {
                    size_bytes: Self::verify_existing_backup_artifact(destination)?,
                    created: false,
                });
            }
            Ok(_) => {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CONSTRAINT,
                    "relay backup destination already exists",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to inspect relay backup destination",
                ));
            }
        }

        let temporary_nonce = rand::random::<u64>();
        let temporary_name = Self::backup_artifact_namespace()
            .temporary_recovery_image_name(now_secs(), temporary_nonce);
        let temporary = backup_directory.join(temporary_name.as_str());
        Self::reserve_private_backup_file(&temporary)?;

        let mut destination_published = false;
        let outcome = (|| {
            let mut backup_conn = Connection::open(&temporary).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CANTOPEN,
                    "unable to open private relay backup file",
                )
            })?;
            Self::restrict_sqlite_file_permissions(&temporary)?;
            {
                let source = self.conn.lock();
                Self::copy_sqlite_backup(&source, &mut backup_conn)?;
            }
            Self::normalize_sqlite_backup_journal(&backup_conn)?;
            Self::verify_sqlite_backup(&backup_conn)?;
            drop(backup_conn);

            Self::restrict_sqlite_file_permissions(&temporary)?;
            std::fs::File::open(&temporary)
                .and_then(|file| file.sync_all())
                .map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_IOERR,
                        "unable to synchronize relay backup file",
                    )
                })?;

            match std::fs::hard_link(&temporary, destination) {
                Ok(()) => destination_published = true,
                Err(error)
                    if reuse_existing && error.kind() == std::io::ErrorKind::AlreadyExists =>
                {
                    Self::remove_sqlite_artifact(&temporary);
                    return Ok(ChatRelayBackupReceipt {
                        size_bytes: Self::verify_existing_backup_artifact(destination)?,
                        created: false,
                    });
                }
                Err(_) => {
                    return Err(Self::backup_io_error(
                        rusqlite::ffi::SQLITE_CONSTRAINT,
                        "unable to publish relay backup without replacement",
                    ));
                }
            }

            Self::restrict_sqlite_file_permissions(destination)?;
            std::fs::remove_file(&temporary).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to finalize relay backup publication",
                )
            })?;
            Self::sync_backup_parent(backup_directory)?;
            let metadata = std::fs::symlink_metadata(destination).map_err(|_| {
                Self::backup_io_error(
                    rusqlite::ffi::SQLITE_IOERR,
                    "unable to inspect published relay backup artifact",
                )
            })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(Self::backup_io_error(
                    rusqlite::ffi::SQLITE_CORRUPT,
                    "published relay backup artifact is invalid",
                ));
            }
            Ok(ChatRelayBackupReceipt {
                size_bytes: metadata.len(),
                created: true,
            })
        })();

        Self::remove_sqlite_artifact(&temporary);
        if outcome.is_err() && destination_published {
            // Never remove a destination created by another replay. Ownership
            // begins only after this invocation's hard-link publication.
            Self::remove_sqlite_artifact(destination);
        }
        outcome
    }

    fn remove_sqlite_artifact(path: &Path) {
        let _ = std::fs::remove_file(path);
        for suffix in ["-journal", "-wal", "-shm"] {
            let mut sidecar = path.as_os_str().to_os_string();
            sidecar.push(suffix);
            let _ = std::fs::remove_file(PathBuf::from(sidecar));
        }
    }

    fn sync_backup_parent(parent: &Path) -> ChatRelayResult<()> {
        LocalBackupFilesystem.sync_backup_parent(parent)
    }

    fn backup_copy_retry_policy() -> BoundedBackupCopyRetryPolicy {
        BoundedBackupCopyRetryPolicy::new(
            CHAT_RELAY_BACKUP_BUSY_TIMEOUT,
            CHAT_RELAY_BACKUP_BUSY_RETRY_DELAY,
        )
    }

    fn backup_copy_progress(step: StepResult) -> BackupCopyProgress {
        match step {
            StepResult::Done => BackupCopyProgress::Complete,
            StepResult::More => BackupCopyProgress::More,
            StepResult::Busy => BackupCopyProgress::Busy,
            StepResult::Locked => BackupCopyProgress::Locked,
            _ => BackupCopyProgress::Unsupported,
        }
    }

    fn map_backup_copy_policy_error(error: BackupCopyPolicyError) -> ChatRelayError {
        match error {
            BackupCopyPolicyError::BusyTimeout => Self::backup_io_error(
                rusqlite::ffi::SQLITE_BUSY,
                "relay backup remained busy",
            ),
            BackupCopyPolicyError::ObservationTimeRegressed => Self::backup_io_error(
                rusqlite::ffi::SQLITE_ABORT,
                "relay backup retry observation time regressed",
            ),
            BackupCopyPolicyError::UnsupportedProgress => Self::backup_io_error(
                rusqlite::ffi::SQLITE_ERROR,
                "unsupported relay backup step result",
            ),
        }
    }

    fn copy_sqlite_backup(
        source: &Connection,
        destination: &mut Connection,
    ) -> ChatRelayResult<()> {
        let backup = Backup::new(source, destination)?;
        let retry_policy = Self::backup_copy_retry_policy();
        let mut retry_state = BackupCopyRetryState::default();
        loop {
            let progress =
                Self::backup_copy_progress(backup.step(CHAT_RELAY_BACKUP_PAGES_PER_STEP)?);
            let action = retry_policy
                .transition(&mut retry_state, progress, Instant::now())
                .map_err(Self::map_backup_copy_policy_error)?;
            match action {
                BackupCopyAction::Complete => return Ok(()),
                BackupCopyAction::Continue => {}
                BackupCopyAction::RetryAfter(delay) => std::thread::sleep(delay),
            }
        }
    }

    fn verify_sqlite_backup_logical_integrity(conn: &Connection) -> ChatRelayResult<()> {
        let installed_version = conn
            .query_row(
                "SELECT schema_version
                 FROM relay_schema_features
                 WHERE feature = ?1",
                params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if installed_version != Some(DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_schema_sentinel",
            });
        }

        let verified_submit_version = conn
            .query_row(
                "SELECT schema_version
                 FROM relay_schema_features
                 WHERE feature = ?1",
                params![VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if verified_submit_version != Some(VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_verified_submit_schema_sentinel",
            });
        }
        let invalid_verified_submit_rows = conn.query_row(
            "SELECT COUNT(*) FROM relay_verified_submit_responses
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(envelope_fingerprint) != 32
                OR LENGTH(response_nonce) != 24
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > 528
                OR completed_at < 0",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_verified_submit_rows != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_verified_submit_rows",
            });
        }
        let invalid_verified_submit_reservations = conn.query_row(
            "SELECT COUNT(*) FROM relay_verified_submit_reservations
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(envelope_fingerprint) != 32
                OR reserved_at < 0
                OR owner_epoch IS NULL
                OR TYPEOF(owner_epoch) != 'blob'
                OR LENGTH(owner_epoch) != 16
                OR TYPEOF(owner_acquired_at) != 'integer'
                OR owner_acquired_at < reserved_at",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_verified_submit_reservations != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_verified_submit_reservations",
            });
        }

        // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] A recovery image
        // must carry the same route side-effect boundary as the live database.
        // Validate only fixed shape here; AEAD authenticity remains bound to
        // the node secret and is checked when an exact response is recovered.
        let blind_route_version = conn
            .query_row(
                "SELECT schema_version
                 FROM relay_schema_features
                 WHERE feature = ?1",
                params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if blind_route_version != Some(BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_blind_route_schema_sentinel",
            });
        }
        let invalid_blind_route_responses = conn.query_row(
            "SELECT COUNT(*) FROM relay_blind_route_responses
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(request_fingerprint) != 32
                OR LENGTH(response_nonce) != 24
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > 2064
                OR completed_at < 0",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        let invalid_blind_route_reservations = conn.query_row(
            "SELECT COUNT(*) FROM relay_blind_route_reservations
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(request_fingerprint) != 32
                OR reserved_at < 0
                OR owner_epoch IS NULL
                OR TYPEOF(owner_epoch) != 'blob'
                OR LENGTH(owner_epoch) != 16
                OR TYPEOF(owner_acquired_at) != 'integer'
                OR owner_acquired_at < reserved_at
                OR (effect_started_at IS NOT NULL
                    AND (TYPEOF(effect_started_at) != 'integer'
                         OR effect_started_at < reserved_at))",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_blind_route_responses != 0 || invalid_blind_route_reservations != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_blind_route_rows",
            });
        }

        DirectPeerCircuitDomain::<SqliteDirectPeerCircuitRepository>::validate_checkpoint(
            conn,
            now_secs(),
        )?;

        let stored_usage = Self::read_storage_usage(conn)?;
        let canonical_usage = Self::read_canonical_storage_usage(conn)?;
        if stored_usage != canonical_usage {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_storage_usage",
            });
        }

        let (last_sequence, max_sequence, missing_sequences) = conn.query_row(
            "SELECT
                (SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1),
                (SELECT COALESCE(MAX(queue_sequence), 0) FROM pending_messages),
                (SELECT COUNT(*) FROM pending_messages WHERE queue_sequence IS NULL)",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )?;
        if last_sequence < 0
            || max_sequence < 0
            || missing_sequences != 0
            || last_sequence < max_sequence
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_queue_sequence",
            });
        }
        Ok(())
    }

    fn verify_sqlite_backup(conn: &Connection) -> ChatRelayResult<()> {
        Self::verify_sqlite_integrity(conn, "sqlite_backup_integrity")?;
        Self::verify_sqlite_backup_logical_integrity(conn).map_err(|_| {
            ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_logical_integrity",
            }
        })
    }

    fn normalize_sqlite_backup_journal(conn: &Connection) -> ChatRelayResult<()> {
        // [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] SQLite may copy
        // the source WAL mode into the isolated image. A later read-only open
        // can then create `-wal`/`-shm`, violating immutable artifact replay.
        // Normalize only the offline copy; restored service startup explicitly
        // re-enables WAL + FULL before accepting custody.
        let journal_mode = conn
            .query_row("PRAGMA journal_mode=DELETE", [], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_journal_mode",
            })?;
        if !journal_mode.eq_ignore_ascii_case("delete") {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_journal_mode",
            });
        }
        Ok(())
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
            Self::restrict_sqlite_file_permissions(Path::new(&config.db_path))?;
        }
        // A short bounded wait absorbs transient locks from an operator backup
        // or diagnostic reader without allowing relay requests to hang forever.
        conn.busy_timeout(Duration::from_secs(5))?;
        Self::verify_sqlite_integrity(&conn, "sqlite_startup_integrity")?;
        let synchronous_level = Self::configure_sqlite_durability(&conn)?;

        let dedup_capacity = config.dedup_lru_capacity;
        let relay_enabled = config.enabled;
        let pull_cursor_codec = ChatPullCursorCodec::new(&node_secret)?;
        let pending_pull = PendingMessagePullDomain::new();
        let pending_custody = PendingMessageCustodyDomain::new(&config);
        let expired_notification_delivery = ExpiredNotificationDelivery::new();
        let blob_custody = EncryptedBlobCustodyDomain::new(node_secret, &config);
        let durable_quarantine = DurableQuarantineDomain::new(&config);
        let cleanup = RelayCleanupDomain::new(
            &config,
            VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
            BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
        );
        let verified_submit_replay = VerifiedSubmitReplay::new(node_secret, dedup_capacity)?;
        let blind_route_replay = BlindRouteReplay::new(node_secret)?;
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
            pull_cursor_codec,
            pending_pull,
            pending_custody,
            expired_notification_delivery,
            blob_custody,
            durable_quarantine,
            cleanup,
            verified_submit_replay,
            blind_route_replay,
            replay_process_epoch,
            dedup: MessageDedup::new(dedup_capacity),
            peer_telemetry: PeerRelayTelemetryDomain::new(peer_status),
            direct_peer_relay_circuit: DirectPeerCircuitDomain::default(),
            maintenance_status: RwLock::new(ChatRelayMaintenanceStatus::default()),
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
        Self::init_pending_message_schema(&mut conn)?;
        Self::init_blob_and_notification_schema(&conn)?;
        self.durable_quarantine.init_schema(&conn)?;
        Self::init_usage_schema(&conn)?;
        self.direct_peer_relay_circuit
            .init_schema(&mut conn, now_secs())?;
        Self::init_verified_submit_response_schema(&mut conn, now_secs())?;
        Self::init_blind_relay_route_replay_schema(&mut conn, now_secs())?;
        Self::reconcile_storage_usage(&conn)?;
        let retained_quarantine_events = self.durable_quarantine.retained_count(&conn)?;
        drop(conn);
        self.maintenance_status.write().quarantine_events_retained =
            u64::try_from(retained_quarantine_events).unwrap_or(u64::MAX);
        Ok(())
    }

    fn init_verified_submit_response_schema(
        conn: &mut Connection,
        now: u64,
    ) -> ChatRelayResult<()> {
        // [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] Completed
        // responses and unfinished reservations contain only node-secret HMACs,
        // sealed bytes, retention timestamps, and random process ownership.
        // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Schema v3
        // fences abandoned reservations so a replacement process can recover
        // entry custody without repeating an uncertain onion side effect.
        let response_table_existed = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'relay_verified_submit_responses'
             )",
            [],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let reservation_table_existed = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'relay_verified_submit_reservations'
             )",
            [],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_verified_submit_responses (
                cache_key            BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                envelope_fingerprint BLOB    NOT NULL CHECK(LENGTH(envelope_fingerprint) = 32),
                response_nonce       BLOB    NOT NULL CHECK(LENGTH(response_nonce) = 24),
                response_ciphertext  BLOB    NOT NULL CHECK(
                    LENGTH(response_ciphertext) > 16
                    AND LENGTH(response_ciphertext) <= 528
                ),
                completed_at         INTEGER NOT NULL CHECK(completed_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_verified_submit_response_retention
                ON relay_verified_submit_responses(completed_at);

            CREATE TABLE IF NOT EXISTS relay_verified_submit_reservations (
                cache_key            BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                envelope_fingerprint BLOB    NOT NULL CHECK(LENGTH(envelope_fingerprint) = 32),
                reserved_at          INTEGER NOT NULL CHECK(reserved_at >= 0),
                owner_epoch         BLOB    NOT NULL CHECK(LENGTH(owner_epoch) = 16),
                owner_acquired_at   INTEGER NOT NULL CHECK(owner_acquired_at >= reserved_at)
            );
            CREATE INDEX IF NOT EXISTS idx_verified_submit_reservation_retention
                ON relay_verified_submit_reservations(reserved_at);
            ",
        )?;
        let installed_version = tx
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if installed_version.is_some_and(|version| {
            version != VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION
                && version != VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION
                && version != VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION
        }) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_installation_version",
            });
        }
        if !response_table_existed && installed_version.is_some() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_table",
            });
        }
        if matches!(
            installed_version,
            Some(
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION
                    | VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION
            )
        ) && !reservation_table_existed
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_table",
            });
        }
        let owner_epoch_exists =
            Self::verified_submit_reservation_column_exists(&tx, "owner_epoch")?;
        let owner_acquired_at_exists =
            Self::verified_submit_reservation_column_exists(&tx, "owner_acquired_at")?;
        if installed_version == Some(VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION)
            && (!owner_epoch_exists || !owner_acquired_at_exists)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_columns",
            });
        }
        if !owner_epoch_exists {
            tx.execute_batch(
                "ALTER TABLE relay_verified_submit_reservations
                 ADD COLUMN owner_epoch BLOB",
            )?;
        }
        if !owner_acquired_at_exists {
            tx.execute_batch(
                "ALTER TABLE relay_verified_submit_reservations
                 ADD COLUMN owner_acquired_at INTEGER",
            )?;
        }
        if installed_version != Some(VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION) {
            // A pre-v3 owner is deliberately foreign to this process. Its
            // immutable reservation age becomes the first takeover lease.
            tx.execute(
                "UPDATE relay_verified_submit_reservations
                 SET owner_epoch = zeroblob(?1), owner_acquired_at = reserved_at",
                params![i64::try_from(REPLAY_PROCESS_EPOCH_BYTES).unwrap_or(i64::MAX)],
            )?;
        }
        let invalid_rows = tx.query_row(
            "SELECT COUNT(*) FROM relay_verified_submit_responses
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(envelope_fingerprint) != 32
                OR LENGTH(response_nonce) != 24
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > 528
                OR completed_at < 0",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_rows != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_row_shape",
            });
        }
        let invalid_reservations = tx.query_row(
            "SELECT COUNT(*) FROM relay_verified_submit_reservations
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(envelope_fingerprint) != 32
                OR reserved_at < 0
                OR owner_epoch IS NULL
                OR TYPEOF(owner_epoch) != 'blob'
                OR LENGTH(owner_epoch) != 16
                OR TYPEOF(owner_acquired_at) != 'integer'
                OR owner_acquired_at < reserved_at",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_reservations != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_row_shape",
            });
        }
        if installed_version.is_none()
            && tx.execute(
                "INSERT INTO relay_schema_features (feature, schema_version, installed_at)
                 VALUES (?1, ?2, ?3)",
                params![
                    VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
                    VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
                    sqlite_integer(now, "verified_submit_response_schema_installed_at")?
                ],
            )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_installation_marker",
            });
        }
        if matches!(
            installed_version,
            Some(
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION
                    | VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION
            )
        ) && tx.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
                 WHERE feature = ?2 AND schema_version IN (?3, ?4)",
            params![
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_migration_marker",
            });
        }

        let cutoff = sqlite_integer(
            now.saturating_sub(VERIFIED_SUBMIT_RESPONSE_TTL_SECS),
            "verified_submit_response_startup_cutoff",
        )?;
        tx.execute(
            "DELETE FROM relay_verified_submit_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_verified_submit_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn verified_submit_reservation_column_exists(
        tx: &Transaction<'_>,
        expected_column: &str,
    ) -> ChatRelayResult<bool> {
        let mut stmt = tx.prepare("PRAGMA table_info(relay_verified_submit_reservations)")?;
        let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for column in columns {
            if column? == expected_column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn init_blind_relay_route_replay_schema(
        conn: &mut Connection,
        now: u64,
    ) -> ChatRelayResult<()> {
        // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] These tables carry
        // only node-secret HMACs, sealed ACK bytes, and retention timestamps.
        // An installed schema marker plus missing table is corruption, not a
        // first-run migration, because silently recreating it would erase the
        // route side-effect boundary after an operator accident.
        let response_table_existed = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'relay_blind_route_responses'
             )",
            [],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let reservation_table_existed = conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'relay_blind_route_reservations'
             )",
            [],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_blind_route_responses (
                cache_key           BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                request_fingerprint BLOB    NOT NULL CHECK(LENGTH(request_fingerprint) = 32),
                response_nonce      BLOB    NOT NULL CHECK(LENGTH(response_nonce) = 24),
                response_ciphertext BLOB    NOT NULL CHECK(
                    LENGTH(response_ciphertext) > 16
                    AND LENGTH(response_ciphertext) <= 2064
                ),
                completed_at        INTEGER NOT NULL CHECK(completed_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_blind_route_response_retention
                ON relay_blind_route_responses(completed_at);

            CREATE TABLE IF NOT EXISTS relay_blind_route_reservations (
                cache_key           BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                request_fingerprint BLOB    NOT NULL CHECK(LENGTH(request_fingerprint) = 32),
                reserved_at         INTEGER NOT NULL CHECK(reserved_at >= 0),
                owner_epoch         BLOB    NOT NULL CHECK(LENGTH(owner_epoch) = 16),
                owner_acquired_at   INTEGER NOT NULL CHECK(owner_acquired_at >= reserved_at),
                effect_started_at   INTEGER CHECK(
                    effect_started_at IS NULL OR effect_started_at >= reserved_at
                )
            );
            CREATE INDEX IF NOT EXISTS idx_blind_route_reservation_retention
                ON relay_blind_route_reservations(reserved_at);
            ",
        )?;
        let installed_version = tx
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if installed_version.is_some_and(|version| {
            version != BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION
                && version != BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION
                && version != BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION
        }) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_installation_version",
            });
        }
        if installed_version.is_some() && (!response_table_existed || !reservation_table_existed) {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_table",
            });
        }
        let owner_epoch_exists = Self::blind_relay_reservation_column_exists(&tx, "owner_epoch")?;
        let owner_acquired_at_exists =
            Self::blind_relay_reservation_column_exists(&tx, "owner_acquired_at")?;
        let effect_started_at_exists =
            Self::blind_relay_reservation_column_exists(&tx, "effect_started_at")?;
        if installed_version == Some(BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION)
            && (!owner_epoch_exists || !owner_acquired_at_exists || !effect_started_at_exists)
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_reservation_columns",
            });
        }
        if !owner_epoch_exists {
            tx.execute_batch(
                "ALTER TABLE relay_blind_route_reservations
                 ADD COLUMN owner_epoch BLOB",
            )?;
        }
        if !effect_started_at_exists {
            tx.execute_batch(
                "ALTER TABLE relay_blind_route_reservations
                 ADD COLUMN effect_started_at INTEGER",
            )?;
        }
        if !owner_acquired_at_exists {
            tx.execute_batch(
                "ALTER TABLE relay_blind_route_reservations
                 ADD COLUMN owner_acquired_at INTEGER",
            )?;
        }
        if installed_version == Some(BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION) {
            // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] A v1 row may
            // have crossed an external side-effect boundary before upgrade.
            // Mark every legacy claim armed; only v2 claims created with an
            // explicit process epoch may participate in safe takeover.
            tx.execute(
                "UPDATE relay_blind_route_reservations
                 SET owner_epoch = zeroblob(?1), effect_started_at = reserved_at",
                params![i64::try_from(REPLAY_PROCESS_EPOCH_BYTES).unwrap_or(i64::MAX)],
            )?;
        }
        if installed_version != Some(BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION) {
            // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Evidence age is
            // immutable. Process ownership gets a separate lease timestamp so
            // recovery cannot extend replay retention by taking over a claim.
            tx.execute(
                "UPDATE relay_blind_route_reservations
                 SET owner_acquired_at = reserved_at",
                [],
            )?;
        }
        let invalid_responses = tx.query_row(
            "SELECT COUNT(*) FROM relay_blind_route_responses
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(request_fingerprint) != 32
                OR LENGTH(response_nonce) != 24
                OR LENGTH(response_ciphertext) <= 16
                OR LENGTH(response_ciphertext) > 2064
                OR completed_at < 0",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        let invalid_reservations = tx.query_row(
            "SELECT COUNT(*) FROM relay_blind_route_reservations
             WHERE LENGTH(cache_key) != 32
                OR LENGTH(request_fingerprint) != 32
                OR reserved_at < 0
                OR owner_epoch IS NULL
                OR TYPEOF(owner_epoch) != 'blob'
                OR LENGTH(owner_epoch) != 16
                OR TYPEOF(owner_acquired_at) != 'integer'
                OR owner_acquired_at < reserved_at
                OR (effect_started_at IS NOT NULL
                    AND (TYPEOF(effect_started_at) != 'integer'
                         OR effect_started_at < reserved_at))",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if invalid_responses != 0 || invalid_reservations != 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_row_shape",
            });
        }
        if installed_version.is_none()
            && tx.execute(
                "INSERT INTO relay_schema_features (feature, schema_version, installed_at)
                 VALUES (?1, ?2, ?3)",
                params![
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
                    sqlite_integer(now, "blind_relay_route_replay_schema_installed_at")?
                ],
            )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_installation_marker",
            });
        }
        if matches!(
            installed_version,
            Some(BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION)
                | Some(BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION)
        ) && tx.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
                 WHERE feature = ?2 AND schema_version IN (?3, ?4)",
            params![
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_replay_migration_marker",
            });
        }
        let cutoff = sqlite_integer(
            now.saturating_sub(BLIND_RELAY_ROUTE_REPLAY_TTL_SECS),
            "blind_relay_route_replay_startup_cutoff",
        )?;
        tx.execute(
            "DELETE FROM relay_blind_route_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_blind_route_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn blind_relay_reservation_column_exists(
        tx: &Transaction<'_>,
        expected_column: &str,
    ) -> ChatRelayResult<bool> {
        let mut stmt = tx.prepare("PRAGMA table_info(relay_blind_route_reservations)")?;
        let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for column in columns {
            if column? == expected_column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn init_pending_message_schema(conn: &mut Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_messages (
                message_id   BLOB(16) PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                timestamp    INTEGER  NOT NULL,
                envelope     BLOB     NOT NULL,
                received_at  INTEGER  NOT NULL,
                status       INTEGER  NOT NULL DEFAULT 0,
                queue_sequence INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status
                ON pending_messages(receiver, status);
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status_message_id
                ON pending_messages(receiver, status, message_id);
            CREATE INDEX IF NOT EXISTS idx_pm_received_at
                ON pending_messages(received_at);
            CREATE INDEX IF NOT EXISTS idx_pm_cleanup
                ON pending_messages(status, received_at, message_id);
            ",
        )?;

        // Upgrade legacy queues atomically. Existing positive unique sequence
        // values are stable across restarts; only missing/invalid/duplicate
        // values are assigned above the current maximum. No routing metadata
        // leaves SQLite during this migration.
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if !Self::pending_message_column_exists(&tx, "queue_sequence")? {
            tx.execute(
                "ALTER TABLE pending_messages ADD COLUMN queue_sequence INTEGER",
                [],
            )?;
        }
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_queue_sequence (
                singleton     INTEGER PRIMARY KEY CHECK(singleton = 1),
                last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0)
            );
            INSERT OR IGNORE INTO relay_queue_sequence (singleton, last_sequence)
            VALUES (1, 0);
            ",
        )?;

        let mut seen_sequences = HashSet::new();
        let mut max_sequence = 0_i64;
        let mut rowids_to_assign = Vec::new();
        {
            let mut stmt = tx.prepare(
                "SELECT rowid, queue_sequence
                 FROM pending_messages
                 ORDER BY rowid ASC",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?))
            })?;
            for row in rows {
                let (rowid, sequence) = row?;
                match sequence {
                    Some(sequence) if sequence > 0 && seen_sequences.insert(sequence) => {
                        max_sequence = max_sequence.max(sequence);
                    }
                    _ => rowids_to_assign.push(rowid),
                }
            }
        }

        let persisted_last = tx.query_row(
            "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if persisted_last < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "relay_queue_sequence_negative",
            });
        }
        max_sequence = max_sequence.max(persisted_last);
        for rowid in rowids_to_assign {
            max_sequence = max_sequence
                .checked_add(1)
                .ok_or(ChatRelayError::QueueSequenceExhausted)?;
            if tx.execute(
                "UPDATE pending_messages SET queue_sequence = ?1 WHERE rowid = ?2",
                params![max_sequence, rowid],
            )? != 1
            {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "pending_message_sequence_backfill_count",
                });
            }
        }
        tx.execute(
            "UPDATE relay_queue_sequence
             SET last_sequence = ?1
             WHERE singleton = 1",
            params![max_sequence],
        )?;
        tx.execute_batch(
            "
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_queue_sequence
                ON pending_messages(queue_sequence);
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_snapshot_v2
                ON pending_messages(receiver, status, queue_sequence, timestamp);
            ",
        )?;
        tx.commit()?;
        Ok(())
    }

    fn pending_message_column_exists(
        tx: &Transaction<'_>,
        expected_column: &str,
    ) -> ChatRelayResult<bool> {
        let mut stmt = tx.prepare("PRAGMA table_info(pending_messages)")?;
        let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for column in columns {
            if column? == expected_column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn init_blob_and_notification_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_blobs (
                blob_id      TEXT PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                data         BLOB     NOT NULL,
                size         INTEGER  NOT NULL,
                received_at  INTEGER  NOT NULL,
                downloaded   INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_pb_received_at
                ON pending_blobs(received_at);
            CREATE INDEX IF NOT EXISTS idx_pb_receiver
                ON pending_blobs(receiver);

            CREATE TABLE IF NOT EXISTS expired_notifications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sender      BLOB(32) NOT NULL,
                receiver    BLOB(32) NOT NULL,
                message_ids BLOB     NOT NULL,
                created_at  INTEGER  NOT NULL,
                pushed      INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_en_sender_pushed
                ON expired_notifications(sender, pushed);
            CREATE INDEX IF NOT EXISTS idx_en_sender_pull_order
                ON expired_notifications(sender, pushed, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_en_cleanup
                ON expired_notifications(pushed, created_at, id);
            ",
        )?;
        Ok(())
    }

    fn init_usage_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_storage_usage (
                singleton              INTEGER PRIMARY KEY CHECK(singleton = 1),
                pending_message_count  INTEGER NOT NULL CHECK(pending_message_count >= 0),
                pending_message_bytes  INTEGER NOT NULL CHECK(pending_message_bytes >= 0),
                pending_blob_count     INTEGER NOT NULL CHECK(pending_blob_count >= 0),
                pending_blob_bytes     INTEGER NOT NULL CHECK(pending_blob_bytes >= 0)
            );

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_insert
            AFTER INSERT ON pending_messages
            WHEN NEW.status = 0
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = pending_message_count + 1,
                    pending_message_bytes = pending_message_bytes + LENGTH(NEW.envelope)
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_delete
            AFTER DELETE ON pending_messages
            WHEN OLD.status = 0
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = MAX(0, pending_message_count - 1),
                    pending_message_bytes = MAX(
                        0,
                        pending_message_bytes - LENGTH(OLD.envelope)
                    )
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_status
            AFTER UPDATE OF status ON pending_messages
            WHEN OLD.status != NEW.status
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = MAX(
                        0,
                        pending_message_count
                        + CASE
                            WHEN OLD.status = 0 AND NEW.status != 0 THEN -1
                            WHEN OLD.status != 0 AND NEW.status = 0 THEN 1
                            ELSE 0
                          END
                    ),
                    pending_message_bytes = MAX(
                        0,
                        pending_message_bytes
                        + CASE
                            WHEN OLD.status = 0 AND NEW.status != 0
                                THEN -LENGTH(OLD.envelope)
                            WHEN OLD.status != 0 AND NEW.status = 0
                                THEN LENGTH(NEW.envelope)
                            ELSE 0
                          END
                    )
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_blob_usage_insert
            AFTER INSERT ON pending_blobs
            BEGIN
                UPDATE relay_storage_usage
                SET pending_blob_count = pending_blob_count + 1,
                    pending_blob_bytes = pending_blob_bytes + NEW.size
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_blob_usage_delete
            AFTER DELETE ON pending_blobs
            BEGIN
                UPDATE relay_storage_usage
                SET pending_blob_count = MAX(0, pending_blob_count - 1),
                    pending_blob_bytes = MAX(0, pending_blob_bytes - OLD.size)
                WHERE singleton = 1;
            END;
            ",
        )?;
        Ok(())
    }

    fn reconcile_storage_usage(conn: &Connection) -> ChatRelayResult<()> {
        // Reconcile from canonical rows at every startup. This makes upgrades
        // and restored databases deterministic even if an older process never
        // maintained the aggregate usage row.
        let usage = Self::read_canonical_storage_usage(conn)?;
        conn.execute(
            "INSERT OR REPLACE INTO relay_storage_usage (
                singleton,
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             )
             VALUES (1, ?1, ?2, ?3, ?4)",
            params![
                sqlite_integer(usage.pending_messages, "pending_message_count")?,
                sqlite_integer(usage.pending_message_bytes, "pending_message_bytes")?,
                sqlite_integer(usage.pending_blobs, "pending_blob_count")?,
                sqlite_integer(usage.pending_blob_bytes, "pending_blob_bytes")?,
            ],
        )?;
        Ok(())
    }

    // ============================================
    // Opaque ChatPullV2 cursor protection
    // ============================================

    fn encode_pull_cursor_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: PullCursorV2,
    ) -> ChatRelayResult<Vec<u8>> {
        self.pull_cursor_codec
            .encode(receiver, after_timestamp, cursor)
    }

    fn decode_pull_cursor_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded: &[u8],
    ) -> ChatRelayResult<PullCursorV2> {
        self.pull_cursor_codec
            .decode(receiver, after_timestamp, encoded)
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

    fn verified_submit_cache_key(&self, request: &ChatRelayVerifiedSubmitRequestV1) -> [u8; 32] {
        self.verified_submit_replay.cache_key(request)
    }

    fn verified_submit_envelope_fingerprint(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> [u8; 32] {
        self.verified_submit_replay.envelope_fingerprint(request)
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
        self.verified_submit_replay.lock(request).await
    }

    /// Looks up a completed response after request authentication.
    pub(crate) fn verified_submit_cache_lookup(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayResult<VerifiedSubmitCacheLookup> {
        let cache_key = self.verified_submit_cache_key(request);
        let envelope_fingerprint = self.verified_submit_envelope_fingerprint(request);
        let memory_lookup = self
            .verified_submit_replay
            .lookup_cached(&cache_key, &envelope_fingerprint);
        if !matches!(memory_lookup, VerifiedSubmitCacheLookup::Miss) {
            return Ok(memory_lookup);
        }

        let now = sqlite_integer(now_secs(), "verified_submit_response_lookup_time")?;
        let durable_row = {
            let conn = self.conn.lock();
            conn.query_row(
                "SELECT envelope_fingerprint, response_nonce,
                        response_ciphertext, completed_at
                 FROM relay_verified_submit_responses
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?
        };
        let Some((stored_fingerprint, nonce, ciphertext, completed_at)) = durable_row else {
            let reservation = {
                let conn = self.conn.lock();
                conn.query_row(
                    "SELECT envelope_fingerprint, reserved_at
                     FROM relay_verified_submit_reservations
                     WHERE cache_key = ?1",
                    params![cache_key.as_slice()],
                    |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?)),
                )
                .optional()?
            };
            let Some((stored_fingerprint, reserved_at)) = reservation else {
                return Ok(VerifiedSubmitCacheLookup::Miss);
            };
            if reserved_at < 0
                || now.saturating_sub(reserved_at)
                    > i64::try_from(VERIFIED_SUBMIT_RESPONSE_TTL_SECS).unwrap_or(i64::MAX)
            {
                let conn = self.conn.lock();
                conn.execute(
                    "DELETE FROM relay_verified_submit_reservations
                     WHERE cache_key = ?1 AND reserved_at = ?2",
                    params![cache_key.as_slice(), reserved_at],
                )?;
                return Ok(VerifiedSubmitCacheLookup::Miss);
            }
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "verified_submit_reservation_envelope_fingerprint",
                    })?;
            return if stored_fingerprint == envelope_fingerprint {
                Ok(VerifiedSubmitCacheLookup::Pending)
            } else {
                Ok(VerifiedSubmitCacheLookup::Conflict)
            };
        };
        if completed_at < 0
            || now.saturating_sub(completed_at)
                > i64::try_from(VERIFIED_SUBMIT_RESPONSE_TTL_SECS).unwrap_or(i64::MAX)
        {
            let conn = self.conn.lock();
            conn.execute(
                "DELETE FROM relay_verified_submit_responses
                 WHERE cache_key = ?1 AND completed_at = ?2",
                params![cache_key.as_slice(), completed_at],
            )?;
            return Ok(VerifiedSubmitCacheLookup::Miss);
        }
        let stored_fingerprint: [u8; 32] =
            stored_fingerprint
                .try_into()
                .map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "verified_submit_response_envelope_fingerprint",
                })?;
        if stored_fingerprint != envelope_fingerprint {
            return Ok(VerifiedSubmitCacheLookup::Conflict);
        }
        let response = self.verified_submit_replay.recover_response(
            &cache_key,
            &envelope_fingerprint,
            &nonce,
            &ciphertext,
        )?;
        response
            .validate_for_request(request)
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_request_binding",
            })?;
        self.verified_submit_replay.remember_cached(
            cache_key,
            envelope_fingerprint,
            response.clone(),
        );
        Ok(VerifiedSubmitCacheLookup::Exact(response))
    }

    /// Atomically reserves one private replay slot before any external effect.
    pub(crate) fn reserve_verified_submit(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayResult<VerifiedSubmitAdmission> {
        // [CRASH-SAFE-VERIFIED-SUBMIT-ADMISSION 2026-08-24 by Codex] A
        // reservation is the durable intent boundary. Capacity is checked
        // inside the same IMMEDIATE transaction and no unexpired row is ever
        // evicted to admit a new route/custody attempt.
        let cache_key = self.verified_submit_cache_key(request);
        let envelope_fingerprint = self.verified_submit_envelope_fingerprint(request);
        let reserved_at = sqlite_integer(now_secs(), "verified_submit_reservation_time")?;
        let cutoff = reserved_at
            .saturating_sub(i64::try_from(VERIFIED_SUBMIT_RESPONSE_TTL_SECS).unwrap_or(i64::MAX));
        let capacity = sqlite_integer(
            u64::try_from(self.config.dedup_lru_capacity.max(1)).unwrap_or(u64::MAX),
            "verified_submit_reservation_capacity",
        )?;

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM relay_verified_submit_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_verified_submit_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;

        let completed_fingerprint = tx
            .query_row(
                "SELECT envelope_fingerprint FROM relay_verified_submit_responses
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        if let Some(stored_fingerprint) = completed_fingerprint {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "verified_submit_response_envelope_fingerprint",
                    })?;
            let outcome = if stored_fingerprint == envelope_fingerprint {
                VerifiedSubmitAdmission::Completed
            } else {
                VerifiedSubmitAdmission::Conflict
            };
            tx.commit()?;
            return Ok(outcome);
        }

        let existing_reservation = tx
            .query_row(
                "SELECT envelope_fingerprint, reserved_at, owner_epoch,
                        owner_acquired_at
                 FROM relay_verified_submit_reservations
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        if let Some((stored_fingerprint, stored_at, owner_epoch, owner_acquired_at)) =
            existing_reservation
        {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "verified_submit_reservation_envelope_fingerprint",
                    })?;
            if stored_fingerprint != envelope_fingerprint {
                tx.commit()?;
                return Ok(VerifiedSubmitAdmission::Conflict);
            }
            let owner_epoch: [u8; REPLAY_PROCESS_EPOCH_BYTES] = owner_epoch
                .try_into()
                .map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "verified_submit_reservation_owner_epoch",
                })?;
            if stored_at < 0 || owner_acquired_at < stored_at {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "verified_submit_reservation_state",
                });
            }
            let reclaim_at = owner_acquired_at.saturating_add(
                i64::try_from(VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS)
                    .unwrap_or(i64::MAX),
            );
            let outcome = if owner_epoch != self.replay_process_epoch
                && reserved_at >= reclaim_at
            {
                // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] The
                // owner CAS fences a still-running predecessor. Recovery owns
                // only entry custody; it never repeats path selection.
                if tx.execute(
                    "UPDATE relay_verified_submit_reservations
                     SET owner_epoch = ?1, owner_acquired_at = ?2
                     WHERE cache_key = ?3
                       AND envelope_fingerprint = ?4
                       AND reserved_at = ?5
                       AND owner_epoch = ?6
                       AND owner_acquired_at = ?7",
                    params![
                        self.replay_process_epoch.as_slice(),
                        reserved_at,
                        cache_key.as_slice(),
                        envelope_fingerprint.as_slice(),
                        stored_at,
                        owner_epoch.as_slice(),
                        owner_acquired_at,
                    ],
                )? != 1
                {
                    return Err(ChatRelayError::CorruptStoredData {
                        field: "verified_submit_reservation_takeover",
                    });
                }
                VerifiedSubmitAdmission::ReservedForEntryRecovery
            } else {
                VerifiedSubmitAdmission::Pending
            };
            tx.commit()?;
            drop(conn);
            if matches!(
                outcome,
                VerifiedSubmitAdmission::ReservedForEntryRecovery
            ) {
                // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex]
                // Admission is the only authoritative attempted transition.
                self.record_verified_submit_recovery_attempted(
                    u64::try_from(reserved_at).unwrap_or(u64::MAX),
                );
            }
            return Ok(outcome);
        }

        let retained = tx.query_row(
            "SELECT
                (SELECT COUNT(*) FROM relay_verified_submit_responses)
              + (SELECT COUNT(*) FROM relay_verified_submit_reservations)",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if retained < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_retained_count",
            });
        }
        if retained >= capacity {
            tx.commit()?;
            return Ok(VerifiedSubmitAdmission::CapacityExhausted);
        }
        if tx.execute(
            "INSERT INTO relay_verified_submit_reservations (
                cache_key, envelope_fingerprint, reserved_at,
                owner_epoch, owner_acquired_at
             ) VALUES (?1, ?2, ?3, ?4, ?3)",
            params![
                cache_key.as_slice(),
                envelope_fingerprint.as_slice(),
                reserved_at,
                self.replay_process_epoch.as_slice(),
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_reservation_insert",
            });
        }
        tx.commit()?;
        Ok(VerifiedSubmitAdmission::Reserved)
    }

    /// Retains one completed response for exact retry replay across restarts.
    pub(crate) fn remember_verified_submit_response(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
        response: &ChatRelayVerifiedSubmitResponseV1,
    ) -> ChatRelayResult<()> {
        let cache_key = self.verified_submit_cache_key(request);
        let envelope_fingerprint = self.verified_submit_envelope_fingerprint(request);
        response
            .validate_for_request(request)
            .map_err(|_| ChatRelayError::VerifiedSubmitProtectionFailed)?;
        let protected = self.verified_submit_replay.protect_response(
            &cache_key,
            &envelope_fingerprint,
            response,
        )?;
        let completed_at = sqlite_integer(now_secs(), "verified_submit_response_completed_at")?;
        let durable_result: ChatRelayResult<()> = (|| {
            let mut conn = self.conn.lock();
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute(
                "INSERT OR IGNORE INTO relay_verified_submit_responses (
                    cache_key, envelope_fingerprint, response_nonce,
                    response_ciphertext, completed_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    cache_key.as_slice(),
                    envelope_fingerprint.as_slice(),
                    protected.nonce.as_slice(),
                    protected.ciphertext,
                    completed_at,
                ],
            )?;
            let stored_fingerprint = tx.query_row(
                "SELECT envelope_fingerprint
                 FROM relay_verified_submit_responses
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )?;
            if stored_fingerprint.as_slice() != envelope_fingerprint.as_slice() {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "verified_submit_response_insert_conflict",
                });
            }
            let removed_reservation = tx.execute(
                "DELETE FROM relay_verified_submit_reservations
                 WHERE cache_key = ?1
                   AND envelope_fingerprint = ?2
                   AND owner_epoch = ?3",
                params![
                    cache_key.as_slice(),
                    envelope_fingerprint.as_slice(),
                    self.replay_process_epoch.as_slice(),
                ],
            )?;
            if removed_reservation != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "verified_submit_reservation_completion",
                });
            }
            tx.commit()?;
            Ok(())
        })();

        // Preserve same-process retry safety even if the durable write fails.
        // The caller receives the storage error and records only its fixed
        // reason bucket; no request-derived values enter logs or health.
        self.verified_submit_replay.remember_cached(
            cache_key,
            envelope_fingerprint,
            response.clone(),
        );
        durable_result?;
        Ok(())
    }

    fn blind_relay_route_cache_key(&self, route_id: &[u8; 16]) -> [u8; 32] {
        self.blind_route_replay.cache_key(route_id)
    }

    fn blind_relay_route_fingerprint(&self, request_commitment: &[u8; 32]) -> [u8; 32] {
        self.blind_route_replay
            .request_fingerprint(request_commitment)
    }

    /// Reserves one authenticated blind route before peel, forward, or store.
    pub(crate) fn reserve_blind_relay_route(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
    ) -> ChatRelayResult<BlindRelayRouteAdmission> {
        // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Reservation and
        // capacity admission share one IMMEDIATE transaction.
        // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Ownership may move
        // after a short process grace period without changing immutable route
        // evidence age. Armed takeover is exposed separately so the caller can
        // reconcile only exact idempotent work and recover a lost sealed ACK.
        let cache_key = self.blind_relay_route_cache_key(route_id);
        let request_fingerprint = self.blind_relay_route_fingerprint(request_commitment);
        let reserved_at = sqlite_integer(now_secs(), "blind_relay_route_reserved_at")?;
        let cutoff = reserved_at
            .saturating_sub(i64::try_from(BLIND_RELAY_ROUTE_REPLAY_TTL_SECS).unwrap_or(i64::MAX));
        let capacity = i64::try_from(BLIND_RELAY_ROUTE_REPLAY_CAPACITY).unwrap_or(i64::MAX);
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "DELETE FROM relay_blind_route_responses WHERE completed_at < ?1",
            params![cutoff],
        )?;
        tx.execute(
            "DELETE FROM relay_blind_route_reservations WHERE reserved_at < ?1",
            params![cutoff],
        )?;

        let completed = tx
            .query_row(
                "SELECT request_fingerprint, response_nonce,
                        response_ciphertext, completed_at
                 FROM relay_blind_route_responses WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        if let Some((stored_fingerprint, nonce, ciphertext, completed_at)) = completed {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "blind_relay_route_response_fingerprint",
                    })?;
            if stored_fingerprint != request_fingerprint {
                tx.commit()?;
                return Ok(BlindRelayRouteAdmission::Conflict);
            }
            let completed_at =
                u64::try_from(completed_at).map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_response_completed_at",
                })?;
            let response = self.blind_route_replay.recover_response(
                &cache_key,
                &request_fingerprint,
                &nonce,
                &ciphertext,
            )?;
            tx.commit()?;
            return Ok(BlindRelayRouteAdmission::Completed {
                response,
                completed_at,
            });
        }

        let reservation = tx
            .query_row(
                "SELECT request_fingerprint, reserved_at, owner_epoch,
                        owner_acquired_at, effect_started_at
                 FROM relay_blind_route_reservations
                 WHERE cache_key = ?1",
                params![cache_key.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<i64>>(4)?,
                    ))
                },
            )
            .optional()?;
        if let Some((
            stored_fingerprint,
            stored_at,
            owner_epoch,
            owner_acquired_at,
            effect_started_at,
        )) = reservation
        {
            let stored_fingerprint: [u8; 32] =
                stored_fingerprint
                    .try_into()
                    .map_err(|_| ChatRelayError::CorruptStoredData {
                        field: "blind_relay_route_reservation_fingerprint",
                    })?;
            if stored_fingerprint != request_fingerprint {
                tx.commit()?;
                return Ok(BlindRelayRouteAdmission::Conflict);
            }
            let owner_epoch: [u8; REPLAY_PROCESS_EPOCH_BYTES] = owner_epoch
                .try_into()
                .map_err(|_| ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_reservation_owner_epoch",
                })?;
            if stored_at < 0
                || owner_acquired_at < stored_at
                || effect_started_at.is_some_and(|started_at| started_at < stored_at)
            {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "blind_relay_route_reservation_state",
                });
            }
            let reclaim_at = owner_acquired_at.saturating_add(
                i64::try_from(BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS).unwrap_or(i64::MAX),
            );
            if owner_epoch != self.replay_process_epoch && reserved_at >= reclaim_at {
                if tx.execute(
                    "UPDATE relay_blind_route_reservations
                     SET owner_epoch = ?1, owner_acquired_at = ?2
                     WHERE cache_key = ?3
                       AND request_fingerprint = ?4
                       AND reserved_at = ?5
                       AND owner_epoch = ?6
                       AND owner_acquired_at = ?7",
                    params![
                        self.replay_process_epoch.as_slice(),
                        reserved_at,
                        cache_key.as_slice(),
                        request_fingerprint.as_slice(),
                        stored_at,
                        owner_epoch.as_slice(),
                        owner_acquired_at,
                    ],
                )? != 1
                {
                    return Err(ChatRelayError::CorruptStoredData {
                        field: "blind_relay_route_reservation_takeover",
                    });
                }
                let admission = if effect_started_at.is_some() {
                    BlindRelayRouteAdmission::ReservedForRecovery
                } else {
                    BlindRelayRouteAdmission::Reserved
                };
                tx.commit()?;
                drop(conn);
                if matches!(admission, BlindRelayRouteAdmission::ReservedForRecovery) {
                    self.peer_telemetry.record_blind_route_recovery(
                        u64::try_from(reserved_at).unwrap_or(u64::MAX),
                        BlindRouteRecoveryEvent::Attempted,
                    );
                }
                return Ok(admission);
            }
            tx.commit()?;
            return Ok(BlindRelayRouteAdmission::Pending);
        }

        let retained = tx.query_row(
            "SELECT
                (SELECT COUNT(*) FROM relay_blind_route_responses)
              + (SELECT COUNT(*) FROM relay_blind_route_reservations)",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if retained < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_retained_count",
            });
        }
        if retained >= capacity {
            tx.commit()?;
            return Ok(BlindRelayRouteAdmission::CapacityExhausted);
        }
        if tx.execute(
            "INSERT INTO relay_blind_route_reservations (
                cache_key, request_fingerprint, reserved_at, owner_epoch,
                owner_acquired_at, effect_started_at
             ) VALUES (?1, ?2, ?3, ?4, ?3, NULL)",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                reserved_at,
                self.replay_process_epoch.as_slice(),
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_reservation_insert",
            });
        }
        tx.commit()?;
        Ok(BlindRelayRouteAdmission::Reserved)
    }

    /// Arms an owned route claim immediately before its first external effect.
    pub(crate) fn arm_blind_relay_route_effect(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        started_at: u64,
    ) -> ChatRelayResult<()> {
        let cache_key = self.blind_relay_route_cache_key(route_id);
        let request_fingerprint = self.blind_relay_route_fingerprint(request_commitment);
        let started_at = sqlite_integer(started_at, "blind_relay_route_effect_started_at")?;
        let conn = self.conn.lock();
        if conn.execute(
            "UPDATE relay_blind_route_reservations
             SET effect_started_at = MAX(?1, reserved_at)
             WHERE cache_key = ?2
               AND request_fingerprint = ?3
               AND owner_epoch = ?4
               AND effect_started_at IS NULL",
            params![
                started_at,
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                self.replay_process_epoch.as_slice(),
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_effect_admission",
            });
        }
        Ok(())
    }

    /// Releases only this process's claim when no external effect was armed.
    pub(crate) fn release_unarmed_blind_relay_route(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
    ) -> ChatRelayResult<bool> {
        let cache_key = self.blind_relay_route_cache_key(route_id);
        let request_fingerprint = self.blind_relay_route_fingerprint(request_commitment);
        let conn = self.conn.lock();
        Ok(conn.execute(
            "DELETE FROM relay_blind_route_reservations
             WHERE cache_key = ?1
               AND request_fingerprint = ?2
               AND owner_epoch = ?3
               AND effect_started_at IS NULL",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                self.replay_process_epoch.as_slice(),
            ],
        )? == 1)
    }

    /// Atomically replaces one route reservation with its sealed exact ACK.
    pub(crate) fn remember_blind_relay_route_response(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        response: &[u8],
        completed_at: u64,
    ) -> ChatRelayResult<()> {
        let cache_key = self.blind_relay_route_cache_key(route_id);
        let request_fingerprint = self.blind_relay_route_fingerprint(request_commitment);
        let protected =
            self.blind_route_replay
                .protect_response(&cache_key, &request_fingerprint, response)?;
        let completed_at = sqlite_integer(completed_at, "blind_relay_route_completed_at")?;
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if tx.execute(
            "INSERT OR IGNORE INTO relay_blind_route_responses (
                cache_key, request_fingerprint, response_nonce,
                response_ciphertext, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                protected.nonce.as_slice(),
                protected.ciphertext,
                completed_at,
            ],
        )? > 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_insert",
            });
        }
        let stored_fingerprint = tx.query_row(
            "SELECT request_fingerprint FROM relay_blind_route_responses
             WHERE cache_key = ?1",
            params![cache_key.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        )?;
        if stored_fingerprint.as_slice() != request_fingerprint.as_slice() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_insert_conflict",
            });
        }
        if tx.execute(
            "DELETE FROM relay_blind_route_reservations
             WHERE cache_key = ?1
               AND request_fingerprint = ?2
               AND owner_epoch = ?3",
            params![
                cache_key.as_slice(),
                request_fingerprint.as_slice(),
                self.replay_process_epoch.as_slice(),
            ],
        )? != 1
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_reservation_completion",
            });
        }
        tx.commit()?;
        Ok(())
    }

    // ============================================
    // Message store / pull / ack
    // ============================================

    fn read_storage_usage(conn: &Connection) -> ChatRelayResult<ChatRelayStorageUsage> {
        let counters = conn.query_row(
            "SELECT
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             FROM relay_storage_usage
             WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )?;
        Ok(ChatRelayStorageUsage {
            pending_messages: nonnegative_sqlite_value(counters.0, "pending_message_count")?,
            pending_message_bytes: nonnegative_sqlite_value(counters.1, "pending_message_bytes")?,
            pending_blobs: nonnegative_sqlite_value(counters.2, "pending_blob_count")?,
            pending_blob_bytes: nonnegative_sqlite_value(counters.3, "pending_blob_bytes")?,
        })
    }

    fn read_canonical_storage_usage(conn: &Connection) -> ChatRelayResult<ChatRelayStorageUsage> {
        let counters = conn.query_row(
            "SELECT
                (SELECT COUNT(*) FROM pending_messages WHERE status = 0),
                (SELECT COALESCE(SUM(LENGTH(envelope)), 0)
                   FROM pending_messages WHERE status = 0),
                (SELECT COUNT(*) FROM pending_blobs),
                (SELECT COALESCE(SUM(size), 0) FROM pending_blobs)",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )?;
        Ok(ChatRelayStorageUsage {
            pending_messages: nonnegative_sqlite_value(
                counters.0,
                "canonical_pending_message_count",
            )?,
            pending_message_bytes: nonnegative_sqlite_value(
                counters.1,
                "canonical_pending_message_bytes",
            )?,
            pending_blobs: nonnegative_sqlite_value(counters.2, "canonical_pending_blob_count")?,
            pending_blob_bytes: nonnegative_sqlite_value(
                counters.3,
                "canonical_pending_blob_bytes",
            )?,
        })
    }

    fn record_pull_quarantine(
        &self,
        now: u64,
        pending_messages: usize,
        expired_notifications: usize,
        removed_events: usize,
        retained_events: usize,
    ) {
        let mut status = self.maintenance_status.write();
        status.quarantined_pending_messages_total = status
            .quarantined_pending_messages_total
            .saturating_add(u64::try_from(pending_messages).unwrap_or(u64::MAX));
        status.quarantined_expired_notifications_total = status
            .quarantined_expired_notifications_total
            .saturating_add(u64::try_from(expired_notifications).unwrap_or(u64::MAX));
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(u64::try_from(removed_events).unwrap_or(u64::MAX));
        status.quarantine_events_retained = u64::try_from(retained_events).unwrap_or(u64::MAX);
        status.last_quarantine_at = Some(now);
    }

    /// Stores a pending offline message for a receiver that is not currently online.
    ///
    /// # Errors
    ///
    /// Returns an item-size or durable-capacity error before insertion, or a
    /// serialization/SQLite error if encoding or the atomic write fails.
    pub fn store_pending(&self, envelope: &ChatEnvelope) -> ChatRelayResult<()> {
        let write = self
            .pending_custody
            .prepare_store(envelope, now_secs())?;
        let mut conn = self.conn.lock();
        let outcome = self.pending_custody.store(&mut conn, write)?;
        drop(conn);

        if let PendingMessageStoreOutcome::Stored { encoded_bytes } = outcome {
            debug!(encoded_bytes, "[CHAT_RELAY] Message stored pending");
        }
        Ok(())
    }

    fn quarantine_pending_pull_rows(
        &self,
        conn: &mut Connection,
        corrupt_rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        if corrupt_rows.is_empty() {
            return Ok(());
        }

        let quarantine_now = now_secs();
        let outcome = self.durable_quarantine.replace_rows(
            conn,
            QuarantineRowTarget::PendingMessage,
            corrupt_rows,
            quarantine_now,
        )?;

        self.record_pull_quarantine(
            quarantine_now,
            outcome.quarantined_rows,
            0,
            outcome.removed_events,
            outcome.retained_events,
        );
        warn!(
            quarantined_pending_messages = outcome.quarantined_rows,
            "[CHAT_RELAY] Corrupt pending rows isolated during pull"
        );
        Ok(())
    }

    /// Retrieves a page of pending messages for the given receiver wallet.
    ///
    /// The v1 wire cursor contains only `message_id`, so rows must be ordered
    /// by that same key. Ordering by timestamp first can permanently skip a
    /// later row whose random ID sorts below the previous page's cursor.
    ///
    /// # Errors
    ///
    /// Corrupt rows are atomically replaced by de-identified quarantine events
    /// so one poison row cannot permanently block a receiver's mailbox.
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        limit: u32,
    ) -> ChatRelayResult<(Vec<PendingMessage>, bool)> {
        let page_limit = usize::try_from(limit.clamp(1, 100)).unwrap_or(100);
        let mut conn = self.conn.lock();
        let page = self.pending_pull.read_legacy_page(
            &conn,
            receiver,
            after_timestamp,
            cursor,
            page_limit,
        )?;
        self.quarantine_pending_pull_rows(&mut conn, &page.corrupt_rows)?;
        drop(conn);

        let mut messages = page.messages;
        let has_more = page.raw_has_more || messages.len() > page_limit;
        messages.truncate(page_limit);
        Ok((messages, has_more))
    }

    /// Retrieves one stable monotonic snapshot page for ChatPullV2.
    ///
    /// An empty cursor captures the current receiver-specific sequence ceiling.
    /// Later inserts receive larger sequences and cannot move into that snapshot,
    /// preventing duplicate/skip behavior while the client paginates. The
    /// sequence and ceiling remain node-internal inside an AEAD-protected cursor.
    ///
    /// # Errors
    ///
    /// Returns [`ChatRelayError::InvalidPullCursor`] for tampered, cross-wallet,
    /// cross-filter, malformed, or foreign-node cursors. Corrupt durable rows are
    /// atomically quarantined using the same path as v1 pulls.
    pub fn pull_pending_v2(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        encoded_cursor: &[u8],
        limit: u32,
    ) -> ChatRelayResult<PendingMessagePageV2> {
        let page_limit = usize::try_from(limit.clamp(1, 100)).unwrap_or(100);
        let mut conn = self.conn.lock();
        let cursor = if encoded_cursor.is_empty() {
            PullCursorV2 {
                position: 0,
                ceiling: self.pending_pull.capture_snapshot_ceiling(
                    &conn,
                    receiver,
                    after_timestamp,
                )?,
            }
        } else {
            self.decode_pull_cursor_v2(receiver, after_timestamp, encoded_cursor)?
        };
        let page = self.pending_pull.read_snapshot_page(
            &conn,
            receiver,
            after_timestamp,
            cursor.position,
            cursor.ceiling,
            page_limit,
        )?;
        self.quarantine_pending_pull_rows(&mut conn, &page.corrupt_rows)?;
        drop(conn);

        let mut valid_messages = page.messages;
        let valid_overflow = valid_messages.len() > page_limit;
        let has_more = page.raw_has_more || valid_overflow;
        let next_position = if valid_overflow {
            valid_messages
                .get(page_limit.saturating_sub(1))
                .map(|(sequence, _)| *sequence)
                .unwrap_or(cursor.position)
        } else if has_more {
            page.raw_max_sequence.unwrap_or(cursor.position)
        } else {
            cursor.ceiling
        };
        valid_messages.truncate(page_limit);
        let messages = valid_messages
            .into_iter()
            .map(|(_, message)| message)
            .collect();
        let next_cursor = self.encode_pull_cursor_v2(
            receiver,
            after_timestamp,
            PullCursorV2 {
                position: next_position,
                ceiling: cursor.ceiling,
            },
        )?;

        Ok(PendingMessagePageV2 {
            messages,
            next_cursor,
            has_more,
        })
    }

    /// Acknowledges delivery of a batch of messages, deleting them from the store.
    ///
    /// Only deletes rows where `receiver = receiver_wallet`.
    ///
    /// # Errors
    ///
    /// Returns an oversized-batch or `SQLite` error. The transaction is atomic.
    pub fn ack_messages(
        &self,
        message_ids: &[[u8; 16]],
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        let Some(batch) = self
            .pending_custody
            .prepare_acknowledgement(message_ids)?
        else {
            return Ok(0);
        };
        let deleted = self.pending_custody.acknowledge(
            &mut self.conn.lock(),
            &batch,
            receiver_wallet,
        )?;

        debug!(count = deleted, "[CHAT_RELAY] Messages ACKed and deleted");
        Ok(deleted)
    }

    // ============================================
    // Blob store / get / delete
    // ============================================

    /// Stores one opaque encrypted blob under node-wide and receiver quotas.
    ///
    /// # Errors
    ///
    /// Returns an item-size, capacity, serialization, or `SQLite` error.
    pub fn put_blob(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        data: &[u8],
        file_hash: &[u8; 32],
    ) -> ChatRelayResult<String> {
        let write =
            self.blob_custody
                .prepare_put(sender, receiver, data, file_hash, now_secs())?;
        let mut conn = self.conn.lock();
        let outcome = self.blob_custody.put(&mut conn, write)?;
        drop(conn);

        if let EncryptedBlobStoreOutcome::Stored { size, .. } = &outcome {
            info!(size = *size, "[CHAT_RELAY] Encrypted blob stored");
        }
        Ok(outcome.blob_id().to_owned())
    }

    /// Retrieves an opaque encrypted blob by its HMAC-derived identifier.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error or [`ChatRelayError::BlobNotFound`].
    pub fn get_blob(&self, blob_id: &str) -> ChatRelayResult<Vec<u8>> {
        let conn = self.conn.lock();
        let data = self.blob_custody.get(&conn, blob_id)?;
        drop(conn);
        debug!(size = data.len(), "[CHAT_RELAY] Encrypted blob retrieved");
        Ok(data)
    }

    /// Deletes an encrypted blob when requested by its original sender.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite`, not-found, or authorization error.
    pub fn delete_blob(&self, blob_id: &str, requester: &[u8; 32]) -> ChatRelayResult<()> {
        let conn = self.conn.lock();
        self.blob_custody.delete(&conn, blob_id, requester)?;
        drop(conn);
        info!("[CHAT_RELAY] Encrypted blob deleted by authorized sender");
        Ok(())
    }

    // ============================================
    // Expired notifications
    // ============================================

    /// Retrieves one bounded page of expiry notifications for a sender.
    ///
    /// The extra row is used only to compute `has_more`; it is never returned.
    /// Invalid durable rows are atomically replaced by de-identified quarantine
    /// evidence so one poison row cannot permanently block sender control flow.
    ///
    /// # Errors
    ///
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<(Vec<ExpiredNotification>, bool)> {
        let mut conn = self.conn.lock();
        let page = self.expired_notification_delivery.read_page(&conn, sender)?;

        if page.corrupt_rows.is_empty() {
            drop(conn);
        } else {
            let quarantine_now = now_secs();
            let outcome = self.durable_quarantine.replace_rows(
                &mut conn,
                QuarantineRowTarget::ExpiredNotification,
                &page.corrupt_rows,
                quarantine_now,
            )?;
            drop(conn);

            self.record_pull_quarantine(
                quarantine_now,
                0,
                outcome.quarantined_rows,
                outcome.removed_events,
                outcome.retained_events,
            );
            warn!(
                quarantined_expired_notifications = outcome.quarantined_rows,
                "[CHAT_RELAY] Corrupt expiry notifications isolated during pull"
            );
        }

        Ok((page.notifications, page.has_more))
    }

    /// Compatibility wrapper for callers that do not consume pagination yet.
    ///
    /// New runtime code should use [`Self::pull_pending_notifications`] and
    /// propagate its `has_more` flag.
    ///
    /// # Errors
    ///
    /// Returns a storage, decoding, or durable-data integrity error.
    pub fn get_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<Vec<ExpiredNotification>> {
        self.pull_pending_notifications(sender)
            .map(|(notifications, _)| notifications)
    }

    /// Atomically marks a successfully written notification page as pushed.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error and rolls back the whole page on failure.
    pub fn mark_notifications_pushed(&self, ids: &[i64]) -> ChatRelayResult<()> {
        let Some(batch) = self
            .expired_notification_delivery
            .prepare_acknowledgement(ids)
        else {
            return Ok(());
        };
        self.expired_notification_delivery
            .mark_pushed(&mut self.conn.lock(), &batch)
    }

    // ============================================
    // TTL cleanup
    // ============================================

    /// Runs one TTL cleanup cycle (synchronous — call from `spawn_blocking`).
    ///
    /// Mutations run in a bounded sequence of `SQLite` IMMEDIATE transactions.
    /// Each committed batch releases the connection before the next begins.
    /// Returns `(expired_messages, expired_blobs)`.
    ///
    /// # Errors
    ///
    /// Returns a storage, serialization, or durable-data integrity error. A
    /// failed batch is rolled back and counted in maintenance evidence. Earlier
    /// committed batches remain durable and are included in aggregate counters.
    pub fn run_cleanup(&self) -> ChatRelayResult<(usize, usize)> {
        self.run_cleanup_with_batch_budget(CLEANUP_MAX_BATCHES_PER_RUN)
    }

    fn run_cleanup_with_batch_budget(&self, max_batches: usize) -> ChatRelayResult<(usize, usize)> {
        let now = now_secs();
        let cleanup_now = i64::try_from(now).unwrap_or(i64::MAX);
        let (summary, failure) = self.run_cleanup_at(cleanup_now, max_batches.max(1));

        self.record_cleanup_run(now, summary, failure.as_ref());

        let Some(error) = failure else {
            return Ok((summary.expired_messages, summary.expired_blobs));
        };
        Err(error)
    }

    fn record_cleanup_run(
        &self,
        now: u64,
        summary: CleanupRunSummary,
        failure: Option<&ChatRelayError>,
    ) {
        let mut status = self.maintenance_status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_batches_total = status
            .cleanup_batches_total
            .saturating_add(u64::try_from(summary.successful_batches).unwrap_or(u64::MAX));
        if summary.backlog_deferred {
            status.cleanup_backlog_deferred_total =
                status.cleanup_backlog_deferred_total.saturating_add(1);
        }
        status.expired_messages_total = status
            .expired_messages_total
            .saturating_add(u64::try_from(summary.expired_messages).unwrap_or(u64::MAX));
        status.expired_blobs_total = status
            .expired_blobs_total
            .saturating_add(u64::try_from(summary.expired_blobs).unwrap_or(u64::MAX));
        status.expired_notifications_removed_total = status
            .expired_notifications_removed_total
            .saturating_add(u64::try_from(summary.removed_notifications).unwrap_or(u64::MAX));
        status.quarantined_pending_messages_total =
            status.quarantined_pending_messages_total.saturating_add(
                u64::try_from(summary.quarantined_pending_messages).unwrap_or(u64::MAX),
            );
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(u64::try_from(summary.removed_quarantine_events).unwrap_or(u64::MAX));
        if summary.successful_batches > 0 {
            status.quarantine_events_retained =
                u64::try_from(summary.retained_quarantine_events).unwrap_or(u64::MAX);
        }
        if summary.quarantined_pending_messages > 0 {
            status.last_quarantine_at = Some(now);
        }
        status.last_cleanup_at = Some(now);
        status.last_cleanup_batches = u64::try_from(summary.successful_batches).unwrap_or(u64::MAX);
        status.last_cleanup_backlog_deferred = summary.backlog_deferred;
        status.last_cleanup_quarantined_pending_messages =
            u64::try_from(summary.quarantined_pending_messages).unwrap_or(u64::MAX);
        match failure {
            None => {
                status.last_cleanup_status = Some("succeeded".to_string());
                status.last_cleanup_failure_reason = None;
            }
            Some(error) => {
                status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
                status.last_cleanup_status = Some("failed".to_string());
                status.last_cleanup_failure_reason = Some(error.reason_bucket().to_string());
            }
        }
    }

    fn run_cleanup_at(
        &self,
        now: i64,
        max_batches: usize,
    ) -> (CleanupRunSummary, Option<ChatRelayError>) {
        let cutoffs = self.cleanup.cutoffs(now);

        let mut summary = CleanupRunSummary::default();
        let mut failure = None;
        for batch_index in 0..max_batches {
            let batch_result = {
                let mut conn = self.conn.lock();
                self.run_cleanup_transaction(&mut conn, now, cutoffs)
            };
            match batch_result {
                Ok(batch) => {
                    let has_more = batch.has_more;
                    summary.absorb(batch);
                    if !has_more {
                        break;
                    }
                    if batch_index + 1 == max_batches {
                        summary.backlog_deferred = true;
                    }
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        if summary.removed_anything() || summary.backlog_deferred {
            info!(
                expired_messages = summary.expired_messages,
                expired_blobs = summary.expired_blobs,
                removed_notifications = summary.removed_notifications,
                quarantined_pending_messages = summary.quarantined_pending_messages,
                removed_quarantine_events = summary.removed_quarantine_events,
                removed_verified_submit_responses = summary.removed_verified_submit_responses,
                removed_verified_submit_reservations = summary.removed_verified_submit_reservations,
                removed_blind_route_responses = summary.removed_blind_route_responses,
                removed_blind_route_reservations = summary.removed_blind_route_reservations,
                retained_quarantine_events = summary.retained_quarantine_events,
                committed_batches = summary.successful_batches,
                backlog_deferred = summary.backlog_deferred,
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Bounded cleanup run complete"
            );
        } else {
            debug!(
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Cleanup: nothing to expire"
            );
        }
        if summary.quarantined_pending_messages > 0 {
            warn!(
                quarantined_pending_messages = summary.quarantined_pending_messages,
                "[CHAT_RELAY] Corrupt pending rows isolated during cleanup"
            );
        }

        (summary, failure)
    }

    fn run_cleanup_transaction(
        &self,
        conn: &mut Connection,
        now: i64,
        cutoffs: RelayCleanupCutoffs,
    ) -> ChatRelayResult<CleanupBatchOutcome> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let transaction_result =
            self.cleanup
                .run_batch(&tx, &self.durable_quarantine, now, cutoffs);

        match transaction_result {
            Ok(counts) => {
                tx.commit()?;
                Ok(counts)
            }
            Err(error) => Err(error),
        }
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
        self.maintenance_status.read().clone()
    }

    /// Records a blocking-worker failure that occurred outside `run_cleanup`.
    ///
    /// Tokio join failures are deliberately converted to stable buckets so a
    /// heartbeat never exposes panic payloads or other runtime internals.
    pub(crate) fn record_maintenance_worker_failure(&self, reason: &'static str) {
        let mut status = self.maintenance_status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
        status.last_cleanup_at = Some(now_secs());
        status.last_cleanup_status = Some("failed".to_string());
        status.last_cleanup_failure_reason = Some(reason.to_string());
        status.last_cleanup_batches = 0;
        status.last_cleanup_backlog_deferred = false;
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
        Self::read_storage_usage(&conn)
    }

    /// Creates an owner-private, verified recovery image of relay custody.
    ///
    /// The artifact is confined to `.aeronyx-relay-backups` beside the active
    /// database; callers cannot redirect encrypted custody to an arbitrary
    /// path. `SQLite`'s online-backup API includes committed WAL pages. Before
    /// publication, the isolated image must pass physical integrity, schema
    /// sentinel, canonical usage, queue sequence, and anonymous circuit-state
    /// validation. Existing artifacts are never replaced.
    ///
    /// [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] The source connection
    /// mutex remains held only during page copying. Validation and publication
    /// operate on the isolated artifact and never inspect E2E ciphertext,
    /// wallet identities, routes, or message identifiers. This is synchronous
    /// storage work; async operator surfaces must invoke it through
    /// `spawn_blocking`, never from an Axum request executor directly.
    ///
    /// # Errors
    ///
    /// Returns a path-free SQLite/storage error if the service uses an
    /// in-memory database, the private backup area cannot be secured, the
    /// source remains locked, validation fails, or durable publication fails.
    pub fn create_verified_backup(&self) -> ChatRelayResult<PathBuf> {
        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let created_at = now_secs();
        let nonce = rand::random::<u64>();
        let artifact_name =
            Self::backup_artifact_namespace().unique_recovery_image_name(created_at, nonce);
        let destination = backup_directory.join(artifact_name.as_str());
        self.create_verified_backup_artifact(&backup_directory, &destination, false)?;
        Ok(destination)
    }

    /// Creates or reuses one verified backup for an audited operation ID.
    ///
    /// The raw ID is never persisted. A domain-separated HMAC under the stable
    /// node secret selects one private destination, so CMS replay after ACK
    /// loss or process restart cannot create a second recovery image. An
    /// existing artifact is opened read-only with `NOFOLLOW` and fully verified
    /// before it is accepted. Corrupt or non-private artifacts fail closed.
    ///
    /// [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] This method returns
    /// aggregate receipt data only; callers cannot learn the artifact key or
    /// path. It is synchronous and must run through `spawn_blocking`.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the operation ID is empty or too
    /// large, the private boundary is unavailable, snapshot certification or
    /// publication fails, or a replay artifact cannot be re-verified.
    pub(crate) fn create_verified_backup_for_operation(
        &self,
        operation_id: &str,
    ) -> ChatRelayResult<ChatRelayBackupReceipt> {
        let artifact_name = Self::backup_artifact_namespace()
            .idempotent_recovery_image_name(&self.node_secret, operation_id)
            .map_err(Self::map_backup_namespace_error)?;

        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let destination = backup_directory.join(artifact_name.as_str());
        self.create_verified_backup_artifact(&backup_directory, &destination, true)
    }

    /// Audits the configured retention policy without creating or deleting
    /// recovery images.
    ///
    /// Every managed recovery image is fully re-verified before the first
    /// capacity calculation. Incomplete private SQLite files left by an
    /// interrupted backup are reported, while unknown entries, symlinks,
    /// permission drift, or corrupt recovery images fail closed. At least the
    /// newest verified image is counted as retained even if it alone exceeds
    /// the configured byte budget.
    ///
    /// [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] This operation shares
    /// one lock with publication and replay, so the inspection cannot race a
    /// backup that has not yet produced its audit receipt. This method has no
    /// filesystem deletion path.
    /// This is synchronous filesystem/SQLite work; async callers must use a
    /// blocking worker rather than an application request executor.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the private boundary cannot be
    /// inspected, its entry count is unbounded, an entry is unmanaged or not
    /// owner-private, or a recovery image fails full SQLite verification.
    pub fn audit_verified_backup_retention(
        &self,
    ) -> ChatRelayResult<ChatRelayBackupRetentionReceipt> {
        let _operation = self.backup_operations.lock();
        let backup_directory = self.private_backup_directory()?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Ok(
            Self::inspect_verified_backup_retention(&self.config, &backup_directory, now_secs())?
                .receipt,
        )
    }

    /// Audits one configured private backup boundary without opening the live
    /// relay database.
    ///
    /// This host-local entry point exists for the CLI. It takes the same
    /// cross-process lock as online backup creation and returns aggregate-only
    /// state without writing an audit record or deleting an artifact.
    pub fn audit_verified_backup_retention_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayBackupRetentionReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        Ok(Self::inspect_verified_backup_retention(config, &backup_directory, now_secs())?.receipt)
    }

    /// Verifies the private relay-custody maintenance audit from genesis.
    ///
    /// [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] The verifier authenticates
    /// every bounded record under the node identity secret, checks sequence and
    /// hash-chain continuity across immutable segments, authenticates each
    /// SHA-256 segment checkpoint, rejects schema drift, and detects file length
    /// changes during reads. It returns aggregate phase/segment counts only.
    /// Audit MACs, checkpoint MACs, hashes, operation identifiers, paths,
    /// payloads, and custody metadata never cross this service boundary. An
    /// absent audit is a valid empty history; this method does not create,
    /// rotate, recover, or repair audit files.
    ///
    /// The host-local CLI is the intended caller. This synchronous filesystem
    /// operation must run on a blocking worker when invoked from async code.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error if the private control boundary is
    /// unsafe, the audit exceeds its fixed limits, a record/checkpoint is
    /// malformed or unauthenticated, chain continuity fails, or any retained
    /// segment changes while it is being verified.
    pub fn verify_backup_maintenance_audit_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayBackupAuditVerificationReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        Ok(Self::verify_backup_audit_chain(parent, node_secret)?
            .state
            .into_receipt())
    }

    /// Creates a portable identity-signed anchor for the latest immutable
    /// relay-custody audit checkpoint.
    ///
    /// [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] The returned protocol frame
    /// commits to the private checkpoint through a domain-separated opaque
    /// digest. It exposes only the producer identity, monotonic checkpoint
    /// generation, aggregate covered records/bytes, and signature. The frame is
    /// deterministic for a checkpoint; independent witnesses own time evidence.
    /// Active records are deliberately excluded until their segment is
    /// checkpointed. An interrupted rotation must be recovered by the next
    /// locked maintenance append before an anchor can be issued.
    ///
    /// # Errors
    /// Returns a path-free storage error when the private chain is unsafe,
    /// unauthenticated, has no immutable checkpoint, or rotation publication is
    /// incomplete.
    pub fn create_backup_maintenance_audit_anchor_for_config(
        config: &ChatRelayConfig,
        identity: &IdentityKeyPair,
    ) -> ChatRelayResult<CustodyAuditAnchorV1> {
        Ok(Self::hold_backup_maintenance_audit_anchor_for_config(config, identity)?.into_anchor())
    }

    /// Holds the cross-process custody maintenance lock while exposing the
    /// exact current producer-signed checkpoint anchor.
    ///
    /// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] This is the
    /// transactional boundary for host-local receipt import. Keeping the guard
    /// alive prevents a concurrent backup/prune operation from publishing a
    /// newer immutable checkpoint between exact-anchor validation and evidence
    /// persistence. It performs no network I/O and exposes no private audit
    /// material.
    ///
    /// # Errors
    /// Returns a path-free storage error under the same conditions as
    /// [`Self::create_backup_maintenance_audit_anchor_for_config`], or when the
    /// cross-process maintenance lock is already held.
    pub fn hold_backup_maintenance_audit_anchor_for_config(
        config: &ChatRelayConfig,
        identity: &IdentityKeyPair,
    ) -> ChatRelayResult<ChatRelayCustodyAuditAnchorGuard> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let parent = backup_directory.parent().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_CANTOPEN,
                "relay backup directory has no private audit parent",
            )
        })?;
        let node_secret = derive_node_secret(&identity.to_bytes());
        let chain = Self::verify_backup_audit_chain(parent, &node_secret)?;
        if chain.pending_rotation.is_some() {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_BUSY,
                "relay backup maintenance audit rotation must complete before anchoring",
            ));
        }
        let anchor_digest = Self::backup_audit_anchor_digest(&chain.state)?;
        let receipt = chain.state.receipt();
        let anchor = CustodyAuditAnchorV1::signed(
            receipt.checkpoint_count,
            receipt.archived_record_count,
            receipt.archived_bytes,
            anchor_digest,
            identity,
        )
        .map_err(|_| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_AUTH,
                "unable to sign relay backup maintenance audit anchor",
            )
        })?;
        Ok(ChatRelayCustodyAuditAnchorGuard {
            _filesystem_lock: filesystem_lock,
            anchor,
        })
    }

    /// Verifies whether the newest private recovery image is ready for a
    /// separately approved host-local restore operation.
    ///
    /// The command fully verifies every managed backup under the same
    /// cross-process lock used by publication and prune, then inspects the
    /// configured active main file and sidecars through metadata only. It never
    /// opens active storage, writes a maintenance audit record, or changes
    /// custody/backup artifacts. The shared control lock may be initialized.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when the private boundary is unsafe,
    /// any managed image is corrupt, or active-file metadata cannot be trusted.
    pub fn audit_latest_restore_readiness_for_config(
        config: &ChatRelayConfig,
    ) -> ChatRelayResult<ChatRelayRestoreReadinessReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, now_secs())?;
        let verified_backup_count = Self::verified_restore_backup_count(&inspection)?;
        let selected_backup_bytes = inspection
            .newest_backup
            .as_ref()
            .map(BackupArtifactSnapshot::size_bytes)
            .unwrap_or_default();
        let active = Self::inspect_active_restore_boundary(config)?;
        let blocker = if inspection.newest_backup.is_none() {
            Some("no_verified_backup")
        } else if active.sidecars_present {
            Some("active_sqlite_sidecars_present")
        } else {
            None
        };

        Ok(ChatRelayRestoreReadinessReceipt {
            ready: blocker.is_none(),
            verified_backup_count,
            selected_backup_bytes,
            active_database_present: active.present,
            active_database_bytes: active.size_bytes,
            active_sidecars_present: active.sidecars_present,
            blocker,
        })
    }

    /// Creates a short-lived authenticated plan for the newest verified image.
    ///
    /// The plan commits to private artifact identity, active database identity,
    /// aggregate recovery state, a random nonce, and a fixed ten-minute expiry.
    /// It does not copy, replace, remove, or open active custody. The plan is
    /// not sufficient authorization for a future restore operation.
    ///
    /// # Errors
    ///
    /// Returns a path-free storage error when no verified image exists, active
    /// `SQLite` sidecars remain, private metadata is unsafe, or the commitment
    /// cannot be produced.
    pub fn create_latest_restore_plan_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        Self::create_latest_restore_plan_at(config, node_secret, now_secs())
    }

    fn create_latest_restore_plan_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        issued_at: u64,
    ) -> ChatRelayResult<ChatRelayRestorePlanReceipt> {
        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, issued_at)?;
        let backup = inspection.newest_backup.as_ref().ok_or_else(|| {
            Self::backup_io_error(
                rusqlite::ffi::SQLITE_NOTFOUND,
                "relay restore plan requires a verified backup",
            )
        })?;
        let active = Self::inspect_active_restore_boundary(config)?;
        if active.sidecars_present {
            return Err(Self::backup_io_error(
                rusqlite::ffi::SQLITE_BUSY,
                "relay restore plan requires an inactive SQLite boundary",
            ));
        }

        let mut nonce = [0u8; CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let aggregate = RestorePlanAggregate {
            verified_backup_count: u64::try_from(Self::verified_restore_backup_count(&inspection)?)
                .map_err(|_| {
                    Self::backup_io_error(
                        rusqlite::ffi::SQLITE_FULL,
                        "relay restore-plan backup count exceeds wire format",
                    )
                })?,
            selected_backup_bytes: backup.size_bytes(),
            active_database_present: active.present,
            active_database_bytes: active.size_bytes,
        };
        HmacRestorePlanAuthenticator
            .issue(
                node_secret,
                issued_at,
                aggregate,
                Self::restore_plan_private_boundary(config, backup, &active),
                nonce,
            )
            .map_err(Self::map_restore_plan_error)
    }

    /// Re-verifies that an authenticated restore plan remains current.
    ///
    /// Verification repeats full backup integrity checks and active-boundary
    /// metadata inspection under the shared filesystem lock. Any expiry,
    /// tampering, backup rotation, config drift, or active database change
    /// invalidates the plan. This remains a read-only check and does not grant
    /// permission to restore. The shared lock is released on return; a future
    /// executor must re-verify and replace storage inside one uninterrupted
    /// lock scope rather than treating this method as TOCTOU-safe authority.
    ///
    /// # Errors
    ///
    /// Returns a generic path-free authentication error for an invalid,
    /// expired, or stale plan, and a storage error when the private boundary
    /// itself cannot be inspected safely.
    pub fn verify_latest_restore_plan_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
    ) -> ChatRelayResult<()> {
        Self::verify_latest_restore_plan_at(config, node_secret, plan, now_secs())
    }

    fn verify_latest_restore_plan_at(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        plan: &ChatRelayRestorePlanReceipt,
        now_unix_secs: u64,
    ) -> ChatRelayResult<()> {
        HmacRestorePlanAuthenticator
            .validate_public_contract(plan, now_unix_secs)
            .map_err(Self::map_restore_plan_error)?;

        let backup_directory = Self::private_backup_directory_for_config(config)?;
        let _filesystem_lock = Self::acquire_backup_filesystem_lock(&backup_directory)?;
        let inspection =
            Self::inspect_verified_backup_retention(config, &backup_directory, now_unix_secs)?;
        let backup = inspection
            .newest_backup
            .as_ref()
            .ok_or_else(|| Self::map_restore_plan_error(RestorePlanError::InvalidOrStale))?;
        let active = Self::inspect_active_restore_boundary(config)?;
        let current_count = u64::try_from(Self::verified_restore_backup_count(&inspection)?)
            .map_err(|_| Self::map_restore_plan_error(RestorePlanError::InvalidOrStale))?;
        if active.sidecars_present {
            return Err(Self::map_restore_plan_error(
                RestorePlanError::InvalidOrStale,
            ));
        }
        let aggregate = RestorePlanAggregate {
            verified_backup_count: current_count,
            selected_backup_bytes: backup.size_bytes(),
            active_database_present: active.present,
            active_database_bytes: active.size_bytes,
        };
        HmacRestorePlanAuthenticator
            .verify(
                node_secret,
                plan,
                now_unix_secs,
                aggregate,
                Self::restore_plan_private_boundary(config, backup, &active),
            )
            .map_err(Self::map_restore_plan_error)
    }

    /// Runs a host-local retention dry-run or explicitly-confirmed prune.
    ///
    /// Dry-run is the default request state. Deletion requires the exact
    /// confirmation phrase and an operator assertion that the node is stopped;
    /// every mutation is bracketed by a private HMAC-chained aggregate audit.
    pub fn prune_verified_backup_retention_for_config(
        config: &ChatRelayConfig,
        node_secret: &[u8; 32],
        request: &ChatRelayBackupPruneRequest,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        Self::prune_verified_backup_retention_at(config, node_secret, request, now_secs())
    }

    /// Runs the same prune contract through an initialized relay service.
    pub fn prune_verified_backup_retention(
        &self,
        request: &ChatRelayBackupPruneRequest,
    ) -> ChatRelayResult<ChatRelayBackupPruneReceipt> {
        let _operation = self.backup_operations.lock();
        Self::prune_verified_backup_retention_at(
            &self.config,
            &self.node_secret,
            request,
            now_secs(),
        )
    }
}

fn nonnegative_sqlite_value(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

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

/// Derives a stable 32-byte node secret from the node's Ed25519 private key.
pub fn derive_node_secret(ed25519_sk_bytes: &[u8; 32]) -> [u8; 32] {
    use hkdf::Hkdf;
    let hk = Hkdf::<Sha256>::new(Some(b"aeronyx-chat-relay-v1"), ed25519_sk_bytes);
    let mut okm = [0u8; 32];
    hk.expand(b"", &mut okm)
        .expect("HKDF expand with 32-byte output always succeeds");
    okm
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
#[path = "chat_relay_tests.rs"]
mod tests;
