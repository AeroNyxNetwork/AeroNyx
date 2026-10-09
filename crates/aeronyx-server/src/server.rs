// ============================================
// File: crates/aeronyx-server/src/server.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   [LINUX-TUN-TRAIT-SCOPE 2026-10-04 by Codex] Restore the Linux TUN
//   trait method scope shared by the extracted session/data-plane children.
//   [CHILD-SELECTION-GUARD 2026-10-02 by Codex] Require one named successful
//   child test; preserve intentional crash exit checks independently.
//   [CHAT-PULL-ROUTE-AUTHORITY 2026-10-01 by Codex] Preserves signed
//   cross-identity Pull queries without granting portable route authority.
//   [WITNESS-SAFETY-RUNTIME-SPLIT 2026-09-25 by Codex] Separates Chat Relay
//   custody witness renewal from MemChain commitment witness coordination.
//   [PEER-CACHE-RUNTIME 2026-09-25 by Codex] Extracts signed local peer
//   cache, witness-fenced persistence, bounded restore, and retry lifecycle.
//   [DATA-PLANE-RUNTIME 2026-09-25 by Codex] Extracts UDP/TUN session
//   ingress, authenticated teardown, snapshots, and keepalive lifecycles.
//   [CHAT-OUTBOUND-RUNTIME 2026-09-25 by Codex] Extracts outbound chat
//   route selection, signed receipts, and ambiguity-safe retry with tests.
//   [DISCOVERY-GOSSIP-RUNTIME 2026-09-25 by Codex] Moves gossip state,
//   scheduling, optional proof exchange, and synthetic route probes into a
//   focused child module while preserving shared client-delivery helpers.
//   [HANDSHAKE-PRE-ADMISSION-VERSION 2026-09-21 by Codex] Rejects unknown
//   ClientHello versions before limiter, voucher, crypto, or resource state.
//   [SERVER-DECOMPOSITION-PHASE1 2026-09-14 by Codex] Moves typed runtime
//   supervision and shutdown policy into a private focused child module.
//   [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex] Runs bounded
//   custody and source-journal retention before readiness and in one
//   supervised, non-overlapping, aggregate-only runtime maintenance task.
//   [PROTOCOL-V2-ADMISSION-HOTFIX 2026-09-13 by Codex] Removed unauthenticated
//   identity eviction from UDP ingress and finalizes only evictions returned
//   by the authenticated, lifecycle-fenced handshake admission transition.
//   [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] Separates bounded,
//   durable completed-response replay from fresh effect admission.
//   [BLIND-VAULT-MANAGEMENT-RUNTIME 2026-08-31 by Codex] Passes the live
//   Blind Vault service explicitly into management readiness reporting.
//   [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Bound startup,
//   gossip, and heartbeat replica advertisement to live admission readiness.
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
//  58. Bounds every discovery, relay-ACK, public-IP, bootstrap, and peer-cache
//      read so an untrusted HTTP peer or oversized local recovery file cannot
//      grow node memory without a protocol-defined ceiling.
//  59. Uses the shared API-layer bounded decoder for every peer acknowledgement
//      and applies route-specific public request ceilings before JSON parsing.
//  60. Reports aggregate durable chat queue usage/capacity, while removing
//      stable message, wallet, receiver, blob, and session identifiers from
//      relay logs so node operators cannot reconstruct a social graph.
//  61. Mounts the signature-authenticated encrypted blob API on loopback/VPN
//      client listeners only; the public node-peer listener remains unchanged.
//  62. Delivers bounded expiry notifications during authenticated chat pulls,
//      retains failed writes for retry, and treats UDP write failure as an
//      offline delivery failure instead of a successful route.
//  63. Schedules the configured ChatRelay TTL cleanup on Tokio's blocking
//      pool, exposes aggregate execution evidence, and skips missed-tick bursts.
//  64. Adds backward-compatible ChatPullV2 dispatch with a signed bounded
//      opaque cursor and stable monotonic mailbox snapshot pagination.
//  65. Reports checkpoint evidence above a follower's recovered local tip as
//      deferred, allowing block sync to run without presenting old evidence
//      as current convergence or divergence.
//  66. Uses a validated configurable follower pages-per-round budget so
//      catch-up remains bounded, restartable, and operable on low-I/O nodes.
//  67. Exercises the coordinator startup policy with a real signed divergent
//      witness checkpoint while proving the local canonical chain is immutable.
//  68. Retains trusted-witness same-height conflicts as durable incidents and
//      blocks coordinator startup before listeners even after later convergence.
//  69. Retains an operator-pinned witness's divergent shared-prefix proof as a
//      sticky incident and atomically halts local commitment production.
//  70. Exchanges fully audited checkpoint certificates only after startup;
//      imported bundles cannot replace the live pinned-witness startup gate.
//  71. Requires one fresh, signed coordinator lease from every configured
//      witness before startup and renewal may authorize local block production.
//  72. Releases the exact process lease from every witness after SIGINT/SIGTERM
//      so planned restarts hand over immediately without weakening crash TTLs.
//  73. Reports monotonic lease authority, consecutive renewal failures, and
//      recovery evidence so witness partitions remain observable and fail closed.
//  74. Reports witness-certificate coverage and uncertified block lag without
//      exposing hashes, witness identities, signatures, or claiming finality.
//  75. Triggers witness reconciliation immediately after a canonical tip
//      advance through a bounded local channel; the low-frequency schedule
//      remains the recovery path and block production never waits on witnesses.
//  76. Warms at most three direct blind-relay candidates concurrently on the
//      first gossip round, then returns to one candidate per recovery cadence.
//  77. Persists only fresh descriptor-bound routeability success evidence in
//      the local peer cache, restores it fail-closed, and still revalidates all
//      startup candidates with bounded direct probes.
//  78. Joins peer-cache persistence and discovery gossip tasks during graceful
//      shutdown so signed route evidence is durably fsynced before process exit.
//  79. Signs and restores an independent bounded two-hop synthetic proof window
//      only after current descriptor-bound routeability rebuilds the path.
//  80. Records proof-cache authentication and successful signed persistence as
//      structured admission evidence instead of parsing bootstrap log detail.
//  81. Uses fresh terminal-signed delivery receipts to unlock authenticated
//      App two-hop onion routing while retaining the legacy encrypted relay
//      path for mixed-version peers and never classifying probes as user work.
//  82. Persists only a signed aggregate verified-client delivery count and
//      latest timestamp, then requires fresh current receipt-capable peers
//      before restart recovery may report authenticated two-hop readiness.
//  83. Binds aggregate delivery evidence to a monotonic cache generation and
//      independent signed local anchor, rejecting older valid cache snapshots
//      without coupling descriptor, routeability, or proof recovery.
//  84. Optionally witnesses the opaque digest of each signed recovery anchor
//      on pinned peer nodes. The v2 anchor also commits independently to the
//      two-hop and three-hop proof sections, serializing generations so
//      whole-host rollback and missed-generation gaps fail closed without
//      exporting counts, timestamps, routes, messages, payloads, or clients.
//  85. Optionally opens a producer-pinned SQLite Directory Chain journal,
//      audits every signed block before listeners start, and transactionally
//      reconciles authenticated descriptor commitments during runtime/shutdown.
//  86. Shares producer-isolated replica synchronization observations with a
//      privacy-tiered status route: aggregate on the public listener and
//      truncated producer fingerprints on local/VPN operator listeners.
//  87. Performs bounded multi-page replica catch-up while reserving the
//      worst-case per-page request cost beneath the peer rate limit.
//  88. Reconciles the validated Directory witness pins and threshold into a
//      node-identity-signed, hash-linked local policy epoch after full replica
//      audit and before synchronization or listeners can start.
//  89. Externally anchors the opaque current Directory witness-policy head at
//      its exact pinned witnesses and audits monotonic remote heads plus signed
//      local receipts before listeners start, without exporting policy members.
//  90. Mounts a rate-limited local/VPN-only Directory Mirror carrier smoke
//      route that verifies retained signed evidence without importing it.
//  91. [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] Shares one Directory
//      runtime between witness-carrier peer routes and status routes, exposing
//      only mutually exclusive process aggregates and never request identity.
//  92. [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] Adds one bounded,
//      replica-audited descriptor inclusion proof to outbound gossip while
//      always retaining the legacy self announcement for rolling upgrades.
//  93. [DIRECTORY-GOSSIP-NEGOTIATION 2026-07-27 by Codex] Negotiates the
//      optional proof frame through a bounded public summary before sending it,
//      while treating missing/invalid hints as legacy-only rather than failure.
//  94. [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] Classifies optional
//      proof failures into privacy-safe buckets, records per-round convergence,
//      and tries at most one independently audited fallback proof before the
//      mandatory legacy descriptor exchange.
//  95. [GOSSIP-OUTCOME-INTEGRITY 2026-07-28 by Codex] Separates optional proof
//      transmission from mandatory legacy exchange and preserves the proof
//      result when a later descriptor or snapshot request fails.
//  96. [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] Bounds concurrent peer
//      fan-out and total per-peer work, reserves legacy compatibility time from
//      optional proof work, and replaces free-form internal failures with
//      typed privacy-safe buckets.
//  98. [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] Generates alternates
//      across producer namespaces and skips a known receiving producer's own
//      anchor without exposing peer-specific telemetry.
//  97. [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] Publishes proof gossip
//      only from audited blocks old enough for independent replica convergence;
//      exact-anchor admission remains unchanged and fail-closed.
//  99. [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] Uses a receiving
//      producer hint only while one canonical gossip URL maps to exactly one
//      verified node identity; endpoint collisions fall back without guessing.
// 100. [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Separates Directory
//      synchronization and operator-smoke HTTP profiles so production replica
//      failover keeps its historical 10-second request deadline while the
//      bounded operator diagnostic retains its 12-second budget.
// 101. [STARTUP-READINESS 2026-07-29 by Codex] Binds every required node API
//      listener before startup can succeed and reports READY/STOPPING through
//      the systemd notify socket without adding a new runtime dependency.
// 102. [RUNTIME-SUPERVISION 2026-07-29 by Codex] Supervises the node, VPN, and
//      public API listeners as one required runtime group. Unexpected listener
//      loss now performs graceful cleanup and exits non-zero for systemd
//      recovery instead of leaving a false-positive active process.
// 103. [TASK-SHUTDOWN 2026-07-29 by Codex] Joins long-lived runtime tasks
//      concurrently under explicit grace periods, aborts timed-out tasks, and
//      verifies cancellation instead of silently detaching their JoinHandles.
// 104. [FOLLOWER-CERTIFICATE-SYNC 2026-07-29 by Codex] Replicates an audited
//      checkpoint certificate from the pinned coordinator only after follower
//      chain convergence, without making evidence availability a chain gate.
// 105. [RUNTIME-IDENTITY-POLICY 2026-07-29 by Codex] Fails startup before any
//      mutable service or listener when a coordinator/follower trust policy
//      treats the node's own identity as an external authority.
// 106. [FOLLOWER-CERTIFICATE-READINESS 2026-07-29 by Codex] Publishes exact
//      current-policy readiness after cryptographic validation without
//      exposing witness identities or certificate material.
// 107. [FOLLOWER-CERTIFICATE-TIP-BINDING 2026-07-29 by Codex] Prevents a
//      certificate evaluated for an older audited tip from remaining ready
//      while follower synchronization advances the local chain.
// 108. [MANAGEMENT-RUNTIME-OWNERSHIP 2026-07-30 by Codex] Returns command,
//      heartbeat, and session-reporting task handles to the main runtime so
//      unexpected exits trigger recovery and shutdown confirms cancellation.
// 109. [PRE-READY-RUNTIME-GATE 2026-07-30 by Codex] Refuses systemd READY
//      when a required worker already failed during startup or every required
//      supervisor disappeared before the readiness transition.
// 110. [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Treats configured
//      Directory replica initialization and task liveness as required runtime
//      state, preventing READY after mirror/synchronization silently vanished.
// 111. [STARTUP-TASK-REGISTRY 2026-07-30 by Codex] Registers every spawned
//      process task immediately so any later startup error aborts owned work
//      instead of detaching JoinHandles from the failed startup transaction.
// 112. [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] Classifies Tokio task join
//      failures into fixed typed reasons so panic payloads cannot cross the
//      process-health, systemd status, or structured shutdown-log boundary.
// 113. [TWO-HOP-PROBE-OUTCOME 2026-07-31 by Codex] Separates probe execution,
//      mixed-version route acceptance, and terminal-signed delivery evidence
//      so operator smoke success cannot overstate a failed or legacy-only path.
// 114. [CLIENT-ONION-PARTIAL-SUCCESS 2026-07-31 by Codex] Separates verified
//      terminal delivery from full replica completion so one successful blind
//      route never triggers a privacy-widening legacy direct-relay fallback.
// 115. [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex] Runs a bounded, fully
//      onion-wrapped entry -> middle -> middle -> terminal delivery probe only
//      after two-hop receipt verification, keeping route members and payloads
//      out of status while exposing independent three-hop runtime evidence.
// 116. [BOUNDED-DISCOVERY-HEARTBEAT 2026-08-02 by Codex] Projects the local
//      PeerStore into a stable aggregate heartbeat contract and exports signed
//      descriptors in a fixed-size batch so growing local diagnostics cannot
//      make the backend discard every discovery health field.
// 117. [THREE-HOP-FEATURE-NEGOTIATION 2026-08-02 by Codex] Negotiates
//      multihop terminal-receipt support before selecting the first middle of
//      a synthetic three-hop route, preventing rolling-upgrade incompatibility
//      from becoming false route-health failure evidence.
// 118. [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Requires receipt v2 and a
//      purpose-separated opaque payload commitment before synthetic or real
//      App onion delivery becomes verified; legacy receipts remain forwardable.
// 119. [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Separates the
//      unsigned v2 bootstrap hint from process-local verified route authority;
//      three-hop probes require fresh v2 evidence for every selected relay.
// 120. [RECEIPT-EVIDENCE-SURFACE-BINDING 2026-08-10 by Codex] Records v2
//      receipt authority against the exact signed descriptors used by each
//      verified route so concurrent endpoint/KEM rotation fails closed.
// 121. [ROUTE-SUCCESS-SURFACE-BINDING 2026-08-10 by Codex] Binds successful
//      probe and encrypted-forward observations to the signed descriptor used
//      for the request so stale endpoints cannot authorize a rotated route.
// 122. [CLIENT-DELIVERY-ATOMIC-ROUTE-EVIDENCE 2026-08-11 by Codex] Validates
//      receipt freshness at response time and commits real two-hop delivery
//      only while both selected signed route surfaces remain current together.
// 123. [DIRECTORY-SHUTDOWN-DURABILITY 2026-08-12 by Codex] Gives the
//      non-cancellable Directory Chain blocking reconciliation enough bounded
//      shutdown time to finish below the service manager stop deadline.
// 124. [BACKGROUND-SHUTDOWN-COOPERATION 2026-08-12 by Codex] Makes initial
//      traffic waits interruptible and adds shutdown checkpoints around each
//      bounded discovery gossip stage instead of relying on forced aborts.
// 125. [KEEPALIVE-SHUTDOWN-COOPERATION 2026-08-12 by Codex] Reuses the
//      interruptible startup delay for VPN keepalive probes so an idle node
//      never waits out the fixed thirty-second warm-up during service stop.
// 126. [PEER-CACHE-DIRTY-RECOVERY 2026-08-12 by Codex] Restores the pending
//      delivery-evidence dirty state when atomic peer-cache persistence fails,
//      preventing one failed write from acknowledging data it never stored.
// 127. [PEER-CACHE-RETRY-STATE 2026-08-12 by Codex] Gives cache persistence a
//      typed persisted/deferred result and caps retry pressure with exponential
//      backoff instead of turning a coalesced notification into a hot loop.
// 128. [FAIL-CLOSED-SHUTDOWN-SIGNALS 2026-08-12 by Codex] Derives the VPN API
//      listener from the configured gateway and converts signal-registration
//      failures into supervised shutdown instead of panicking past cleanup.
// 129. [MINER-STARTUP-ERROR 2026-08-12 by Codex] Propagates SaaS miner stub
//      resource failures through the normal typed startup transaction instead
//      of panicking after required node resources have already been acquired.
// 130. [MANAGEMENT-CLIENT-STARTUP 2026-08-12 by Codex] Treats management HTTP
//      client construction as a typed startup boundary before spawning any
//      management worker, preventing a panic or partial control plane.
// 131. [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Installs the
//      operator-validated immutable commitment authority root before startup
//      audit so readiness cannot precede exact-height proposer verification.
// 132. [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Recovers exact-next
//      authority proofs through bounded operator pins while keeping transport
//      identity independent from the dual-signed coordinator transition.
// 133. [AUTHORITY-CARRIER-POLICY 2026-08-14 by Codex] Gives handover-proof
//      transport a dedicated follower pin set, with an explicit compatibility
//      fallback that does not merge transport and witness authorization.
// 134. [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Resolves one typed
//      authority carrier policy and propagates configured follower
//      initialization failures through the startup transaction.
// 135. [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] Makes explicit
//      Chat Relay enablement fail startup when its durable service cannot be
//      initialized, using only a privacy-safe reason bucket.
// 136. [CHAT-SESSION-SENDER-BINDING 2026-08-15 by Codex] Requires every
//      client-tunnel ChatEnvelope sender to match the authenticated VPN
//      session identity before dedupe, route announcement, or peer relay.
// 137. [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Retries one exact
//      target-bound v3 request after transport ambiguity or HTTP 425, relying
//      on the target's bounded commitment-keyed custody ACK cache.
// 138. [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] Treats bounded ACK-body
//      truncation as transport ambiguity and retries the exact v3 request,
//      while deterministic protocol, size, and receipt failures remain final.
// 139. [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] Publishes only
//      aggregate direct-relay retry recovery, exhaustion, and deterministic
//      failure evidence through the existing relay health contract.
// 140. [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] Proves in a fresh
//      process that post-install checkpoint table loss rejects Chat Relay
//      activation instead of resetting the outage circuit to closed.
// 141. [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] Optionally requires
//      fresh exact-anchor signed receipts from independent operator pins before
//      PeerStore bootstrap or listeners, with no startup network transmission.
// 142. [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Consumes one
//      typed SQLite snapshot for both vault audit telemetry and startup policy,
//      preventing CLI/runtime interpretation or snapshot drift.
// 143. [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Optionally
//      re-audits exact-anchor custody evidence from the local durable vault
//      while running and asks the required-task supervisor to stop the process
//      when strict readiness is lost, without collecting over the network.
// 144. [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] Derives the exact aggregate
//      lifetime of the accepted receipt threshold and warns locally before the
//      strict runtime gate reaches its fail-closed boundary.
// 145. [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] Edge-triggers local
//      renewal warnings, suppresses duplicate warning spam for the same quorum
//      horizon, and records recovery after explicit evidence refresh.
// 146. [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Optionally renews
//      an expiring exact-anchor receipt threshold through the already pinned,
//      bounded, durable witness transport without expanding discovery into
//      authority or changing the default local-only runtime behavior.
// 147. [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Converts direct and
//      authenticated-onion relay diagnostics into validated aggregate reason
//      values before heartbeat state can observe them.
// 148. [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Signs active,
//      descriptor-bound route quarantine into peer-cache v2 and restores it
//      before route admission so restart cannot revive a recently failed peer.
// 149. [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Extends the signed
//      monotonic recovery anchor to v3 with an opaque commitment to the exact
//      routeability/quarantine section, rejecting old or unanchored route
//      state while preserving independently verified peer descriptors.
// 150. [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] Applies adverse
//      external recovery-anchor evidence to the complete v3 restart-readiness
//      bundle before listeners start, retaining descriptors while revoking
//      route, quarantine, proof, and aggregate delivery recovery.
// 151. [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] Requires the
//      local recovery cache and witnessed anchor to represent the exact same
//      generation, closing the interrupted cache-before-anchor write window.
// 152. [RECOVERY-ANCHOR-HEARTBEAT 2026-08-21 by Codex] Projects the shared
//      exact-generation recovery-anchor status into the signed management
//      heartbeat through one bounded, testable aggregate builder.
// 153. [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex] Dispatches chat
//      runtime frames independently from optional MemChain persistence while
//      rejecting storage-owned frames through a typed privacy-safe gate.
// 154. [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex] Opens and
//      cleans explicitly enabled custody storage before readiness, then
//      supervises non-overlapping custody/source retention with typed retry
//      and fatal failure behavior; disabled storage has no path or task effect.
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
//   - Witness-carrier counters are transport telemetry only. They must not
//     become peer reputation, authority weight, route ranking, or consensus.
//   - Directory-authenticated gossip proof announcements are additive public
//     evidence during mixed-version rollout. A rejected proof must never
//     suppress the legacy self announcement, and proof carriers gain no
//     producer, witness, routing, voting, or consensus authority.
//   - Directory proof fallback is bounded to two frames per capable peer per
//     round. Only an exact-evidence HTTP 422 may advance to the second audited
//     candidate; replica, rate-limit, protocol, and transport failures stop
//     proof fallback while the legacy exchange still runs.
//   - Optional proof outcomes and mandatory legacy exchange outcomes have
//     independent failure domains. Never discard an accepted/rejected proof
//     observation merely because a later descriptor or snapshot step failed.
//   - Outbound gossip concurrency and per-peer lifetime are bounded. Optional
//     proof work may consume at most one third of a peer deadline; it must
//     never starve the mandatory descriptor/snapshot compatibility exchange.
//   - New-tip notifications are bounded process-local hints. Reconciliation
//     must always read and verify the current audited storage tip, and periodic
//     polling must remain available when notifications are coalesced or closed.
//   - Checkpoint freshness comes only from audited, currently applicable
//     durable signed evidence. Deferred historical frames, failed attempts,
//     and inbound served requests cannot refresh it.
//   - Witness round states are operator evidence only. Never use their counts
//     as votes, quorum, finality, leader election, or fork choice.
//   - Commitment coordinators must confirm SQLite FULL-or-stronger durability
//     before startup audit and before any block producer can start.
//   - The signed commitment tip anchor detects an older/replaced SQLite chain
//     only while the host-side anchor remains current. It does not detect a
//     whole-host snapshot rollback and is not consensus, quorum, or finality.
//   - The checkpoint certificate anchor has the same local-only boundary. It
//     must run after the full evidence-vault audit and before networking.
//   - External startup witnesses are explicit identity trust pins. Discovery
//     may rotate their signed endpoints, but unpinned peers stay evidence-only.
//     A witness pin must never equal the local runtime identity.
//   - Delivery-cache witnesses are not Memory Chain checkpoint witnesses. They
//     retain one aggregate-only opaque high-water row per node and must never
//     become consensus, finality, routing authority, or user traffic telemetry.
//   - The minimum verified witness count is an operator startup threshold over
//     distinct pins. It is not consensus, quorum, finality, or fork choice.
//   - Custody-receipt strict startup is local-only and default-off. It must
//     hold the maintenance anchor guard, audit the complete durable vault, and
//     reject future-dated evidence beyond the fixed clock-skew allowance.
//   - Custody readiness counters are not an open-coded API. Startup and tools
//     must use the typed atomic snapshot and reject inconsistent aggregates.
//   - Runtime custody strictness is separately default-off and requires the
//     startup gate. Its audit is local-only; never add implicit witness
//     collection, discovery authority, or identity-bearing failure output.
//   - Certificate exchange is post-startup evidence transport only. Never use
//     an imported historical bundle to satisfy the live startup witness gate.
//   - Follower certificate synchronization runs only after signed convergence.
//     Its source is transport, not authority; local witness pins and threshold
//     remain mandatory, and mixed-version absence must not undo a verified tip.
//   - Follower certificate readiness may become `ready` only after the exact
//     current tip, current pins, and current threshold pass local validation.
//     The management projection must preserve the evaluated tip height so
//     operators can distinguish current coverage from scheduler lag.
//   - Coordinator leases are a short-lived duplicate-writer safety control,
//     not consensus, leader election, finality, or fork choice. A partial
//     witness round must never extend the local production deadline.
//   - Graceful lease release must run only after an in-flight renewal finishes.
//     Never cancel a renewal request and race a release against its late grant.
//   - Outbound peer HTTP responses must pass the shared bounded readers in
//     api/mod.rs; discovery recovery files use this file's bounded file reader.
//     Never reintroduce response.json(), response.bytes(), response.text(), or
//     tokio::fs::read() on these paths.
//   - encrypted_message_counter is aggregate only and never stores payload,
//     destination, DNS, URL, voucher, wallet, or client public IP details.
//   - Chat relay logs and heartbeat capacity telemetry are aggregate-only.
//     Never add message IDs, wallet prefixes, sender/receiver keys, blob IDs,
//     session IDs, payload data, or endpoint values to those surfaces.
//   - Chat expiry notifications are bounded durable control events. Mark them
//     pushed only after a successful transport write and preserve `has_more`
//     whenever a page remains or delivery/marking fails.
//   - ChatRelay SQLite cleanup is synchronous and must remain inside
//     spawn_blocking. The dedicated timer uses the configured interval and
//     MissedTickBehavior::Skip to prevent catch-up storms after a slow cycle.
//   - Voice relay logs follow the same blind-node boundary: no wallet target,
//     virtual IP, session ID, endpoint, or raw cryptographic error values.
//   - dns_proxy forwards opaque DNS UDP payloads only; it does not parse,
//     log, store, or report queried domains.
//   - If vpn.dns_proxy_enabled=false, an external gateway DNS listener such as
//     systemd-resolved must own gateway_ip:53. The health endpoint verifies
//     that listener independently.
//   - Tip announcement retries are advisory process-local work. Never let them
//     gate block production or expose peer identity/endpoint data in telemetry.
//   - A newer audited commitment tip may cancel an older in-flight announcement
//     round. Cancellation is delivery prioritization only: it must not mutate
//     the canonical chain, witness evidence, lease state, or production gates.
//   - A middle-hop ACK is not terminal-authenticated routeability evidence.
//     Startup warm-up must probe each candidate directly and remain bounded;
//     never mark a terminal healthy solely because a middle claims forwarding.
//   - Real client-delivery evidence must use PeerStore's atomic two-hop route
//     transition. Never publish the aggregate count after separately updating
//     middle and terminal capability or route-health state.
//   - Public Directory Replica status must remain aggregate-only. Never expose
//     full producer identities, endpoints, descriptors, routes, or user data.
//   - Multi-page Directory Replica catch-up must reserve the worst-case next
//     page request cost before continuing; never rely on average block size.
//   - Directory Replica producers synchronize through the dedicated bounded
//     coordinator. Preserve independent producer budgets and the concurrency
//     cap so one slow peer cannot block or amplify the complete pinned set.
//   - Directory Replica schema v2 persists bounded retry state. A successful
//     authenticated import must clear that state in the same SQLite transaction.
//   - Directory Replica schema v7 anchors signed witness-policy history in
//     metadata. Policy epochs are local operator evidence configuration, not a
//     validator set, vote, quorum, fork choice, consensus, or finality.
//   - Directory Replica schema v8 records only opaque signed policy-head
//     anchors. First observation is TOFU; rollback, same-epoch conflict, and
//     forward history gaps fail closed and never mutate the accepted head.
//   - [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] Required production
//     discovery routers receive the local node id as private runtime context so
//     public onion pools can exclude candidates collocated with the entry.
//   - [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] Publish fine-grained wire
//     features only through backward-compatible signed descriptor metadata.
//     Do not add a closed `NodeCapability` variant for response framing.
//   - [SIGNED-RECEIPT-NEGOTIATION 2026-08-11 by Codex] Prefer the signed
//     purpose-bound receipt probe claim over the legacy unsigned summary so a
//     network intermediary cannot suppress v2 probing with a false negative.
//   - [DIRECTORY-SHUTDOWN-DURABILITY 2026-08-12 by Codex] Keep the Directory
//     persistence grace below systemd `TimeoutStopSec`. Its blocking SQLite
//     reconciliation cannot be cancelled safely after it has started.
//   - [BACKGROUND-SHUTDOWN-COOPERATION 2026-08-12 by Codex] Every long initial
//     delay and multi-stage gossip round must observe shutdown between bounded
//     operations; adding another await requires adding the same checkpoint.
//   - [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Install the immutable
//     authority root before the first chain audit. Never derive it from mutable
//     storage, log its identity, or bypass dual-signed coordinator handover.
//   - [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Operator-pinned
//     handover carriers transport exact dual-signed proofs only. They never
//     gain coordinator, witness, voting, fork-choice, or consensus authority.
//   - [AUTHORITY-CARRIER-POLICY 2026-08-14 by Codex] Use the effective
//     authority carrier set only for proof transport. Checkpoint block and
//     certificate recovery must continue using the independent witness set.
//   - [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] A configured
//     commitment follower must return a supervised task or fail startup.
//     Never convert malformed coordinator/carrier state into `None`.
//   - [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] Never turn an
//     explicitly enabled Chat Relay initialization failure into `None`.
//     Capability readiness and the configured service contract must agree.
//   - [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] HTTP success is not
//     durable custody. When the signed target descriptor advertises receipt
//     v2, require a fresh target signature bound to the exact authenticated
//     request before route health records success. Legacy negotiation remains
//     descriptor-driven and must never be inferred from an HTTP response.
//   - [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Build the signed v3
//     request separately for every selected peer because target identity is
//     part of its authorization. Never reuse one target-bound request across
//     fanout peers or infer v3 support from an HTTP response.
//   - [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Retry only the exact
//     signed v3 request and only once. v1/v2 do not advertise target-bound
//     replay semantics, so they must retain their historical single attempt.
//   - [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] A v3 attempt owns request
//     send, bounded ACK read, and negotiated ACK verification as one operation.
//     Only ambiguous transport/body failures and HTTP 425 may cross the retry
//     edge; explicit rejection, oversized ACKs, or invalid receipts must not.
//   - [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] A failed aggregate v3 SLO
//     opens a source-blind circuit. While cooling, direct fallback fails closed
//     into local pending storage and must never downgrade to v2/v1. Half-open
//     recovery admits one generation-bound v3 delivery at a time.
//   - [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] The anonymous schema
//     marker is part of relay startup integrity. If it proves prior checkpoint
//     installation but the checkpoint table is absent, startup must fail.
//   - [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Route-local
//     diagnostics and logs may use reviewed stable buckets, but heartbeat must
//     receive the validated reason type. Never pass through a raw reqwest
//     error, endpoint, response body, request id, message id, or payload text.
//   - [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Cache v2 binds positive
//     routeability and active quarantine under one node signature. Keep v1
//     readable for rolling upgrades, but never infer missing quarantine state
//     from a v1 cache or persist failure reasons, endpoints, routes, or payloads.
//   - [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Only recovery-anchor
//     v3 may authorize persisted routeability/quarantine after restart. v1/v2
//     anchors preserve compatibility for their historical evidence sections,
//     but route readiness must be rebuilt by fresh probes.
//   - [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] The historical
//     verified_delivery_witness wire/config name remains backward compatible,
//     but a witnessed v3 anchor covers all committed restart-readiness state.
//     Any signed adverse result must revoke the complete bundle at startup.
//   - [CHAT-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-23 by Codex] Authenticate
//     before entering the private single-flight lane. Hold its guard through
//     cache lookup, onion delivery, entry custody, and response insertion so
//     concurrent exact retries cannot create contradictory delivery results.
//   - [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] A replacement
//     process may repeat only exact idempotent local entry custody. It must not
//     re-announce a wallet route, select another onion path, or claim terminal
//     evidence that was not durably retained before the predecessor stopped.
//   - [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] Recovery telemetry
//     is aggregate and process-local. Never attach request, message, wallet,
//     route, peer, receipt, endpoint, ciphertext, or payload dimensions.
//
// Last Modified:
//   [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Suppressed the
//     signed replica capability whenever policy, issuer, logical capacity, or
//     physical disk readiness cannot safely admit a new anonymous lease.
//   [BLIND-VAULT-LEASE-INVENTORY 2026-08-28 by Codex] Advertised private,
//     terminal-signed encrypted-object inventory commitments.
//   [BLIND-VAULT-LEASE-STATUS 2026-08-28 by Codex] Advertised private,
//     administration-authorized terminal-signed lease status observations.
//   [BLIND-VAULT-LEASE-RENEWAL 2026-08-28 by Codex] Advertised blind-authorized
//     lease renewal and included bounded renewal-marker cleanup in aggregate
//     maintenance telemetry.
//   [BLIND-VAULT-LEASE-RETIRE 2026-08-28 by Codex] Advertised exact anonymous
//     lease-retirement support and included bounded retry-marker cleanup in
//     aggregate-only maintenance telemetry.
//   [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] Classified each
//     owner-fenced recovery as completed, failed, or deferred after combining
//     entry-custody and exact-response persistence outcomes.
//   [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Routed abandoned
//     verified submissions through owner-fenced custody-only recovery while
//     preserving the existing fail-closed pending and capacity boundaries.
//   [CHAT-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-23 by Codex] Replayed the first
//     request-bound response for exact retries and rejected request-id reuse
//     with another envelope before any route or durable-state mutation.
//   [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] Prevented a
//     valid older anchor from authorizing restored state from a newer cache
//     generation after an interrupted two-file durability update.
//   [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] Closed the whole-host
//     rollback gap by applying adverse external witness evidence to route,
//     quarantine, two/three-hop proof, and delivery readiness before listeners.
//   [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Bound signed route state
//     to recovery-anchor v3 and rejected stale, missing, invalid, conflicting,
//     or legacy-unanchored route evidence without discarding descriptors.
//   [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Added signed active-route
//     quarantine recovery and prompt persistence for security-state changes.
//   [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Added a typed,
//     allowlisted export boundary for direct and authenticated-onion failures.
//   [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex] Added one privacy-safe,
//     process-lifetime custody runtime snapshot to Chat Relay heartbeat status.
//   [CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex] Separated strict local
//     audits from identity-jittered witness retry backoff and retained the
//     post-round durable audit even when collection reports partial failure.
//   [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Added explicit opt-in
//     pre-expiry renewal inside the supervised strict custody runtime task.
//   [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] Added a process-local
//     renewal warning/recovery state machine without networking or new APIs.
//   [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] Added local-only exact quorum
//     expiry telemetry and a bounded pre-expiry operator warning window.
//   [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Added a default-off,
//     local-only runtime re-audit with supervised fail-closed process recovery.
//   [CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex] Added a local aggregate-only
//     startup eligibility plan without enabling witness network transmission.
//   [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Installed independent
//     custody requester pins and advertised the fail-closed peer route.
//   [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] Added a two-process
//     crash drill proving destructive checkpoint table loss cannot reactivate
//     direct relay with an invented closed circuit.
//   [DIRECT-RELAY-HALF-OPEN-PROGRESS 2026-08-15 by Codex] Added a
//     three-process drill proving one committed half-open success survives a
//     crash, requires one new serial probe, and durably closes the circuit.
//   [DIRECT-RELAY-HALF-OPEN-CRASH 2026-08-15 by Codex] Extended the
//     multi-process restart drill through an abruptly interrupted half-open
//     recovery lease, which must be counted failed and reopened on restart.
//   [DIRECT-RELAY-CRASH-DRILL 2026-08-15 by Codex] Added a test-only,
//     multi-process crash/restart drill proving encrypted mailbox custody and
//     no-downgrade circuit state survive abrupt source-node termination.
//   [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Wired source-blind open and
//     half-open admission into direct relay without protocol downgrade.
//   [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] Extended exact v3 retries
//     through bounded ACK-body reads and negotiated target ACK verification.
//   [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Added one safe exact
//     v3 retry for transport ambiguity and in-flight custody publication.
//   [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Added signed target
//     binding and descriptor-negotiated v3 direct relay transport.
//   [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Added target-signed direct
//     relay durable-custody evidence with explicit rolling compatibility.
//   [CHAT-SESSION-SENDER-BINDING 2026-08-15 by Codex] Bound every accepted
//     client ChatRelay sender to its authenticated VPN session identity.
//   [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] Made configured Chat
//     Relay storage initialization a privacy-safe startup gate.
//   [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Made follower task
//     construction fail closed and consume one validated carrier policy.
//   [AUTHORITY-CARRIER-POLICY 2026-08-14 by Codex] Separated authority-proof
//     transport pins from checkpoint witness policy with legacy fallback.
//   [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Added direct-first,
//     bounded proof recovery with an isolated process-lifetime carrier circuit.
//   [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Bound startup readiness
//     to the configured commitment authority root and proposer history audit.
//   v2.8.84-ChatSessionSenderBinding - Bound client ChatRelay senders to the
//     authenticated VPN session before route mutation or relay fan-out.
//   v2.8.83-ManagementClientStartup - Made management HTTP client creation
//     fail-closed through ServerError before worker startup.
//   v2.8.82-MinerStartupError - Made SaaS miner scheduler construction
//     fail-closed through ServerError instead of process panic.
//   v2.8.81-FailClosedShutdownSignals - Removed production panic paths from
//     API address construction and shutdown-signal supervision.
//   v2.8.80-PeerCacheRetryState - Added typed persistence outcomes and bounded
//     retry scheduling for failed or witness-deferred cache generations.
//   v2.8.79-PeerCacheDirtyRecovery - Kept verified delivery evidence pending
//     after peer-cache write failures and added a regression test.
//   v2.8.78-KeepaliveShutdownCooperation - Made the VPN keepalive warm-up
//     observe shutdown instead of requiring supervisor cancellation.
//   v2.8.77-BackgroundShutdownCooperation - Made traffic startup delay
//     interruptible and bounded discovery shutdown at stage boundaries.
//   v2.8.76-DirectoryShutdownDurability - Added a bounded, service-manager-safe
//     grace window for the final signed Directory Chain reconciliation.
//   v2.8.75-SignedReceiptNegotiation - Bound purpose-receipt probe eligibility
//     to signed descriptors while retaining the legacy summary fallback.
//   v2.8.74-SignedProtocolFeatures - Advertised authenticated hop-local failure
//     receipts without changing descriptor schema or capability discriminants;
//     registered the dormant self-descriptor privacy regression test.
//   v2.8.73-ClientDeliveryAtomicRouteEvidence - Bound live terminal receipts,
//     route success, and capability evidence to one current signed path state.
//   v2.8.72-RouteSuccessSurfaceBinding - Bound probe and live forward success
//     to the selected signed route surface instead of node identity alone.
//   v2.8.71-ReceiptEvidenceSurfaceBinding - Bound verified receipt evidence to
//     the selected signed route descriptors instead of node identity alone.
//   v2.8.70-PurposeBoundReceiptNegotiation - Prevented v1-only relay framing
//     from authorizing or being penalized by v2 multi-hop delivery probes.
//   v2.8.69-PurposeBoundReceipt - Prevented cross-workload relay-proof reuse
//     without exposing the route-purpose label to intermediary nodes.
//   v2.8.68-BlindVaultReplicaCapability - Added rollout-gated, runtime-honest
//     signed advertisement for admitted anonymous ciphertext replicas.
//   v2.8.67-RouteDomainAttestedSelection - Applies the audited local
//     attestor/quorum policy to multi-hop probes and live relay selection,
//     while preserving legacy direct single-hop routing.
//   v2.8.66-RouteDomainAttestorPolicyHistory - Reconciles the canonical local
//     attestor/quorum policy before listeners and logs aggregate evidence only.
//   v2.8.65-OnionEntryAntiAffinity - Injected local entry identity into both
//     required discovery listeners without serializing it in public status.
//   v2.8.64-PathProofRollbackAnchor - Bound two-hop and three-hop proof
//   digests independently into the monotonic local/external recovery anchor.
//   v2.8.63-ThreeHopSignedRecovery - Added independent signed three-hop proof
//     snapshot persistence, strict startup revalidation, and aggregate status.
//   v2.8.62-ThreeHopFeatureNegotiation - Added fail-closed mixed-version
//     selection for the first middle of three-hop runtime probes.
//   v2.8.61-BoundedDiscoveryHeartbeat - Added a bounded aggregate PeerStore
//                                      heartbeat projection and record batch
//   v2.8.60-ThreeHopRuntimeProof - Added bounded live three-hop onion delivery
//     verification with terminal-signed receipts and isolated proof history.
//   v2.8.59-FollowerEffectiveReadiness - Added fail-closed composite follower
//     readiness to local status and signed heartbeat telemetry.
//   v2.8.58-FollowerCertificateRetry - Replaced the follower round tuple with a
//     typed outcome and added bounded retry scheduling for deferred certificates.
//   v2.8.57-CertificatePersistenceTruth - Kept verified-but-unpersisted follower
//     evidence separate from durable coordinator/carrier recovery.
//   v2.8.56-StickySecurityEvidence - Preserved role-isolated security-stop times
//   v2.8.55-CertificateBackfillTelemetry - Published role-isolated,
//     source-blind coordinator certificate recovery evidence
//   v2.8.54-CertificateCarrierRecovery - Unified coordinator and follower
//     carrier recovery so security faults cannot be masked by later sources
//   v2.8.53-TypedCarrierCircuit - Retained independent process-lifetime block
//     and certificate carrier circuits in the follower runtime
//   v2.8.52-BlockCarrierCircuitTelemetry - Forwarded anonymous carrier circuit
//     cooling, skip, and half-open aggregates into signed heartbeat status
//   v2.8.50-BlockCarrierCircuitBreaker - Added process-only fixed-slot cooldown
//     and half-open recovery for repeated follower carrier availability faults
//   v2.8.50-RouteDomainCertificateRecovery - [ROUTE-DOMAIN-CERTIFICATE-RECOVERY
//     2026-08-03 by Codex] Persist and reverify bounded portable route-domain
//     certificates without exposing trust metadata on public discovery
//   v2.8.49-FollowerCertificateTipBinding - Bound readiness heartbeat state to
//     the exact fully audited local tip
//   v2.8.48-FollowerCertificateReadiness - Added identity-blind current-policy
//     readiness to local status and management heartbeat
//   v2.8.47-RuntimeIdentityPolicy - Rejected self-referential coordinator and
//     witness trust pins before mutable runtime initialization
//   v2.8.45-FollowerCertificateTelemetry - Reported source-blind aggregate
//     checkpoint-certificate retrieval and carrier-recovery outcomes
//   v2.8.44-FollowerCertificateCarrier - Recovered current-tip certificates
//     through exact operator-pinned witness carriers when the coordinator
//     transport is unavailable, without granting carrier chain authority
//   v2.8.43-FollowerCertificateSync - Replicated current-tip checkpoint
//     certificates to converged followers under their local witness policy
//   v2.8.42-TaskShutdown - Bounded concurrent task joins with explicit abort
//     confirmation after graceful-shutdown timeout
//   v2.8.41-RuntimeSupervision - Required API listener-group failures now
//     trigger graceful process failure and systemd recovery
//   v2.8.40-StartupReadiness - Pre-bound required API listeners and reported
//     listener-backed systemd readiness
//   v2.8.39-DiscoveryIdentityAmbiguity - Failed closed when independently
//     signed peer descriptors claim the same canonical gossip endpoint
//   v2.8.38-DirectoryProofDiversity - Added producer-first fallback rotation
//     and verified peer-aware self-producer suppression
//   v2.8.37-DirectoryProofMaturity - Added a safe proof publication maturity
//     window without delaying legacy descriptor gossip
//   v2.8.36-DiscoveryGossipIsolation - Bounded concurrent peer fan-out with
//     typed failures and compatibility-reserved per-peer deadlines
//   v2.8.35-DirectoryMirrorCarrierCapability - Added rollout-gated signed
//     self-advertisement for audited non-authoritative replica carriers
//   v1.0.0-BlindVaultService - Fail-closed initialization and bounded cleanup
//     for the independent anonymous encrypted-object store
//   v2.8.34-DirectoryPolicyHeadAnchor - Anchored opaque witness-policy heads at
//     exact current pins with restart-audited monotonic evidence
//   v2.8.33-DirectoryWitnessPolicyEpoch - Reconciled signed hash-linked local
//     witness policy history during fail-closed Directory Replica startup
//   v2.8.32-DirectoryMatureWitnessPipelineStatus - Passed the configured witness
//     maturity delay into additive privacy-tiered Directory status
//   v2.8.31-DirectoryWitnessThreshold - Wired the validated independent
//     checkpoint receipt target into scheduling and privacy-tiered status
//   v2.8.30-DurableDirectoryReplicaBackoff - Audit and restore producer retry state across restarts
//   v2.8.29-DirectoryReplicaCoordinator - Extract bounded concurrent replica scheduling
//   v2.8.28-DirectoryReplicaStatus - Privacy-tiered status and request-budgeted multi-page catch-up
//   v2.8.27-VerifiedDeliveryRollbackAnchor - Detect locally replaced older signed delivery caches
//   v2.8.26-VerifiedDeliveryRestartContinuity - Signed aggregate receipt evidence with fail-closed live readiness
//   v2.8.25-LegacyFallbackReceiptSearch - Do not let legacy onward-envelope ACKs mask receipt-capable paths
//   v2.8.24-ReceiptProbeFallbackSearch - Continue bounded mixed-version probes until a signed terminal receipt is found
//   v2.8.23-RouteNetworkAntiAffinity - Fail closed on unreachable or endpoint-collocated relay paths
//   v2.8.22-DiscoveryTaskLifecycle - Await peer-cache and gossip shutdown tasks
//   v2.8.21-RouteEvidenceCache - Descriptor-bound warm-restart routeability evidence
//   v2.8.20-RouteWarmup - Bounded direct startup routeability probes
//   v2.8.19-TipSupersessionIntegration - Verify latest-tip delivery over real HTTP
//   v2.8.18-TipSupersession - Prioritize newer audited tip announcements
//   v2.8.17-TipRetryQueue - Bounded transient follower wake-up retries
//   v2.8.15-AnnouncementReceipts - Exact coordinator tip-delivery result evidence
//   v2.8.14-SyncObservability - Event-driven follower trigger and announcement evidence
//   v2.8.12-LeaseFailClosedTelemetry - Partition/recovery lease evidence
//   v2.8.11-CoordinatorLeaseRelease - Signed graceful lease handover on SIGINT/SIGTERM
//   v2.8.10-CoordinatorLease - Strict all-witness short-lived production authority
//   v2.8.9-CoordinatorProductionFence - Reject duplicate local block producers
//   v2.8.8-CertificateRollbackGuard - Signed local certificate-vault high-water gate
//   v2.8.7-CertificateExchange - Post-startup audited certificate exchange
//   v2.8.5-TrustedDivergenceHalt - Sticky trusted fork evidence and live production halt
//   v2.8.3-WitnessDivergence - Signed divergent startup-gate integration test
//   v2.8.24-DirectorySyncServing - Mounted pinned authenticated Directory Chain peer reads
//   v2.8.23-DirectoryChainStore - Transactional local descriptor commitment ledger
//   v2.8.1-BlockFollowerOps - Configurable bounded catch-up round budget
//   v2.8.0-ChatPullV2 - Signed opaque-cursor stable mailbox snapshots
//   v2.7.22-ChatRelayMaintenance - Scheduled TTL cleanup with health evidence
//   v2.7.21-OfflineControlReliability - Bounded ChatExpired pull delivery
//   v2.7.20-ChatRelayStorageGuard - Global durable quotas and route-safe telemetry
//   v2.7.19-PublicApiBounds - Shared peer decoder and pre-parse request ceilings
//   v2.7.18-BoundedPeerReads - Bounded untrusted peer responses and recovery files
//   v2.7.17-SectionSafeAuth - Resolve API secrets before Server construction
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

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures::StreamExt;
#[cfg(target_os = "linux")]
use nix::sys::socket::{
    sendto, socket, AddressFamily, MsgFlags, SockFlag, SockProtocol, SockType, UnixAddr,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{broadcast, mpsc, Mutex as TokioMutex};
use tokio::task::{JoinHandle, JoinSet};
use tracing::{debug, error, info, trace, warn};

use aeronyx_core::protocol::auth::{
    verify_signed_message, DOMAIN_CHAT_ACK, DOMAIN_CHAT_PULL, DOMAIN_CHAT_PULL_V2,
    DOMAIN_DEVICE_REGISTER, DOMAIN_SESSION_CLOSE_V1, DOMAIN_WALLET_PRESENCE,
};
use aeronyx_core::protocol::chat::{
    custody_audit_anchor_frame_sha256, encode_envelope, BlindRelayDeliveryReceipt,
    BlindRelayEnvelope, ChatContentType, ChatEnvelope,
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
use aeronyx_core::protocol::discovery::RouteDomainAttestationCertificateV1;
use aeronyx_core::protocol::memchain::{
    chat_verified_submit_route_id, encode_memchain, ChatRelayVerifiedSubmitRequestV1,
    ChatRelayVerifiedSubmitResponseV1, MemChainMessage, MAX_CHAT_PULL_CURSOR_V2_BYTES,
};
use aeronyx_core::protocol::messages::CLIENT_HELLO_SIZE;
use aeronyx_core::protocol::{
    DataPacket, MessageType, NodeBootstrapSnapshot, NodeCapability, NodeCapacity, NodeDescriptor,
    NodeDiscoveryMessage, NodePolicy, NodeProtocolFeature, OnionRouteFailureDisposition,
    OnionRoutePlanError, OnionRoutePurpose, SignedNodeDescriptor, VerifiedOnionRoute,
    PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2,
};
use aeronyx_transport::traits::{Transport, TunConfig};
use aeronyx_transport::UdpTransport;

// [LINUX-TUN-TRAIT-SCOPE 2026-10-04 by Codex] Child modules use super::*;
// LinuxTun up/name/read/write are trait methods, not inherent methods.
#[cfg(target_os = "linux")]
use aeronyx_transport::traits::TunDevice;
#[cfg(target_os = "linux")]
use aeronyx_transport::LinuxTun;

use rand::RngCore;
use rusqlite::OptionalExtension;

use crate::api::auth::ensure_jwt_secret;
use crate::api::blind_vault::{
    build_blind_vault_router_with_admission_runtime, BlindVaultApiAdmissionRuntime,
};
use crate::api::chat_anonymous_mailbox_source::build_chat_anonymous_mailbox_source_router;
use crate::api::chat_handlers::build_chat_router;
#[cfg(test)]
use crate::api::chat_peer::blind_relay_delivery_receipt_is_valid;
use crate::api::chat_peer::{
    build_chat_peer_router_with_anonymous_mailbox, prepare_peer_blind_relay_http_request_with,
    prepare_peer_chat_relay_request_v1, prepare_peer_chat_relay_request_v2,
    prepare_peer_chat_relay_request_v3, verify_blind_relay_delivery_receipt,
    verify_peer_chat_relay_receipt, BlindRelayDeliveryReceiptVerificationFailure,
    BlindRelayRequestPreparationError, DirectRelayReceiptVerificationFailure,
    PeerBlindRelayRequest, PeerBlindRelayResponse, PeerChatRelayResponse, PeerChatRelayResponseV2,
    PreparedAuthenticatedPeerChatRelayHttpRequest,
};
use crate::api::directory_chain_peer::build_directory_chain_peer_router_with_replica_and_runtime;
use crate::api::directory_replica_status::{
    build_directory_replica_status_router_with_witness_carrier, DirectoryReplicaStatusScope,
};
use crate::api::directory_replica_sync::{
    run_directory_carrier_cold_bootstrap_smoke, run_directory_mirror_carrier_smoke,
    DirectoryCarrierColdBootstrapSmokeReport, DirectoryMirrorCarrierSmokeReport,
    DirectoryReplicaSyncCoordinator, DirectoryReplicaSyncPolicy, DirectoryReplicaSyncResources,
    DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS, DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
};
use crate::api::discovery::{
    blind_relay_runtime_status_value,
    build_discovery_router_with_local_entry_and_attestation_inbox,
    discovery_readiness_status_value, recovery_anchor_status_value, DiscoveryApiPolicy,
    DiscoveryBlindVaultCapabilityObservation, DiscoveryLocalCapabilityStatus, GossipResponse,
};
use crate::api::memchain_peer::{
    announce_current_record_commitment_tip, build_memchain_peer_router_with_runtime,
    plan_custody_audit_witnesses, publish_current_descriptor_to_commitment_witnesses,
    pull_record_commitment_checkpoint, pull_record_commitment_page_with_carrier_runtime_bounded,
    reconcile_record_commitment_pinned_witnesses_with_certificate_threshold,
    reconcile_record_commitment_witnesses,
    recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime,
    release_record_commitment_coordinator_lease, request_record_commitment_coordinator_lease,
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime,
    sync_next_record_coordinator_handover_with_carrier_runtime,
    witness_custody_audit_anchor_round_durable, witness_verified_delivery_anchor,
    CommitmentAuthorityCarrierCircuitBreaker, CommitmentAuthorityCarrierCursor,
    CommitmentAuthoritySyncSource, CommitmentBlockCarrierCircuitBreaker,
    CommitmentBlockCarrierCursor, CommitmentCertificateCarrierCircuitBreaker,
    CommitmentCertificateCarrierRecoveryDisposition, CommitmentCheckpointRelation,
    CommitmentFollowerCertificateSyncOutcome, CommitmentReconciliationOutcome,
    CommitmentSyncPageSource, CustodyAuditWitnessRound, VerifiedDeliveryAnchorWitnessRound,
    MAX_BLOCKS_PER_RESPONSE_WIRE,
};
use crate::api::mpi::{
    build_mpi_router, build_mpi_router_with_source, BaselineSnapshot, Mode, MpiState,
    SessionEmbeddingCache,
};
use crate::api::public_node_router::{
    build_public_node_router, public_endpoint_flow_context, PublicNodeRouterDependencies,
};
use crate::api::voice::build_voice_router;
use crate::api::vpn_health::{
    build_vpn_health_router_with_anonymous_mailbox_readiness,
    collect_node_operator_status_value_with_anonymous_mailbox_readiness,
    collect_vpn_health_value_with_anonymous_mailbox_readiness, AnonymousMailboxReadinessProjection,
};
use crate::api::{
    canonical_peer_http_url, decode_bounded_json_response, peer_endpoint_is_permitted,
    privacy_safe_peer_http_client_builder, read_bounded_http_response, BoundedHttpResponseError,
    PEER_ACK_RESPONSE_MAX_BYTES,
};
use crate::config::{
    DiscoveryConfig, MemChainConfig, MemChainMode, ServerConfig, VectorQuantizationMode,
};
use crate::error::{Result, RuntimeTaskJoinFailureKind, ServerError};
use crate::handlers::packet::DecryptedPayload;
use crate::handlers::PacketHandler;
use crate::management::{
    reporter::{SessionEventSender, SessionQuality},
    CommandHandler, HeartbeatReporter, ManagementClient, SessionReporter,
};
use crate::miner::ReflectionMiner;
use crate::services::chat_relay::{
    derive_node_secret, ChatRelayOutboundFailureReason, ChatRelayPeerStatus, ChatRelayService,
    ExpiredNotification, VerifiedSubmitAdmission, VerifiedSubmitCacheLookup,
    VerifiedSubmitRecoveryOutcome, MAX_CHAT_ACK_MESSAGE_IDS,
};
use crate::services::chat_relay_anonymous_mailbox_source::{
    AnonymousMailboxSourceCleanupReport, AnonymousMailboxSourceCoordinator,
    AnonymousMailboxSourceError, SqliteAnonymousMailboxSourceJournal,
};
use crate::services::chat_relay_mailbox::{
    AnonymousMailboxCustodyRepository, AnonymousMailboxStoreError, SqliteAnonymousMailboxStore,
};
use crate::services::discovery_endpoint_promotion_coordinator::{
    build_endpoint_possession_responder, PermissionlessPromotionCoordinator,
};
use crate::services::discovery_peer_sampling::sample_public_gossip_peers;
use crate::services::memchain::derive_rawlog_key;
use crate::services::memchain::derive_record_key;
use crate::services::memchain::EmbedEngine;
use crate::services::memchain::NerEngine;
use crate::services::memchain::RerankerEngine;
use crate::services::memchain::{
    custody_witness_renewal_warning_window_secs, ensure_volumes_config,
    CustodyAuditWitnessPolicyReadiness, CustodyAuditWitnessReadinessError,
    CustodyAuditWitnessReceiptReadinessSnapshot, StoragePool, SystemDb, VectorIndexPool,
    VolumeRouter,
};
#[allow(deprecated)]
use crate::services::memchain::{AofWriter, MemPool, MemoryStorage, VectorIndex};
use crate::services::memchain::{
    LlmRouter, RecordCommitmentCertificateBackfillDisposition, TaskWorker,
};
use crate::services::peer_store::{
    PeerStoreDirectoryProofGossipRound, PeerStoreRouteQuarantineCacheEvidence,
    PeerStoreRouteabilityCacheEvidence, PeerStoreSignedPeerRecordsStatus, PeerStoreStatus,
    PeerStoreTwoHopPathProofEvent, PeerStoreVerifiedClientDeliveryCacheEvidence,
    PeerStoreVerifiedDeliveryWitnessRound, AUTHENTICATED_CHAT_MIDDLE_CANDIDATE_LIMIT,
    AUTHENTICATED_CHAT_TERMINAL_FANOUT_LIMIT, ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION,
    ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION, ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES,
    ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION, ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION,
    THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION, TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
    VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION,
};
use crate::services::{
    start_dns_proxy, BlindVaultService, DirectoryChainAppendReport, DirectoryChainStore,
    DirectoryReplicaGossipAnnouncement, DirectoryReplicaStore, DirectoryReplicaSyncRuntime,
    DiscoveryEndpointAttestationInboxConfig, DiscoveryEndpointEvidenceStoreConfig,
    HandshakeService, IpPoolService, NodePolicyRuntime, PeerStore, RoutingService, SessionManager,
    SessionTermination, SqliteDiscoveryEndpointAttestationInbox,
    SqliteDiscoveryEndpointEvidenceStore,
};
// v1.0.0-Membership
use crate::services::deny_list::DenyList;
use crate::services::session::StatsSnapshot;
use crate::services::traffic_tracker::TrafficTracker;
use crate::voucher_verifier::VoucherVerifier;

// [PEER-CACHE-BACKUP-DURABILITY 2026-09-21 by Codex] Keep the bounded backup
// rotation and primary publication in one blocking filesystem transaction.
mod peer_cache_backup_io;
use peer_cache_backup_io::publish_peer_cache_snapshot;
// [SERVER-DECOMPOSITION-PHASE12A 2026-09-21 by Codex] Pure persistence
// outcomes and signed recovery-anchor validation remain independent from I/O.
mod peer_cache_recovery;
// [PEER-CACHE-RUNTIME 2026-09-25 by Codex] Preserve the existing parent
// document alias consumed by signed recovery and compatibility tests.
mod peer_cache_runtime;
use self::peer_cache_runtime::PeerStoreCacheDocument;
use peer_cache_recovery::{
    PeerStoreCachePersistOutcome, PeerStoreCachePersistReport,
    PeerStoreVerifiedClientDeliveryAnchor, PeerStoreVerifiedClientDeliveryAnchorState,
    PeerStoreVerifiedClientDeliveryExternalWitnessDecision,
    VERIFIED_CLIENT_DELIVERY_ANCHOR_MAX_BYTES,
};
#[cfg(test)]
use peer_cache_recovery::{
    VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT,
    VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT,
};
// [CHAT-OUTBOUND-RUNTIME 2026-09-25 by Codex] Keep bounded outbound chat
// routing and receipt decisions together without changing Server startup.
mod chat_outbound_runtime;
// [WITNESS-SAFETY-RUNTIME-SPLIT 2026-09-25 by Codex] These independent
// operator-pinned safety machines keep their original Server::run call order.
mod chat_custody_witness_runtime;
mod memchain_commitment_runtime;
use self::chat_custody_witness_runtime::*;
use self::memchain_commitment_runtime::*;
// [CHAT-OUTBOUND-RUNTIME 2026-09-25 by Codex] Discovery probes reuse the
// outbound scheduler's typed attribution without a public API change.
use self::chat_outbound_runtime::OnionRouteFailureAttribution;
#[cfg(test)]
use self::chat_outbound_runtime::HTTP_TOO_EARLY_STATUS_CODE;
// [DATA-PLANE-RUNTIME 2026-09-25 by Codex] Startup owns task ordering;
// the child owns the unchanged transport/session task bodies.
mod data_plane_runtime;
// [DISCOVERY-GOSSIP-RUNTIME 2026-09-25 by Codex] Keep the public server
// composition stable while gossip state, scheduling, and probes live together.
mod discovery_gossip_runtime;
// [SERVER-DECOMPOSITION-PHASE1 2026-09-14 by Codex] Keep the public server
// composition stable while runtime supervision lives in a focused child module.
mod runtime_supervision;
// [SERVER-BACKGROUND-TASKS-SPLIT 2026-09-25 by Codex] Keep periodic cleanup,
// directory persistence/sync, and shutdown-aware task bodies together while
// preserving the existing Server startup call sites and method signatures.
mod background_tasks;
// [SERVER-API-RUNTIME-SPLIT 2026-09-25 by Codex] Keep router/listener
// assembly and management task construction in a focused child while keeping
// startup call sites and failure ordering stable.
mod api_runtime;
// [SERVER-SESSION-RUNTIME-SPLIT 2026-09-25 by Codex] Keep VPN service
// initialization and transport shutdown in one focused child; data-plane
// handshake/session task bodies remain in data_plane_runtime.
mod session_runtime;
use background_tasks::{
    anonymous_mailbox_custody_cleanup_failure_disposition,
    anonymous_mailbox_source_cleanup_failure_disposition, run_anonymous_mailbox_cleanup_loop,
    AnonymousMailboxCleanupCycleOutcome, AnonymousMailboxCleanupFailureDisposition,
    AnonymousMailboxCleanupLoopDirective, AnonymousMailboxCleanupLoopExit, ManagementRuntime,
};

use self::runtime_supervision::{
    custody_witness_runtime_failure, data_plane_receive_failure_action,
    required_runtime_supervisor_channel_closed, retry_required_data_plane_receive,
    take_pre_ready_runtime_failure, CriticalRuntimeFailure, DataPlaneReceiveFailureAction,
    RequiredApiListenerExit, RuntimeTaskRegistry, RuntimeTaskShutdownOutcome,
    RuntimeTaskShutdownReport,
};

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
/// First delay after a required data-plane receive failure.
const DATA_PLANE_RECV_RETRY_BASE_MILLIS: u64 = 25;
/// Upper bound for one required data-plane receive retry delay.
const DATA_PLANE_RECV_RETRY_MAX_MILLIS: u64 = 1_000;
/// Consecutive receive failures that make the data plane process-unhealthy.
const DATA_PLANE_RECV_FAILURE_LIMIT: u32 = 8;
// [CHAT-OUTBOUND-RUNTIME 2026-09-25 by Codex] Both client delivery and
// discovery probes share this unchanged route-selection bound.
const ONION_ROUTE_SELECTION_CANDIDATE_LIMIT: usize = 8;
const TWO_HOP_PROBE_REQUEST_LIMIT: usize = 8;
const THREE_HOP_PROBE_REQUEST_LIMIT: usize = 4;
/// Keep signed discovery evidence bounded independently from peer-store size.
const HEARTBEAT_SIGNED_PEER_RECORD_LIMIT: usize = 8;
const BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS: u64 = 15 * 60;
const BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS: u64 = 60;
/// Manual mirror-carrier verification is local-only but remains low frequency.
const DIRECTORY_MIRROR_CARRIER_SMOKE_COOLDOWN_SECS: u64 = 30;
/// [MIRROR-CARRIER-SMOKE 2026-07-25 by Codex] Bound the complete
/// multi-producer verification round, including descriptor hydration. Per-request
/// HTTP timeouts alone cannot cap a paginated or multi-carrier smoke.
const DIRECTORY_MIRROR_CARRIER_SMOKE_DEADLINE_SECS: u64 = 35;
#[cfg(test)]
const BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS: u64 = 120;
/// Previous signed aggregate delivery schema accepted during rolling upgrades.
const VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION: u16 = 1;
/// Direct startup probes are bounded independently of untrusted peer count.
const BLIND_RELAY_STARTUP_WARMUP_MAX_CANDIDATES: usize = 3;

/// Aggregate-only result of one bounded two-hop relay probe round.
///
/// [TWO-HOP-PROBE-OUTCOME 2026-07-31 by Codex] These dimensions intentionally
/// remain independent:
/// - `attempted` drives the background cooldown after network work;
/// - `route_accepted` preserves rolling-upgrade control-path compatibility;
/// - `terminal_delivery_verified` requires a fresh terminal-signed receipt and
///   is the only condition that may satisfy operator smoke `success`.
///
/// Never add selected hops, node ids, endpoints, route ids, receiver keys,
/// payloads, or client metadata to this type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TwoHopBlindRelayProbeOutcome {
    attempted: bool,
    route_accepted: bool,
    terminal_delivery_verified: bool,
}

/// Public IP services should return one textual IP address and whitespace.
const PUBLIC_IP_RESPONSE_MAX_BYTES: usize = 256;

/// Gossip is bounded independently from the operator's peer-store capacity.
const DISCOVERY_GOSSIP_RESPONSE_MAX_BYTES: usize = 1024 * 1024;
/// Capability negotiation reads only the compact public discovery summary.
const DISCOVERY_NEGOTIATION_SUMMARY_MAX_BYTES: usize = 64 * 1024;
/// [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Unsigned capability
/// hints must never stall route proof or legacy fallback.
const DISCOVERY_NEGOTIATION_HINT_TIMEOUT: Duration = Duration::from_secs(2);
/// At most two producer-diverse alternates follow exact-evidence misses.
const DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT: usize = 3;

/// Bootstrap/cache snapshots may contain thousands of signed descriptors.
const DISCOVERY_SNAPSHOT_MAX_BYTES: usize = 8 * 1024 * 1024;

/// Build the bounded PeerStore projection sent in the management heartbeat.
///
/// [BOUNDED-DISCOVERY-HEARTBEAT 2026-08-02 by Codex] Full route candidates,
/// per-peer health rows, and audit rings remain available from the local node
/// API. Repeating them in every heartbeat caused the backend's 64 KiB safety
/// gate to reject the entire discovery snapshot. This projection keeps the
/// aggregate contracts consumed by nodeboard while excluding those heavy
/// local diagnostics. Never add endpoints, full node ids, route ids, payloads,
/// receiver identities, client IPs, or user traffic here.
fn peer_store_heartbeat_status_value(status: &PeerStoreStatus) -> serde_json::Value {
    serde_json::json!({
        "snapshot": &status.snapshot,
        "runtime": &status.runtime,
        "blind_relay_quality": &status.blind_relay_quality,
        "two_hop_path_proof_history": &status.two_hop_path_proof_history,
        "three_hop_path_proof_history": &status.three_hop_path_proof_history,
        "max_peers": status.max_peers,
        "recent_peer_events": &status.recent_peer_events,
        "bootstrap": &status.bootstrap,
        "stability": &status.stability,
        "route_governance": &status.route_governance,
        "peer_quorum": &status.peer_quorum,
        "network_story": &status.network_story,
    })
}

/// Builds the bounded discovery object carried by the signed management heartbeat.
///
/// [RECOVERY-ANCHOR-HEARTBEAT 2026-08-21 by Codex] Keep all derived readiness
/// in shared API helpers so the local status endpoint and backend heartbeat
/// cannot disagree about external-witness generation alignment. The signed
/// peer-record batch is intentionally passed in after its own bounded export;
/// this function must not add local audit rows, route candidates, witness
/// identities/endpoints, anchor material, or user traffic.
fn discovery_heartbeat_status_value(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
    signed_peer_records: PeerStoreSignedPeerRecordsStatus,
) -> serde_json::Value {
    serde_json::json!({
        "generated_at": generated_at,
        "peer_store": peer_store_heartbeat_status_value(status),
        "route_governance": &status.route_governance,
        "blind_relay_runtime": blind_relay_runtime_status_value(
            generated_at,
            status,
            local_capabilities,
        ),
        "recovery_anchor": recovery_anchor_status_value(status),
        "signed_peer_records": signed_peer_records,
        "local_capabilities": local_capabilities,
        "discovery_readiness": discovery_readiness_status_value(status, local_capabilities),
        "source": "rust_peer_store",
        "privacy_boundary": "aggregate node discovery, blind relay, and recovery-anchor state plus bounded signed node-level discovery descriptors for central verification; no client IPs, destinations, DNS contents, packet payloads, chat plaintext, voucher secrets, private keys, or wallet-level traffic"
    })
}

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

/// One immutable process-lifetime peer HTTP transport budget.
///
/// [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Transport budgets are named
/// values rather than duplicated builder chains. This keeps production,
/// tests, startup logs, and standalone Directory constructors on one contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PeerHttpProfileSpec {
    name: &'static str,
    connect_timeout_secs: Option<u64>,
    request_timeout_secs: u64,
    pool_max_idle_per_host: usize,
    pool_idle_timeout_secs: Option<u64>,
}

impl PeerHttpProfileSpec {
    fn build(&self) -> Result<Arc<reqwest::Client>> {
        let mut builder = privacy_safe_peer_http_client_builder()
            .timeout(Duration::from_secs(self.request_timeout_secs))
            .pool_max_idle_per_host(self.pool_max_idle_per_host);
        if let Some(connect_timeout_secs) = self.connect_timeout_secs {
            builder = builder.connect_timeout(Duration::from_secs(connect_timeout_secs));
        }
        if let Some(pool_idle_timeout_secs) = self.pool_idle_timeout_secs {
            builder = builder.pool_idle_timeout(Duration::from_secs(pool_idle_timeout_secs));
        }
        builder.build().map(Arc::new).map_err(|error| {
            ServerError::startup_failed(format!(
                "privacy-safe peer HTTP {} client initialization failed: {error}",
                self.name
            ))
        })
    }
}

const PEER_CONTROL_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "control",
    connect_timeout_secs: Some(3),
    request_timeout_secs: 5,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
const DIRECTORY_SYNC_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "directory_sync",
    connect_timeout_secs: Some(DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS),
    request_timeout_secs: DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
const DIRECTORY_OPERATOR_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "directory_operator",
    connect_timeout_secs: Some(3),
    request_timeout_secs: 12,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
const MEMCHAIN_SYNC_HTTP_PROFILE: PeerHttpProfileSpec = PeerHttpProfileSpec {
    name: "memchain_sync",
    connect_timeout_secs: Some(5),
    request_timeout_secs: 15,
    pool_max_idle_per_host: 1,
    pool_idle_timeout_secs: None,
};
// [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Replica availability recovery
// must fail over before the operator-only diagnostic request budget expires.
const _: () = assert!(
    DIRECTORY_SYNC_HTTP_PROFILE.request_timeout_secs
        < DIRECTORY_OPERATOR_HTTP_PROFILE.request_timeout_secs
);

/// Process-lifetime HTTP transports for authenticated node-to-node traffic.
///
/// [PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex] Each profile preserves its
/// historical timeout and pool budget while sharing one connection pool for
/// the complete server lifetime. Building every profile before mutable
/// services start prevents a configured protocol capability from disappearing
/// later because one background task failed to initialize its own client.
#[derive(Clone)]
struct PeerHttpClients {
    /// Short control requests: relay, witnesses, leases, and cache anchors.
    control: Arc<reqwest::Client>,
    /// Directory replica synchronization with its historical failover budget.
    directory_sync: Arc<reqwest::Client>,
    /// Operator-only Directory diagnostics with a larger bounded deadline.
    directory_operator: Arc<reqwest::Client>,
    /// Long-running MemChain page and checkpoint synchronization requests.
    sync: Arc<reqwest::Client>,
    /// Discovery gossip with its operator-configured fetch timeout and pool.
    gossip: Arc<reqwest::Client>,
}

impl PeerHttpClients {
    fn build(config: &ServerConfig) -> Result<Self> {
        let control = PEER_CONTROL_HTTP_PROFILE.build()?;
        let directory_sync = DIRECTORY_SYNC_HTTP_PROFILE.build()?;
        let directory_operator = DIRECTORY_OPERATOR_HTTP_PROFILE.build()?;
        let sync = MEMCHAIN_SYNC_HTTP_PROFILE.build()?;
        let gossip = PeerHttpProfileSpec {
            name: "gossip",
            connect_timeout_secs: None,
            request_timeout_secs: config.discovery.fetch_timeout_secs,
            pool_max_idle_per_host: 8,
            pool_idle_timeout_secs: Some(90),
        }
        .build()?;
        Ok(Self {
            control,
            directory_sync,
            directory_operator,
            sync,
            gossip,
        })
    }
}

/// Process-lifetime bridge to systemd's datagram notification protocol.
///
/// [STARTUP-READINESS 2026-07-29 by Codex] `Type=notify` must reflect the
/// actual listener barrier, not process creation. Manual/non-systemd starts
/// remain backward compatible because an absent `NOTIFY_SOCKET` is a no-op.
/// The implementation uses the existing `nix` dependency and supports both
/// filesystem and Linux abstract namespace notify sockets.
#[derive(Clone, Debug, Default)]
struct SystemdNotifier {
    socket: Option<OsString>,
}

impl SystemdNotifier {
    fn from_environment() -> Self {
        Self {
            socket: std::env::var_os("NOTIFY_SOCKET"),
        }
    }

    #[cfg(test)]
    fn from_socket(socket: Option<OsString>) -> Self {
        Self { socket }
    }

    fn status(&self, status: &str) -> Result<bool> {
        self.send(&format!("STATUS={}", Self::sanitize_status(status)))
    }

    fn ready(&self, status: &str) -> Result<bool> {
        self.send(&format!(
            "READY=1\nSTATUS={}",
            Self::sanitize_status(status)
        ))
    }

    fn stopping(&self, status: &str) -> Result<bool> {
        self.send(&format!(
            "STOPPING=1\nSTATUS={}",
            Self::sanitize_status(status)
        ))
    }

    fn sanitize_status(status: &str) -> String {
        status
            .chars()
            .map(|character| match character {
                '\r' | '\n' => ' ',
                other => other,
            })
            .collect()
    }

    fn send(&self, payload: &str) -> Result<bool> {
        let Some(socket_name) = self.socket.as_ref() else {
            return Ok(false);
        };

        #[cfg(target_os = "linux")]
        {
            let socket_bytes = socket_name.as_os_str().as_bytes();
            if socket_bytes.is_empty() {
                return Err(ServerError::startup_failed(
                    "systemd NOTIFY_SOCKET is empty",
                ));
            }

            let address = if let Some(abstract_name) = socket_bytes.strip_prefix(b"@") {
                if abstract_name.is_empty() {
                    return Err(ServerError::startup_failed(
                        "systemd abstract NOTIFY_SOCKET name is empty",
                    ));
                }
                UnixAddr::new_abstract(abstract_name).map_err(|error| {
                    ServerError::startup_failed(format!(
                        "invalid systemd abstract NOTIFY_SOCKET: {error}"
                    ))
                })?
            } else {
                UnixAddr::new(Path::new(socket_name)).map_err(|error| {
                    ServerError::startup_failed(format!(
                        "invalid systemd filesystem NOTIFY_SOCKET: {error}"
                    ))
                })?
            };
            let datagram = socket(
                AddressFamily::Unix,
                SockType::Datagram,
                SockFlag::SOCK_CLOEXEC,
                None::<SockProtocol>,
            )
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "failed to create systemd notification socket: {error}"
                ))
            })?;
            let sent = sendto(
                datagram.as_raw_fd(),
                payload.as_bytes(),
                &address,
                MsgFlags::empty(),
            )
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "failed to send systemd readiness notification: {error}"
                ))
            })?;
            if sent != payload.len() {
                return Err(ServerError::startup_failed(format!(
                    "partial systemd readiness notification: sent {sent} of {} bytes",
                    payload.len()
                )));
            }
            Ok(true)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = payload;
            Ok(false)
        }
    }
}

// [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex] Keep the source
// coordinator and its exact journal handle in one server-local composition.
// The coordinator remains the only API-facing value; maintenance receives no
// route, target, request, or ciphertext projection.
struct AnonymousMailboxSourceRuntime {
    coordinator: Arc<AnonymousMailboxSourceCoordinator>,
    journal: Arc<SqliteAnonymousMailboxSourceJournal>,
}

pub struct Server {
    config: ServerConfig,
    identity: IdentityKeyPair,
    config_path: Option<PathBuf>,
    shutdown: Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<()>,
    custody_witness_runtime: Arc<CustodyWitnessRuntimeTelemetry>,
}

// [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex] Keep optional
// persistence requirements explicit instead of using the existence of one
// storage tuple as an implicit gate for every MemChain wire variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemChainStorageRequirement {
    IndependentRuntime,
    FactAof,
    RecordStore,
}

#[derive(Clone, Copy)]
enum MemChainStorageAccess<'a> {
    IndependentRuntime,
    FactAof {
        mempool: &'a Arc<MemPool>,
        aof_writer: &'a Arc<TokioMutex<AofWriter>>,
    },
    RecordStore {
        storage: &'a Arc<MemoryStorage>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemChainDispatchGateError {
    FactAofUnavailable,
    RecordStoreUnavailable,
}

impl MemChainDispatchGateError {
    const fn reason_bucket(self) -> &'static str {
        match self {
            Self::FactAofUnavailable => "fact_aof_unavailable",
            Self::RecordStoreUnavailable => "record_store_unavailable",
        }
    }

    const fn family_bucket(self) -> &'static str {
        match self {
            Self::FactAofUnavailable => "fact",
            Self::RecordStoreUnavailable => "record",
        }
    }
}

impl MemChainStorageRequirement {
    fn for_message(message: &MemChainMessage) -> Self {
        match message {
            MemChainMessage::ChatRelay(_)
            | MemChainMessage::ChatRelayVerifiedSubmitV1(_)
            | MemChainMessage::ChatRelayVerifiedSubmitResponseV1(_)
            | MemChainMessage::ChatPull { .. }
            | MemChainMessage::ChatPullV2 { .. }
            | MemChainMessage::ChatAck { .. }
            | MemChainMessage::DeviceRegister { .. }
            | MemChainMessage::WalletPresence { .. } => Self::IndependentRuntime,
            MemChainMessage::BroadcastRecord(_)
            | MemChainMessage::SyncRecordRequest { .. }
            | MemChainMessage::SyncRecordResponse { .. } => Self::RecordStore,
            // Preserve the historical storage gate for every non-chat variant.
            // New wire variants therefore fail closed until they are explicitly
            // assigned to an independently configured runtime.
            _ => Self::FactAof,
        }
    }

    fn authorize<'a>(
        self,
        mempool: Option<&'a Arc<MemPool>>,
        aof_writer: Option<&'a Arc<TokioMutex<AofWriter>>>,
        storage: &'a Option<Arc<MemoryStorage>>,
    ) -> std::result::Result<MemChainStorageAccess<'a>, MemChainDispatchGateError> {
        match self {
            Self::IndependentRuntime => Ok(MemChainStorageAccess::IndependentRuntime),
            Self::FactAof => match (mempool, aof_writer) {
                (Some(mempool), Some(aof_writer)) => Ok(MemChainStorageAccess::FactAof {
                    mempool,
                    aof_writer,
                }),
                _ => Err(MemChainDispatchGateError::FactAofUnavailable),
            },
            Self::RecordStore => {
                // Record messages historically passed through the same
                // MemPool/AOF outer gate before consulting MemoryStorage.
                // Preserve that prerequisite even though their handler only
                // needs the record store after admission.
                if mempool.is_none() || aof_writer.is_none() {
                    return Err(MemChainDispatchGateError::FactAofUnavailable);
                }
                storage
                    .as_ref()
                    .map(|storage| MemChainStorageAccess::RecordStore { storage })
                    .ok_or(MemChainDispatchGateError::RecordStoreUnavailable)
            }
        }
    }
}

// [PERMISSIONLESS-ENDPOINT-EVIDENCE 2026-09-24 by Codex] Keep blocking
// private-database admission out of the listener composition and make the
// default-off zero-side-effect boundary directly testable.
async fn open_endpoint_evidence_store(
    discovery: &DiscoveryConfig,
    identity: &IdentityKeyPair,
) -> Result<Option<Arc<SqliteDiscoveryEndpointEvidenceStore>>> {
    if !discovery.permissionless_endpoint_evidence_enabled {
        return Ok(None);
    }
    let config = DiscoveryEndpointEvidenceStoreConfig {
        db_path: discovery
            .permissionless_endpoint_evidence_db_path
            .clone()
            .into(),
        max_entries: discovery.permissionless_endpoint_evidence_max_entries,
        retention_ttl_secs: discovery.permissionless_endpoint_evidence_ttl_secs,
        cleanup_batch_size: discovery.permissionless_endpoint_evidence_cleanup_batch,
    };
    let verifier_node_id = identity.public_key_bytes();
    let expected_context = public_endpoint_flow_context(verifier_node_id);
    let store = tokio::task::spawn_blocking(move || {
        SqliteDiscoveryEndpointEvidenceStore::open(config, verifier_node_id, expected_context)
    })
    .await
    .map_err(|_| ServerError::startup_failed("Endpoint evidence startup task failed"))?
    .map_err(|_| ServerError::startup_failed("Endpoint evidence store unavailable"))?;
    Ok(Some(Arc::new(store)))
}

// [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX-COMPOSITION 2026-09-24 by Codex]
// Open and audit the independent quarantine before any listener is bound.
// Disabled configuration returns before deriving a path or touching disk.
async fn open_endpoint_attestation_inbox(
    discovery: &DiscoveryConfig,
) -> Result<Option<Arc<SqliteDiscoveryEndpointAttestationInbox>>> {
    if !discovery.permissionless_endpoint_attestation_inbox_enabled {
        return Ok(None);
    }
    let config = DiscoveryEndpointAttestationInboxConfig {
        db_path: discovery
            .permissionless_endpoint_attestation_inbox_db_path
            .clone()
            .into(),
        max_entries: discovery.permissionless_endpoint_attestation_inbox_max_entries,
        max_logical_bytes: discovery.permissionless_endpoint_attestation_inbox_max_bytes,
        retention_ttl_secs: discovery.permissionless_endpoint_attestation_inbox_ttl_secs,
        cleanup_batch_size: discovery.permissionless_endpoint_attestation_inbox_cleanup_batch,
    };
    let inbox =
        tokio::task::spawn_blocking(move || SqliteDiscoveryEndpointAttestationInbox::open(config))
            .await
            .map_err(|_| ServerError::startup_failed("Endpoint attestation startup task failed"))?
            .map_err(|_| ServerError::startup_failed("Endpoint attestation inbox unavailable"))?;
    Ok(Some(Arc::new(inbox)))
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod bootstrap_import;
mod discovery_advertisement;
mod session_ingress;
mod startup_chat_mailbox;
mod startup_directory;
mod startup_files;
mod startup_memchain;
mod startup_models;
mod startup_network;
mod verified_submit_ingress;

impl Server {
    pub fn new(
        config: ServerConfig,
        identity: IdentityKeyPair,
        config_path: Option<PathBuf>,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        let custody_witness_runtime = Arc::new(CustodyWitnessRuntimeTelemetry::new(
            config.discovery.custody_audit_witness_runtime_required,
            config.discovery.custody_audit_witness_auto_renewal_enabled,
            config.discovery.custody_audit_witness_max_age_secs,
        ));
        Self {
            config,
            identity,
            config_path,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
            custody_witness_runtime,
        }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting AeroNyx server v{}", env!("CARGO_PKG_VERSION"));
        // [RUNTIME-IDENTITY-POLICY 2026-07-29 by Codex] Static config parsing
        // cannot compare trust pins with the public key derived from the
        // server key. Fail before transports, storage, listeners, or
        // background tasks make a self-referential node look operational.
        self.config
            .memchain
            .validate_runtime_identity(&self.identity.public_key_bytes())?;
        self.config
            .discovery
            .validate_runtime_identity(&self.identity.public_key_bytes())?;
        let systemd_notifier = SystemdNotifier::from_environment();
        systemd_notifier.status("Auditing encrypted state and initializing protocol services")?;
        // [RUNTIME-SUPERVISION 2026-07-29 by Codex] A bounded channel carries
        // only the first critical runtime failure. The receiver stays in the
        // main task so required listener loss can terminate the process.
        let (critical_failure_tx, mut critical_failure_rx) = mpsc::channel(1);
        // [STARTUP-TASK-REGISTRY 2026-07-30 by Codex] Create ownership before
        // the first process task can be spawned. Every later `?` then aborts
        // work already started by this startup transaction.
        let mut tasks = RuntimeTaskRegistry::default();

        let peer_http_clients = PeerHttpClients::build(&self.config)?;
        info!(
            profiles = 5,
            proxy = false,
            redirects = false,
            directory_sync_request_timeout_secs = DIRECTORY_SYNC_HTTP_PROFILE.request_timeout_secs,
            directory_operator_request_timeout_secs =
                DIRECTORY_OPERATOR_HTTP_PROFILE.request_timeout_secs,
            "[PEER_HTTP] Process-lifetime privacy-safe transports initialized"
        );

        // Onion routing forward secrecy: initialize the in-memory rotating onion
        // key at startup. It is never persisted, so a restart yields a fresh key
        // and the old secret is unrecoverable. The grace window is tied to the
        // descriptor TTL so a previous key outlives any descriptor still in
        // circulation. See services::onion_keys.
        crate::services::onion_keys::init_shared(
            unix_now_secs(),
            self.config.discovery.descriptor_ttl_secs,
        );

        let (ip_pool, sessions, routing) = self.init_services()?;

        let (storage, vector_index, mempool, aof_writer) = if self.config.memchain.is_enabled() {
            let (st, vi, mp, aw) = self.init_memchain().await?;
            (Some(st), Some(vi), Some(mp), Some(aw))
        } else {
            info!("[MEMCHAIN] Disabled (mode=off)");
            (None, None, None, None)
        };
        // [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Complete the
        // provider transaction before self descriptors, peer cache, gossip,
        // or AgentRelay capability can leave this process.
        let llm_router: Option<Arc<LlmRouter>> = self.init_llm_router()?;
        if let Some(ref commitment_storage) = storage {
            commitment_storage.configure_record_commitment_sync(
                self.config.memchain.commitment_coordinator_enabled,
                self.config.memchain.commitment_sync_enabled,
            );
            commitment_storage.configure_record_commitment_sync_readiness_freshness(
                self.config.memchain.commitment_sync_interval_secs,
            );
            commitment_storage.configure_record_commitment_certificate_policy(
                self.config.memchain.commitment_witness_node_ids.len(),
                self.config.memchain.commitment_witness_min_verified,
            );
            commitment_storage.configure_record_commitment_coordinator_lease(
                self.config.memchain.commitment_coordinator_lease_required,
                self.config.memchain.commitment_witness_node_ids.len(),
            );
        }

        let chat_relay_enabled = self.config.memchain.is_chat_relay_enabled();
        let chat_relay = self.init_chat_relay_service()?;
        let anonymous_mailbox = self.init_anonymous_mailbox_store().await?;
        // [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] Strict mode
        // consumes only already-durable local receipts. Startup does not
        // contact witnesses or let permissionless discovery supply authority.
        if self.config.discovery.custody_audit_witness_startup_required {
            let custody_storage = storage.as_deref().ok_or_else(|| {
                ServerError::startup_failed(
                    "Chat Relay custody witness startup guard: local_storage_unavailable",
                )
            })?;
            self.verify_chat_relay_custody_witness_startup(custody_storage)
                .await?;
        }
        // [BLIND-VAULT-SERVICE 2026-07-23 by Codex] This store is independent
        // from identity-indexed MemChain and receiver-indexed ChatRelay state.
        // Explicit enablement is fail-closed: a configured database error must
        // stop startup rather than silently dropping encrypted recovery data.
        let blind_vault = if self.config.blind_vault.enabled {
            let service =
                BlindVaultService::new(self.config.blind_vault.clone(), self.identity.clone())
                    .map_err(|error| {
                        ServerError::startup_failed(format!(
                            "Blind Vault initialization failed: {error}"
                        ))
                    })?;
            info!("[BLIND_VAULT] Anonymous encrypted-object store initialized");
            Some(Arc::new(service))
        } else {
            info!("[BLIND_VAULT] Disabled");
            None
        };
        let chat_relay_runtime_ready = chat_relay.is_some();
        let anonymous_mailbox_runtime_ready = anonymous_mailbox.is_some();
        // [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Admission
        // readiness is an observed service state, not a config synonym. Run
        // the SQLite/filesystem observation off the async startup worker.
        let blind_vault_runtime_ready =
            Self::observe_blind_vault_admission_readiness(blind_vault.clone(), unix_now_secs())
                .await;

        let peer_store = self
            .init_peer_store_with_storage_runtime(
                chat_relay_runtime_ready,
                blind_vault_runtime_ready,
                anonymous_mailbox_runtime_ready,
                peer_http_clients.control.as_ref(),
            )
            .await?;
        if self
            .config
            .memchain
            .chat_relay
            .anonymous_mailbox_source
            .enabled
            && !(storage.is_some()
                && vector_index.is_some()
                && mempool.is_some()
                && aof_writer.is_some())
        {
            // [ANONYMOUS-MAILBOX-SOURCE-WIRING 2026-09-03 by Codex] The
            // source surface is deliberately authenticated by the VPN MPI
            // app. Refuse before deriving a key or reserving its database if
            // that app cannot exist; source must never silently become a
            // node-peer/public route or leave an unusable private journal.
            return Err(ServerError::startup_failed(
                "Anonymous mailbox source requires full authenticated VPN MPI runtime",
            ));
        }
        let anonymous_mailbox_source_runtime = self
            .init_anonymous_mailbox_source_coordinator(Arc::clone(&peer_store))
            .await?;
        let anonymous_mailbox_source = anonymous_mailbox_source_runtime
            .as_ref()
            .map(|runtime| Arc::clone(&runtime.coordinator));
        let anonymous_mailbox_source_journal =
            anonymous_mailbox_source_runtime.map(|runtime| runtime.journal);
        // [ANONYMOUS-MAILBOX-READINESS-PROJECTION 2026-09-14 by Codex]
        // Health remains unreported until the same composition root has both
        // registered cleanup supervision and built the actual terminal/source
        // routers. Earlier management snapshots must not infer readiness from
        // configuration or successfully opened storage alone.
        let anonymous_mailbox_readiness = AnonymousMailboxReadinessProjection::default();
        if self.config.discovery.custody_audit_witness_runtime_required {
            // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Runtime
            // custody starts only after authenticated PeerStore bootstrap.
            // This preserves the original local audit gate while giving the
            // separately enabled renewal path an exact pinned transport view.
            let custody_storage = storage.as_ref().cloned().ok_or_else(|| {
                ServerError::startup_failed(
                    "Chat Relay custody witness runtime guard: local_storage_unavailable",
                )
            })?;
            let custody_runtime_task = self.spawn_chat_relay_custody_witness_runtime_guard(
                custody_storage,
                Arc::clone(&peer_store),
                Arc::clone(&peer_http_clients.control),
                critical_failure_tx.clone(),
            );
            tasks.push((
                "custody-witness-runtime",
                Self::supervise_required_runtime_task(
                    "custody-witness-runtime",
                    custody_runtime_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
        }
        self.publish_memchain_commitment_descriptor_preflight(
            &peer_store,
            peer_http_clients.control.as_ref(),
        )
        .await;
        // [VPN-BEFORE-DIRECTORY 2026-10-09 by Claude] Bring the VPN data
        // plane (UDP, TUN, gateway DNS, sessions, management, keepalive) up
        // before the Directory Chain/replica startup audits. Nothing below
        // depends on the directory stores; on JP1 those audits held VPN
        // users offline for ~98 s per restart and grew with chain length.
        // Directory features still start only after their audits pass.
        let udp = Arc::new(
            UdpTransport::bind_addr(self.config.listen_addr())
                .await
                .map_err(|e| ServerError::startup_failed(format!("UDP bind: {}", e)))?,
        );
        info!("UDP transport listening on {}", self.config.listen_addr());

        // [VPN-OPTIONAL-ROLE 2026-10-09 by Claude] The relay/mailbox role
        // (`vpn.enabled = false`) never creates a TUN device, so it runs in a
        // container without NET_ADMIN or /dev/net/tun.
        #[cfg(target_os = "linux")]
        let tun = if self.config.vpn_enabled() {
            Some(self.init_tun().await?)
        } else {
            info!("[VPN] Data plane disabled by vpn.enabled=false; no TUN device");
            None
        };

        let server_pubkey_hex = hex::encode(self.identity.public_key_bytes());

        // AeroNyx client readiness requires DNS to be available at the tunnel
        // gateway. When enabled, this proxy forwards opaque UDP DNS bytes only
        // and never records queried domains, DNS contents, destinations, or
        // client IPs. Operators may disable it when systemd-resolved or another
        // hardened host resolver intentionally owns gateway_ip:53.
        if self.config.dns_proxy_enabled() {
            // [DNS-STARTUP-READINESS 2026-07-30 by Codex] Bind before
            // systemd READY so the node cannot advertise a usable privacy
            // data plane while its configured DNS listener is unavailable.
            let dns_task = start_dns_proxy(self.config.gateway_ip(), self.shutdown_tx.subscribe())
                .await
                .map_err(|error| {
                    ServerError::startup_failed(format!(
                        "required VPN DNS listener failed to bind: {error}"
                    ))
                })?;
            tasks.push((
                "dns-proxy",
                Self::supervise_required_runtime_task(
                    "dns-proxy",
                    dns_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
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
        let ManagementRuntime {
            session_events: session_event_sender,
            tasks: management_tasks,
        } = self
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
                blind_vault.clone(),
                chat_relay_enabled,
                anonymous_mailbox_readiness.clone(),
            )
            .await?;
        for (name, task) in management_tasks {
            tasks.push((
                name,
                Self::supervise_required_runtime_task(
                    name,
                    task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
        }

        let udp_task = self.spawn_udp_task(
            Arc::clone(&udp),
            #[cfg(target_os = "linux")]
            tun.clone(),
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
            Arc::clone(&peer_http_clients.control),
            Arc::clone(&traffic_tracker),
        );
        tasks.push((
            "udp",
            Self::supervise_required_runtime_task(
                "udp",
                udp_task,
                Arc::clone(&self.shutdown),
                critical_failure_tx.clone(),
            ),
        ));

        #[cfg(target_os = "linux")]
        if let Some(tun) = &tun {
            let tun_task = self.spawn_tun_task(
                Arc::clone(tun),
                Arc::clone(&udp),
                Arc::clone(&packet_handler),
            );
            tasks.push((
                "tun",
                Self::supervise_required_runtime_task(
                    "tun",
                    tun_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
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

        if self.config.vpn_enabled() {
            let keepalive_task = self.spawn_keepalive_probe_task(
                Arc::clone(&sessions),
                Arc::clone(&udp),
                Arc::clone(&packet_handler),
                self.config.gateway_ip(),
            );
            tasks.push(("vpn-keepalive", keepalive_task));
        }

        let directory_chain_store = self.init_directory_chain(&peer_store).await?;
        let directory_replica_store = self.init_directory_replica().await?;
        let directory_replica_sync_runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
        if let Some(ref commitment_storage) = storage {
            self.verify_memchain_commitment_startup_witnesses(
                commitment_storage,
                &peer_store,
                peer_http_clients.control.as_ref(),
            )
            .await?;
        }
        let commitment_coordinator_lease_instance = if let Some(ref commitment_storage) = storage {
            self.acquire_memchain_commitment_coordinator_lease(
                commitment_storage,
                &peer_store,
                peer_http_clients.control.as_ref(),
            )
            .await?
        } else {
            None
        };
        if let Some(task) = self.spawn_peer_store_persistence_task(
            Arc::clone(&peer_store),
            Arc::clone(&peer_http_clients.control),
        ) {
            tasks.push(("peer-cache-persistence", task));
        }
        if let Some(task) = self.spawn_directory_chain_persistence_task(
            Arc::clone(&peer_store),
            directory_chain_store.clone(),
        ) {
            tasks.push(("directory-chain-persistence", task));
        }
        if let Some(task) = self.spawn_directory_replica_sync_task(
            Arc::clone(&peer_store),
            directory_replica_store.clone(),
            Arc::clone(&directory_replica_sync_runtime),
            Arc::clone(&peer_http_clients.directory_sync),
        )? {
            tasks.push((
                "directory-replica-sync",
                Self::supervise_required_runtime_task(
                    "directory-replica-sync",
                    task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
        }
        if let Some(task) = self.spawn_discovery_gossip_task(
            Arc::clone(&peer_store),
            directory_replica_store.clone(),
            chat_relay_runtime_ready,
            blind_vault.clone(),
            anonymous_mailbox_runtime_ready,
            Arc::clone(&peer_http_clients.gossip),
        ) {
            tasks.push(("discovery-gossip", task));
        }


        if let Some(ref relay) = chat_relay {
            let relay_cleanup_task = self.spawn_chat_relay_cleanup_task(Arc::clone(relay));
            tasks.push(("chat-relay-cleanup", relay_cleanup_task));

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

        let anonymous_mailbox_cleanup_runtime_supervised = if let Some(cleanup_task) = self
            .spawn_anonymous_mailbox_cleanup_task(
                anonymous_mailbox.clone(),
                anonymous_mailbox_source_journal,
            ) {
            tasks.push((
                "anonymous-mailbox-cleanup",
                Self::supervise_required_runtime_task(
                    "anonymous-mailbox-cleanup",
                    cleanup_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
            true
        } else {
            false
        };

        if let Some(ref vault) = blind_vault {
            let cleanup_task = self.spawn_blind_vault_cleanup_task(Arc::clone(vault));
            tasks.push(("blind-vault-cleanup", cleanup_task));
        }

        // [NODE-API-LIFECYCLE 2026-07-23 by Codex] The protocol node API is
        // infrastructure, not a MemChain side effect. Track whether the rich
        // MemChain branch started it and fall back to the same router without
        // MPI routes when memory services are disabled.
        let mut node_api_started = false;
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

            if llm_router.is_some() {
                // [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex] Startup and the
                // live worker must agree on when a processing claim is expired.
                // The grace interval prevents a second process from stealing a
                // task while its original owner persists the timeout outcome.
                let timeout_secs = self
                    .config
                    .memchain
                    .supernode
                    .worker
                    .stale_claim_recovery_secs();
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
                    parking_lot::RwLock::new(SessionEmbeddingCache::default()),
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

            // The coordinator's local tip channel and the follower's remote
            // announcement channel are deliberately separate. Each is bounded
            // to one pending height because its consumer always re-reads the
            // current audited chain rather than trusting event payload state.
            let (commitment_sync_tip_tx, commitment_sync_tip_rx) = mpsc::channel(1);
            let commitment_sync_tip_notifier = self
                .config
                .memchain
                .commitment_sync_enabled
                .then_some(commitment_sync_tip_tx);
            let (commitment_tip_tx, commitment_tip_rx) = mpsc::channel(1);

            let api_task = self
                .start_combined_api(
                    self.config.memchain.api_listen_addr,
                    Some(Arc::clone(&mpi_state)),
                    Arc::clone(&ip_pool),
                    Arc::clone(&sessions),
                    Arc::clone(&node_policy),
                    Arc::clone(&voucher_verifier),
                    Arc::clone(&encrypted_message_counter),
                    Arc::clone(&packet_handler),
                    Arc::clone(&peer_store),
                    directory_chain_store.clone(),
                    directory_replica_store.clone(),
                    Arc::clone(&directory_replica_sync_runtime),
                    chat_relay.clone(),
                    blind_vault.clone(),
                    anonymous_mailbox.clone(),
                    anonymous_mailbox_source.clone(),
                    Arc::clone(&udp),
                    &peer_http_clients,
                    commitment_sync_tip_notifier,
                    anonymous_mailbox_cleanup_runtime_supervised,
                    anonymous_mailbox_readiness.clone(),
                    critical_failure_tx.clone(),
                )
                .await?;
            tasks.push((
                "node-api",
                Self::supervise_required_runtime_task(
                    "node-api",
                    api_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
            node_api_started = true;

            if let Some(sync_task) = self.spawn_memchain_commitment_sync_task(
                Arc::clone(st),
                Arc::clone(&peer_store),
                commitment_sync_tip_rx,
                Arc::clone(&peer_http_clients.sync),
            )? {
                tasks.push((
                    "memchain-block-sync",
                    Self::supervise_required_runtime_task(
                        "memchain-block-sync",
                        sync_task,
                        Arc::clone(&self.shutdown),
                        critical_failure_tx.clone(),
                    ),
                ));
            }
            if let Some(reconciliation_task) = self.spawn_memchain_commitment_reconciliation_task(
                Arc::clone(st),
                Arc::clone(&peer_store),
                commitment_tip_rx,
                Arc::clone(&peer_http_clients.sync),
            ) {
                tasks.push(("memchain-checkpoint-witness", reconciliation_task));
            }
            if let Some(lease_task) = self.spawn_memchain_commitment_coordinator_lease_task(
                Arc::clone(st),
                Arc::clone(&peer_store),
                commitment_coordinator_lease_instance,
                Arc::clone(&peer_http_clients.control),
            ) {
                tasks.push(("memchain-coordinator-lease", lease_task));
            }

            if let Some(ref router) = llm_router {
                let worker = TaskWorker::new(
                    Arc::clone(st),
                    Arc::clone(router),
                    self.config.memchain.supernode.worker.clone(),
                );
                let worker_shutdown = self.shutdown_tx.subscribe();
                // [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Once the
                // operator enables SuperNode, its worker is required runtime.
                // Unexpected return or panic must revoke process readiness.
                let worker_task = tokio::spawn(async move {
                    worker.run(worker_shutdown).await;
                });
                tasks.push((
                    "supernode-worker",
                    Self::supervise_required_runtime_task(
                        "supernode-worker",
                        worker_task,
                        Arc::clone(&self.shutdown),
                        critical_failure_tx.clone(),
                    ),
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
                        .await?;
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
                    )
                    .with_commitment_tip_notifier(commitment_tip_tx);

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

        if !node_api_started {
            let api_task = self
                .start_combined_api(
                    self.config.memchain.api_listen_addr,
                    None,
                    Arc::clone(&ip_pool),
                    Arc::clone(&sessions),
                    Arc::clone(&node_policy),
                    Arc::clone(&voucher_verifier),
                    Arc::clone(&encrypted_message_counter),
                    Arc::clone(&packet_handler),
                    Arc::clone(&peer_store),
                    directory_chain_store.clone(),
                    directory_replica_store.clone(),
                    Arc::clone(&directory_replica_sync_runtime),
                    chat_relay.clone(),
                    blind_vault.clone(),
                    anonymous_mailbox.clone(),
                    anonymous_mailbox_source.clone(),
                    Arc::clone(&udp),
                    &peer_http_clients,
                    None,
                    anonymous_mailbox_cleanup_runtime_supervised,
                    anonymous_mailbox_readiness.clone(),
                    critical_failure_tx.clone(),
                )
                .await?;
            tasks.push((
                "node-api",
                Self::supervise_required_runtime_task(
                    "node-api",
                    api_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
        }

        // [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] Required-task
        // wrappers and the API listener group own the remaining senders. If
        // every supervisor disappears without reporting, receiver closure is
        // itself fatal.
        drop(critical_failure_tx);
        // [PRE-READY-RUNTIME-GATE 2026-07-30 by Codex] Startup can be long
        // enough for a required worker to fail before all optional services
        // are initialized. Never advertise READY when that failure is already
        // queued or every required-task supervisor has disappeared.
        let runtime_failure =
            if let Some(failure) = take_pre_ready_runtime_failure(&mut critical_failure_rx) {
                Some(failure)
            } else {
                systemd_notifier.ready("AeroNyx privacy node is ready")?;
                info!("Server started successfully");
                self.wait_for_shutdown(&mut critical_failure_rx).await
            };
        if let Some(failure) = runtime_failure.as_ref() {
            error!(
                task = failure.task,
                reason = %failure.reason,
                "[RUNTIME] Required task failed; initiating process recovery"
            );
        }
        let stopping_status = if runtime_failure.is_some() {
            "AeroNyx privacy node is stopping after a critical runtime failure"
        } else {
            "AeroNyx privacy node is stopping"
        };
        if let Err(error) = systemd_notifier.stopping(stopping_status) {
            warn!(%error, "[STARTUP] Failed to report systemd stopping state");
        }
        info!("Shutting down server...");

        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());

        // [TASK-SHUTDOWN 2026-07-29 by Codex] Every task received the same
        // broadcast already, so join them concurrently. This bounds total
        // shutdown by the longest grace period instead of the sum of all
        // per-task deadlines.
        for report in Self::shutdown_runtime_tasks(tasks.take_for_shutdown()).await {
            match report.outcome {
                RuntimeTaskShutdownOutcome::Completed => {
                    debug!(task = report.name, "Runtime task completed");
                }
                RuntimeTaskShutdownOutcome::JoinFailed(failure) => {
                    warn!(
                        task = report.name,
                        failure = ?failure,
                        "Runtime task join failed"
                    );
                }
                RuntimeTaskShutdownOutcome::CancelledAfterTimeout => {
                    warn!(
                        task = report.name,
                        "Runtime task exceeded shutdown grace period and was cancelled"
                    );
                }
                RuntimeTaskShutdownOutcome::CompletedAfterTimeout => {
                    warn!(
                        task = report.name,
                        "Runtime task completed while timeout cancellation was being applied"
                    );
                }
                RuntimeTaskShutdownOutcome::CancellationUnconfirmed => {
                    error!(
                        task = report.name,
                        "Runtime task did not confirm cancellation before the abort deadline"
                    );
                }
            }
        }

        Self::shutdown_udp_transport(udp.as_ref()).await;

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

        if let Some(failure) = runtime_failure {
            return Err(ServerError::runtime_failed(failure.task, failure.reason));
        }

        Ok(())
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
    use super::data_plane_runtime::{
        client_hello_wire_version_is_supported, handshake_rejection_class, log_handshake_rejection,
        HandshakeRejectionClass,
    };
    use crate::services::memchain::{
        CustodyAuditWitnessPolicyReadiness, CustodyAuditWitnessReceiptPolicyEvidence,
        CustodyAuditWitnessReceiptReadinessSnapshot, CustodyAuditWitnessReceiptVaultAudit,
    };

    // [DISCOVERY-GOSSIP-RUNTIME 2026-09-25 by Codex] Keep existing server
    // compatibility tests against the extracted runtime projections.
    use super::discovery_gossip_runtime::{
        DirectoryProofGossipOutcome, DirectoryProofGossipPeerState, DirectoryProofGossipResult,
        DiscoveryGossipExecution, DiscoveryGossipFailure, DiscoveryGossipFailureKind,
        DiscoveryGossipPhase, DiscoveryGossipRoundAccumulator, DiscoveryGossipSampleRequest,
        DiscoveryPeerGossipReport, DiscoveryPeerIdentityHints,
    };

    use super::{
        await_commitment_tip_announcement_or_newer,
        commitment_coordinator_lease_degraded_retry_delay,
        commitment_coordinator_lease_production_valid_for, commitment_follower_success_retry_delay,
        commitment_witness_startup_decision, custody_witness_auto_renewal_due,
        custody_witness_readiness_decision, custody_witness_renewal_status,
        custody_witness_renewal_warning_window_secs, custody_witness_runtime_audit_interval_secs,
        custody_witness_runtime_failure, data_plane_receive_failure_action,
        discovery_heartbeat_status_value, memchain_index_rejection_reason,
        open_endpoint_attestation_inbox, open_endpoint_evidence_store,
        peer_store_heartbeat_status_value, prefix_to_netmask,
        required_runtime_supervisor_channel_closed, retry_required_data_plane_receive,
        take_pre_ready_runtime_failure, unix_now_secs, CommitmentCoordinatorLeaseRound,
        CommitmentFollowerRoundOutcome, CommitmentSyncTaskLivenessGuard,
        CommitmentTipAnnouncementWaitOutcome, CommitmentWitnessStartupBlockReason,
        CommitmentWitnessStartupDecision, CriticalRuntimeFailure, CustodyWitnessAuditEvidence,
        CustodyWitnessReadinessBlockReason, CustodyWitnessRenewalLogAction,
        CustodyWitnessRenewalLogState, CustodyWitnessRenewalRetryAction,
        CustodyWitnessRenewalRetrySchedule, CustodyWitnessRenewalRetryState,
        CustodyWitnessRenewalStatus, CustodyWitnessRuntimeTelemetry, DataPlaneReceiveFailureAction,
        DirectoryChainStore, MemChainDispatchGateError, MemChainStorageRequirement,
        PeerHttpClients, PeerStoreCacheDocument, PeerStoreCachePersistOutcome,
        PeerStoreVerifiedClientDeliveryAnchor, PeerStoreVerifiedClientDeliveryAnchorState,
        PeerStoreVerifiedClientDeliveryCacheEvidence,
        PeerStoreVerifiedClientDeliveryExternalWitnessDecision, RequiredApiListenerExit,
        RuntimeTaskRegistry, RuntimeTaskShutdownOutcome, RuntimeTaskShutdownReport, Server,
        SystemdNotifier, BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS,
        BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS, BLIND_RELAY_STARTUP_WARMUP_MAX_CANDIDATES,
        COORDINATOR_LEASE_PRODUCTION_SAFETY_SECS, DATA_PLANE_RECV_FAILURE_LIMIT,
        DIRECTORY_OPERATOR_HTTP_PROFILE, DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS,
        DIRECTORY_SYNC_HTTP_PROFILE, DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
        HTTP_TOO_EARLY_STATUS_CODE, MEMCHAIN_SYNC_HTTP_PROFILE,
        ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION,
        ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION, ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION,
        ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION, THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
        TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION, VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT,
        VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT,
        VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION,
        VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION,
    };
    use crate::api::chat_peer::{
        build_chat_peer_router, PeerBlindRelayRequest, PeerBlindRelayResponse,
        PeerChatRelayReceiptV2, PeerChatRelayRequest, PeerChatRelayRequestV2,
        PeerChatRelayRequestV3, PeerChatRelayResponse, PeerChatRelayResponseV2,
    };
    use crate::api::memchain_peer::{
        announce_current_record_commitment_tip_for_test, build_memchain_peer_router,
        CommitmentReconciliationOutcome,
    };
    use crate::api::{
        decode_bounded_json_response, read_bounded_http_response, BoundedHttpResponseError,
        PEER_ACK_RESPONSE_MAX_BYTES,
    };
    use crate::config_chat_relay::ChatRelayConfig;
    use crate::error::{RuntimeTaskJoinFailureKind, ServerError};
    use crate::services::chat_relay::{
        VerifiedSubmitAdmission, VerifiedSubmitCacheLookup,
        VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS,
    };
    use crate::services::chat_relay_anonymous_mailbox_source::ExactAnonymousMailboxTargetPin;
    use aeronyx_core::crypto::handshake::{
        create_client_hello, verify_server_hello, DefaultHandshakeCrypto, HandshakeCrypto,
    };
    use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto};
    use aeronyx_core::crypto::{EphemeralKeyPair, IdentityKeyPair, IdentityPublicKey};
    use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};
    use aeronyx_core::ledger::{RecordCommitmentBlockV1, GENESIS_PREV_HASH};
    use aeronyx_core::protocol::anonymous_mailbox::{
        encode_anonymous_mailbox_terminal_frame, AnonymousMailboxLeaseCreateV1,
        AnonymousMailboxTerminalFrameV1, AnonymousMailboxTicketIssueV1,
    };
    use aeronyx_core::protocol::auth::TIMESTAMP_WINDOW_SECS;
    use aeronyx_core::protocol::chat::{
        encode_envelope, BlindRelayDeliveryReceipt, ChatContentType, ChatEnvelope,
    };
    use aeronyx_core::protocol::discovery::{
        AnonymousMailboxWorkPolicyError, DirectoryCommitmentBlockV1,
        DirectoryDescriptorCommitmentV1, DirectoryDescriptorInclusionProofV1,
        RouteDomainAttestationCertificateV1, RouteDomainAttestationV1,
    };
    use aeronyx_core::protocol::memchain::{
        ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1, MemChainMessage,
        CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1,
        CHAT_VERIFIED_SUBMIT_REJECTED_V1,
    };
    use aeronyx_core::protocol::messages::CLIENT_HELLO_SIZE;
    use aeronyx_core::protocol::onion::is_onion_blob;
    use aeronyx_core::protocol::{
        MessageType, NodeBootstrapSnapshot, NodeCapability, NodeCapacity, NodeDescriptor,
        NodeDiscoveryMessage, NodeProtocolFeature, OnionRoutePurpose, SignedNodeDescriptor,
        PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2,
    };
    use aeronyx_transport::{Transport, UdpTransport};
    use axum::{
        body::{to_bytes, Body},
        extract::{Request, State},
        http::{header::CONTENT_LENGTH, HeaderValue, StatusCode},
        middleware::{self, Next},
        response::Response,
        routing::{get, post},
        Json, Router,
    };
    use sha2::{Digest, Sha256};
    use std::io::{self, Write};
    use std::net::Ipv4Addr;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use tokio::net::TcpListener;
    use tracing_subscriber::{fmt::MakeWriter, prelude::*};

    #[cfg(target_os = "linux")]
    use std::os::unix::net::UnixDatagram;

    use crate::api::discovery::{DiscoveryLocalCapabilityStatus, GossipResponse};
    use crate::config::{DiscoveryConfig, MemChainConfig, MemChainMode, ServerConfig};
    use crate::services::chat_relay_mailbox::{
        AnonymousMailboxCustodyRepository, AnonymousMailboxStoreError,
        AnonymousMailboxTicketIssueOutcome,
    };

    // [SERVER-DECOMPOSITION-PHASE0 2026-09-14 by Codex] Keep the first
    // behavior-preserving extraction limited to required-listener tests.
    mod anonymous_mailbox;
    mod capability_endpoint;
    mod chat_pull;
    mod chat_relay;
    mod commitment_custody;
    mod discovery_gossip;
    mod handshake_http;
    mod memchain;
    mod peer_cache;
    mod remaining;
    mod runtime_supervision;
    mod session_vpn;
    mod startup_runtime;
    mod verified_submit;

    use crate::services::memchain::MemoryStorage;
    use crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound;
    use crate::services::{
        AofWriter, ChatRelayService, DirectoryReplicaGossipAnnouncement, DirectoryReplicaStore,
        DirectoryReplicaSyncRuntime, HandshakeLimiter, MemPool, PeerStore, PeerStoreImportReport,
        SessionManager,
    };
    use crate::voucher_verifier::VoucherVerifier;
    use tokio::sync::Mutex as TokioMutex;

    #[derive(Clone, Default)]
    struct CapturedServerLogs(Arc<Mutex<Vec<u8>>>);

    struct CapturedServerLogWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedServerLogWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("captured server log mutex")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'writer> MakeWriter<'writer> for CapturedServerLogs {
        type Writer = CapturedServerLogWriter;

        fn make_writer(&'writer self) -> Self::Writer {
            CapturedServerLogWriter(Arc::clone(&self.0))
        }
    }

    fn capture_server_info_logs(operation: impl FnOnce()) -> String {
        let captured = CapturedServerLogs::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .without_time()
            .with_writer(captured.clone())
            .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
                // [SERVER-LOG-TEST-TARGET 2026-09-25 by Codex] The handshake
                // rejection emitter lives in the extracted data-plane module.
                metadata.level() <= &tracing::Level::INFO
                    && (metadata.target().ends_with("server")
                        || metadata.target().ends_with("server::data_plane_runtime"))
            }));
        let subscriber = tracing_subscriber::registry().with(layer);
        crate::with_scoped_test_subscriber(subscriber, operation);
        let bytes = captured
            .0
            .lock()
            .expect("captured server log mutex")
            .clone();
        String::from_utf8(bytes).expect("captured server logs are UTF-8")
    }

    fn capture_core_handshake_debug_logs(operation: impl FnOnce()) -> String {
        let captured = CapturedServerLogs::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .without_time()
            .with_writer(captured.clone())
            .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
                metadata.level() <= &tracing::Level::DEBUG
                    && metadata.target().ends_with("crypto::handshake")
            }));
        let subscriber = tracing_subscriber::registry().with(layer);
        crate::with_scoped_test_subscriber(subscriber, operation);
        let bytes = captured
            .0
            .lock()
            .expect("captured core handshake log mutex")
            .clone();
        String::from_utf8(bytes).expect("captured core handshake logs are UTF-8")
    }

    fn test_peer_http_client() -> Arc<reqwest::Client> {
        // [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Exercise the same
        // proxy-free construction boundary as production; `Client::new()` may
        // consult host proxy state and is not valid for isolated node tests.
        Arc::new(
            super::privacy_safe_peer_http_client_builder()
                .build()
                .unwrap(),
        )
    }

    fn test_chat_relay_config(db_path: &std::path::Path) -> ChatRelayConfig {
        let mut config = ChatRelayConfig::default();
        config.enabled = true;
        config.db_path = db_path.to_string_lossy().into_owned();
        config
    }

    fn test_chat_relay_service(
        db_path: &std::path::Path,
        node_secret: [u8; 32],
    ) -> Arc<ChatRelayService> {
        Arc::new(
            ChatRelayService::new(test_chat_relay_config(db_path), node_secret)
                .expect("initialize test chat relay service"),
        )
    }

    const ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW: u64 = 1_800_020_000;

    fn test_anonymous_mailbox_custody_server(enabled: bool, db_path: &std::path::Path) -> Server {
        let mut config = ServerConfig::default();
        config.memchain.chat_relay.anonymous_mailbox.enabled = enabled;
        config.memchain.chat_relay.anonymous_mailbox.db_path =
            db_path.to_string_lossy().into_owned();
        config
            .memchain
            .chat_relay
            .anonymous_mailbox
            .ticket_issue_work_bits = 1;
        Server::new(
            config,
            IdentityKeyPair::from_bytes(&[0xc1; 32]).expect("cleanup test identity"),
            None,
        )
    }

    fn test_anonymous_mailbox_ticket_issue(
        server: &Server,
        expires_at: u64,
    ) -> AnonymousMailboxTicketIssueV1 {
        let depositor = IdentityKeyPair::from_bytes(&[0xc2; 32]).expect("depositor");
        let reader = IdentityKeyPair::from_bytes(&[0xc3; 32]).expect("reader");
        let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
            &[0xc4; 32],
            &depositor.public_key_bytes(),
            &reader.public_key_bytes(),
            2,
            32,
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW + 1_000,
        );
        for nonce in 0..u64::MAX {
            let request = AnonymousMailboxTicketIssueV1::new(
                [0xc5; 16],
                [0xc6; 16],
                server.identity.public_key_bytes(),
                claims,
                ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
                expires_at,
                nonce,
            )
            .expect("ticket issue request");
            if request.proof_digest().expect("proof digest")[0] & 0x80 == 0 {
                return request;
            }
        }
        unreachable!("one-bit work proof is reachable")
    }

    // [ANONYMOUS-MAILBOX-SOURCE-STARTUP-ACCEPTANCE 2026-09-05 by Codex]
    // These fixtures call the production coordinator initializer directly, but
    // own only a synthetic temporary path and an empty in-memory peer view.
    // They neither start Server::run nor contact a peer.
    fn test_anonymous_mailbox_source_server(enabled: bool, db_path: &std::path::Path) -> Server {
        let mut config = ServerConfig::default();
        config.memchain.chat_relay.anonymous_mailbox_source.enabled = enabled;
        config.memchain.chat_relay.anonymous_mailbox_source.db_path =
            db_path.to_string_lossy().into_owned();
        config
            .memchain
            .chat_relay
            .anonymous_mailbox_source
            .terminal_retention_secs = 10;
        Server::new(
            config,
            IdentityKeyPair::from_bytes(&[0xd1; 32]).expect("source cleanup identity"),
            None,
        )
    }

    fn test_anonymous_mailbox_source_descriptor(
        target: &IdentityKeyPair,
        now: u64,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 1_000,
            "source-cleanup-target",
        )
        .with_x25519_kem(target.x25519_public_key_bytes())
        .with_protocol_features([
            NodeProtocolFeature::AnonymousMailboxV1,
            NodeProtocolFeature::OnionReplyV1,
            NodeProtocolFeature::BlindRelaySuccessReceiptV1,
            NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
        ]);
        descriptor.public_endpoint = Some("https://1.1.1.1:443".into());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, target).expect("source cleanup descriptor")
    }

    fn test_anonymous_mailbox_source_terminal_frame(
        target: &IdentityKeyPair,
        request_byte: u8,
    ) -> Vec<u8> {
        let request = AnonymousMailboxTicketIssueV1::new(
            [request_byte; 16],
            [request_byte.wrapping_add(1); 16],
            target.public_key_bytes(),
            [request_byte.wrapping_add(2); 32],
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW + 30,
            u64::from(request_byte),
        )
        .expect("source cleanup request");
        encode_anonymous_mailbox_terminal_frame(&AnonymousMailboxTerminalFrameV1::TicketIssue(
            request,
        ))
        .expect("source cleanup terminal frame")
    }

    fn test_verified_submit_request(
        sender: &IdentityKeyPair,
        request_id: [u8; 16],
        message_id: [u8; 16],
        request_timestamp: u64,
        marker: u8,
    ) -> ChatRelayVerifiedSubmitRequestV1 {
        let mut envelope = ChatEnvelope {
            message_id,
            sender: sender.public_key_bytes(),
            receiver: [marker.wrapping_add(1); 32],
            timestamp: request_timestamp,
            ciphertext: vec![marker; 32],
            nonce: [marker.wrapping_add(2); 24],
            content_type: ChatContentType::Text,
            signature: [0_u8; 64],
        };
        envelope.signature = sender.sign(&envelope.sign_data());
        ChatRelayVerifiedSubmitRequestV1::signed(request_id, envelope, request_timestamp, sender)
            .expect("sign test verified submit request")
    }

    fn test_verified_submit_session(
        sender: &IdentityKeyPair,
        marker: u8,
    ) -> Arc<crate::services::Session> {
        Arc::new(crate::services::Session::new(
            aeronyx_common::types::SessionId::generate(),
            sender.public_key(),
            aeronyx_core::crypto::SessionKey::from_bytes([marker; 32]),
            Ipv4Addr::new(100, 64, 0, marker.max(1)),
            format!("127.0.0.1:{}", 10_000_u16 + u16::from(marker))
                .parse()
                .expect("parse test verified submit endpoint"),
        ))
    }

    fn seed_completed_verified_submit(
        relay: &ChatRelayService,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> ChatRelayVerifiedSubmitResponseV1 {
        assert_eq!(
            relay
                .reserve_verified_submit(request)
                .expect("reserve completed verified submit fixture"),
            VerifiedSubmitAdmission::Reserved
        );
        let response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            request.envelope.message_id,
            false,
            true,
            None,
        );
        relay
            .remember_verified_submit_response(request, &response)
            .expect("persist completed verified submit fixture");
        response
    }

    fn assert_no_verified_submit_delivery_effects(
        relay: &ChatRelayService,
        sender: &IdentityKeyPair,
    ) {
        assert_eq!(
            relay
                .storage_usage()
                .expect("read verified submit storage usage")
                .pending_messages,
            0
        );
        assert!(relay
            .wallet_routes
            .lookup(&sender.public_key_bytes())
            .is_empty());
        assert_eq!(relay.peer_status().outbound_rounds, 0);
    }

    // [CHAT-PULL-ROUTE-AUTHORITY 2026-10-01 by Codex] Exercise operation
    // authorization through the real dispatcher and encrypted UDP replies.
    // Sessions are fixtures, not a claim to exercise the handshake here.
    struct PullRouteFixture {
        relay: Arc<ChatRelayService>,
        wallet: IdentityKeyPair,
        node: IdentityKeyPair,
        session: Arc<crate::services::Session>,
        sessions: Arc<SessionManager>,
        client: Arc<UdpTransport>,
        server: Arc<UdpTransport>,
        _directory: tempfile::TempDir,
    }

    impl PullRouteFixture {
        async fn new(same_identity: bool) -> Self {
            let directory = tempfile::tempdir().unwrap();
            let relay = test_chat_relay_service(&directory.path().join("pull.sqlite3"), [0x67; 32]);
            let wallet = IdentityKeyPair::generate();
            let transport = if same_identity {
                wallet.clone()
            } else {
                IdentityKeyPair::generate()
            };
            let client = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
            let server = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
            let sessions = Arc::new(SessionManager::new(4, Duration::from_secs(60)));
            let session = sessions
                .create(
                    aeronyx_common::types::SessionId::generate(),
                    transport.public_key(),
                    aeronyx_core::crypto::SessionKey::from_bytes([0x68; 32]),
                    Ipv4Addr::new(100, 64, 0, 86),
                    client.local_addr().unwrap(),
                )
                .unwrap();
            Self {
                relay,
                wallet,
                node: IdentityKeyPair::generate(),
                session,
                sessions,
                client,
                server,
                _directory: directory,
            }
        }

        fn pull(&self, v2: bool) -> MemChainMessage {
            use aeronyx_core::protocol::auth::{
                signed_message_digest, DOMAIN_CHAT_PULL, DOMAIN_CHAT_PULL_V2,
            };
            let wallet = self.wallet.public_key_bytes();
            let now = unix_now_secs();
            let after = 0u64.to_le_bytes();
            let limit = 1u32.to_le_bytes();
            let timestamp = now.to_le_bytes();
            if v2 {
                let cursor_len = 0u16.to_le_bytes();
                let signature = self.wallet.sign(&signed_message_digest(
                    DOMAIN_CHAT_PULL_V2,
                    &[&wallet, &after, &cursor_len, &[], &limit, &timestamp],
                ));
                MemChainMessage::ChatPullV2 {
                    wallet,
                    after_timestamp: 0,
                    cursor: Vec::new(),
                    limit: 1,
                    request_timestamp: now,
                    signature,
                }
            } else {
                let cursor = [0; 16];
                let signature = self.wallet.sign(&signed_message_digest(
                    DOMAIN_CHAT_PULL,
                    &[&wallet, &after, &cursor, &limit, &timestamp],
                ));
                MemChainMessage::ChatPull {
                    wallet,
                    after_timestamp: 0,
                    cursor,
                    limit: 1,
                    request_timestamp: now,
                    signature,
                }
            }
        }

        async fn dispatch(&self, message: MemChainMessage) {
            let mut config = MemChainConfig::default();
            config.mode = MemChainMode::Off;
            Server::handle_memchain_message(
                message,
                None,
                None,
                &None,
                &None,
                &config,
                "unused",
                &self.session,
                &self.server,
                &DefaultTransportCrypto::new(),
                &self.sessions,
                &Some(Arc::clone(&self.relay)),
                &Arc::new(PeerStore::new()),
                &self.node.public_key_bytes(),
                &self.node,
                None,
            )
            .await;
        }

        async fn receive_pull(&self, v2: bool) -> Vec<ChatEnvelope> {
            let mut datagram = vec![0; 65_535];
            let (len, _) =
                tokio::time::timeout(Duration::from_secs(2), self.client.recv(&mut datagram))
                    .await
                    .expect("bounded pull response")
                    .unwrap();
            let packet =
                aeronyx_core::protocol::codec::decode_data_packet(&datagram[..len]).unwrap();
            assert_eq!(packet.session_id, *self.session.id.as_bytes());
            let mut clear = vec![0; packet.encrypted_payload.len()];
            let len = DefaultTransportCrypto::new()
                .decrypt(
                    &self.session.session_key,
                    packet.counter,
                    self.session.id.as_bytes(),
                    &packet.encrypted_payload,
                    &mut clear,
                )
                .unwrap();
            let response =
                aeronyx_core::protocol::memchain::decode_memchain(&clear[1..len]).unwrap();
            match response {
                MemChainMessage::ChatPullResponse { envelopes, .. } if !v2 => envelopes,
                MemChainMessage::ChatPullResponseV2 { envelopes, .. } if v2 => envelopes,
                _ => panic!("unexpected pull response kind"),
            }
        }
    }

    async fn assert_cross_identity_pull_route_authority(v2: bool, existing: bool) {
        let fixture = PullRouteFixture::new(false).await;
        let wallet = fixture.wallet.public_key_bytes();
        let sender = IdentityKeyPair::generate();
        let mut envelope =
            test_verified_submit_request(&sender, [0x71; 16], [0x72; 16], unix_now_secs(), 0x73)
                .envelope;
        envelope.receiver = wallet;
        envelope.signature = sender.sign(&envelope.sign_data());
        fixture.relay.store_pending(&envelope).unwrap();
        if existing {
            // An existing delegated route must not have its endpoint rewritten
            // by a portable Pull signature. announce also refreshes its TTL.
            assert!(fixture.relay.wallet_routes.announce(
                &wallet,
                fixture.session.id.clone(),
                "127.0.0.1:9".parse().unwrap()
            ));
        }
        let before = fixture.relay.wallet_routes.lookup(&wallet);
        fixture.dispatch(fixture.pull(v2)).await;
        let received = fixture.receive_pull(v2).await;
        assert_eq!(
            received.len(),
            1,
            "cross-identity signed query remains available"
        );
        assert!(encode_envelope(&received[0]).unwrap() == encode_envelope(&envelope).unwrap());
        assert!(
            fixture.relay.wallet_routes.lookup(&wallet) == before,
            "portable Pull signature must not create or refresh a cross-identity route"
        );
    }

    /// Captures exact v3 ACK bodies and truncates the first one after custody.
    ///
    /// [DIRECT-RELAY-ACK-LOSS 2026-08-15 by Codex] This test-only middleware
    /// runs outside the real target router. `next.run` therefore completes
    /// SQLite custody and replay-cache publication before the injected stream
    /// error reaches the source node.
    #[derive(Default)]
    struct DirectRelayAckLossInjection {
        successful_ack_bodies: Mutex<Vec<Vec<u8>>>,
        successful_acks_seen: AtomicUsize,
    }

    async fn truncate_first_direct_relay_ack(
        State(injection): State<Arc<DirectRelayAckLossInjection>>,
        request: Request,
        next: Next,
    ) -> Response {
        let targets_v3 = request.uri().path() == "/api/chat/peer/relay-v3";
        let response = next.run(request).await;
        if !targets_v3 || !response.status().is_success() {
            return response;
        }

        let (mut parts, body) = response.into_parts();
        let body = to_bytes(body, PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .expect("real target ACK must remain bounded");
        injection
            .successful_ack_bodies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(body.to_vec());
        let ack_index = injection
            .successful_acks_seen
            .fetch_add(1, AtomicOrdering::SeqCst);
        parts.headers.insert(
            CONTENT_LENGTH,
            HeaderValue::from_str(&body.len().to_string()).expect("bounded ACK length header"),
        );
        if ack_index > 0 {
            return Response::from_parts(parts, Body::from(body));
        }

        let prefix_len = body.len().saturating_div(2).max(1).min(body.len());
        let prefix = body.slice(..prefix_len);
        let truncated = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(prefix),
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "injected custody ACK loss",
            )),
        ]);
        Response::from_parts(parts, Body::from_stream(truncated))
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

    fn signed_chat_relay_peer_descriptor(
        endpoint: String,
        sequence: u64,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        signed_chat_relay_peer_descriptor_with_features(endpoint, sequence, expires_at, &[])
    }

    fn signed_chat_relay_peer_descriptor_with_features(
        endpoint: String,
        sequence: u64,
        expires_at: u64,
        features: &[NodeProtocolFeature],
    ) -> SignedNodeDescriptor {
        let peer_identity = IdentityKeyPair::generate();
        signed_chat_relay_peer_descriptor_for_identity(
            endpoint,
            sequence,
            expires_at,
            features,
            &peer_identity,
        )
    }

    fn signed_chat_relay_peer_descriptor_for_identity(
        endpoint: String,
        sequence: u64,
        expires_at: u64,
        features: &[NodeProtocolFeature],
        peer_identity: &IdentityKeyPair,
    ) -> SignedNodeDescriptor {
        // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Receipt tests need the
        // signed descriptor and HTTP ACK to share one explicit target identity.
        let mut descriptor = NodeDescriptor::new(
            peer_identity.public_key_bytes(),
            sequence,
            sequence,
            expires_at,
            "test-peer",
        )
        .with_protocol_features(features.iter().copied());
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        SignedNodeDescriptor::sign(descriptor, peer_identity).unwrap()
    }

    fn signed_probe_peer_descriptor(
        endpoint: String,
        sequence: u64,
        issued_at: u64,
        expires_at: u64,
        capabilities: Vec<NodeCapability>,
        kem_public: [u8; 32],
    ) -> SignedNodeDescriptor {
        let peer_identity = IdentityKeyPair::generate();
        let mut descriptor = NodeDescriptor::new(
            peer_identity.public_key_bytes(),
            sequence,
            issued_at,
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

    const DIRECT_RELAY_RESTART_DRILL_STAGE_ENV: &str =
        "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_STAGE";
    const DIRECT_RELAY_RESTART_DRILL_DB_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_DB";
    const DIRECT_RELAY_RESTART_DRILL_V3_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_V3";
    const DIRECT_RELAY_RESTART_DRILL_V2_ENV: &str = "AERONYX_TEST_DIRECT_RELAY_RESTART_DRILL_V2";
    // [ARCH-SPLIT-VERIFY 2026-10-02 by Codex] Match the worker's extracted test module.
    const DIRECT_RELAY_RESTART_DRILL_WORKER: &str =
        "server::tests::chat_relay::target_bound_v3_restart_drill_subprocess_worker";
    const DIRECT_RELAY_RESTART_DRILL_CRASH_EXIT_CODE: i32 = 73;

    async fn run_direct_relay_restart_drill_child(
        stage: &str,
        db_path: &std::path::Path,
        v3_endpoint: Option<&str>,
        v2_endpoint: Option<&str>,
    ) -> std::process::Output {
        // [DIRECT-RELAY-CRASH-DRILL 2026-08-15 by Codex] Re-execute this exact
        // test binary so each phase owns a genuinely fresh process and no
        // process-local relay state can cross the restart boundary. Reuse the
        // production child isolation policy so the drill cannot inherit
        // systemd readiness, watchdog, or socket-activation authority.
        let executable = std::env::current_exe().expect("resolve restart drill test binary");
        let mut child = crate::isolated_child_command(executable);
        child
            .arg(DIRECT_RELAY_RESTART_DRILL_WORKER)
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            // [CHILD-SELECTION-GUARD 2026-10-02 by Codex] Freeze parsed output.
            .arg("--format=pretty")
            .arg("--color=never")
            .env(DIRECT_RELAY_RESTART_DRILL_STAGE_ENV, stage)
            .env(DIRECT_RELAY_RESTART_DRILL_DB_ENV, db_path)
            .kill_on_drop(true);
        if let Some(endpoint) = v3_endpoint {
            child.env(DIRECT_RELAY_RESTART_DRILL_V3_ENV, endpoint);
        }
        if let Some(endpoint) = v2_endpoint {
            child.env(DIRECT_RELAY_RESTART_DRILL_V2_ENV, endpoint);
        }
        tokio::time::timeout(Duration::from_secs(20), child.output())
            .await
            .expect("restart drill child exceeded its bounded deadline")
            .expect("start restart drill child")
    }

    fn assert_restart_drill_child_crashed(output: &std::process::Output) {
        assert_eq!(
            output.status.code(),
            Some(DIRECT_RELAY_RESTART_DRILL_CRASH_EXIT_CODE),
            "seed worker missed the intentional crash boundary with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn assert_restart_drill_child_succeeded(stage: &str, output: &std::process::Output) {
        assert!(
            output.status.success() && restart_drill_selected_one_worker(&output.stdout),
            "{stage} worker rejected durable restart invariants with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // [CHILD-SELECTION-GUARD 2026-10-02 by Codex] An exit-zero libtest
    // process may have selected no tests. Check the named result and exact
    // counts on NORMAL completion only; crash phases deliberately lack them.
    // This bounds parsing, not the pre-existing child output capture.
    fn restart_drill_selected_one_worker(bytes: &[u8]) -> bool {
        if bytes.len() > 64 * 1024 {
            return false;
        }
        let Ok(output) = std::str::from_utf8(bytes) else {
            return false;
        };
        if output
            .lines()
            .filter(|line| *line == "running 1 test")
            .count()
            != 1
        {
            return false;
        }
        let expected = format!("test {} ... ok", DIRECT_RELAY_RESTART_DRILL_WORKER);
        let mut results = output
            .lines()
            .filter(|line| line.starts_with("test ") && !line.starts_with("test result:"));
        if results.next() != Some(expected.as_str()) || results.next().is_some() {
            return false;
        }
        let mut summaries = output
            .lines()
            .filter(|line| line.starts_with("test result:"));
        let Some(summary) = summaries.next() else {
            return false;
        };
        if summaries.next().is_some() {
            return false;
        }
        let Some(tail) =
            summary.strip_prefix("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; ")
        else {
            return false;
        };
        let Some((filtered, elapsed)) = tail.split_once(" filtered out; finished in ") else {
            return false;
        };
        filtered.parse::<u64>().is_ok()
            && elapsed
                .strip_suffix('s')
                .and_then(|value| value.parse::<f64>().ok())
                .is_some_and(|value| value.is_finite() && value >= 0.0)
    }

    #[test]
    fn successful_child_selection_requires_one_named_test() {
        let valid = format!(
            "\nrunning 1 test\ntest {} ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2170 filtered out; finished in 0.01s\n",
            DIRECT_RELAY_RESTART_DRILL_WORKER,
        );
        assert!(restart_drill_selected_one_worker(valid.as_bytes()));
        assert!(restart_drill_selected_one_worker(
            valid.replace("2170 filtered", "0 filtered").as_bytes()
        ));
        for invalid in [
            valid.replace(DIRECT_RELAY_RESTART_DRILL_WORKER, "wrong::worker"),
            valid.replace("1 passed", "0 passed"),
            valid.replace("1 passed", "11 passed"),
            valid.replace("0 failed", "1 failed"),
            valid.replace("0 ignored", "1 ignored"),
            valid.replace("running 1 test", "running 0 tests"),
            valid.replace(" ... ok", " ... ignored"),
            valid.replace("test result: ok.", "test result: FAILED."),
            valid.replace("0.01s", "NaNs"),
            valid.replace("2170 filtered", "not-a-count filtered"),
            format!("{valid}{valid}"),
            "running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 2171 filtered out; finished in 0.00s\n".into(),
            "Usage: test-binary [OPTIONS]\n".into(),
            format!("{}: test\n1 test, 0 benchmarks\n", DIRECT_RELAY_RESTART_DRILL_WORKER),
            format!("running 1 test\ntest {} ... ok\n", DIRECT_RELAY_RESTART_DRILL_WORKER),
        ] {
            assert!(!restart_drill_selected_one_worker(invalid.as_bytes()));
        }
        assert!(!restart_drill_selected_one_worker(&[0xff]));
        assert!(!restart_drill_selected_one_worker(&vec![
            b' ';
            64 * 1024 + 1
        ]));
    }

    #[tokio::test]
    async fn successful_child_selection_rejects_real_zero_test_exit() {
        // [CHILD-SELECTION-GUARD 2026-10-02 by Codex] Calibrate the gate
        // with an actual exit-zero child that runs no worker, not --list.
        let mut child = crate::isolated_child_command(
            std::env::current_exe().expect("resolve child selection test binary"),
        );
        child
            .arg("__aeronyx_intentionally_missing_worker_selection_guard__")
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--format=pretty")
            .arg("--color=never")
            .kill_on_drop(true);
        let output = tokio::time::timeout(Duration::from_secs(20), child.output())
            .await
            .expect("zero-test child exceeded bounded deadline")
            .expect("start zero-test child");
        assert!(output.status.success(), "negative fixture must exit zero");
        assert!(String::from_utf8_lossy(&output.stdout).contains("running 0 tests"));
        assert!(std::panic::catch_unwind(|| {
            assert_restart_drill_child_succeeded("known zero-test negative", &output);
        })
        .is_err());
    }

    fn open_direct_relay_restart_drill_circuit(relay: &ChatRelayService, first_failure_at: u64) {
        for offset in 0..3 {
            let observed_at = first_failure_at.saturating_add(offset);
            let permit = relay
                .begin_direct_peer_delivery(observed_at)
                .expect("closed circuit should admit failure seed");
            let allows_more =
                relay.complete_direct_peer_delivery(observed_at, permit, false, false, true);
            assert_eq!(allows_more, offset < 2);
        }
    }

    fn open_direct_relay_restart_drill_circuit_at_recovery(
        relay: &ChatRelayService,
        recovery_at: u64,
    ) {
        let cooldown = relay
            .peer_status()
            .direct_peer_retry
            .circuit
            .cooldown_seconds;
        let first_failure_at = recovery_at.saturating_sub(cooldown.saturating_add(2));
        open_direct_relay_restart_drill_circuit(relay, first_failure_at);
    }

    fn terminate_direct_relay_restart_drill_process() -> ! {
        // `process::exit` intentionally skips Rust destructors while preserving
        // committed SQLite transactions. This is a bounded crash injection,
        // not a graceful service recreation.
        std::process::exit(DIRECT_RELAY_RESTART_DRILL_CRASH_EXIT_CODE);
    }

    fn seed_direct_relay_restart_drill_crash(relay: &ChatRelayService) -> ! {
        let now = unix_now_secs();
        relay
            .store_pending(&signed_test_chat_envelope(now))
            .expect("persist encrypted custody before crash");
        open_direct_relay_restart_drill_circuit(relay, now.saturating_sub(2));
        let status = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(status.state, "open");
        assert!(status.restart_protected);

        // [DIRECT-RELAY-CRASH-DRILL 2026-08-15 by Codex] `process::exit`
        // crosses a real process boundary after both safety writes commit.
        terminate_direct_relay_restart_drill_process();
    }

    fn seed_direct_relay_half_open_crash(relay: &ChatRelayService) -> ! {
        let now = unix_now_secs();
        open_direct_relay_restart_drill_circuit_at_recovery(relay, now);

        let permit = relay
            .begin_direct_peer_delivery(now)
            .expect("expired cooldown should commit one half-open lease");
        assert!(permit.is_half_open());
        let circuit = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(circuit.state, "half_open");
        assert_eq!(circuit.half_open_attempted_total, 1);
        assert_eq!(circuit.half_open_failed_total, 0);
        assert!(circuit.restart_protected);

        // [DIRECT-RELAY-HALF-OPEN-CRASH 2026-08-15 by Codex] Deliberately
        // abandon the durable lease without cancellation or completion.
        terminate_direct_relay_restart_drill_process();
    }

    fn seed_direct_relay_half_open_progress_crash(relay: &ChatRelayService) -> ! {
        let now = unix_now_secs();
        open_direct_relay_restart_drill_circuit_at_recovery(relay, now);

        let permit = relay
            .begin_direct_peer_delivery(now)
            .expect("expired cooldown should commit the first recovery probe");
        assert!(permit.is_half_open());
        assert!(relay.complete_direct_peer_delivery(now, permit, false, true, false));
        let circuit = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(circuit.state, "half_open");
        assert_eq!(circuit.opened_total, 1);
        assert_eq!(circuit.half_open_consecutive_successes, 1);
        assert_eq!(circuit.half_open_attempted_total, 1);
        assert_eq!(circuit.half_open_succeeded_total, 1);
        assert_eq!(circuit.half_open_failed_total, 0);
        assert_eq!(circuit.recovered_total, 0);
        assert!(circuit.restart_protected);

        // [DIRECT-RELAY-HALF-OPEN-PROGRESS 2026-08-15 by Codex] Terminate only
        // after the successful completion transition has reached SQLite.
        terminate_direct_relay_restart_drill_process();
    }

    fn seed_direct_relay_checkpoint_table_loss_crash(
        relay: &ChatRelayService,
        db_path: &std::path::Path,
    ) -> ! {
        let circuit = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(circuit.state, "closed");
        assert!(circuit.restart_protected);

        let conn =
            rusqlite::Connection::open(db_path).expect("open relay database for crash drill");
        conn.execute("DROP TABLE relay_direct_peer_circuit_checkpoint", [])
            .expect("remove installed checkpoint table before crash");
        drop(conn);

        // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] Exit without
        // destructors after SQLite commits the destructive schema change.
        terminate_direct_relay_restart_drill_process();
    }

    fn verify_missing_checkpoint_table_restart(db_path: &std::path::Path) {
        assert!(matches!(
            ChatRelayService::new(test_chat_relay_config(db_path), [0x79; 32]),
            Err(
                crate::services::chat_relay::ChatRelayError::CorruptStoredData {
                    field: "direct_peer_circuit_checkpoint_table"
                }
            )
        ));
    }

    async fn verify_direct_relay_restart_drill(relay: &ChatRelayService) {
        let circuit = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(circuit.state, "open");
        assert!(circuit.restart_protected);
        assert!(circuit.checkpoint_loaded_at.is_some());
        assert!(circuit.checkpoint_persisted_at.is_some());

        let receiver = [0x66; 32];
        let (pending, has_more) = relay
            .pull_pending(&receiver, 0, &[0; 16], 2)
            .expect("recover encrypted mailbox after restart");
        assert_eq!(pending.len(), 1);
        assert!(!has_more);
        let recovered_envelope = pending[0].envelope.clone();
        relay
            .store_pending(&recovered_envelope)
            .expect("exact post-restart retry remains idempotent");
        let (after_retry, has_more) = relay
            .pull_pending(&receiver, 0, &[0; 16], 2)
            .expect("read idempotent mailbox after restart");
        assert_eq!(after_retry.len(), 1);
        assert!(!has_more);

        let v3_endpoint =
            std::env::var(DIRECT_RELAY_RESTART_DRILL_V3_ENV).expect("restart drill v3 endpoint");
        let v2_endpoint =
            std::env::var(DIRECT_RELAY_RESTART_DRILL_V2_ENV).expect("restart drill v2 endpoint");
        let now = unix_now_secs();
        let v3_descriptor = signed_chat_relay_peer_descriptor_for_identity(
            v3_endpoint,
            now.saturating_sub(1),
            now.saturating_add(300),
            &[NodeProtocolFeature::DirectPeerRelayTargetBindingV3],
            &IdentityKeyPair::generate(),
        );
        let v2_descriptor = signed_chat_relay_peer_descriptor_for_identity(
            v2_endpoint,
            now.saturating_sub(1),
            now.saturating_add(300),
            &[NodeProtocolFeature::DirectPeerRelayAuthV2],
            &IdentityKeyPair::generate(),
        );
        let v3_node_id = v3_descriptor.node_id();
        let v2_node_id = v2_descriptor.node_id();
        let peer_store = PeerStore::new();
        peer_store
            .upsert_verified(v3_descriptor, now)
            .expect("install restart drill v3 descriptor");
        peer_store
            .upsert_verified(v2_descriptor, now)
            .expect("install restart drill v2 descriptor");
        peer_store.record_route_forward_success(&v3_node_id, now.saturating_sub(1));
        peer_store.record_route_forward_success(&v2_node_id, now.saturating_sub(1));

        let client = test_peer_http_client();
        let accepted = Server::relay_chat_envelope_to_discovered_peers(
            Some(client.as_ref()),
            Some(relay),
            &peer_store,
            &IdentityKeyPair::generate(),
            &signed_test_chat_envelope(now),
        )
        .await;
        assert_eq!(accepted, 0);
        let post_attempt = relay.peer_status();
        assert_eq!(post_attempt.direct_peer_retry.circuit.state, "open");
        assert_eq!(post_attempt.last_outbound_attempted, 0);
        assert_eq!(
            post_attempt.last_outbound_failure_reason.as_deref(),
            Some("peer_relay_circuit_open")
        );

        assert_eq!(
            relay
                .ack_messages(&[recovered_envelope.message_id], &receiver)
                .expect("ack recovered encrypted custody"),
            1
        );
        let (after_ack, has_more) = relay
            .pull_pending(&receiver, 0, &[0; 16], 1)
            .expect("verify recovered custody deletion");
        assert!(after_ack.is_empty());
        assert!(!has_more);
    }

    fn verify_interrupted_half_open_restart(relay: &ChatRelayService) {
        let now = unix_now_secs();
        let circuit = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(circuit.state, "open");
        assert!(circuit.restart_protected);
        assert!(circuit.checkpoint_loaded_at.is_some());
        assert!(circuit.checkpoint_persisted_at.is_some());
        assert_eq!(circuit.opened_total, 2);
        assert_eq!(circuit.half_open_attempted_total, 1);
        assert_eq!(circuit.half_open_succeeded_total, 0);
        assert_eq!(circuit.half_open_failed_total, 1);
        let remaining = circuit
            .open_remaining_seconds
            .expect("interrupted probe must restart a bounded cooldown");
        assert!(remaining > 0);
        assert!(remaining <= circuit.cooldown_seconds);

        let blocked_before = circuit.blocked_total;
        assert!(relay.begin_direct_peer_delivery(now).is_none());
        let blocked = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(blocked.state, "open");
        assert_eq!(blocked.blocked_total, blocked_before.saturating_add(1));
        assert_eq!(blocked.half_open_failed_total, 1);
    }

    fn resume_half_open_progress_after_restart(relay: &ChatRelayService) {
        let now = unix_now_secs();
        let restored = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(restored.state, "half_open");
        assert_eq!(restored.opened_total, 1);
        assert_eq!(restored.half_open_consecutive_successes, 1);
        assert_eq!(restored.half_open_attempted_total, 1);
        assert_eq!(restored.half_open_succeeded_total, 1);
        assert_eq!(restored.half_open_failed_total, 0);
        assert_eq!(restored.recovered_total, 0);
        assert!(restored.restart_protected);
        assert!(restored.checkpoint_loaded_at.is_some());
        assert!(restored.checkpoint_persisted_at.is_some());

        let permit = relay
            .begin_direct_peer_delivery(now)
            .expect("restored progress should admit exactly one serial probe");
        assert!(permit.is_half_open());
        assert!(relay.complete_direct_peer_delivery(now, permit, false, true, false));
        let closed = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(closed.state, "closed");
        assert_eq!(closed.opened_total, 1);
        assert_eq!(closed.half_open_consecutive_successes, 0);
        assert_eq!(closed.half_open_attempted_total, 2);
        assert_eq!(closed.half_open_succeeded_total, 2);
        assert_eq!(closed.half_open_failed_total, 0);
        assert_eq!(closed.recovered_total, 1);
        assert!(closed.restart_protected);
    }

    fn verify_closed_circuit_after_restart(relay: &ChatRelayService) {
        let closed = relay.peer_status().direct_peer_retry.circuit;
        assert_eq!(closed.state, "closed");
        assert_eq!(closed.opened_total, 1);
        assert_eq!(closed.half_open_consecutive_successes, 0);
        assert_eq!(closed.half_open_attempted_total, 2);
        assert_eq!(closed.half_open_succeeded_total, 2);
        assert_eq!(closed.half_open_failed_total, 0);
        assert_eq!(closed.recovered_total, 1);
        assert!(closed.open_remaining_seconds.is_none());
        assert!(closed.restart_protected);
        assert!(closed.checkpoint_loaded_at.is_some());
        assert!(closed.checkpoint_persisted_at.is_some());
    }

    fn directory_gossip_announcement(now: u64) -> DirectoryReplicaGossipAnnouncement {
        let producer = IdentityKeyPair::generate();
        let descriptor =
            signed_chat_relay_peer_descriptor("http://127.0.0.1:9".to_string(), now, now + 300);
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let descriptor_hash = commitment.descriptor_hash;
        let block =
            DirectoryCommitmentBlockV1::new_signed(1, now, [0u8; 32], vec![commitment], &producer)
                .unwrap();
        let block_hash = block.hash();
        let proof =
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, now).unwrap();
        DirectoryReplicaGossipAnnouncement {
            producer: producer.public_key_bytes(),
            block_hash,
            descriptor_hash,
            proof,
        }
    }

    fn directory_gossip_announcements_for_receiver(
        now: u64,
        receiver_producer: [u8; 32],
    ) -> [DirectoryReplicaGossipAnnouncement; 3] {
        // [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] Keep peer-aware
        // fallback test fixtures explicit without obscuring the test flow.
        let mut receiver_announcement = directory_gossip_announcement(now);
        receiver_announcement.producer = receiver_producer;
        let mut first_alternate = directory_gossip_announcement(now);
        first_alternate.producer = [0x92; 32];
        let mut second_alternate = directory_gossip_announcement(now);
        second_alternate.producer = [0x93; 32];
        [receiver_announcement, first_alternate, second_alternate]
    }

    fn gossip_execution<'a>(
        client: &'a reqwest::Client,
        peer_store: &'a PeerStore,
        directory_announcements: &'a [DirectoryReplicaGossipAnnouncement],
        now: u64,
        peer_timeout: Duration,
    ) -> DiscoveryGossipExecution<'a> {
        DiscoveryGossipExecution {
            client,
            peer_store,
            directory_announcements,
            peer_identity_hints: None,
            now,
            snapshot_limit: 8,
            peer_timeout,
        }
    }

    async fn spawn_legacy_gossip_mock(
        delay: Duration,
        calls: Arc<AtomicUsize>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        // [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] Shared mock keeps
        // concurrency tests focused on scheduling rather than route boilerplate.
        let app = Router::new().route(
            "/api/discovery/gossip",
            post(move |Json(_message): Json<NodeDiscoveryMessage>| {
                let calls = Arc::clone(&calls);
                async move {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                    tokio::time::sleep(delay).await;
                    Json(GossipResponse {
                        applied: PeerStoreImportReport::empty(),
                        response: None,
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/api/discovery/gossip",
            listener.local_addr().unwrap()
        );
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, task)
    }
}
