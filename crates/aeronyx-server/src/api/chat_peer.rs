// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer.rs
// ============================================================================
//! # Inter-Node Encrypted Chat Relay API
//!
//! ## Creation Reason
//! Phase 9 connects AeroNyx node discovery to real encrypted message movement.
//! Discovery tells a node which peers advertise `NodeCapability::ChatRelay`;
//! this module exposes the receiving side for those peers.
//!
//! ## Main Functionality
//! - `POST /api/chat/peer/relay`: accepts a signed `ChatEnvelope` from another
//!   AeroNyx node
//! - `POST /api/chat/peer/relay-v2`: additionally authenticates the immediate
//!   previous-hop node over the exact encrypted envelope
//! - `POST /api/chat/peer/relay-v3`: also binds that signature to the selected
//!   target node, preventing cross-node replay of an authenticated request
//! - `POST /api/chat/peer/blind-relay`: accepts a signed `BlindRelayEnvelope`
//!   and forwards only opaque encrypted bytes toward `next_hop`
//! - Verifies the envelope signature before doing any delivery or storage
//! - Durably queues every accepted peer envelope before attempting local live
//!   delivery; the authenticated receiver retires it with `ChatAck`
//! - Delivers the already-durable envelope to locally online receiver devices
//!   when possible, while preserving pull-after-restart recovery
//! - Returns a terminal-signed receipt bound to the exact opaque payload after
//!   successful onion terminal store-and-forward; middle hops only propagate it
//! - [SIGNED-FAILURE-RECEIPT 2026-08-11 by Codex] Signs hop-local failure ACKs
//!   against the exact request while keeping deeper onion topology hidden
//! - [FAILURE-RECEIPT-ANTI-DOWNGRADE 2026-08-11 by Codex] Requires the signed
//!   failure receipt when the exact next-hop descriptor advertises support,
//!   while preserving the legacy path for peers that do not advertise it
//! - [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Signs receipt v2 with a
//!   purpose-separated opaque payload commitment after terminal acceptance;
//!   v1 remains signature-verifiable for rolling-upgrade relay compatibility
//! - [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] Accepts a bounded,
//!   signed Blind Vault `Put` frame as an alternative onion terminal payload,
//!   reusing the existing anonymous lease, quota, TTL, and idempotency service
//! - [MULTIHOP-RECEIPT-VALIDATION 2026-08-01 by Codex] Validates terminal ACKs
//!   against the immediate next hop while allowing a forwarded ACK to carry a
//!   valid downstream terminal receipt through three-hop and longer paths
//! - [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Carries the actual
//!   response-observation time out of retry handling so receipt freshness and
//!   route-health evidence never reuse a stale request-ingress timestamp
//! - [DURABLE-TERMINAL-REPLAY-WINDOW 2026-08-11 by Codex] Starts terminal
//!   receipt and replay retention at durable acceptance, while generation-tags
//!   replay queue entries so a forgotten route cannot evict a newer reuse
//! - [REPLAY-GENERATION-COMPACTION 2026-08-11 by Codex] Uses unique local
//!   generations instead of second-resolution timestamps for replay eviction,
//!   and compacts stale generations under a strict memory bound
//! - [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] Distinguishes in-flight route
//!   retries from completed delivery replays and retains the exact bounded ACK,
//!   so a lost response cannot erase a terminal delivery receipt or create a
//!   false acceptance while the original attempt is still unresolved
//! - [BLIND-RELAY-NO-EVICTION-ADMISSION 2026-08-24 by Codex] Preserves every
//!   unexpired in-flight claim and completed ACK under capacity pressure;
//!   saturation rejects only the new route before relay or terminal effects
//! - [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Binds replay admission
//!   to the complete accepted request and persists only node-secret HMACs
//!   plus an AEAD-sealed ACK, preserving at-most-once effects across restart
//! - [DURABLE-BLIND-RELAY-ADMISSION 2026-08-24 by Codex] Fails the public
//!   blind-relay HTTP gate closed before body parsing when its durable replay
//!   store is unavailable instead of silently falling back to process memory
//! - [SIGNED-ONWARD-ENVELOPE 2026-08-24 by Codex] Verifies the previous-hop
//!   signature on an optional legacy onward envelope before route admission,
//!   preventing ciphertext substitution before this node re-signs the frame
//! - [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Reconciles a crashed
//!   armed claim by repeating the exact idempotent request; deterministic
//!   onion forwarding lets the next hop replay its sealed ACK without effects
//! - [RELAY-ROUTE-RAII 2026-08-11 by Codex] Owns every newly admitted route
//!   through an RAII lease so cancellation, shutdown, and future early-return
//!   paths release in-flight replay state unless a durable ACK is committed
//! - [PEER-RELAY-ADMISSION 2026-08-15 by Codex] Applies configurable,
//!   node-global direct-relay admission before JSON parsing without creating
//!   privacy-sensitive sender, receiver, wallet, or source-address buckets
//! - [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] Applies the same
//!   identity-independent parser-front ceiling to blind relay so permissionless
//!   callers cannot bypass resource protection by rotating node keys
//! - [BLIND-RELAY-BUCKET-FAIRNESS 2026-08-21 by Codex] Makes fixed-memory
//!   previous-hop eviction expiration-aware and preserves active quarantine
//!   evidence under permissionless identity churn
//! - [BLIND-RELAY-MONOTONIC-ABUSE-CLOCK 2026-08-21 by Codex] Enforces
//!   previous-hop rate, decay, quarantine, and LRU windows with process-local
//!   monotonic time so host clock corrections cannot extend or reset policy
//! - [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Runs previous-hop
//!   signature verification and request commitment hashing on a bounded
//!   blocking pool boundary before any identity attribution or signed failure
//! - [PEER-ACK-PRIVACY 2026-08-15 by Codex] Normalizes successful direct
//!   relay ACKs so peers cannot probe receiver presence, device count, or
//!   mailbox/dedup state through legacy compatibility fields
//! - [PREVIOUS-HOP-ATTRIBUTION 2026-08-15 by Codex] Authenticates the claimed
//!   blind-relay previous hop before touching per-node rate, reputation, or
//!   quarantine state, preventing forged node-id poisoning
//! - [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] Adds a separately negotiated
//!   direct-relay endpoint whose node signature binds the complete canonical
//!   encrypted envelope; the legacy endpoint remains available during rollout
//! - [AUTHENTICATED-PEER-FAIRNESS 2026-08-15 by Codex] Applies a bounded
//!   per-node fairness ceiling only after direct-relay v2 authentication while
//!   retaining the global parser-front ceiling against identity rotation
//! - [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Signs a privacy-minimal
//!   receipt after durable direct-relay v2 acceptance so the sender can verify
//!   the selected target node, exact request, and fresh custody evidence
//! - [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Negotiates a v3
//!   request whose previous-hop signature commits to the selected target node,
//!   while retaining v1/v2 endpoints for rolling fleet compatibility
//! - [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Retains a bounded,
//!   short-lived exact custody ACK by opaque request commitment so an ACK-loss
//!   retry cannot consume quota or repeat durable/live delivery
//! - [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Converts parser-front,
//!   authentication, validation, and durable-store rejection diagnostics into
//!   a validated aggregate reason before relay health can export it
//! - [CHAT-PEER-ADMISSION-DOMAIN 2026-08-26 by Codex] Composes global
//!   admission, authenticated fairness, and exact ACK replay through a private
//!   trait-based domain instead of retaining policy inside HTTP orchestration
//! - [CHAT-PEER-ACK-COMPLETION-TTL 2026-08-26 by Codex] Starts exact completed
//!   ACK retention when durable acceptance finishes rather than request ingress
//! - [BLIND-TRANSPORT-DOMAIN 2026-08-26 by Codex] Composes bounded outbound
//!   HTTP through a replaceable trait while retaining route and receipt policy
//! - [BLIND-RESPONSE-DOMAIN 2026-08-26 by Codex] Interprets bounded responses
//!   through a pure policy while orchestration owns I/O and aggregate effects
//! - [BLIND-FORWARD-OBSERVER 2026-08-26 by Codex] Emits write-only aggregate
//!   forwarding observations through a replaceable persistence capability
//! - [PREPARED-TERMINAL-EFFECT 2026-08-30 by Codex] Decodes and authenticates
//!   terminal payloads before arming durable effect recovery; read-only Blind
//!   Vault replies never consume mutation-recovery capacity
//! - [DIRECT-RELAY-VERIFY-ADMISSION 2026-08-30 by Codex] Runs direct previous-
//!   hop and sender signature verification behind bounded blocking admission
//!   so hostile cryptographic work cannot stall asynchronous relay I/O
//! - [RELAY-STORAGE-ADMISSION 2026-08-30 by Codex] Reserves bounded ChatRelay
//!   or Blind Vault execution before arming terminal effects, then keeps each
//!   permit inside its blocking worker through durable completion
//! - [BLIND-RELAY-CRYPTO-DOMAIN 2026-08-30 by Codex] Composes previous-hop
//!   verification, onion peeling, and deterministic onward signing behind one
//!   bounded CPU domain outside the asynchronous I/O runtime
//! - [AUTHENTICATED-ONWARD-DOMAIN 2026-08-30 by Codex] Preserves the private
//!   authenticated request boundary through legacy forwarding so onion-middle
//!   metadata is not redundantly signature-verified on a Tokio I/O worker
//! - [BLIND-RESPONSE-CRYPTO-COMPLETION 2026-08-30 by Codex] Evaluates bounded
//!   downstream ACKs and verifies their hop-local receipts outside Tokio I/O,
//!   with fair completion admission after an outbound route effect is armed
//! - [BLIND-SUCCESS-SIGNING-COMPLETION 2026-08-30 by Codex] Signs hop-local
//!   success receipts in the same fair completion domain without cloning the
//!   opaque request envelope or response carrier
//! - [BLIND-TERMINAL-PROOF-COMPLETION 2026-08-30 by Codex] Commits and signs
//!   terminal delivery evidence together with its hop-local success response
//!   in one worker after durable terminal acceptance
//! - [BLIND-FAILURE-SIGNING-COMPLETION 2026-08-30 by Codex] Signs only
//!   authenticated failure receipts outside Tokio and degrades worker faults
//!   to retryable unsigned backpressure rather than false downgrade evidence
//! - [DIRECT-RECEIPT-SIGNING-COMPLETION 2026-08-31 by Codex] Signs direct
//!   custody receipts in the bounded direct-crypto domain after durable
//!   acceptance, with fair completion admission and retryable failure
//! - [SINGLE-PASS-DIRECT-REQUEST-COMMITMENT 2026-08-31 by Codex] Derives
//!   direct request signatures and replay commitments from one canonical
//!   envelope encoding instead of serializing large ciphertext twice
//! - [OUTBOUND-DIRECT-REQUEST-PREPARATION 2026-08-31 by Codex] Prepares
//!   bounded v1/v2/v3 JSON bodies and authenticated commitments behind
//!   fail-fast CPU admission, leaving only peer selection and I/O in Server
//! - [OUTBOUND-DIRECT-RECEIPT-VERIFICATION 2026-08-31 by Codex] Verifies
//!   bounded custody receipts outside Tokio while preserving local worker
//!   failures as non-peer-attributable typed outcomes
//! - [OUTBOUND-BLIND-REQUEST-PREPARATION 2026-08-31 by Codex] Serializes each
//!   opaque blind-relay request once behind bounded CPU admission and carries
//!   immutable HTTP bytes without retaining a second large request graph
//! - [OUTBOUND-BLIND-RECEIPT-VERIFICATION 2026-08-31 by Codex] Verifies
//!   terminal delivery receipts outside Tokio and distinguishes invalid peer
//!   evidence from local verifier shutdown or worker loss
//! - [CHAT-PEER-OUTBOUND-SPLIT 2026-09-25 by Codex] Keeps immutable outbound
//!   carrier preparation, receipt checks, and next-hop forwarding in a private
//!   module while retaining the existing `chat_peer` call surface
//! - [BLIND-RELAY-CPU-RESERVATION 2026-08-31 by Codex] Composes a process-wide
//!   blind-crypto ceiling with an ingress-only sub-quota so public verification
//!   and completion traffic cannot consume all outbound progress capacity
//! - [PREPARED-BLIND-FORWARD-CARRIER 2026-08-31 by Codex] Serializes and
//!   bounds each hop-to-hop request before arming route effects, then reuses
//!   the same immutable body across exact transport retries
//! - [TERMINAL-DECODE-CPU-DOMAIN 2026-08-31 by Codex] Classifies, decodes,
//!   and validates opaque terminal workloads behind blind CPU admission while
//!   retaining original owned bytes for clone-free purpose-bound receipts
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/chat.rs: `ChatEnvelope`, `BlindRelayEnvelope`,
//!   and bounded envelope encoding
//! - aeronyx-core/src/protocol/memchain.rs: wraps envelope for client delivery
//! - aeronyx-server/src/services/chat_relay.rs: pending queue and dedup logic
//! - aeronyx-server/src/services/blind_vault.rs: anonymous encrypted-object
//!   persistence for receiver-independent store-and-forward
//! - aeronyx-server/src/services/peer_store.rs: verified node descriptors for
//!   next-hop routing
//! - aeronyx-server/src/services/session.rs: active receiver sessions
//! - aeronyx-transport/src/udp.rs: encrypted client packet send path
//!
//! ## Main Logical Flow
//! 1. Peer node posts an already end-to-end encrypted `ChatEnvelope`
//! 2. This node checks size and sender signature
//! 3. The complete signed envelope enters the idempotent SQLite pending queue;
//!    same-ID/different-envelope collisions fail before any receipt is signed
//! 4. Duplicate live deliveries are ignored only after durable byte equality
//!    has been established
//! 5. Online receiver sessions get the durable envelope through the existing
//!    encrypted client transport; `ChatAck` removes it after client persistence
//! 6. Blind relay requests verify the previous-hop signature, decrement TTL,
//!    re-sign with this node key, and POST to the verified `next_hop`
//! 7. Direct-relay v2 requests verify node-key possession before durable relay
//!    processing; v3 additionally rejects requests signed for another target
//!    node; the inner sender signature remains independently mandatory
//!
//! ## Important Note for Next Developer
//! - Never decrypt, inspect, log, store, or report ciphertext contents.
//! - Do not add client public IPs, destination domains, DNS contents, URLs,
//!   browsing history, voucher secrets, private keys, or wallet-level traffic
//!   analytics to this endpoint.
//! - The endpoint is node-to-node plumbing only. Client wire format remains
//!   `MemChainMessage::ChatRelay(ChatEnvelope)`.
//! - Blind relay keeps the relay invariant: route_id / next_hop / ttl /
//!   encrypted_blob / timestamp / signature are handled as routing metadata;
//!   encrypted_blob stays opaque and must not be parsed.
//! - Blind relay rejects immediate self/previous-hop loops using only node-level
//!   route metadata, preserving the "blind relay" invariant while preparing
//!   for future controlled multi-hop/onion routing.
//! - Blind relay keeps a bounded local `route_id` replay cache. Advertised relay
//!   nodes additionally use the node-private ChatRelay SQLite store so replay
//!   reservations and exact sealed ACKs survive restart; diagnostic mode without
//!   ChatRelay remains memory-only. Durable rows contain only node-secret HMACs,
//!   AEAD ciphertext, and timestamps, never raw route ids, endpoints, peers, or
//!   payloads. [BLIND-RELAY-NO-EVICTION-ADMISSION 2026-08-24 by Codex] Capacity
//!   is an admission bound: after expired entries are removed, a full cache
//!   rejects the new route and never evicts an unexpired ACK or live claim.
//! - Blind relay applies one identity-independent parser-front rate ceiling,
//!   followed by previous-hop rate limiting and short quarantine only after
//!   signature verification. This protects commercial nodes from identity
//!   rotation and noisy verified peers without parsing encrypted blobs.
//! - Blind-relay abuse enforcement uses process-local monotonic deadlines.
//!   Unix timestamps are observability projections only and must never become
//!   the authority for inbound request admission, failure decay, process-local
//!   quarantine lifetime, or LRU.
//! - Blind relay reports privacy-safe previous-hop health buckets to PeerStore
//!   so nodeboard can show protection status without route ids, endpoints,
//!   encrypted blobs, or user metadata.
//! - Blind relay only forwards to peers that explicitly advertise
//!   `NodeCapability::ChatRelay`; valid discovery peers without that capability
//!   are treated as unavailable routes.
//! - Blind relay validates next-hop ACK bodies before marking a route forward
//!   successful. HTTP 2xx with `accepted=false` or an unreadable ACK is treated
//!   as `forward_failed`, preserving delivery correctness without logging
//!   route ids, endpoints, encrypted blobs, or user metadata.
//! - Blind relay tests cover rejected and malformed next-hop ACKs so future
//!   routing work cannot accidentally count a bad HTTP 200 response as
//!   successful encrypted message movement.
//! - Blind relay tests cover unresponsive next-hop endpoints so timeout
//!   failures stay retryable, aggregate-only, and never leak endpoint URLs into
//!   audit status.
//! - Blind relay rejects stale or too-far-in-the-future routing timestamps so
//!   old opaque route frames cannot be replayed indefinitely. This uses only
//!   envelope routing metadata and never inspects encrypted blob contents.
//! - Blind relay supports an optional `onward_envelope` for controlled two-hop
//!   experiments. A middle hop accepts an outer frame addressed to itself, then
//!   forwards only the already-opaque onward frame to its next node without
//!   parsing the encrypted blob or learning any user-level receiver identity.
//! - Blind relay forwards only to peers with fresh routeability evidence.
//!   Signed descriptors prove identity/capability, but routeability probes
//!   prove the next hop can actually receive encrypted relay work.
//! - True onion middle-hop recovery may forward to a fresh signed terminal
//!   descriptor before routeability is proven, but it still refuses route-health
//!   quarantined peers. This breaks cold-start proof deadlocks without relaxing
//!   the ordinary blind relay forwarding gate.
//! - Onion terminal hops must successfully hand the peeled `ChatEnvelope` to
//!   the existing chat relay store-and-forward path before ACKing the previous
//!   hop. A successful peel alone is not enough to claim real encrypted message
//!   movement.
//! - [DURABLE-RECEIPT-BOUNDARY 2026-08-15 by Codex] Peer message acceptance
//!   must call `store_pending` before the live-only message-id dedupe check.
//!   Reversing this order can sign a terminal receipt for a conflicting payload
//!   that never entered durable storage. Online delivery remains at-least-once
//!   and is retired by the receiver's existing authenticated `ChatAck`.
//! - Duplicate route IDs are treated as idempotent replay drops, not previous-hop
//!   abuse. Lost ACK retries must not quarantine an otherwise healthy relay.
//! - Relay logs are route-safe: they must not include message IDs, receiver
//!   prefixes, endpoint URLs, raw transport errors, route IDs, encrypted blobs,
//!   or payload-derived strings. Use stable reason buckets only.
//! - Route-specific request ceilings and concurrency gates must wrap the Axum
//!   handlers. Putting them inside a `Json<T>` handler is too late because an
//!   attacker can consume memory and parser work before the guard executes.
//! - Direct-relay v2 authentication is domain-separated and binds the complete
//!   canonical `ChatEnvelope`. Never weaken it to a bearer header or sign only
//!   a message id; either would permit ciphertext substitution or replay across
//!   protocol surfaces.
//! - Per-node direct-relay admission must run after outer node authentication.
//!   Invalid signatures must not create or mutate node buckets. Keep the
//!   process-global parser-front ceiling because permissionless node keys are
//!   not Sybil resistance.
//! - A direct-relay receipt proves only that one node accepted one opaque
//!   authenticated request into durable custody. It must never include user,
//!   receiver, message-id, online-state, endpoint, or payload-size fields.
//! - Next-hop acknowledgement bodies are untrusted and must use the shared
//!   bounded decoder. Never call `Response::json()` directly on peer traffic.
//! - Durable queue count/byte exhaustion is a retryable capacity condition,
//!   not an internal server fault. Preserve the service's privacy-safe reason
//!   bucket while returning HTTP 503 to the previous hop.
//! - Delivery receipts authenticate terminal acceptance only. They must not add
//!   sender, receiver, endpoint, online-state, mailbox-state, or payload-size
//!   fields. Intermediates verify route, freshness, and signature, but only the
//!   source knows the complete route and final payload commitment and can
//!   therefore enforce the final terminal/payload binding.
//! - A successful downstream ACK must describe exactly one completed action:
//!   terminal acceptance or onward forwarding. `accepted=true` without either
//!   action is not delivery evidence and must never improve route reputation.
//! - A structurally valid peer-declared failure proves only that the immediate
//!   transport responded. A failure receipt authenticates only that immediate
//!   response; it must not become blame evidence for a deeper participant.
//!   Invalid, stale, replayed, or wrong-signer receipts are direct protocol
//!   failure evidence against the immediate next-hop route surface.
//! - [FAILURE-RECEIPT-ANTI-DOWNGRADE 2026-08-11 by Codex] A next hop that
//!   advertises `BlindRelayFailureReceiptV1` in its signed descriptor must not
//!   omit the receipt from a handled protocol failure. Treat omission as a
//!   direct downgrade violation against that exact route surface. Peers without
//!   the advertisement remain on the explicit mixed-version compatibility path.
//! - Blind Vault's detailed storage receipt contains stable replica-local lease
//!   metadata and therefore must not be exposed in a multi-hop JSON ACK. The
//!   existing delivery receipt is signed only after `BlindVaultService::put`
//!   succeeds and is bound to the exact encoded Put frame; the source can prove
//!   replica acceptance without revealing the lease or object to middle hops.
//! - Blind Vault terminal failures expose only permanent rejection, replica
//!   capacity, or temporary unavailability. Never forward service errors.
//! - Receipt v2 purpose separation must stay inside the opaque commitment.
//!   Do not add a clear workload label to the propagated ACK: low-cardinality
//!   route purpose would become visible metadata at every middle hop.
//! - [ROUTE-SUCCESS-SURFACE-BINDING 2026-08-10 by Codex] Successful next-hop
//!   forwarding must be recorded against the exact descriptor used to build
//!   the request URL; a concurrent endpoint/KEM rotation must fail closed.
//! - [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Retry completion,
//!   receipt freshness, previous-hop success, and route-health evidence must
//!   use one projected response-observation time returned by the forwarder.
//! - [DURABLE-TERMINAL-REPLAY-WINDOW 2026-08-11 by Codex] Replay eviction must
//!   compare the queued generation with the current route entry;
//!   terminal receipts and replay retention begin only after durable success.
//! - [REPLAY-GENERATION-COMPACTION 2026-08-11 by Codex] A replay queue entry
//!   must carry a unique process-local generation. Timestamps are not unique:
//!   fail/retry can reuse one route id within the same second. Keep the queue
//!   bounded independently from the live route map to prevent stale-entry DoS.
//! - [PEER-RELAY-ADMISSION 2026-08-15 by Codex] Legacy direct relay has no
//!   authenticated previous-hop identity. Its admission limiter must remain
//!   node-global until a negotiated signed v2 contract exists; never emulate
//!   peer identity with user/sender/receiver keys or source IP addresses.
//! - [PEER-ACK-PRIVACY 2026-08-15 by Codex] Direct relay success proves only
//!   durable custody of the opaque envelope. Keep actual duplicate and online
//!   delivery counts in aggregate local health; never return them on peer wire.
//! - [PREVIOUS-HOP-ATTRIBUTION 2026-08-15 by Codex] An unverified claimed
//!   previous-hop key has no attribution authority. Invalid keys/signatures may
//!   increment aggregate rejection telemetry only; they must never consume a
//!   node bucket or mutate that node's route reputation/quarantine state.
//! - [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] The ACK replay cache
//!   stores only request commitments and signed ACKs. Keep its entries and
//!   generation queue independently bounded; stale owners must never mutate a
//!   newer generation, and retries must bypass authenticated quota only after
//!   exact request authentication succeeds.
//! - [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Inbound relay health
//!   accepts only the validated reason type. Keep raw store errors local and
//!   never export request, endpoint, identity, or payload-derived text.
//! - [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Never queue unbounded
//!   signature work or sign a failure receipt for an unauthenticated request.
//!   The owned admission permit must remain inside the blocking worker so task
//!   cancellation cannot release capacity while verification is still active.
//! - Direct-peer ACK replay ownership is isolated in `chat_peer_admission.rs`.
//!   Keep HTTP extraction, signatures, and wire responses in this file while
//!   admission policy remains replaceable and free of user-level dimensions.
//! - [BLIND-REPLAY-CODEC-DOMAIN 2026-08-26 by Codex] Restart-durable blind ACK
//!   encoding, legacy reads, and completed-state validation are isolated in
//!   `chat_peer_replay.rs`; public HTTP errors remain compatibility-stable.
//! - [BLIND-RETRY-DOMAIN 2026-08-26 by Codex] Forward retry policy is isolated
//!   in `chat_peer_retry.rs`; it may use only coarse transport state and signed
//!   route metadata, while I/O and observability remain composed here.
//! - [BLIND-TRANSPORT-DOMAIN 2026-08-26 by Codex] HTTP request execution and
//!   bounded ACK decoding are isolated in `chat_peer_transport.rs`; keep
//!   receipt verification, retry decisions, and route evidence in this file.
//! - [BLIND-RESPONSE-DOMAIN 2026-08-26 by Codex] Receipt verification and
//!   response interpretation are isolated in `chat_peer_response.rs`; this
//!   module only executes typed decisions and records aggregate effects.
//! - [BLIND-FORWARD-OBSERVER 2026-08-26 by Codex] Aggregate retry and route
//!   health writes are isolated in `chat_peer_observer.rs`. Keep the observer
//!   write-only: persistence must never influence forwarding control flow.
//!
//! ## Last Modified
//! v0.83.0-OutboundTransportSplit - Isolate bounded request preparation,
//! receipt verification, and exact next-hop forwarding without wire changes
//! v0.82.0-TerminalDecodeCpuDomain - Classify, decode, and validate terminal
//! workloads in the blind CPU partition while preserving owned proof bytes
//! v0.81.0-PreparedBlindForwardCarrier - Bound one serialized hop request
//! before effect arming and reuse its immutable bytes across exact retries
//! v0.80.0-BlindCryptoReservation - Reserve total blind CPU progress outside
//! the public ingress sub-quota on multi-worker nodes
//! v0.79.0-FailureSigningCompletion - Complete failure response fields and
//! move authenticated receipt signing into bounded completion workers
//! v0.78.0-TerminalProofCompletion - Co-locate terminal payload commitment and
//! both receipt signatures in one bounded completion operation
//! v0.77.0-SuccessSigningCompletion - Move success receipt hashing and Ed25519
//! signing outside Tokio while preserving durable completion ordering
//! v0.76.0-ResponseCryptoCompletion - Move downstream response policy and
//! receipt verification into bounded fair crypto completion workers
//! v0.75.0-EnvelopeSizePreflight - Validate canonical blind-envelope bounds
//! without allocating another ciphertext-sized encoding buffer
//! v0.74.0-StreamingRequestCommitment - Preserve canonical replay commitments
//! while hashing large authenticated requests without a second payload buffer
//! v0.73.0-AuthenticatedOnwardDomain - Remove duplicate onion-middle signature
//! verification and bound every legacy next-hop re-signing operation
//! v0.72.0-BlindRelayCryptoDomain - Bound onion peel and deterministic onward
//! signing with the same fail-fast CPU admission as blind authentication
//! v0.71.0-RelayStorageAdmission - Move synchronous terminal SQLite work out
//! of async workers and reserve bounded execution before effect arming
//! v0.70.0-DirectRelayVerifyAdmission - Bound all direct-relay signature work
//! outside Tokio workers with immediate backpressure and coarse diagnostics
//! v0.69.0-PreparedTerminalEffect - Separate terminal parsing, authentication,
//! effect arming, and execution so malformed work remains safely releasable
//! v0.68.0-OnionDeleteReply - Dispatch signed Blind Vault deletion requests
//! and propagate their fixed-size encrypted terminal receipts
//! v0.67.0-OnionTerminalReply - Propagate fixed-size encrypted terminal
//! responses through durable blind-relay acknowledgements
//! v0.66.0-BlindForwardObserver - Compose aggregate forwarding observations
//! behind a write-only trait without changing route-health attribution
//! v0.65.0-BlindResponseDomain - Compose receipt validation and response
//! decisions behind a pure policy while preserving all observable contracts
//! v0.64.0-BlindTransportDomain - Compose bounded outbound HTTP behind a
//! replaceable trait without changing response, retry, or telemetry contracts
//! v0.63.0-BlindRetryDomain - Compose payload-blind retry classification and
//! deterministic jitter behind a replaceable policy trait
//! v0.62.0-BlindReplayCodecDomain - Move versioned durable ACK storage rules
//! into the replay domain without changing wire or SQLite compatibility
//! v0.61.0-DirectPeerAdmissionDomain - Compose monotonic admission and exact
//! ACK replay; completed ACK TTL now begins at durable completion
//! v0.60.0-RecoverableBlindRelayClaim - Persist a fenced effect boundary so
//! unarmed restart claims recover while ambiguous side effects stay fail-closed
//! v0.59.0-BlindRelayBodyAdmissionOrder - Preserve the fixed 413 contract for
//! known oversized requests before durable replay availability is evaluated
//! v0.58.0-BlindRelayTestAdmissionIsolation - Keep focused route tests bounded
//! without racing for the production-global signature verification semaphore
//! v0.57.0-DurableBlindRelayAdmission - Require the node-private durable replay
//! store before accepting public blind-relay parser or forwarding work
//! v0.56.0-DurableBlindRelayReplay - Persist private route reservations and
//! sealed exact ACKs across restart, including signed legacy onward envelopes
//! v0.55.0-BlindRelayNoEvictionAdmission - Reject new routes at replay-cache
//! saturation without evicting unexpired completed or in-flight evidence
//! v0.54.0-BlindRelayVerifyAdmission - Isolate signature verification and
//! request commitment hashing behind bounded CPU admission; unsigned rejection
//! is mandatory until previous-hop authentication succeeds
//! v0.53.0-BlindRelayMonotonicAbuseClock - Enforce previous-hop rate, decay,
//! quarantine, and LRU windows independently from host wall-clock corrections
//! v0.52.0-BlindRelayBucketFairness - Evict expired/LRU non-quarantined peer
//! buckets without letting one active FIFO head retain stale attacker state
//! v0.51.0-BlindRelayGlobalAdmission - Bound aggregate blind-relay request rate
//! before JSON parsing so permissionless node-key rotation cannot evade limits
//! v0.50.0-RelayHealthReasonBoundary - Enforce validated aggregate inbound
//! failure reasons while preserving legacy heartbeat JSON values
//! v0.49.0-DirectRelayIdempotentRetry - Return exact bounded custody ACKs for
//! authenticated same-request retries without repeating relay side effects
//! v0.48.0-DirectRelayTargetBindingV3 - Bind authenticated direct relay work to
//! the selected target node without breaking v1/v2 rolling compatibility
//! v0.47.0-DirectRelayReceiptV2 - Sign exact target-authored durable-custody
//! evidence for descriptor-negotiated direct relay v2 responses
//! v0.46.0-AuthenticatedPeerFairness - Add bounded post-signature per-node
//! admission while retaining global parser-front protection
//! v0.45.0-DirectRelayAuthV2 - Authenticate direct relay previous-hop node
//! identity through a signed, descriptor-negotiated rolling-upgrade endpoint
//! v0.44.0-PreviousHopAttribution - Verify blind-relay node identity before
//! per-node admission and failure scoring to prevent forged-id quarantine
//! v0.43.0-PeerAckPrivacy - Normalize direct relay success ACKs to durable
//! custody without receiver presence, device-count, or mailbox-state signals
//! v0.42.0-PeerRelayAdmission - Bound direct compatibility relay request rate
//! before JSON parsing using aggregate-only, monotonic process state
//! v0.41.0-PeerRelayReplayWindow - Apply bounded timestamp freshness to direct
//! signed envelopes so captured requests cannot be admitted indefinitely
//! v0.40.0-DurableReceiptBoundary - Persist exact signed peer envelopes before
//! live dedupe so terminal receipts cannot attest to an unstored ID collision
//! v0.39.0-FailureReceiptAntiDowngrade - Enforce signed failure receipts for
//! descriptor-negotiated peers while preserving legacy relay compatibility
//! v0.38.0-SignedFailureReceipt - Authenticate exact hop-local failure ACKs
//! without exposing deeper onion topology or breaking legacy peers
//! v0.37.0-DownstreamFailureAttribution - Keep valid peer-declared downstream
//! failures out of immediate-next-hop reputation while preserving retry classes
//! v0.36.0-RelayAckStateMachine - Require successful downstream ACKs to prove
//! exactly one terminal or forwarding disposition before recording success
//! v0.35.0-ReplayGenerationCompaction - Make same-second route reuse safe and
//! bound stale replay generations independently from live route capacity
//! v0.34.0-DurableTerminalReplayWindow - Bind terminal receipt time to durable
//! acceptance and make replay-cache eviction generation-safe
//! v0.33.0-RelayResponseObservationTime - Bind retry ACK validation and route
//! evidence to response time rather than request-ingress time
//! v0.32.0-RouteSuccessSurfaceBinding - Bound next-hop success observations to
//! the exact signed descriptor used for each opaque forward
//! v0.31.0-PurposeBoundReceipt - Sign terminal workload into opaque receipt v2 commitments
//! v0.30.0-BlindVaultRetryClass - Stop retrying permanently invalid anonymous
//! writes while preserving coarse capacity and availability failover signals
//! v0.29.0-BlindVaultOnionDispatch - Persist signed anonymous Blind Vault Put
//! frames at onion terminals without exposing lease/object metadata in ACKs
//! v0.28.0-MultihopReceiptValidation - Keep direct-terminal signer checks while
//! accepting valid downstream terminal receipts propagated through longer paths
//! v0.27.0-PeerEndpointPolicy - Enforce canonical public-IP-only next-hop URLs
//! v0.26.0-SignedDeliveryReceipt - Sign exact terminal payload acceptance and propagate verified receipts
//! v0.25.0-DurableQueueCapacity - Classify global pending-store quota exhaustion
//! v0.24.0-PublicRequestBounds - Bound peer bodies and concurrency before JSON extraction
//! v0.23.0-RouteSafeRelayLogs - Remove user/route-adjacent values and raw transport errors from chat peer logs
//! v0.22.0-BlindRelayDuplicateIdempotence - Keep duplicate route drops out of previous-hop quarantine scoring
//! v0.21.0-OnionMiddleRouteabilityRecovery - Allow true onion middle recovery through fresh signed descriptors unless route-quarantined
//! v0.20.0-OnionTerminalDeliveryAck - Require terminal onion delivery before accepted ACK
//! v0.19.0-BlindRelayDescriptorHint - Allow signed next-hop descriptor hints for controlled two-hop proofs
//! v0.18.0-BlindRelayRouteabilityGate - Require fresh routeability evidence before next-hop forwarding
//! v0.17.0-BlindRelayOnwardEnvelope - Add optional two-hop middle-hop forwarding
//! v0.16.0-BlindRelayTimestampFreshness - Reject stale/future opaque route frames
//! v0.15.0-BlindRelayTimeoutTest - Cover unresponsive next-hop retry exhaustion
//! v0.14.0-BlindRelayMalformedAckTest - Cover malformed 2xx next-hop ACK as forward_failed
//! v0.13.0-BlindRelayAckValidation - Require accepted next-hop ACK before route success
//! v0.12.0-BlindRelayCapabilityGate - Require next hop to advertise ChatRelay before forwarding
//! v0.11.0-PeerHealthSummary - Report previous-hop abuse buckets to PeerStore
//! v0.10.0-BlindRelayAbuseGuard - Add previous-hop rate limit and quarantine
//! v0.9.0-BlindRelayReplayGuard - Drop duplicate route_id frames idempotently
//! v0.8.0-BlindRelayLoopGuard - Reject immediate self/previous-hop relay loops
//! v0.7.0-BlindRelayRetryStats - Report retry recovery/exhaustion to PeerStore status
//! v0.6.0-BlindRelayRetryJitter - Retry transient next-hop blind relay failures with privacy-safe jitter
//! v0.5.0-BlindRelayRouteHealth - Feed next-hop success/failure back into PeerStore scoring
//! v0.4.0-BlindRelayBackpressure - Added blind relay in-flight pressure gate
//! v0.3.0-BlindRelayEndpoint - Added node-to-node opaque blind relay endpoint
//! v0.2.0-PeerRelayHealth - Record inbound peer relay health counters
//! v0.1.0-DiscoveryPhase9 - Initial inter-node encrypted chat relay endpoint
// ============================================================================

use std::{
    io::{self, Write},
    sync::{atomic::AtomicUsize, Arc, OnceLock},
    time::Instant,
};

#[cfg(test)]
use std::time::Duration;

use aeronyx_core::crypto::transport::{
    DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD,
};
use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::{
    decode_envelope, encode_envelope, validate_blind_relay_envelope_size,
    BlindRelayDeliveryReceipt, BlindRelayEnvelope, BlindRelayFailureReceipt,
    BlindRelaySuccessReceipt, ChatEnvelope, BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION,
};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::discovery::{
    NodeProtocolFeature, SignedNodeDescriptor, SignedPrivateOnionRecipientAuthorizationV1,
};
use aeronyx_core::protocol::memchain::MEMCHAIN_MAGIC;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{
    is_onion_blob, try_open_onion_layer, OnionRoutePurpose, VerifiedOnionRoute,
};
use aeronyx_core::protocol::{
    decode_blind_vault_frame, is_blind_vault_frame, is_onion_reply_request, BlindVaultFrame,
    BlindVaultPutRequest, DataPacket, NodeCapability, OnionReplyProofMode,
};
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;
use axum::{
    body::HttpBody,
    extract::{DefaultBodyLimit, Extension, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{debug, warn};

#[cfg(test)]
use super::chat_peer_abuse_guard::PREVIOUS_HOP_FAILURE_THRESHOLD as BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD;
use super::chat_peer_abuse_guard::{
    BlindRelayAbuseDecision, BlindRelayAbuseDomain, BlindRelayAbusePolicy,
};
use super::chat_peer_admission::{
    AuthenticatedPeerRelayReplayStart, DirectPeerAdmissionDomain, DirectPeerAdmissionPolicy,
};
use super::chat_peer_anonymous_mailbox::{
    AnonymousMailboxTerminalFailure, PreparedAnonymousMailboxTerminal,
};
#[cfg(test)]
use super::chat_peer_replay::REPLAY_CAPACITY_FOR_TESTS as MAX_BLIND_RELAY_SEEN_ROUTES;
use super::chat_peer_replay::{
    decode_durable_blind_relay_response, encode_durable_blind_relay_response,
    validate_completed_blind_relay_response, BlindRelayReplayDomain, BlindRelayReplayMutation,
    BlindRelayReplayRegistry, BlindRelayRouteReplayDecision,
};
use super::chat_peer_response::BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS;
#[cfg(test)]
use super::chat_peer_response::{
    validate_downstream_delivery_receipt, validate_downstream_failure_receipt,
    BLIND_RELAY_FAILURE_RECEIPT_MAX_AGE_SECS, BLIND_RELAY_FAILURE_RECEIPT_MAX_FUTURE_SKEW_SECS,
};
use super::chat_peer_retry::BlindRelayDownstreamFailure;
#[cfg(test)]
use super::chat_peer_retry::DEFAULT_MAX_ATTEMPTS_FOR_TESTS as MAX_BLIND_RELAY_FORWARD_ATTEMPTS;
use super::chat_peer_terminal_reply::{
    prepare_blind_vault_inline_reply, PreparedTerminalReply, TerminalReplyFailure,
};
use crate::api::InFlightRequestGuard;
use crate::config_chat_relay::{
    DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE, DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE,
};
use crate::services::chat_relay::{
    BlindRelayRouteAdmission, ChatRelayError, ChatRelayInboundFailureReason,
};
use crate::services::chat_relay_mailbox::AnonymousMailboxCustodyRepository;
use crate::services::peer_store::PeerStore;
use crate::services::{
    BlindVaultPutFailureClass, BlindVaultServiceError, ChatRelayService, Session, SessionManager,
    SharedBlindVaultService,
};
use crate::services::reverse_onion_queue::{
    ReverseOnionQueueAdmission, ReverseOnionQueueItem,
};
use crate::services::reverse_onion_queue_db::{ReverseOnionQueueDb, ReverseOnionQueueDbError};

mod outbound_transport;
use outbound_transport::{
    blind_peer_relay_url, blind_relay_response_observed_at, forward_blind_relay_with_retry,
    prepare_blind_relay_forward_request,
};
pub(crate) use outbound_transport::{
    blind_relay_delivery_receipt_is_valid, prepare_exact_peer_blind_relay_http_request,
    prepare_peer_blind_relay_http_request_with, prepare_peer_chat_relay_request_v1,
    prepare_peer_chat_relay_request_v2, prepare_peer_chat_relay_request_v3,
    verify_blind_relay_delivery_receipt, verify_peer_chat_relay_receipt,
    BlindRelayDeliveryReceiptVerificationFailure, BlindRelayRequestPreparationError,
    BlindRelayRequestPreparationFailure, DirectRelayReceiptVerificationFailure,
    DirectRelayRequestPreparationFailure, PreparedAuthenticatedPeerChatRelayHttpRequest,
    PreparedPeerBlindRelayHttpRequest, PreparedPeerChatRelayHttpRequest,
};

// ============================================
// Constants
// ============================================

/// Maximum bincode-encoded envelope bytes accepted from another node.
///
/// This mirrors the protocol decode limit and protects the JSON endpoint from
/// carrying huge opaque payloads. Encrypted files should use blob storage, not
/// the peer envelope relay path.
const MAX_PEER_CHAT_ENVELOPE_BYTES: usize = 128 * 1024;

/// Maximum ordinary peer-relay JSON body accepted before deserialization.
///
/// `ChatEnvelope` has a 128 KiB binary ceiling, while JSON byte arrays are
/// substantially larger. This transport allowance preserves valid envelopes
/// without allowing an untrusted peer to allocate an unbounded request body.
const PEER_CHAT_REQUEST_BODY_MAX_BYTES: usize = 768 * 1024;

/// Maximum blind-relay JSON body accepted before deserialization.
///
/// A two-hop request can contain two independently bounded 192 KiB opaque
/// blobs plus a signed descriptor. JSON byte arrays can expand to roughly four
/// characters per byte, so 2 MiB is the narrow safe ceiling for that contract.
const PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES: usize = 2 * 1024 * 1024;

/// Domain separator for the direct peer-relay v2 node-auth signature.
const PEER_CHAT_RELAY_AUTH_V2_DOMAIN: &[u8] = b"AeroNyx/peer-chat-relay-auth/v2";

/// Domain separator for target-bound direct peer-relay v3 authentication.
const PEER_CHAT_RELAY_AUTH_V3_DOMAIN: &[u8] = b"AeroNyx/peer-chat-relay-auth/v3";

/// Domain separator for exact direct peer-relay request commitments.
const PEER_CHAT_RELAY_REQUEST_COMMITMENT_V2_DOMAIN: &[u8] =
    b"AeroNyx/peer-chat-relay-request-commitment/v2";

/// Domain separator for exact target-bound direct relay commitments.
const PEER_CHAT_RELAY_REQUEST_COMMITMENT_V3_DOMAIN: &[u8] =
    b"AeroNyx/peer-chat-relay-request-commitment/v3";

/// Domain separator for target-authored direct peer-relay receipts.
const PEER_CHAT_RELAY_RECEIPT_V2_DOMAIN: &[u8] = b"AeroNyx/peer-chat-relay-receipt/v2";

/// Current direct peer-relay durable receipt version.
const PEER_CHAT_RELAY_RECEIPT_V2_VERSION: u8 = 2;

/// Direct receipts are online acknowledgements, not durable bearer tokens.
const PEER_CHAT_RELAY_RECEIPT_MAX_AGE_SECS: u64 = 120;

/// Small clock-skew allowance for a target node whose clock is ahead.
const PEER_CHAT_RELAY_RECEIPT_MAX_FUTURE_SKEW_SECS: u64 = 30;

/// Maximum ordinary peer-relay requests allowed in parser/handler execution.
const MAX_IN_FLIGHT_PEER_CHAT_REQUESTS: usize = 64;

/// HTTP 425 remains unavailable as a named constant in the pinned http crate.
const HTTP_TOO_EARLY_STATUS_CODE: u16 = 425;

/// Maximum concurrent blind relay requests handled by this process.
///
/// Blind relay is intentionally opaque and can carry large encrypted blobs, so
/// it needs a hard in-flight cap before future multi-hop routing increases the
/// possible fanout. This is local backpressure only; callers should retry with
/// jitter at the transport/client layer.
const MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS: usize = 64;

/// Hard ceiling for concurrent blind-relay cryptographic workers.
///
/// [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] The runtime derives a
/// smaller CPU-aware value from this cap. Keeping it below the HTTP in-flight
/// ceiling prevents signature, onion peel, or forwarding-signature work from
/// occupying Tokio workers or creating an unbounded blocking-task backlog.
const MAX_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT: usize = 8;

/// Process-wide total admission for CPU-bound blind-relay cryptography.
static BLIND_RELAY_CRYPTO_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

/// Ingress-only sub-quota within the process-wide blind-relay CPU budget.
///
/// [BLIND-RELAY-CPU-RESERVATION 2026-08-31 by Codex] Public ingress must not
/// consume every total permit on multi-worker hosts. Outbound route creation
/// and post-effect receipt completion need one reserved progress edge so a
/// saturated relay can still originate and conclusively finish blind work.
static BLIND_RELAY_INGRESS_CRYPTO_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

/// Hard ceiling for concurrent direct-relay CPU workers.
///
/// [DIRECT-RELAY-VERIFY-ADMISSION 2026-08-30 by Codex] Direct and blind relay
/// have separate half-CPU partitions so one public surface cannot starve the
/// other. Together they approximate host parallelism; one-core nodes retain
/// one worker per surface so either protocol can still make progress.
const MAX_DIRECT_RELAY_CPU_OPERATIONS_IN_FLIGHT: usize = 8;

/// Process-wide admission for direct authentication, signing, and encoding.
static DIRECT_RELAY_CPU_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

/// Maximum concurrent blocking ChatRelay custody operations.
const MAX_CHAT_RELAY_STORAGE_OPERATIONS_IN_FLIGHT: usize = 8;

/// Admission before scheduling synchronous ChatRelay SQLite work.
static CHAT_RELAY_STORAGE_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

/// Maximum concurrent blocking Blind Vault terminal operations.
const MAX_BLIND_VAULT_TERMINAL_OPERATIONS_IN_FLIGHT: usize = 8;

/// Admission before scheduling synchronous Blind Vault crypto/SQLite work.
static BLIND_VAULT_TERMINAL_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

/// Domain for the complete authenticated blind request, including onward data.
const BLIND_RELAY_AUTHENTICATED_REQUEST_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindRelay-AuthenticatedRequest-v1";

/// Maximum accepted age for an opaque blind-relay routing frame.
///
/// This is intentionally based only on `BlindRelayEnvelope.timestamp`, a signed
/// routing metadata field. It does not inspect or derive anything from the
/// encrypted blob, preserving the blind relay invariant while reducing replay
/// risk for commercial node operators.
const BLIND_RELAY_MAX_ENVELOPE_AGE_SECS: u64 = 10 * 60;

/// Small clock-skew allowance for peers whose clocks run slightly ahead.
const BLIND_RELAY_MAX_FUTURE_SKEW_SECS: u64 = 120;

/// Trusted, source-local admission for the one supported private hop shape
/// S -> local relay R -> configured private recipient P.
///
/// [PRIVATE-RECIPIENT-ADMISSION 2026-10-04 by Codex] This capability is never
/// decoded from a peer request. Startup must construct it from configured,
/// authenticated R/P descriptors and P-signed authorization; ordinary public
/// relay remains unchanged when it is `None`.
#[derive(Clone)]
pub(crate) struct PrivateBlindRelayAdmission {
    local_relay_node_id: [u8; 32],
    relay_descriptor: SignedNodeDescriptor,
    recipient_descriptor: SignedNodeDescriptor,
    authorization: SignedPrivateOnionRecipientAuthorizationV1,
    purpose: OnionRoutePurpose,
    allowed_sources: Arc<[[u8; 32]]>,
    queue: Arc<ReverseOnionQueueDb>,
    queue_admission: Arc<Semaphore>,
    authority_commitment: [u8; 32],
    route_cap_secs: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PrivateBlindRelayAdmissionError {
    #[error("private recipient admission rejected")]
    Rejected,
}

impl PrivateBlindRelayAdmission {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        local_relay_node_id: [u8; 32],
        relay_descriptor: SignedNodeDescriptor,
        recipient_descriptor: SignedNodeDescriptor,
        authorization: SignedPrivateOnionRecipientAuthorizationV1,
        purpose: OnionRoutePurpose,
        allowed_sources: Vec<[u8; 32]>,
        queue: Arc<ReverseOnionQueueDb>,
        queue_max_in_flight: usize,
        route_cap_secs: u64,
        now: u64,
    ) -> Result<Self, PrivateBlindRelayAdmissionError> {
        if local_relay_node_id == [0; 32]
            || relay_descriptor.node_id() != local_relay_node_id
            || recipient_descriptor.node_id() == [0; 32]
            || recipient_descriptor.node_id() == local_relay_node_id
            || recipient_descriptor.descriptor.public_endpoint.is_some()
            || allowed_sources.is_empty()
            || queue_max_in_flight == 0
            || route_cap_secs == 0
            || purpose != OnionRoutePurpose::BlindVaultPull
        {
            return Err(PrivateBlindRelayAdmissionError::Rejected);
        }
        if relay_descriptor.verify_at(now).is_err()
            || recipient_descriptor.verify_at(now).is_err()
            || authorization
                .verify_at(
                    &relay_descriptor,
                    &recipient_descriptor,
                    purpose.as_str(),
                    now,
                )
                .is_err()
        {
            return Err(PrivateBlindRelayAdmissionError::Rejected);
        }
        let mut authority_hasher = Sha256::new();
        authority_hasher.update(b"AeroNyx-PrivateBlindRelay-Admission-v1");
        authority_hasher.update(purpose.as_str().as_bytes());
        authority_hasher.update(
            relay_descriptor
                .encode_canonical()
                .map_err(|_| PrivateBlindRelayAdmissionError::Rejected)?,
        );
        authority_hasher.update(
            recipient_descriptor
                .encode_canonical()
                .map_err(|_| PrivateBlindRelayAdmissionError::Rejected)?,
        );
        authority_hasher.update(
            authorization
                .encode_canonical()
                .map_err(|_| PrivateBlindRelayAdmissionError::Rejected)?,
        );
        let authority_commitment: [u8; 32] = authority_hasher.finalize().into();
        for source in &allowed_sources {
            if *source == [0; 32]
                || *source == local_relay_node_id
                || *source == recipient_descriptor.node_id()
                || VerifiedOnionRoute::from_signed_private_recipient_descriptors(
                    *source,
                    &relay_descriptor,
                    &recipient_descriptor,
                    &authorization,
                    purpose,
                    now,
                )
                .is_err()
            {
                return Err(PrivateBlindRelayAdmissionError::Rejected);
            }
        }
        Ok(Self {
            local_relay_node_id,
            relay_descriptor,
            recipient_descriptor,
            authorization,
            purpose,
            allowed_sources: allowed_sources.into(),
            queue,
            queue_admission: Arc::new(Semaphore::new(queue_max_in_flight)),
            authority_commitment,
            route_cap_secs,
        })
    }

    pub(crate) fn recipient_node_id(&self) -> [u8; 32] {
        self.recipient_descriptor.node_id()
    }

    pub(crate) fn source_allowed(&self, source: [u8; 32]) -> bool {
        self.allowed_sources.iter().any(|allowed| *allowed == source)
    }

    pub(crate) fn route_deadline(
        &self,
        envelope_timestamp: u64,
        now: u64,
    ) -> Result<u64, PrivateBlindRelayAdmissionError> {
        let freshness = envelope_timestamp
            .checked_add(BLIND_RELAY_MAX_ENVELOPE_AGE_SECS)
            .ok_or(PrivateBlindRelayAdmissionError::Rejected)?;
        let local_cap = envelope_timestamp
            .checked_add(self.route_cap_secs)
            .ok_or(PrivateBlindRelayAdmissionError::Rejected)?;
        let deadline = self
            .authorization
            .expires_at()
            .min(self.relay_descriptor.descriptor.expires_at)
            .min(self.recipient_descriptor.descriptor.expires_at)
            .min(freshness)
            .min(local_cap);
        if envelope_timestamp >= deadline || deadline <= now {
            return Err(PrivateBlindRelayAdmissionError::Rejected);
        }
        Ok(deadline)
    }

    pub(crate) fn queue(&self) -> &Arc<ReverseOnionQueueDb> {
        &self.queue
    }

    pub(crate) fn try_queue_permit(&self) -> Result<OwnedSemaphorePermit, PrivateBlindRelayAdmissionError> {
        self.queue_admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| PrivateBlindRelayAdmissionError::Rejected)
    }

    pub(crate) fn local_relay_node_id(&self) -> [u8; 32] {
        self.local_relay_node_id
    }

    pub(crate) fn authority_commitment(&self) -> [u8; 32] {
        self.authority_commitment
    }

    pub(crate) fn purpose(&self) -> OnionRoutePurpose {
        self.purpose
    }
}
// ============================================
// State / Request / Response Types
// ============================================

#[derive(Clone)]
struct ChatPeerState {
    chat_relay: Option<Arc<ChatRelayService>>,
    /// Optional anonymous ciphertext store used only for declared Blind Vault
    /// terminal frames. Absence is fail-closed and never falls back to chat.
    blind_vault: Option<SharedBlindVaultService>,
    anonymous_mailbox: Option<Arc<dyn AnonymousMailboxCustodyRepository>>,
    private_recipient_admission: Option<Arc<PrivateBlindRelayAdmission>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_relay_in_flight: Arc<AtomicUsize>,
    blind_relay_replay_registry: Arc<dyn BlindRelayReplayRegistry>,
    /// [CHAT-PEER-ABUSE-DOMAIN 2026-08-26 by Codex] Blind relay rate and
    /// quarantine state are composed behind a payload-blind policy boundary.
    blind_relay_abuse_guard: Arc<dyn BlindRelayAbusePolicy>,
}

#[derive(Clone)]
struct PeerRelayRequestGate {
    in_flight: Arc<AtomicUsize>,
    /// [CHAT-PEER-ADMISSION-DOMAIN 2026-08-26 by Codex] Policy and exact
    /// replay ownership are composed behind one replaceable capability.
    admission: Arc<dyn DirectPeerAdmissionPolicy>,
    chat_relay: Option<Arc<ChatRelayService>>,
}

impl PeerRelayRequestGate {
    fn new(
        requests_per_minute: u32,
        authenticated_requests_per_minute: u32,
        chat_relay: Option<Arc<ChatRelayService>>,
    ) -> Self {
        Self {
            in_flight: Arc::new(AtomicUsize::new(0)),
            admission: Arc::new(DirectPeerAdmissionDomain::new(
                requests_per_minute,
                authenticated_requests_per_minute,
            )),
            chat_relay,
        }
    }

    fn admit(&self, now: Instant) -> bool {
        self.admission.admit_global(now)
    }

    fn record_rejected(&self, reason: ChatRelayInboundFailureReason) {
        if let Some(relay) = self.chat_relay.as_ref() {
            relay.record_peer_relay_inbound_rejected_typed(now_secs(), reason);
        }
    }

    fn admit_authenticated(&self, node_id: [u8; 32], now: Instant) -> bool {
        self.admission.admit_authenticated(node_id, now)
    }

    fn begin_authenticated_replay(
        &self,
        request_commitment: [u8; 32],
        now: Instant,
    ) -> AuthenticatedPeerRelayReplayStart {
        self.admission.begin_replay(request_commitment, now)
    }
}

enum BlindRelayRouteStart {
    Acquired(BlindRelayRouteLease),
    Completed(PeerBlindRelayResponse),
}

/// Owns one in-flight route until its durable outcome is published.
///
/// [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Axum request futures may
/// be dropped during shutdown or transport cancellation. `Drop` releases only
/// work that has not crossed an external effect boundary; armed work remains
/// pending for fail-closed replay safety. Successful paths consume the lease
/// through `complete`, atomically replacing the in-flight marker with the exact
/// bounded ACK before disarming cleanup.
struct BlindRelayRouteLease {
    replay_registry: Option<Arc<dyn BlindRelayReplayRegistry>>,
    owner_generation: Option<u64>,
    durable_relay: Option<Arc<ChatRelayService>>,
    route_id: [u8; 16],
    request_commitment: [u8; 32],
    state: BlindRelayRouteLeaseState,
    recovered: bool,
}

/// Lifecycle of one replay-fenced route ownership claim.
///
/// [BLIND-ROUTE-LEASE-STATE 2026-08-30 by Codex] One enum replaces the former
/// `active`/`effect_started` booleans so impossible combinations cannot weaken
/// Drop recovery. Acquired work is releasable, armed work is fail-closed, and
/// completed work owns a durable exact ACK.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlindRelayRouteLeaseState {
    Acquired,
    Armed,
    Completed,
}

impl BlindRelayRouteLease {
    fn local(
        replay_registry: Arc<dyn BlindRelayReplayRegistry>,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        owner_generation: u64,
    ) -> Self {
        Self {
            replay_registry: Some(replay_registry),
            owner_generation: Some(owner_generation),
            durable_relay: None,
            route_id,
            request_commitment,
            state: BlindRelayRouteLeaseState::Acquired,
            recovered: false,
        }
    }

    fn durable(
        durable_relay: Arc<ChatRelayService>,
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        effect_started: bool,
    ) -> Self {
        // [BLIND-ROUTE-RECOVERY-STATUS 2026-08-25 by Codex] A lease is born
        // armed only when it owns a restart takeover. Fresh work becomes armed
        // later through `arm_effect`, so this snapshot distinguishes recovery
        // without retaining a route or peer identifier in telemetry.
        let recovered = effect_started;
        Self {
            replay_registry: None,
            owner_generation: None,
            durable_relay: Some(durable_relay),
            route_id,
            request_commitment,
            state: if effect_started {
                BlindRelayRouteLeaseState::Armed
            } else {
                BlindRelayRouteLeaseState::Acquired
            },
            recovered,
        }
    }

    fn arm_effect(&mut self, now: u64) -> Result<(), BlindRelayError> {
        match self.state {
            BlindRelayRouteLeaseState::Completed => {
                return Err(BlindRelayError::ReplayProtectionUnavailable);
            }
            BlindRelayRouteLeaseState::Armed => return Ok(()),
            BlindRelayRouteLeaseState::Acquired => {}
        }
        if let Some(relay) = self.durable_relay.as_ref() {
            relay
                .arm_blind_relay_route_effect(&self.route_id, &self.request_commitment, now)
                .map_err(|_| BlindRelayError::ReplayProtectionUnavailable)?;
        }
        self.state = BlindRelayRouteLeaseState::Armed;
        Ok(())
    }

    fn complete(
        mut self,
        now: u64,
        response: PeerBlindRelayResponse,
    ) -> Result<(), BlindRelayError> {
        if self.state == BlindRelayRouteLeaseState::Completed {
            return Err(BlindRelayError::ReplayProtectionUnavailable);
        }
        if let Some(relay) = self.durable_relay.as_ref() {
            let encoded = encode_durable_blind_relay_response(&response)
                .map_err(|_| BlindRelayError::ReplayProtectionUnavailable)?;
            relay
                .remember_blind_relay_route_response(
                    &self.route_id,
                    &self.request_commitment,
                    &encoded,
                    now,
                )
                .map_err(|_| BlindRelayError::ReplayProtectionUnavailable)?;
        } else {
            let (Some(registry), Some(owner_generation)) =
                (self.replay_registry.as_ref(), self.owner_generation)
            else {
                return Err(BlindRelayError::ReplayProtectionUnavailable);
            };
            if registry.complete(
                self.route_id,
                self.request_commitment,
                owner_generation,
                now,
                response,
            ) != BlindRelayReplayMutation::Applied
            {
                return Err(BlindRelayError::ReplayProtectionUnavailable);
            }
        }
        self.state = BlindRelayRouteLeaseState::Completed;
        if self.recovered {
            if let Some(relay) = self.durable_relay.as_ref() {
                relay.record_blind_route_recovery_completed(now);
            }
        }
        Ok(())
    }
}

impl Drop for BlindRelayRouteLease {
    fn drop(&mut self) {
        if self.state == BlindRelayRouteLeaseState::Completed {
            return;
        }
        // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] An armed route is
        // ambiguous after cancellation and must remain pending. An unarmed
        // owner has not crossed an external boundary, so release only the
        // exact process-fenced claim; storage failure safely leaves it pending.
        if let Some(relay) = self.durable_relay.as_ref() {
            match self.state {
                BlindRelayRouteLeaseState::Acquired => {
                    let _ = relay.release_unarmed_blind_relay_route(
                        &self.route_id,
                        &self.request_commitment,
                    );
                }
                BlindRelayRouteLeaseState::Armed if self.recovered => {
                    relay.record_blind_route_recovery_deferred(now_secs());
                }
                BlindRelayRouteLeaseState::Armed | BlindRelayRouteLeaseState::Completed => {}
            }
            return;
        }
        if self.state == BlindRelayRouteLeaseState::Armed {
            return;
        }
        if let (Some(registry), Some(owner_generation)) =
            (self.replay_registry.as_ref(), self.owner_generation)
        {
            let _ = registry.release(self.route_id, self.request_commitment, owner_generation);
        }
    }
}

/// Node-to-node encrypted envelope relay request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerChatRelayRequest {
    /// End-to-end encrypted, sender-signed chat envelope.
    pub envelope: ChatEnvelope,
}

/// Authenticated node-to-node encrypted envelope relay request.
///
/// The inner `ChatEnvelope` remains sender-signed end-to-end content. This
/// outer signature proves only which immediate node submitted the opaque
/// envelope and must never be treated as user identity or content authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerChatRelayRequestV2 {
    /// End-to-end encrypted, sender-signed chat envelope.
    pub envelope: ChatEnvelope,
    /// Ed25519 node id of the immediate previous hop.
    pub previous_hop_node_id: [u8; 32],
    /// Previous-hop signature over the domain-separated canonical request.
    #[serde(with = "peer_relay_signature_serde")]
    pub previous_hop_signature: [u8; 64],
}

impl PeerChatRelayRequestV2 {
    /// Builds a request authenticated by the immediate node identity.
    pub fn sign(
        envelope: ChatEnvelope,
        node_identity: &IdentityKeyPair,
    ) -> Result<Self, bincode::Error> {
        Self::sign_with_commitment(envelope, node_identity).map(|(request, _)| request)
    }

    /// Builds the authenticated request and its replay commitment together.
    ///
    /// The signature and commitment intentionally consume the same canonical
    /// bytes. Outbound callers should use this method when they need both so a
    /// large opaque envelope is encoded exactly once.
    pub fn sign_with_commitment(
        envelope: ChatEnvelope,
        node_identity: &IdentityKeyPair,
    ) -> Result<(Self, [u8; 32]), bincode::Error> {
        let previous_hop_node_id = node_identity.public_key_bytes();
        let signing_data = peer_chat_relay_auth_v2_signing_data(&previous_hop_node_id, &envelope)?;
        let previous_hop_signature = node_identity.sign(&signing_data);
        let request_commitment = peer_chat_relay_request_commitment(
            PEER_CHAT_RELAY_REQUEST_COMMITMENT_V2_DOMAIN,
            &signing_data,
            &previous_hop_signature,
        );
        Ok((
            Self {
                envelope,
                previous_hop_node_id,
                previous_hop_signature,
            },
            request_commitment,
        ))
    }

    /// Verifies node-key possession and exact encrypted-envelope binding.
    #[must_use]
    pub fn verify_previous_hop(&self) -> bool {
        let Ok(public_key) = IdentityPublicKey::from_bytes(&self.previous_hop_node_id) else {
            return false;
        };
        let Ok(signing_data) =
            peer_chat_relay_auth_v2_signing_data(&self.previous_hop_node_id, &self.envelope)
        else {
            return false;
        };
        public_key
            .verify(&signing_data, &self.previous_hop_signature)
            .is_ok()
    }

    /// Returns a commitment to the complete accepted request.
    pub fn request_commitment(&self) -> Result<[u8; 32], bincode::Error> {
        let signing_data =
            peer_chat_relay_auth_v2_signing_data(&self.previous_hop_node_id, &self.envelope)?;
        Ok(peer_chat_relay_request_commitment(
            PEER_CHAT_RELAY_REQUEST_COMMITMENT_V2_DOMAIN,
            &signing_data,
            &self.previous_hop_signature,
        ))
    }

    /// Authenticates the previous hop and returns the exact request commitment.
    #[must_use]
    pub fn verified_request_commitment(&self) -> Option<[u8; 32]> {
        let public_key = IdentityPublicKey::from_bytes(&self.previous_hop_node_id).ok()?;
        let signing_data =
            peer_chat_relay_auth_v2_signing_data(&self.previous_hop_node_id, &self.envelope)
                .ok()?;
        public_key
            .verify(&signing_data, &self.previous_hop_signature)
            .ok()?;
        Some(peer_chat_relay_request_commitment(
            PEER_CHAT_RELAY_REQUEST_COMMITMENT_V2_DOMAIN,
            &signing_data,
            &self.previous_hop_signature,
        ))
    }
}

/// Target-bound authenticated node-to-node encrypted envelope relay request.
///
/// [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Unlike v2, the
/// previous-hop signature includes `target_node_id`. A valid request captured
/// by one relay therefore cannot be submitted to another relay as fresh work.
/// The target id is public node-routing metadata; content remains opaque.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerChatRelayRequestV3 {
    /// End-to-end encrypted, sender-signed chat envelope.
    pub envelope: ChatEnvelope,
    /// Ed25519 node id of the immediate previous hop.
    pub previous_hop_node_id: [u8; 32],
    /// Ed25519 node id of the one relay authorized to accept this request.
    pub target_node_id: [u8; 32],
    /// Previous-hop signature over source, target, and canonical envelope.
    #[serde(with = "peer_relay_signature_serde")]
    pub previous_hop_signature: [u8; 64],
}

impl PeerChatRelayRequestV3 {
    /// Builds a target-bound request authenticated by the immediate node.
    pub fn sign(
        envelope: ChatEnvelope,
        target_node_id: [u8; 32],
        node_identity: &IdentityKeyPair,
    ) -> Result<Self, bincode::Error> {
        Self::sign_with_commitment(envelope, target_node_id, node_identity)
            .map(|(request, _)| request)
    }

    /// Builds one target-bound request and its exact replay commitment.
    pub fn sign_with_commitment(
        envelope: ChatEnvelope,
        target_node_id: [u8; 32],
        node_identity: &IdentityKeyPair,
    ) -> Result<(Self, [u8; 32]), bincode::Error> {
        let previous_hop_node_id = node_identity.public_key_bytes();
        let signing_data = peer_chat_relay_auth_v3_signing_data(
            &previous_hop_node_id,
            &target_node_id,
            &envelope,
        )?;
        let previous_hop_signature = node_identity.sign(&signing_data);
        let request_commitment = peer_chat_relay_request_commitment(
            PEER_CHAT_RELAY_REQUEST_COMMITMENT_V3_DOMAIN,
            &signing_data,
            &previous_hop_signature,
        );
        Ok((
            Self {
                envelope,
                previous_hop_node_id,
                target_node_id,
                previous_hop_signature,
            },
            request_commitment,
        ))
    }

    /// Verifies previous-hop possession and binding to the local target.
    #[must_use]
    pub fn verify_for_target(&self, expected_target_node_id: &[u8; 32]) -> bool {
        if &self.target_node_id != expected_target_node_id {
            return false;
        }
        let Ok(public_key) = IdentityPublicKey::from_bytes(&self.previous_hop_node_id) else {
            return false;
        };
        let Ok(signing_data) = peer_chat_relay_auth_v3_signing_data(
            &self.previous_hop_node_id,
            &self.target_node_id,
            &self.envelope,
        ) else {
            return false;
        };
        public_key
            .verify(&signing_data, &self.previous_hop_signature)
            .is_ok()
    }

    /// Returns a commitment to the complete target-bound request.
    pub fn request_commitment(&self) -> Result<[u8; 32], bincode::Error> {
        let signing_data = peer_chat_relay_auth_v3_signing_data(
            &self.previous_hop_node_id,
            &self.target_node_id,
            &self.envelope,
        )?;
        Ok(peer_chat_relay_request_commitment(
            PEER_CHAT_RELAY_REQUEST_COMMITMENT_V3_DOMAIN,
            &signing_data,
            &self.previous_hop_signature,
        ))
    }

    /// Authenticates this exact target and returns the request commitment.
    #[must_use]
    pub fn verified_request_commitment_for_target(
        &self,
        expected_target_node_id: &[u8; 32],
    ) -> Option<[u8; 32]> {
        if &self.target_node_id != expected_target_node_id {
            return None;
        }
        let public_key = IdentityPublicKey::from_bytes(&self.previous_hop_node_id).ok()?;
        let signing_data = peer_chat_relay_auth_v3_signing_data(
            &self.previous_hop_node_id,
            &self.target_node_id,
            &self.envelope,
        )
        .ok()?;
        public_key
            .verify(&signing_data, &self.previous_hop_signature)
            .ok()?;
        Some(peer_chat_relay_request_commitment(
            PEER_CHAT_RELAY_REQUEST_COMMITMENT_V3_DOMAIN,
            &signing_data,
            &self.previous_hop_signature,
        ))
    }
}

/// Versioned direct-relay request awaiting previous-hop authentication.
///
/// This enum intentionally does not implement `Debug` because it contains an
/// end-to-end encrypted user envelope and routing metadata.
enum DirectPeerAuthenticationRequest {
    V2(PeerChatRelayRequestV2),
    V3 {
        request: PeerChatRelayRequestV3,
        expected_target_node_id: [u8; 32],
    },
}

/// Direct relay request after bounded previous-hop signature verification.
struct AuthenticatedDirectPeerRelayRequest {
    envelope: ChatEnvelope,
    previous_hop_node_id: [u8; 32],
    request_commitment: [u8; 32],
}

/// Coarse pre-custody authentication failure safe for public responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectPeerAuthenticationFailure {
    Invalid,
    Backpressure,
    Unavailable,
}

/// Internal failure of the bounded direct-relay cryptographic worker.
///
/// [DIRECT-RECEIPT-SIGNING-COMPLETION 2026-08-31 by Codex] Keep worker
/// availability separate from protocol authentication. A worker failure after
/// durable custody must produce a retryable transport failure, never an
/// unsigned success or an authentication judgement about the previous hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectRelayCryptoFailure {
    Unavailable,
}

impl DirectPeerAuthenticationFailure {
    const fn status_code(self) -> StatusCode {
        match self {
            Self::Invalid => StatusCode::UNAUTHORIZED,
            Self::Backpressure => StatusCode::TOO_MANY_REQUESTS,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    const fn reason_bucket(self) -> &'static str {
        match self {
            Self::Invalid => "peer_auth_invalid",
            Self::Backpressure => "peer_auth_backpressure",
            Self::Unavailable => "peer_auth_unavailable",
        }
    }
}

/// Serde helper preserving a fixed 64-byte Ed25519 signature representation.
mod peer_relay_signature_serde {
    use serde::ser::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error> {
        // [PANIC-FREE-SIGNATURE-SERDE 2026-08-31 by Codex] Preserve the fixed
        // wire representation while keeping serialization total. Even an
        // internal shape invariant must become a typed Serde error rather
        // than an availability-impacting process panic.
        let (lower, upper) = value.split_at(32);
        let lower: &[u8; 32] = lower.try_into().map_err(S::Error::custom)?;
        let upper: &[u8; 32] = upper.try_into().map_err(S::Error::custom)?;
        (*lower, *upper).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 64], D::Error> {
        let (lower, upper): ([u8; 32], [u8; 32]) = Deserialize::deserialize(deserializer)?;
        let mut signature = [0u8; 64];
        signature[..32].copy_from_slice(&lower);
        signature[32..].copy_from_slice(&upper);
        Ok(signature)
    }
}

/// Node-to-node encrypted envelope relay response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerChatRelayResponse {
    /// Whether this node accepted the envelope as valid relay work.
    pub accepted: bool,
    /// Legacy compatibility field; privacy-safe public responses keep it false.
    pub duplicate: bool,
    /// Legacy compatibility field; privacy-safe public responses keep it zero.
    pub delivered_online: usize,
    /// Whether the node accepted durable custody of the opaque envelope.
    pub stored_pending: bool,
}

/// Target-signed proof of authenticated direct relay ciphertext acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerChatRelayReceiptV2 {
    /// Receipt contract version.
    pub version: u8,
    /// SHA-256 commitment to the exact domain-separated authenticated request.
    pub request_commitment: [u8; 32],
    /// Ed25519 identity of the node that accepted durable custody.
    pub accepting_node_id: [u8; 32],
    /// Node wall-clock time when durable acceptance completed.
    pub accepted_at: u64,
    /// Target-node signature over every preceding receipt field.
    #[serde(with = "peer_relay_signature_serde")]
    pub signature: [u8; 64],
}

impl PeerChatRelayReceiptV2 {
    /// Creates a receipt after durable acceptance has already succeeded.
    #[must_use]
    pub fn accepted(
        request_commitment: [u8; 32],
        accepted_at: u64,
        node_identity: &IdentityKeyPair,
    ) -> Self {
        let mut receipt = Self {
            version: PEER_CHAT_RELAY_RECEIPT_V2_VERSION,
            request_commitment,
            accepting_node_id: node_identity.public_key_bytes(),
            accepted_at,
            signature: [0u8; 64],
        };
        receipt.signature = node_identity.sign(&receipt.signing_data());
        receipt
    }

    /// Verifies signature, target binding, request commitment, and freshness.
    pub fn verify_expected(
        &self,
        request: &PeerChatRelayRequestV2,
        expected_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), &'static str> {
        let request_commitment = request
            .request_commitment()
            .map_err(|_| "receipt_binding_invalid")?;
        self.verify_expected_commitment(&request_commitment, expected_node_id, observed_at)
    }

    /// Verifies this receipt against an independently computed request digest.
    ///
    /// [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Receipt v2 is a
    /// generic custody statement over a domain-separated request commitment.
    /// Accepting the commitment directly lets v3 retain the audited receipt
    /// format without pretending its request bytes follow the v2 contract.
    pub fn verify_expected_commitment(
        &self,
        expected_request_commitment: &[u8; 32],
        expected_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), &'static str> {
        if self.version != PEER_CHAT_RELAY_RECEIPT_V2_VERSION {
            return Err("receipt_version_invalid");
        }
        if &self.accepting_node_id != expected_node_id {
            return Err("receipt_binding_invalid");
        }
        let public_key = IdentityPublicKey::from_bytes(&self.accepting_node_id)
            .map_err(|_| "receipt_signature_invalid")?;
        public_key
            .verify(&self.signing_data(), &self.signature)
            .map_err(|_| "receipt_signature_invalid")?;
        if &self.request_commitment != expected_request_commitment {
            return Err("receipt_binding_invalid");
        }
        if self.accepted_at
            > observed_at.saturating_add(PEER_CHAT_RELAY_RECEIPT_MAX_FUTURE_SKEW_SECS)
        {
            return Err("receipt_timestamp_in_future");
        }
        if observed_at.saturating_sub(self.accepted_at) > PEER_CHAT_RELAY_RECEIPT_MAX_AGE_SECS {
            return Err("receipt_timestamp_expired");
        }
        Ok(())
    }

    fn signing_data(&self) -> Vec<u8> {
        let mut data =
            Vec::with_capacity(PEER_CHAT_RELAY_RECEIPT_V2_DOMAIN.len() + 1 + 32 + 32 + 8);
        data.extend_from_slice(PEER_CHAT_RELAY_RECEIPT_V2_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.request_commitment);
        data.extend_from_slice(&self.accepting_node_id);
        data.extend_from_slice(&self.accepted_at.to_be_bytes());
        data
    }
}

/// Authenticated direct relay response. Flattening preserves the legacy JSON
/// fields so request-auth-only v2/v3 senders can ignore the independently
/// negotiated receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerChatRelayResponseV2 {
    /// Privacy-normalized direct relay acceptance fields.
    #[serde(flatten)]
    pub relay: PeerChatRelayResponse,
    /// Target-signed durable-custody evidence on successful acceptance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<PeerChatRelayReceiptV2>,
}

/// Node-to-node blind relay request.
///
/// `previous_hop_node_id` is transport/auth context for signature
/// verification. It is intentionally outside `BlindRelayEnvelope` so the
/// envelope itself remains the minimal route metadata set documented in
/// aeronyx-core. Do not add user ids, receiver wallet ids, domains, URLs, DNS
/// contents, or payload-derived information here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerBlindRelayRequest {
    /// Opaque encrypted relay envelope. `encrypted_blob` must not be parsed.
    pub envelope: BlindRelayEnvelope,
    /// Ed25519 node id that signed this hop.
    pub previous_hop_node_id: [u8; 32],
    /// Optional already-opaque next routing frame for a no-exit middle hop.
    ///
    /// This field is absent for existing single-hop requests. When present,
    /// only the node named by `envelope.next_hop` may use it, and it must still
    /// verify/re-sign the onward frame as normal blind relay work. Do not place
    /// user identifiers, plaintext, DNS data, domains, URLs, packet payloads,
    /// or full route/social-graph metadata here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onward_envelope: Option<BlindRelayEnvelope>,
    /// Optional signed descriptor for the onward `next_hop`.
    ///
    /// This is used by controlled two-hop path proofs when the middle node has
    /// not yet warmed its local routeability cache for the terminal node. The
    /// descriptor is still node-control-plane metadata: it must verify, match
    /// the onward `next_hop`, and advertise `ChatRelay`. It must never carry
    /// user ids, receiver identities, plaintext, DNS data, domains, URLs,
    /// packet payloads, route ids, or social-graph metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onward_descriptor_hint: Option<SignedNodeDescriptor>,
}

/// A blind-relay request whose claimed previous hop has authenticated the
/// exact envelope and whose failure-receipt commitment is already computed.
///
/// [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] This type is deliberately
/// private and cannot be deserialized from the wire. Construct it only through
/// `authenticate_peer_blind_relay_request_with_admission`; that boundary keeps
/// unauthenticated work out of per-peer state and out of the node signing path.
struct AuthenticatedPeerBlindRelayRequest {
    request: PeerBlindRelayRequest,
    failure_request_commitment: [u8; 32],
    /// Commitment to the entire authenticated request, including optional
    /// onward envelope and signed descriptor hint, for private replay binding.
    request_commitment: [u8; 32],
}

/// Streaming adapter for canonical bincode commitment bytes.
struct Sha256CommitmentWriter<'a>(&'a mut Sha256);

impl Write for Sha256CommitmentWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Node-to-node blind relay response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerBlindRelayResponse {
    /// Whether this node accepted the request as valid relay work.
    pub accepted: bool,
    /// Whether this node is the requested next hop.
    pub terminal: bool,
    /// Whether this node forwarded the opaque envelope to another node.
    pub forwarded: bool,
    /// Remaining TTL observed or forwarded by this node.
    pub ttl_remaining: u8,
    /// Privacy-safe coarse result bucket for nodeboard/audits.
    pub reason: Option<String>,
    /// Optional terminal-signed proof bound to the exact delivered payload.
    ///
    /// Older nodes omit this field. Intermediate nodes may propagate it but
    /// must not infer sender, receiver, online state, or payload contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_receipt: Option<BlindRelayDeliveryReceipt>,
    /// Optional immediate-hop success proof bound to this exact response.
    ///
    /// [BLIND-RELAY-SUCCESS-RECEIPT 2026-08-29 by Codex] Each forwarding node
    /// replaces the downstream value with its own signature. Upstream peers
    /// can therefore authenticate their direct hop without learning which
    /// deeper node produced a source-sealed terminal response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success_receipt: Option<BlindRelaySuccessReceipt>,
    /// Optional immediate-hop signature over a coarse failure response.
    ///
    /// The receipt authenticates this responder and exact opaque request only;
    /// it never identifies or assigns blame to a deeper onion participant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_receipt: Option<BlindRelayFailureReceipt>,
    /// Optional fixed-size encrypted terminal response encoded as base64.
    ///
    /// [ONION-REPLY-INLINE 2026-08-28 by Codex] Middle relays propagate this
    /// value unchanged. The terminal identity, workload response, signature,
    /// and logical payload length remain inside authenticated ciphertext.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opaque_terminal_response_b64: Option<String>,
}

#[derive(Debug, thiserror::Error)]
enum ChatPeerRelayError {
    #[error("chat relay disabled")]
    RelayUnavailable,

    #[error("direct relay signature verification capacity exhausted")]
    VerificationBackpressure,

    #[error("direct relay signature verification unavailable")]
    VerificationUnavailable,

    #[error("chat relay durable storage is busy")]
    StorageBackpressure,

    #[error("invalid envelope signature")]
    InvalidSignature,

    #[error("envelope too large: {size} bytes")]
    EnvelopeTooLarge { size: usize },

    #[error("envelope serialization failed")]
    Serialization,

    #[error("chat envelope timestamp expired")]
    TimestampExpired,

    #[error("chat envelope timestamp is too far in the future")]
    TimestampInFuture,

    #[error("pending store failed")]
    StoreFailed,

    #[error("pending store capacity exhausted")]
    PendingCapacity,
}

#[derive(Debug, thiserror::Error)]
enum BlindRelayError {
    #[error("blind relay verification capacity exhausted")]
    Backpressure,

    #[error("invalid previous hop public key")]
    InvalidPreviousHop,

    #[error("invalid blind envelope signature")]
    InvalidSignature,

    #[error("blind envelope too large")]
    EnvelopeTooLarge,

    #[error("ttl exhausted")]
    TtlExhausted,

    #[error("blind envelope timestamp expired")]
    TimestampExpired,

    #[error("blind envelope timestamp is too far in the future")]
    TimestampInFuture,

    #[error("previous hop rate limited")]
    RateLimited,

    #[error("previous hop quarantined")]
    Quarantined,

    #[error("blind relay route is still in flight")]
    RouteInFlight,

    #[error("blind relay replay cache capacity exhausted")]
    ReplayCapacity,

    #[error("blind relay route id conflicts with another authenticated request")]
    ReplayConflict,

    #[error("blind relay completed response is no longer fresh enough to replay")]
    ReplayResponseExpired,

    #[error("blind relay durable replay protection unavailable")]
    ReplayProtectionUnavailable,

    #[error("blind relay route loop detected")]
    RouteLoop,

    #[error("next hop not found")]
    NoRoute,

    #[error("next hop endpoint missing or invalid")]
    InvalidEndpoint,

    #[error("blind relay forward failed")]
    ForwardFailed,

    #[error("onion layer peel failed")]
    OnionPeelFailed,

    #[error("onion terminal payload rejected")]
    OnionTerminalPayloadRejected,

    #[error("onion terminal replica capacity exhausted")]
    OnionTerminalCapacityExhausted,

    #[error("downstream blind relay rejected request")]
    DownstreamRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelayTimestampError {
    Expired,
    InFuture,
}

impl From<BlindRelayDownstreamFailure> for BlindRelayError {
    fn from(failure: BlindRelayDownstreamFailure) -> Self {
        match failure {
            BlindRelayDownstreamFailure::OnionTerminalCapacityExhausted => {
                Self::OnionTerminalCapacityExhausted
            }
            BlindRelayDownstreamFailure::ForwardFailed => Self::ForwardFailed,
            BlindRelayDownstreamFailure::DownstreamRejected => Self::DownstreamRejected,
        }
    }
}

impl BlindRelayError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidPreviousHop
            | Self::InvalidSignature
            | Self::EnvelopeTooLarge
            | Self::TtlExhausted
            | Self::TimestampExpired
            | Self::TimestampInFuture
            | Self::RouteLoop
            | Self::ReplayConflict
            | Self::OnionPeelFailed
            | Self::OnionTerminalPayloadRejected
            | Self::DownstreamRejected => StatusCode::BAD_REQUEST,
            Self::OnionTerminalCapacityExhausted
            | Self::RouteInFlight
            | Self::ReplayCapacity
            | Self::ReplayProtectionUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::ReplayResponseExpired => StatusCode::CONFLICT,
            Self::Backpressure | Self::RateLimited | Self::Quarantined => {
                StatusCode::TOO_MANY_REQUESTS
            }
            Self::NoRoute | Self::InvalidEndpoint => StatusCode::BAD_GATEWAY,
            Self::ForwardFailed => StatusCode::BAD_GATEWAY,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::Backpressure => "backpressure",
            Self::InvalidPreviousHop => "invalid_previous_hop",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge => "envelope_too_large",
            Self::TtlExhausted => "ttl_exhausted",
            Self::TimestampExpired => "timestamp_expired",
            Self::TimestampInFuture => "timestamp_in_future",
            Self::RateLimited => "rate_limited",
            Self::Quarantined => "quarantined",
            Self::RouteInFlight => "route_in_flight",
            Self::ReplayCapacity => "replay_capacity",
            Self::ReplayConflict => "replay_conflict",
            Self::ReplayResponseExpired => "replay_response_expired",
            Self::ReplayProtectionUnavailable => "replay_protection_unavailable",
            Self::RouteLoop => "route_loop",
            Self::NoRoute => "no_route",
            Self::InvalidEndpoint => "invalid_endpoint",
            Self::ForwardFailed => "forward_failed",
            Self::OnionPeelFailed => "onion_peel_failed",
            Self::OnionTerminalPayloadRejected => "onion_terminal_payload_rejected",
            Self::OnionTerminalCapacityExhausted => "onion_terminal_capacity_exhausted",
            Self::DownstreamRejected => "downstream_rejected",
        }
    }
}

impl ChatPeerRelayError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::RelayUnavailable | Self::VerificationUnavailable | Self::StorageBackpressure => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::VerificationBackpressure => StatusCode::TOO_MANY_REQUESTS,
            Self::InvalidSignature
            | Self::EnvelopeTooLarge { .. }
            | Self::Serialization
            | Self::TimestampExpired
            | Self::TimestampInFuture => StatusCode::BAD_REQUEST,
            Self::PendingCapacity => StatusCode::SERVICE_UNAVAILABLE,
            Self::StoreFailed => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::RelayUnavailable => "relay_unavailable",
            Self::VerificationBackpressure => "signature_backpressure",
            Self::VerificationUnavailable => "signature_verification_unavailable",
            Self::StorageBackpressure => "store_backpressure",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge { .. } => "envelope_too_large",
            Self::Serialization => "envelope_serialization_failed",
            Self::TimestampExpired => "timestamp_expired",
            Self::TimestampInFuture => "timestamp_in_future",
            Self::PendingCapacity => "pending_capacity_exhausted",
            Self::StoreFailed => "store_pending_failed",
        }
    }
}

// ============================================
// Router
// ============================================

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod blind_relay;
mod direct_relay;

use blind_relay::acquire_blind_relay_ingress_crypto;
use blind_relay::attach_blind_relay_success_receipt;
use blind_relay::attach_blind_relay_terminal_success_receipts;
use blind_relay::authenticate_blind_relay_envelope;
use blind_relay::authenticate_peer_blind_relay_request;
use blind_relay::authenticate_peer_blind_relay_request_with_admission;
use blind_relay::authenticate_peer_blind_relay_request_with_permits;
use blind_relay::begin_blind_relay_route;
pub(crate) use blind_relay::blind_relay_authenticated_request_commitment;
use blind_relay::blind_relay_crypto_admission;
use blind_relay::blind_relay_crypto_capacity;
use blind_relay::blind_relay_failure_response;
use blind_relay::blind_relay_ingress_crypto_admission;
use blind_relay::blind_relay_ingress_crypto_capacity;
use blind_relay::blind_relay_reason_counts_toward_quarantine;
use blind_relay::blind_vault_terminal_admission;
use blind_relay::build_blind_relay_failure_response;
use blind_relay::build_forwarded_onion_envelope;
use blind_relay::build_forwarded_onion_envelope_from_seed;
use blind_relay::check_blind_relay_previous_hop_allowed;
use blind_relay::complete_blind_relay_crypto;
use blind_relay::complete_blind_relay_route;
use blind_relay::decode_onion_terminal_payload;
use blind_relay::execute_blind_relay_crypto;
use blind_relay::execute_onion_terminal_payload;
use blind_relay::map_anonymous_mailbox_terminal_failure;
use blind_relay::map_blind_vault_put_error;
use blind_relay::map_terminal_chat_preparation_error;
use blind_relay::map_terminal_reply_failure;
use blind_relay::peer_blind_relay_handler;
use blind_relay::peer_blind_relay_request_gate;
use blind_relay::prepare_onion_terminal_payload;
use blind_relay::process_authenticated_peer_blind_relay;
use blind_relay::process_onion_blind_relay;
use blind_relay::process_onion_middle_blind_relay;
#[cfg(test)]
use blind_relay::process_peer_blind_relay;
use blind_relay::record_blind_relay_previous_hop_success;
use blind_relay::record_blind_relay_replay_protection_failure;
use blind_relay::reject_blind_relay_previous_hop;
use blind_relay::rejected_blind_relay_response;
use blind_relay::rejected_blind_relay_response_with_status;
use blind_relay::resolve_blind_relay_next_hop_descriptor;
use blind_relay::run_blind_relay_crypto;
use blind_relay::sign_blind_relay_success_receipt;
use blind_relay::try_acquire_blind_relay_ingress_crypto;
#[cfg(test)]
use blind_relay::validate_blind_relay_envelope;
use blind_relay::validate_blind_relay_metadata;
use blind_relay::validate_blind_relay_timestamp;
use direct_relay::acquire_chat_relay_storage;
use direct_relay::authenticate_direct_peer_relay_request;
use direct_relay::authenticated_peer_relay_response;
use direct_relay::chat_relay_storage_admission;
use direct_relay::complete_direct_relay_crypto;
use direct_relay::direct_relay_cpu_admission;
use direct_relay::durable_peer_acceptance_response;
use direct_relay::execute_direct_relay_crypto;
use direct_relay::map_pending_store_error;
use direct_relay::peer_chat_relay_auth_v2_signing_data;
use direct_relay::peer_chat_relay_auth_v3_signing_data;
use direct_relay::peer_chat_relay_request_commitment;
use direct_relay::peer_relay_handler;
use direct_relay::peer_relay_request_gate;
use direct_relay::peer_relay_response;
use direct_relay::peer_relay_v2_handler;
use direct_relay::peer_relay_v3_handler;
use direct_relay::process_authenticated_peer_relay;
use direct_relay::process_authenticated_peer_relay_with_storage_permit;
use direct_relay::process_peer_relay;
use direct_relay::record_peer_envelope_rejection;
use direct_relay::reject_direct_peer_authentication;
use direct_relay::rejected_direct_peer_relay_v2_response;
use direct_relay::rejected_peer_relay_response;
use direct_relay::send_envelope_to_session;
use direct_relay::validate_peer_envelope;
use direct_relay::validate_peer_envelope_for_relay;
use direct_relay::validate_relay_timestamp;

// [ARCH-SPLIT-VERIFY 2026-10-02 by Codex] Keep router documentation on its public builder.
/// Builds node-to-node encrypted chat relay routes.
pub fn build_chat_peer_router(
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_vault: Option<SharedBlindVaultService>,
) -> Router {
    build_chat_peer_router_with_anonymous_mailbox(
        chat_relay,
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        blind_vault,
        None,
        None,
    )
}

/// Builds peer routes with an explicitly enabled anonymous mailbox terminal.
/// Existing callers retain the default-off constructor above.
pub fn build_chat_peer_router_with_anonymous_mailbox(
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_vault: Option<SharedBlindVaultService>,
    anonymous_mailbox: Option<Arc<dyn AnonymousMailboxCustodyRepository>>,
) -> Router {
    build_chat_peer_router_with_private_recipient_admission(
        chat_relay,
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        blind_vault,
        anonymous_mailbox,
        None,
    )
}

/// Builds peer routes with an optional trusted direct S -> R -> private-P
/// admission capability. Existing callers remain default-off via the builder
/// above; no wire field or generic endpoint rule opts this path in.
pub(crate) fn build_chat_peer_router_with_private_recipient_admission(
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_vault: Option<SharedBlindVaultService>,
    anonymous_mailbox: Option<Arc<dyn AnonymousMailboxCustodyRepository>>,
    private_recipient_admission: Option<Arc<PrivateBlindRelayAdmission>>,
) -> Router {
    let peer_relay_requests_per_minute = chat_relay
        .as_ref()
        .map(|relay| relay.config().peer_relay_requests_per_minute)
        .unwrap_or(DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE);
    let authenticated_peer_relay_requests_per_minute = chat_relay
        .as_ref()
        .map(|relay| relay.config().peer_relay_authenticated_requests_per_minute)
        .unwrap_or(DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE);
    let peer_relay_gate = Arc::new(PeerRelayRequestGate::new(
        peer_relay_requests_per_minute,
        authenticated_peer_relay_requests_per_minute,
        chat_relay.clone(),
    ));
    let state = ChatPeerState {
        chat_relay,
        blind_vault,
        anonymous_mailbox,
        private_recipient_admission,
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let peer_relay_router = Router::new()
        .route("/api/chat/peer/relay", post(peer_relay_handler))
        .route("/api/chat/peer/relay-v2", post(peer_relay_v2_handler))
        .route("/api/chat/peer/relay-v3", post(peer_relay_v3_handler))
        .route_layer(middleware::from_fn_with_state(
            peer_relay_gate,
            peer_relay_request_gate,
        ))
        .layer(DefaultBodyLimit::max(PEER_CHAT_REQUEST_BODY_MAX_BYTES));
    let blind_relay_router = Router::new()
        .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            peer_blind_relay_request_gate,
        ))
        .layer(DefaultBodyLimit::max(
            PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES,
        ));

    peer_relay_router
        .merge(blind_relay_router)
        .with_state(state)
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn signature_verification_capacity(hard_cap: usize) -> usize {
    // Reserve roughly half the reported hardware parallelism for the rest of
    // the node. A one-core process still receives one verification worker.
    let hardware_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    (hardware_threads.saturating_add(1) / 2)
        .max(1)
        .min(hard_cap)
}

/// Owned admission for one blind-relay CPU operation.
///
/// Ingress work holds both fields. Reserved outbound/completion work holds
/// only the total permit. Keeping ownership in the blocking worker prevents a
/// cancelled async request from releasing either quota while CPU work remains.
struct BlindRelayCryptoPermits {
    _total: OwnedSemaphorePermit,
    _ingress: Option<OwnedSemaphorePermit>,
}

impl BlindRelayCryptoPermits {
    fn total_only(total: OwnedSemaphorePermit) -> Self {
        Self {
            _total: total,
            _ingress: None,
        }
    }
}

#[derive(Clone, Copy)]
struct BlindRelayForwardSeed {
    route_id: [u8; 16],
    ttl: u8,
    timestamp: u64,
}

/// Fully re-signed legacy forwarding unit prepared before route effects arm.
struct PreparedLegacyBlindRelayForward {
    envelope: BlindRelayEnvelope,
    onward_envelope: Option<BlindRelayEnvelope>,
}

impl From<&BlindRelayEnvelope> for BlindRelayForwardSeed {
    fn from(envelope: &BlindRelayEnvelope) -> Self {
        Self {
            route_id: envelope.route_id,
            ttl: envelope.ttl,
            timestamp: envelope.timestamp,
        }
    }
}

/// Payload ownership required to create terminal evidence without a clone.
struct TerminalDeliveryProofInput {
    payload: Vec<u8>,
    purpose: OnionRoutePurpose,
}

/// Persists one terminal onion payload without widening what middle hops can
/// observe. The terminal necessarily learns the replica-local Blind Vault
/// lease used for storage, but neither the sender/receiver identity nor the
/// ciphertext plaintext exists in the Put frame.
struct OnionTerminalDelivery {
    purpose: OnionRoutePurpose,
    proof_mode: OnionReplyProofMode,
    opaque_response_b64: Option<String>,
}

/// Validated terminal workload whose effect class is known before execution.
///
/// This enum deliberately has no `Debug` implementation because every variant
/// contains either ciphertext, private capabilities, or sender routing data.
enum PreparedOnionTerminalPayload {
    AnonymousMailbox {
        request: PreparedAnonymousMailboxTerminal,
        execution_permit: OwnedSemaphorePermit,
    },
    BlindVaultReply {
        reply: PreparedTerminalReply,
        execution_permit: OwnedSemaphorePermit,
    },
    LegacyBlindVaultPut {
        request: BlindVaultPutRequest,
        execution_permit: OwnedSemaphorePermit,
    },
    Message {
        envelope: ChatEnvelope,
        storage_permit: OwnedSemaphorePermit,
    },
}

/// Parsed terminal workload before service admission or durable effects.
enum DecodedOnionTerminalPayload {
    AnonymousMailbox(PreparedAnonymousMailboxTerminal),
    BlindVaultReply(PreparedTerminalReply),
    LegacyBlindVaultPut(BlindVaultPutRequest),
    Message(ChatEnvelope),
}

/// Typed failure from pure terminal classification and validation.
enum OnionTerminalDecodeFailure {
    Protocol(BlindRelayError),
    Message(ChatPeerRelayError),
}

/// Terminal operation plus the exact opaque bytes committed by its receipt.
///
/// [TERMINAL-DECODE-CPU-DOMAIN 2026-08-31 by Codex] Ownership keeps the
/// original payload available for a purpose-bound proof while the decoded
/// operation moves independently into storage execution. No payload clone is
/// needed between parsing, durable acceptance, and receipt construction.
struct PreparedOnionTerminalWork {
    proof_payload: Vec<u8>,
    operation: PreparedOnionTerminalPayload,
}

impl PreparedOnionTerminalPayload {
    const fn requires_durable_guard(&self) -> bool {
        match self {
            Self::AnonymousMailbox { .. } => true,
            Self::BlindVaultReply { reply, .. } => reply.effect().requires_durable_guard(),
            Self::LegacyBlindVaultPut { .. } | Self::Message { .. } => true,
        }
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    mod blind_relay;
    mod direct_relay;
    mod other;

    use super::*;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultAdmissionTicket, BlindVaultLeaseAdmissionRequest, BlindVaultOnionPullSession,
        BlindVaultPullRequest, BLIND_VAULT_PROTOCOL_VERSION,
    };
    use aeronyx_core::protocol::chat::ChatContentType;
    use aeronyx_core::protocol::{
        encode_blind_vault_frame, BlindVaultFrame, BlindVaultLeaseCreateRequest,
        BlindVaultPutRequest, NodeCapability, NodeCapacity, NodeDescriptor, SignedNodeDescriptor,
    };
    use aeronyx_core::protocol::discovery::SignedPrivateOnionRecipientAuthorizationV1;
    use aeronyx_core::protocol::onion::{
        open_onion_layer, OnionRoutePurpose, VerifiedOnionRoute,
    };
    use aeronyx_transport::UdpTransport;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use axum::response::IntoResponse;
    use rusqlite::Connection;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Mutex;
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    use sha2::{Digest, Sha256};

    use crate::api::PEER_ACK_RESPONSE_MAX_BYTES;
    use crate::config::{BlindVaultConfig, ChatRelayConfig};
    use crate::services::{BlindVaultLeaseProvisionOutcome, BlindVaultService};

    fn signed_envelope() -> ChatEnvelope {
        signed_envelope_at(now_secs())
    }

    fn signed_envelope_at(timestamp: u64) -> ChatEnvelope {
        let kp = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [0x11u8; 16],
            sender: kp.public_key_bytes(),
            receiver: [0x22u8; 32],
            timestamp,
            ciphertext: b"opaque encrypted payload".to_vec(),
            nonce: [0x33u8; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        envelope.signature = kp.sign(&envelope.sign_data());
        envelope
    }

    fn test_chat_config(path: String) -> ChatRelayConfig {
        ChatRelayConfig {
            enabled: true,
            db_path: path,
            ..ChatRelayConfig::default()
        }
    }

    fn temp_chat_relay(label: &str) -> (Arc<ChatRelayService>, std::path::PathBuf) {
        temp_chat_relay_with_peer_rate(label, DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE)
    }

    fn temp_chat_relay_with_peer_rate(
        label: &str,
        requests_per_minute: u32,
    ) -> (Arc<ChatRelayService>, std::path::PathBuf) {
        temp_chat_relay_with_rates(
            label,
            requests_per_minute,
            DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE,
        )
    }

    fn temp_chat_relay_with_rates(
        label: &str,
        requests_per_minute: u32,
        authenticated_requests_per_minute: u32,
    ) -> (Arc<ChatRelayService>, std::path::PathBuf) {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aeronyx-{label}-{unique}.db"));
        let mut config = test_chat_config(path.to_string_lossy().to_string());
        config.peer_relay_requests_per_minute = requests_per_minute;
        config.peer_relay_authenticated_requests_per_minute = authenticated_requests_per_minute;
        let relay = Arc::new(ChatRelayService::new(config, [7u8; 32]).unwrap());
        (relay, path)
    }

    fn temp_blind_vault_with_put(
        node_identity: &IdentityKeyPair,
        now_ms: u64,
    ) -> (
        tempfile::TempDir,
        Arc<BlindVaultService>,
        BlindVaultPutRequest,
    ) {
        // [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] Use the production
        // admission and mutation pipeline in relay tests. Bypassing lease
        // provisioning would miss signature, quota, expiry, and authority
        // regressions at the protocol boundary this feature is meant to join.
        let directory = tempfile::tempdir().expect("blind vault temp directory");
        let issuer = IdentityKeyPair::generate();
        let config = BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            admission_issuer_public_keys: vec![hex::encode(issuer.public_key_bytes())],
            db_path: directory
                .path()
                .join("blind-vault.db")
                .display()
                .to_string(),
            ..BlindVaultConfig::default()
        };
        let service = Arc::new(
            BlindVaultService::new(config, node_identity.clone()).expect("blind vault service"),
        );
        let write_key = IdentityKeyPair::generate();
        let admin_key = IdentityKeyPair::generate();
        let lease_id = [0x61; 32];
        let mut lease = BlindVaultLeaseCreateRequest::new(
            lease_id,
            [0x62; 16],
            write_key.public_key_bytes(),
            admin_key.public_key_bytes(),
            Sha256::digest([0x63; 32]).into(),
            now_ms + 24 * 60 * 60 * 1_000,
        );
        lease.sign(&admin_key).expect("sign anonymous lease");
        let mut admission = BlindVaultAdmissionTicket::new(
            [0x64; 32],
            issuer.public_key_bytes(),
            now_ms.saturating_sub(1_000),
            now_ms + 60 * 60 * 1_000,
            2 * 24 * 60 * 60 * 1_000,
        );
        admission.sign(&issuer).expect("sign admission ticket");
        assert_eq!(
            service
                .provision_lease_with_admission(
                    &BlindVaultLeaseAdmissionRequest { admission, lease },
                    now_ms,
                )
                .expect("provision anonymous lease"),
            BlindVaultLeaseProvisionOutcome::Created
        );

        let mut put = BlindVaultPutRequest::new(
            lease_id,
            [0x65; 32],
            [0x66; 16],
            vec![0xa5; 4 * 1024],
            now_ms + 60 * 60 * 1_000,
        );
        put.sign(&write_key);
        (directory, service, put)
    }

    fn signed_chat_relay_peer_descriptor_for(
        peer_identity: &IdentityKeyPair,
        endpoint: String,
        sequence: u64,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        signed_peer_descriptor_for(
            peer_identity,
            endpoint,
            sequence,
            expires_at,
            vec![NodeCapability::ChatRelay],
        )
    }

    fn signed_peer_descriptor_for(
        peer_identity: &IdentityKeyPair,
        endpoint: String,
        sequence: u64,
        expires_at: u64,
        capabilities: Vec<NodeCapability>,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            peer_identity.public_key_bytes(),
            sequence,
            sequence,
            expires_at,
            "test-chat-peer",
        );
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = capabilities;
        descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        SignedNodeDescriptor::sign(descriptor, peer_identity).unwrap()
    }

    #[test]
    fn chat_peer_logs_stay_route_safe() {
        let source = [
            include_str!("chat_peer.rs"),
            include_str!("chat_peer/direct_relay.rs"),
            include_str!("chat_peer/blind_relay.rs"),
        ]
        .concat();
        let message_id_log_pattern = concat!("id = %hex::encode(envelope.", "message_id)");
        let receiver_log_pattern = concat!("receiver = %hex::encode(&envelope.", "receiver");
        let raw_error_log_pattern = concat!("error = %", "error");

        assert!(
            !source.contains(message_id_log_pattern),
            "message ids must not be logged by the relay"
        );
        assert!(
            !source.contains(receiver_log_pattern),
            "receiver prefixes must not be logged by the relay"
        );
        assert!(
            !source.contains(raw_error_log_pattern),
            "raw errors may contain endpoint URLs or route-adjacent context"
        );
        assert!(
            source.contains("reason = error.as_str()"),
            "ACK decode failures should use bounded stable reason buckets"
        );
        assert!(
            source.contains("let reason = error.reason_bucket()"),
            "store failures should use service-owned stable reason buckets"
        );
    }

    #[test]
    fn private_pull_core_shape_is_one_route_and_relay_signed() {
        // [PRIVATE-BLIND-VAULT-PULL-RELAY-FIXTURE 2026-10-04 by Codex]
        // This is the real core private Pull shape: one route id, no optional
        // onward carrier, and R re-signs the peeled ttl=1 envelope.
        const NOW: u64 = 1_800_000_000;
        let source = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let relay = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
        let purpose = OnionRoutePurpose::BlindVaultPull;
        let mut relay_body = NodeDescriptor::new(
            relay.public_key_bytes(),
            1,
            NOW - 1,
            NOW + 10_000,
            "test",
        )
        .with_x25519_kem(relay.x25519_public_key_bytes())
        .with_protocol_features(purpose.required_path_protocol_features().iter().copied());
        relay_body.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
        relay_body.public_endpoint = Some("https://relay.invalid".to_owned());
        let mut recipient_body = NodeDescriptor::new(
            recipient.public_key_bytes(),
            1,
            NOW - 1,
            NOW + 10_000,
            "test",
        )
        .with_x25519_kem(recipient.x25519_public_key_bytes())
        .with_protocol_features(
            purpose
                .required_terminal_protocol_features()
                .iter()
                .copied(),
        );
        recipient_body.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::BlindVaultReplica];
        recipient_body.public_endpoint = None;
        let relay_descriptor = SignedNodeDescriptor::sign(relay_body, &relay).unwrap();
        let recipient_descriptor = SignedNodeDescriptor::sign(recipient_body, &recipient).unwrap();
        let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
            &relay_descriptor,
            &recipient_descriptor,
            purpose.as_str(),
            NOW,
            NOW + 9_000,
            &recipient,
        )
        .unwrap();
        let route = VerifiedOnionRoute::from_signed_private_recipient_descriptors(
            source.public_key_bytes(),
            &relay_descriptor,
            &recipient_descriptor,
            &authorization,
            purpose,
            NOW,
        )
        .unwrap();
        let (terminal, _) = BlindVaultOnionPullSession::prepare(
            [0x94; 16],
            recipient.public_key_bytes(),
            BlindVaultPullRequest {
                version: BLIND_VAULT_PROTOCOL_VERSION,
                lease_id: [0x95; 32],
                read_capability: [0x96; 32],
                continuation_cursor: Vec::new(),
                limit: 1,
            },
        )
        .unwrap();
        let envelope = route
            .build_envelope(&terminal, [0x94; 16], NOW, &source)
            .unwrap();
        let (relay_secret, _) = relay.to_x25519();
        let peeled = open_onion_layer(&envelope.encrypted_blob, &relay_secret).unwrap();
        assert_eq!(envelope.route_id, [0x94; 16]);
        assert_eq!(envelope.ttl, 2);
        assert_eq!(peeled.next_hop, Some(recipient.public_key_bytes()));
        let forwarded = build_forwarded_onion_envelope_from_seed(
            BlindRelayForwardSeed::from(&envelope),
            recipient.public_key_bytes(),
            peeled.inner,
            &relay,
        );
        assert_eq!(forwarded.route_id, envelope.route_id);
        assert_eq!(forwarded.timestamp, envelope.timestamp);
        assert_eq!(forwarded.ttl, 1);
        forwarded
            .verify_signature_from(&IdentityPublicKey::from_bytes(&relay.public_key_bytes()).unwrap())
            .unwrap();
    }
}
