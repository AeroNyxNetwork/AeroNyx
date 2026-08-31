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
    BlindRelaySuccessReceipt, ChatEnvelope,
};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::discovery::{NodeProtocolFeature, SignedNodeDescriptor};
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{is_onion_blob, try_open_onion_layer, OnionRoutePurpose};
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
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    time::sleep,
};
use tracing::{debug, warn};

#[cfg(test)]
use super::chat_peer_abuse_guard::PREVIOUS_HOP_FAILURE_THRESHOLD as BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD;
use super::chat_peer_abuse_guard::{
    BlindRelayAbuseDecision, BlindRelayAbuseDomain, BlindRelayAbusePolicy,
};
use super::chat_peer_admission::{
    AuthenticatedPeerRelayReplayStart, DirectPeerAdmissionDomain, DirectPeerAdmissionPolicy,
};
use super::chat_peer_observer::{BlindRelayForwardObserver, PeerStoreBlindRelayForwardObserver};
#[cfg(test)]
use super::chat_peer_replay::REPLAY_CAPACITY_FOR_TESTS as MAX_BLIND_RELAY_SEEN_ROUTES;
use super::chat_peer_replay::{
    decode_durable_blind_relay_response, encode_durable_blind_relay_response,
    validate_completed_blind_relay_response, BlindRelayReplayDomain, BlindRelayReplayMutation,
    BlindRelayReplayRegistry, BlindRelayRouteReplayDecision,
};
#[cfg(test)]
use super::chat_peer_response::{
    validate_downstream_delivery_receipt, validate_downstream_failure_receipt,
    BLIND_RELAY_FAILURE_RECEIPT_MAX_AGE_SECS, BLIND_RELAY_FAILURE_RECEIPT_MAX_FUTURE_SKEW_SECS,
};
use super::chat_peer_response::{
    BlindRelayInvalidResponseKind, BlindRelayResponseContext, BlindRelayResponseDecision,
    BlindRelayResponseDomain, BlindRelayResponsePolicy, BlindRelayResponseSource,
    BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS, BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS,
};
#[cfg(test)]
use super::chat_peer_retry::DEFAULT_MAX_ATTEMPTS_FOR_TESTS as MAX_BLIND_RELAY_FORWARD_ATTEMPTS;
use super::chat_peer_retry::{
    BlindRelayDownstreamFailure, BlindRelayRetryContext, BlindRelayRetryDomain,
    BlindRelayRetryPolicy,
};
use super::chat_peer_terminal_reply::{
    prepare_blind_vault_inline_reply, PreparedTerminalReply, TerminalReplyFailure,
};
use super::chat_peer_transport::{BlindRelayTransport, ReqwestBlindRelayTransport};
use crate::api::{canonical_peer_http_url, peer_endpoint_is_public_ip, InFlightRequestGuard};
use crate::config_chat_relay::{
    DEFAULT_AUTHENTICATED_PEER_RELAY_REQUESTS_PER_MINUTE, DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE,
};
use crate::services::chat_relay::{
    BlindRelayRouteAdmission, ChatRelayError, ChatRelayInboundFailureReason,
};
use crate::services::peer_store::PeerStore;
use crate::services::{
    BlindVaultPutFailureClass, BlindVaultServiceError, ChatRelayService, Session, SessionManager,
    SharedBlindVaultService,
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

/// Process-wide admission for CPU-bound blind-relay cryptography.
static BLIND_RELAY_CRYPTO_ADMISSION: OnceLock<Arc<Semaphore>> = OnceLock::new();

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
// ============================================
// State / Request / Response Types
// ============================================

#[derive(Clone)]
struct ChatPeerState {
    chat_relay: Option<Arc<ChatRelayService>>,
    /// Optional anonymous ciphertext store used only for declared Blind Vault
    /// terminal frames. Absence is fail-closed and never falls back to chat.
    blind_vault: Option<SharedBlindVaultService>,
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

fn peer_chat_relay_auth_v2_signing_data(
    previous_hop_node_id: &[u8; 32],
    envelope: &ChatEnvelope,
) -> Result<Vec<u8>, bincode::Error> {
    let encoded_envelope = encode_envelope(envelope)?;
    let mut signing_data =
        Vec::with_capacity(PEER_CHAT_RELAY_AUTH_V2_DOMAIN.len() + 32 + 8 + encoded_envelope.len());
    signing_data.extend_from_slice(PEER_CHAT_RELAY_AUTH_V2_DOMAIN);
    signing_data.extend_from_slice(previous_hop_node_id);
    signing_data.extend_from_slice(&(encoded_envelope.len() as u64).to_be_bytes());
    signing_data.extend_from_slice(&encoded_envelope);
    Ok(signing_data)
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

fn peer_chat_relay_request_commitment(
    domain: &[u8],
    signing_data: &[u8],
    previous_hop_signature: &[u8; 64],
) -> [u8; 32] {
    // [SINGLE-PASS-DIRECT-REQUEST-COMMITMENT 2026-08-31 by Codex] This pure
    // helper is the one commitment contract for v2 and v3. Version separation
    // remains explicit in `domain`; callers cannot accidentally hash a second
    // serialization that differs from the bytes already signed.
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((signing_data.len() as u64).to_be_bytes());
    hasher.update(signing_data);
    hasher.update(previous_hop_signature);
    hasher.finalize().into()
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

/// Privacy-safe local failure while preparing an outbound direct request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectRelayRequestPreparationFailure {
    Encoding,
    BodyTooLarge,
    Backpressure,
    Unavailable,
}

/// Local bounded-worker failure or remote cryptographic rejection for one
/// direct custody receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectRelayReceiptVerificationFailure {
    Invalid(&'static str),
    Backpressure,
    Unavailable,
}

/// Privacy-safe local failure while preparing an outbound blind request.
///
/// This is deliberately distinct from a next-hop failure: no HTTP request has
/// been attempted when one of these variants is returned, so callers must not
/// penalize a selected peer or mark the route surface as exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindRelayRequestPreparationFailure {
    Encoding,
    BodyTooLarge,
    Backpressure,
    Unavailable,
}

/// Domain build failure or local carrier-preparation failure.
///
/// Keeping `Build(E)` generic lets the onion route planner preserve its typed
/// refresh/policy/construction dispositions while this module owns only CPU
/// admission and the wire carrier contract.
#[derive(Debug)]
pub(crate) enum BlindRelayRequestPreparationError<E> {
    Build(E),
    Local(BlindRelayRequestPreparationFailure),
}

/// Peer-invalid evidence or a local bounded-verifier failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindRelayDeliveryReceiptVerificationFailure {
    Missing,
    Invalid,
    Unavailable,
}

impl BlindRelayRequestPreparationFailure {
    /// Closed aggregate label safe for relay-health telemetry.
    #[must_use]
    pub(crate) const fn reason_bucket(self) -> &'static str {
        // [OUTBOUND-BLIND-REQUEST-PREPARATION 2026-08-31 by Codex] Preserve
        // one coarse local bucket across encoding, policy, and worker faults;
        // the public health surface must not expose request-size distinctions.
        match self {
            Self::Encoding | Self::BodyTooLarge | Self::Backpressure | Self::Unavailable => {
                "onion_request_build_failed"
            }
        }
    }
}

impl DirectRelayRequestPreparationFailure {
    /// Closed aggregate label safe for route-health telemetry.
    #[must_use]
    pub(crate) const fn reason_bucket(self) -> &'static str {
        match self {
            Self::Encoding | Self::BodyTooLarge => "peer_relay_auth_encode_failed",
            // Keep the established closed telemetry vocabulary during rolling
            // upgrades. The typed variant still controls local recovery, while
            // aggregate route evidence avoids inventing a label older nodes
            // would sanitize to `unknown`.
            Self::Backpressure | Self::Unavailable => "peer_relay_auth_encode_failed",
        }
    }
}

/// Opaque, bounded HTTP carrier prepared outside the async I/O runtime.
///
/// [OUTBOUND-DIRECT-HTTP-BODY-PREPARATION 2026-08-31 by Codex] Serialize once
/// under bounded CPU admission, then reuse the exact immutable bytes for v2
/// fanout or v3 retry without blocking a Tokio I/O worker.
///
/// This type intentionally omits `Debug`: its body contains an end-to-end
/// encrypted user envelope. `Bytes` keeps exact retry and fanout clones O(1).
#[derive(Clone)]
pub(crate) struct PreparedPeerChatRelayHttpRequest {
    body: Bytes,
}

impl PreparedPeerChatRelayHttpRequest {
    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.body.clone()
    }
}

/// Prepared v2/v3 carrier whose commitment is mandatory by construction.
///
/// [AUTHENTICATED-DIRECT-CARRIER-TYPE 2026-08-31 by Codex] A signed request
/// without its exact commitment is not representable. This removes repeated
/// runtime `Option` checks from receipt verification and retry orchestration.
#[derive(Clone)]
pub(crate) struct PreparedAuthenticatedPeerChatRelayHttpRequest {
    request: PreparedPeerChatRelayHttpRequest,
    request_commitment: [u8; 32],
}

/// Opaque blind-relay HTTP carrier prepared outside the async I/O runtime.
///
/// The full request is dropped after serialization. Retaining only the exact
/// route id needed for receipt verification avoids holding both a potentially
/// large onion object graph and its JSON representation during network I/O.
/// This type intentionally omits `Debug` because its body is encrypted user
/// data even though the node cannot decrypt it.
pub(crate) struct PreparedPeerBlindRelayHttpRequest {
    body: Bytes,
    route_id: [u8; 16],
}

impl PreparedPeerBlindRelayHttpRequest {
    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.body.clone()
    }

    #[must_use]
    pub(crate) const fn route_id(&self) -> &[u8; 16] {
        &self.route_id
    }
}

impl PreparedAuthenticatedPeerChatRelayHttpRequest {
    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.request.body()
    }

    #[must_use]
    pub(crate) const fn request_commitment(&self) -> [u8; 32] {
        self.request_commitment
    }
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

fn peer_chat_relay_auth_v3_signing_data(
    previous_hop_node_id: &[u8; 32],
    target_node_id: &[u8; 32],
    envelope: &ChatEnvelope,
) -> Result<Vec<u8>, bincode::Error> {
    let encoded_envelope = encode_envelope(envelope)?;
    let mut signing_data =
        Vec::with_capacity(PEER_CHAT_RELAY_AUTH_V3_DOMAIN.len() + 64 + 8 + encoded_envelope.len());
    signing_data.extend_from_slice(PEER_CHAT_RELAY_AUTH_V3_DOMAIN);
    signing_data.extend_from_slice(previous_hop_node_id);
    signing_data.extend_from_slice(target_node_id);
    signing_data.extend_from_slice(&(encoded_envelope.len() as u64).to_be_bytes());
    signing_data.extend_from_slice(&encoded_envelope);
    Ok(signing_data)
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

fn blind_relay_authenticated_request_commitment(
    request: &PeerBlindRelayRequest,
) -> Result<[u8; 32], bincode::Error> {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Bincode is already the
    // canonical internal encoding used by this protocol crate. Hashing the
    // complete request prevents route-id reuse from swapping optional onward
    // routing material while keeping all durable keys node-secret HMACs.
    // [STREAMING-REPLAY-COMMITMENT 2026-08-30 by Codex] `serialized_size`
    // and `serialize_into` use the same bincode 1.x canonical options as
    // `serialize`. Hashing through `Write` preserves the existing commitment
    // exactly without allocating a second request-sized byte vector.
    let encoded_len = bincode::serialized_size(request)?;
    let mut hasher = Sha256::new();
    hasher.update(BLIND_RELAY_AUTHENTICATED_REQUEST_COMMITMENT_DOMAIN);
    hasher.update(encoded_len.to_be_bytes());
    bincode::serialize_into(Sha256CommitmentWriter(&mut hasher), request)?;
    Ok(hasher.finalize().into())
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

/// Internal result of one accepted next-hop relay round.
///
/// The observation timestamp is deliberately process-local and never enters
/// the peer wire contract. Callers use it for every success-side state write,
/// keeping receipt verification and route evidence on one clock snapshot.
struct BlindRelayForwardOutcome {
    response: PeerBlindRelayResponse,
    observed_at: u64,
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

// ============================================
// Handlers
// ============================================

/// Applies ordinary peer-relay backpressure before Axum reads a JSON body.
async fn peer_relay_request_gate(
    State(gate): State<Arc<PeerRelayRequestGate>>,
    mut request: Request,
    next: Next,
) -> Response {
    // [PEER-RELAY-ADMISSION 2026-08-15 by Codex] Count attempted direct-relay
    // work before body parsing. This intentionally uses only aggregate process
    // state: the legacy wire contract cannot authenticate a previous-hop node,
    // and sender/receiver/IP buckets would create misleading identity state.
    if !gate.admit(Instant::now()) {
        gate.record_rejected(ChatRelayInboundFailureReason::from_bucket("rate_limited"));
        return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
    }

    let Some(_in_flight) =
        InFlightRequestGuard::try_acquire(&gate.in_flight, MAX_IN_FLIGHT_PEER_CHAT_REQUESTS)
    else {
        gate.record_rejected(ChatRelayInboundFailureReason::from_bucket("backpressure"));
        return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
    };

    // [AUTHENTICATED-PEER-FAIRNESS 2026-08-15 by Codex] Pass the exact gate
    // that admitted parser work to the v2 handler. This keeps direct-relay
    // admission ownership out of the shared chat/blind relay runtime state.
    request.extensions_mut().insert(Arc::clone(&gate));
    next.run(request).await
}

fn rejected_peer_relay_response(status: StatusCode) -> Response {
    (
        status,
        Json(PeerChatRelayResponse {
            accepted: false,
            duplicate: false,
            delivered_online: 0,
            stored_pending: false,
        }),
    )
        .into_response()
}

fn durable_peer_acceptance_response() -> PeerChatRelayResponse {
    // [PEER-ACK-PRIVACY 2026-08-15 by Codex] Preserve the legacy JSON schema
    // while returning only the one fact another node needs: the ciphertext is
    // now in durable custody. Revealing duplicate or online-session state lets
    // arbitrary signed senders probe a receiver's presence and device count.
    PeerChatRelayResponse {
        accepted: true,
        duplicate: false,
        delivered_online: 0,
        stored_pending: true,
    }
}

/// Applies blind-relay backpressure before Axum reads a JSON body.
async fn peer_blind_relay_request_gate(
    State(state): State<ChatPeerState>,
    request: Request,
    next: Next,
) -> Response {
    // [BLIND-RELAY-BODY-ADMISSION-ORDER 2026-08-24 by Codex] Reject a declared
    // or exactly-known oversized body before any service-availability signal.
    // Unknown-length streams are not read here: the existing DefaultBodyLimit
    // remains authoritative when the JSON extractor consumes an admitted body.
    let body_limit = u64::try_from(PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES).unwrap_or(u64::MAX);
    let declared_length = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let exact_length = request.body().size_hint().exact();
    if declared_length.is_some_and(|length| length > body_limit)
        || exact_length.is_some_and(|length| length > body_limit)
    {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }

    // [DURABLE-BLIND-RELAY-ADMISSION 2026-08-24 by Codex] A public relay must
    // never accept work whose at-most-once evidence disappears on restart.
    // Reject before JSON/body parsing, signature work, route mutation, or any
    // ciphertext side effect. The process-local cache remains only an internal
    // compatibility primitive for focused unit tests and non-HTTP helpers.
    let Some(relay) = state.chat_relay.as_ref() else {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "replay_protection_unavailable");
        return rejected_blind_relay_response_with_status(
            StatusCode::SERVICE_UNAVAILABLE,
            "replay_protection_unavailable",
        );
    };

    // [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] Permissionless node
    // identities are cheap to rotate, so the verified previous-hop bucket
    // cannot protect parser and process capacity by itself. Count only one
    // aggregate process window before body parsing; never create source-IP,
    // user, receiver, route, endpoint, or ciphertext-derived buckets here.
    let requests_per_minute = relay.config().peer_relay_requests_per_minute;
    let admitted = state
        .blind_relay_abuse_guard
        .admit_global(Instant::now(), requests_per_minute);
    if !admitted {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "rate_limited");
        return rejected_blind_relay_response("rate_limited");
    }

    let Some(_in_flight) = InFlightRequestGuard::try_acquire(
        &state.blind_relay_in_flight,
        MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS,
    ) else {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "backpressure");
        return rejected_blind_relay_response("backpressure");
    };

    next.run(request).await
}

fn rejected_blind_relay_response(reason: &'static str) -> Response {
    rejected_blind_relay_response_with_status(StatusCode::TOO_MANY_REQUESTS, reason)
}

fn rejected_blind_relay_response_with_status(status: StatusCode, reason: &'static str) -> Response {
    (
        status,
        Json(PeerBlindRelayResponse {
            accepted: false,
            terminal: false,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some(reason.to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        }),
    )
        .into_response()
}

async fn peer_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerChatRelayRequest>,
) -> impl IntoResponse {
    peer_relay_response(state, request.envelope).await
}

async fn peer_relay_v2_handler(
    State(state): State<ChatPeerState>,
    Extension(gate): Extension<Arc<PeerRelayRequestGate>>,
    Json(request): Json<PeerChatRelayRequestV2>,
) -> impl IntoResponse {
    // [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] Authenticate the immediate
    // node before the inner envelope reaches durable storage. Invalid claims
    // affect only aggregate health and cannot poison another node's identity.
    let authenticated =
        match authenticate_direct_peer_relay_request(DirectPeerAuthenticationRequest::V2(request))
            .await
        {
            Ok(authenticated) => authenticated,
            Err(failure) => return reject_direct_peer_authentication(&state, failure),
        };

    authenticated_peer_relay_response(
        state,
        gate,
        authenticated.envelope,
        authenticated.previous_hop_node_id,
        authenticated.request_commitment,
    )
    .await
}

async fn peer_relay_v3_handler(
    State(state): State<ChatPeerState>,
    Extension(gate): Extension<Arc<PeerRelayRequestGate>>,
    Json(request): Json<PeerChatRelayRequestV3>,
) -> impl IntoResponse {
    // [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Reject a request
    // signed for another node before durable storage or authenticated quota
    // attribution. Node ids are already public discovery metadata, so this
    // branch does not expose client identity, receiver state, or content.
    let local_node_id = state.node_identity.public_key_bytes();
    if request.target_node_id != local_node_id {
        if let Some(relay) = state.chat_relay.as_ref() {
            relay.record_peer_relay_inbound_rejected_typed(
                now_secs(),
                ChatRelayInboundFailureReason::from_bucket("peer_target_mismatch"),
            );
        }
        return rejected_peer_relay_response(StatusCode::UNAUTHORIZED);
    }
    let authenticated =
        match authenticate_direct_peer_relay_request(DirectPeerAuthenticationRequest::V3 {
            request,
            expected_target_node_id: local_node_id,
        })
        .await
        {
            Ok(authenticated) => authenticated,
            Err(failure) => return reject_direct_peer_authentication(&state, failure),
        };

    authenticated_peer_relay_response(
        state,
        gate,
        authenticated.envelope,
        authenticated.previous_hop_node_id,
        authenticated.request_commitment,
    )
    .await
}

fn reject_direct_peer_authentication(
    state: &ChatPeerState,
    failure: DirectPeerAuthenticationFailure,
) -> Response {
    if let Some(relay) = state.chat_relay.as_ref() {
        relay.record_peer_relay_inbound_rejected_typed(
            now_secs(),
            ChatRelayInboundFailureReason::from_bucket(failure.reason_bucket()),
        );
    }
    rejected_peer_relay_response(failure.status_code())
}

/// Applies post-authentication fairness and emits one common custody response.
///
/// [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] v2 and v3 share the
/// exact durable-storage and signed-receipt boundary. Version-specific code is
/// limited to request authentication, avoiding divergent custody semantics.
async fn authenticated_peer_relay_response(
    state: ChatPeerState,
    gate: Arc<PeerRelayRequestGate>,
    envelope: ChatEnvelope,
    previous_hop_node_id: [u8; 32],
    request_commitment: [u8; 32],
) -> Response {
    let replay_lease = match gate.begin_authenticated_replay(request_commitment, Instant::now()) {
        AuthenticatedPeerRelayReplayStart::Acquired(lease) => lease,
        AuthenticatedPeerRelayReplayStart::Completed(response) => {
            // [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Return the
            // exact ACK produced after durable custody. This path neither
            // consumes per-node quota nor repeats storage/live delivery.
            if let Some(relay) = state.chat_relay.as_ref() {
                relay.record_peer_relay_inbound_accepted(now_secs(), true, 0, false);
            }
            return (StatusCode::OK, Json(response)).into_response();
        }
        AuthenticatedPeerRelayReplayStart::InFlight => {
            gate.record_rejected(ChatRelayInboundFailureReason::from_bucket(
                "peer_auth_retry_in_flight",
            ));
            let status =
                StatusCode::from_u16(HTTP_TOO_EARLY_STATUS_CODE).unwrap_or(StatusCode::CONFLICT);
            return rejected_peer_relay_response(status);
        }
        AuthenticatedPeerRelayReplayStart::Saturated => {
            gate.record_rejected(ChatRelayInboundFailureReason::from_bucket(
                "peer_auth_retry_cache_saturated",
            ));
            return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
        }
    };

    if !gate.admit_authenticated(previous_hop_node_id, Instant::now()) {
        gate.record_rejected(ChatRelayInboundFailureReason::from_bucket(
            "peer_auth_rate_limited",
        ));
        return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
    }

    let node_identity = Arc::clone(&state.node_identity);
    match process_peer_relay(state, envelope).await {
        Ok(relay) => {
            // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Sign only after
            // `process_peer_relay` has established durable custody. The
            // commitment was computed from the already authenticated request.
            let accepted_at = now_secs();
            let receipt = match complete_direct_relay_crypto(move || {
                PeerChatRelayReceiptV2::accepted(
                    request_commitment,
                    accepted_at,
                    node_identity.as_ref(),
                )
            })
            .await
            {
                Ok(receipt) => receipt,
                Err(DirectRelayCryptoFailure::Unavailable) => {
                    // Durable custody is already idempotent. Leave the replay
                    // lease incomplete and ask the sender to retry; the next
                    // attempt can recover the exact custody ACK without
                    // claiming that an unsigned response is authoritative.
                    return rejected_direct_peer_relay_v2_response(StatusCode::SERVICE_UNAVAILABLE);
                }
            };
            let response = PeerChatRelayResponseV2 {
                relay,
                receipt: Some(receipt),
            };
            replay_lease.complete(response.clone());
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error) => (
            error.status_code(),
            Json(PeerChatRelayResponseV2 {
                relay: PeerChatRelayResponse {
                    accepted: false,
                    duplicate: false,
                    delivered_online: 0,
                    stored_pending: false,
                },
                receipt: None,
            }),
        )
            .into_response(),
    }
}

fn rejected_direct_peer_relay_v2_response(status: StatusCode) -> Response {
    (
        status,
        Json(PeerChatRelayResponseV2 {
            relay: PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            },
            receipt: None,
        }),
    )
        .into_response()
}

async fn peer_relay_response(state: ChatPeerState, envelope: ChatEnvelope) -> Response {
    match process_peer_relay(state, envelope).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            error.status_code(),
            Json(PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            }),
        )
            .into_response(),
    }
}

async fn peer_blind_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerBlindRelayRequest>,
) -> impl IntoResponse {
    let failure_route_id = request.envelope.route_id;
    let node_identity = Arc::clone(&state.node_identity);
    let authenticated = match authenticate_peer_blind_relay_request(request).await {
        Ok(authenticated) => authenticated,
        Err(error) => {
            // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] The claimed
            // previous hop has no attribution authority before verification.
            // Record only aggregate health and never sign an oracle response.
            state
                .peer_store
                .record_blind_relay_rejected(now_secs(), error.reason_bucket());
            return blind_relay_failure_response(error, failure_route_id, None, node_identity)
                .await;
        }
    };
    let failure_request_commitment = authenticated.failure_request_commitment;
    match process_authenticated_peer_blind_relay(state, authenticated).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => {
            blind_relay_failure_response(
                error,
                failure_route_id,
                Some(failure_request_commitment),
                node_identity,
            )
            .await
        }
    }
}

/// Builds the stable blind-relay failure shape and signs it only when the
/// caller supplies a commitment produced by the authenticated request type.
///
/// [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] `None` is a security
/// boundary, not legacy absence: pre-authentication failures must stay unsigned.
async fn blind_relay_failure_response(
    error: BlindRelayError,
    route_id: [u8; 16],
    authenticated_request_commitment: Option<[u8; 32]>,
    node_identity: Arc<IdentityKeyPair>,
) -> Response {
    let status = error.status_code();
    let reason = error.reason_bucket();
    let Some(request_commitment) = authenticated_request_commitment else {
        return build_blind_relay_failure_response(status, reason, None);
    };
    let failed_at = now_secs();
    let failure_receipt = match complete_blind_relay_crypto(move || {
        // [BLIND-FAILURE-SIGNING-COMPLETION 2026-08-30 by Codex] Possession
        // of the precomputed commitment proves the request crossed the private
        // authenticated boundary. No envelope or peer-controlled bytes enter
        // this signing worker.
        Ok(BlindRelayFailureReceipt::failed(
            route_id,
            request_commitment,
            reason,
            failed_at,
            node_identity.as_ref(),
        ))
    })
    .await
    {
        Ok(receipt) => receipt,
        Err(_) => {
            // An unsigned declared protocol failure would look like a signed
            // feature downgrade. A bare retryable 429 carries no blame and
            // lets the exact durable request recover on its next attempt.
            return build_blind_relay_failure_response(
                StatusCode::TOO_MANY_REQUESTS,
                BlindRelayError::Backpressure.reason_bucket(),
                None,
            );
        }
    };
    build_blind_relay_failure_response(status, reason, Some(failure_receipt))
}

fn build_blind_relay_failure_response(
    status: StatusCode,
    reason: &'static str,
    failure_receipt: Option<BlindRelayFailureReceipt>,
) -> Response {
    (
        status,
        Json(PeerBlindRelayResponse {
            accepted: false,
            terminal: false,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some(reason.to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt,
            opaque_terminal_response_b64: None,
        }),
    )
        .into_response()
}

async fn process_peer_relay(
    state: ChatPeerState,
    envelope: ChatEnvelope,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let now = now_secs();
    let envelope = validate_peer_envelope_for_relay(&state, envelope, now).await?;
    process_authenticated_peer_relay(state, envelope, now).await
}

/// Verifies one end-to-end sender envelope and records only a coarse rejection.
///
/// [PREPARED-TERMINAL-EFFECT 2026-08-30 by Codex] Onion terminal dispatch uses
/// this same boundary before arming durable route recovery. That prevents an
/// invalid sender signature or replay timestamp from pinning an ambiguous
/// effect claim while keeping direct and onion telemetry semantics aligned.
async fn validate_peer_envelope_for_relay(
    state: &ChatPeerState,
    envelope: ChatEnvelope,
    now: u64,
) -> Result<ChatEnvelope, ChatPeerRelayError> {
    let permit = match direct_relay_cpu_admission().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let error = ChatPeerRelayError::VerificationBackpressure;
            record_peer_envelope_rejection(state, now, &error);
            return Err(error);
        }
    };
    let worker = execute_direct_relay_crypto(permit, move || {
        let result = validate_peer_envelope(&envelope, now);
        (envelope, result)
    })
    .await;
    let (envelope, result) = match worker {
        Ok(result) => result,
        Err(DirectRelayCryptoFailure::Unavailable) => {
            let error = ChatPeerRelayError::VerificationUnavailable;
            record_peer_envelope_rejection(state, now, &error);
            return Err(error);
        }
    };
    if let Err(error) = result {
        record_peer_envelope_rejection(state, now, &error);
        return Err(error);
    }
    Ok(envelope)
}

fn record_peer_envelope_rejection(state: &ChatPeerState, now: u64, error: &ChatPeerRelayError) {
    if let Some(relay) = state.chat_relay.as_ref() {
        relay.record_peer_relay_inbound_rejected_typed(
            now,
            ChatRelayInboundFailureReason::from_bucket(error.reason_bucket()),
        );
    }
}

/// Establishes durable custody for an already authenticated sender envelope.
async fn process_authenticated_peer_relay(
    state: ChatPeerState,
    envelope: ChatEnvelope,
    now: u64,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let storage_permit = acquire_chat_relay_storage(&state, now)?;
    process_authenticated_peer_relay_with_storage_permit(state, envelope, now, storage_permit).await
}

fn acquire_chat_relay_storage(
    state: &ChatPeerState,
    now: u64,
) -> Result<OwnedSemaphorePermit, ChatPeerRelayError> {
    if state.chat_relay.is_none() {
        return Err(ChatPeerRelayError::RelayUnavailable);
    }
    chat_relay_storage_admission()
        .try_acquire_owned()
        .map_err(|_| {
            let error = ChatPeerRelayError::StorageBackpressure;
            record_peer_envelope_rejection(state, now, &error);
            error
        })
}

async fn process_authenticated_peer_relay_with_storage_permit(
    state: ChatPeerState,
    envelope: ChatEnvelope,
    now: u64,
    storage_permit: OwnedSemaphorePermit,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let Some(relay) = state.chat_relay.as_ref().map(Arc::clone) else {
        return Err(ChatPeerRelayError::RelayUnavailable);
    };

    // [DURABLE-RECEIPT-BOUNDARY 2026-08-15 by Codex] Persist the exact signed
    // envelope before consulting the live-delivery dedupe cache. Checking only
    // `message_id` first allowed a conflicting ciphertext to be reported as an
    // accepted retry; an onion terminal could then sign a receipt for bytes it
    // had never stored. `store_pending` is idempotent for byte-identical retries
    // and rejects same-ID/different-envelope collisions atomically.
    // [RELAY-STORAGE-ADMISSION 2026-08-30 by Codex] SQLite custody is
    // synchronous. Keep its owned permit inside the blocking worker so request
    // cancellation cannot create unbounded detached database tasks.
    let store_relay = Arc::clone(&relay);
    let worker = tokio::task::spawn_blocking(move || {
        let _storage_permit = storage_permit;
        let result = store_relay.store_pending(&envelope);
        (envelope, result)
    })
    .await;
    let (envelope, result) = match worker {
        Ok(result) => result,
        Err(_) => {
            warn!("[CHAT_PEER] Pending custody worker failed closed");
            let error = ChatPeerRelayError::StoreFailed;
            record_peer_envelope_rejection(&state, now, &error);
            return Err(error);
        }
    };
    result.map_err(|error| {
        let reason = error.reason_bucket();
        warn!(reason, "[CHAT_PEER] Failed to durably accept peer envelope");
        // [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Preserve the
        // storage diagnostic in the local warning while exporting only a
        // validated aggregate bucket to node health.
        relay.record_peer_relay_inbound_rejected_typed(
            now,
            ChatRelayInboundFailureReason::from_bucket(reason),
        );
        map_pending_store_error(&error)
    })?;

    if relay.is_online_duplicate(&envelope.message_id) {
        debug!("[CHAT_PEER] Duplicate peer envelope ignored");
        relay.record_peer_relay_inbound_accepted(now, true, 0, true);
        return Ok(durable_peer_acceptance_response());
    }

    let target_sessions = state.sessions.get_all_by_wallet(&envelope.receiver);
    let mut delivered_online = 0usize;

    for session in target_sessions {
        if send_envelope_to_session(&envelope, &session, &state.udp).await {
            delivered_online += 1;
        }
    }

    // The authenticated receiver retires this durable copy with ChatAck only
    // after local persistence. This keeps online UDP delivery crash-safe and
    // gives terminal receipts one stable meaning: accepted into durable relay
    // custody, never merely queued to a socket.
    relay.record_peer_relay_inbound_accepted(now, false, delivered_online, true);

    Ok(durable_peer_acceptance_response())
}

fn map_pending_store_error(error: &ChatRelayError) -> ChatPeerRelayError {
    match error {
        ChatRelayError::MessageTooLarge { size, .. } => {
            ChatPeerRelayError::EnvelopeTooLarge { size: *size }
        }
        error if error.is_capacity_exhausted() => ChatPeerRelayError::PendingCapacity,
        _ => ChatPeerRelayError::StoreFailed,
    }
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

fn blind_relay_crypto_admission() -> Arc<Semaphore> {
    Arc::clone(BLIND_RELAY_CRYPTO_ADMISSION.get_or_init(|| {
        // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Reserve roughly
        // half the host for the rest of the node and reject excess work before
        // it enters Tokio's blocking queue.
        Arc::new(Semaphore::new(signature_verification_capacity(
            MAX_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT,
        )))
    }))
}

/// Executes one bounded blind-relay cryptographic operation off the async I/O
/// runtime. Work must remain pure with respect to network and durable storage.
async fn run_blind_relay_crypto<T, F>(work: F) -> Result<T, BlindRelayError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BlindRelayError> + Send + 'static,
{
    let permit = blind_relay_crypto_admission()
        .try_acquire_owned()
        .map_err(|_| BlindRelayError::Backpressure)?;
    execute_blind_relay_crypto(permit, work).await
}

/// Completes pure cryptographic work after an external route effect is armed.
///
/// Unlike preflight admission, this waits fairly for bounded CPU capacity so a
/// successfully returned ACK is not discarded merely because new ingress
/// verification arrived first. The outer HTTP in-flight gate bounds waiters.
async fn complete_blind_relay_crypto<T, F>(work: F) -> Result<T, BlindRelayError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BlindRelayError> + Send + 'static,
{
    let permit = blind_relay_crypto_admission()
        .acquire_owned()
        .await
        .map_err(|_| BlindRelayError::Backpressure)?;
    execute_blind_relay_crypto(permit, work).await
}

async fn execute_blind_relay_crypto<T, F>(
    permit: OwnedSemaphorePermit,
    work: F,
) -> Result<T, BlindRelayError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BlindRelayError> + Send + 'static,
{
    match tokio::task::spawn_blocking(move || {
        // [BLIND-RELAY-CRYPTO-DOMAIN 2026-08-30 by Codex] Keep the owned
        // permit in the worker after request cancellation. This helper must
        // never contain I/O or durable effects, so retries remain safe.
        let _permit = permit;
        work()
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            warn!("[CHAT_PEER] Blind relay crypto worker failed closed");
            Err(BlindRelayError::Backpressure)
        }
    }
}

fn direct_relay_cpu_admission() -> Arc<Semaphore> {
    Arc::clone(DIRECT_RELAY_CPU_ADMISSION.get_or_init(|| {
        Arc::new(Semaphore::new(signature_verification_capacity(
            MAX_DIRECT_RELAY_CPU_OPERATIONS_IN_FLIGHT,
        )))
    }))
}

fn chat_relay_storage_admission() -> Arc<Semaphore> {
    Arc::clone(
        CHAT_RELAY_STORAGE_ADMISSION
            .get_or_init(|| Arc::new(Semaphore::new(MAX_CHAT_RELAY_STORAGE_OPERATIONS_IN_FLIGHT))),
    )
}

fn blind_vault_terminal_admission() -> Arc<Semaphore> {
    Arc::clone(BLIND_VAULT_TERMINAL_ADMISSION.get_or_init(|| {
        Arc::new(Semaphore::new(
            MAX_BLIND_VAULT_TERMINAL_OPERATIONS_IN_FLIGHT,
        ))
    }))
}

async fn authenticate_direct_peer_relay_request(
    request: DirectPeerAuthenticationRequest,
) -> Result<AuthenticatedDirectPeerRelayRequest, DirectPeerAuthenticationFailure> {
    let permit = direct_relay_cpu_admission()
        .try_acquire_owned()
        .map_err(|_| DirectPeerAuthenticationFailure::Backpressure)?;
    execute_direct_relay_crypto(permit, move || {
        // [DIRECT-RELAY-VERIFY-ADMISSION 2026-08-30 by Codex] The owned permit
        // remains in the shared worker after HTTP cancellation. Excess work
        // never queues, and no inner envelope or node identity reaches
        // telemetry.
        match request {
            DirectPeerAuthenticationRequest::V2(request) => {
                let request_commitment = request
                    .verified_request_commitment()
                    .ok_or(DirectPeerAuthenticationFailure::Invalid)?;
                Ok(AuthenticatedDirectPeerRelayRequest {
                    envelope: request.envelope,
                    previous_hop_node_id: request.previous_hop_node_id,
                    request_commitment,
                })
            }
            DirectPeerAuthenticationRequest::V3 {
                request,
                expected_target_node_id,
            } => {
                let request_commitment = request
                    .verified_request_commitment_for_target(&expected_target_node_id)
                    .ok_or(DirectPeerAuthenticationFailure::Invalid)?;
                Ok(AuthenticatedDirectPeerRelayRequest {
                    envelope: request.envelope,
                    previous_hop_node_id: request.previous_hop_node_id,
                    request_commitment,
                })
            }
        }
    })
    .await
    .map_err(|DirectRelayCryptoFailure::Unavailable| DirectPeerAuthenticationFailure::Unavailable)?
}

/// Waits fairly for direct-relay crypto capacity after durable custody.
///
/// Preflight verification intentionally uses `try_acquire_owned` so hostile
/// ingress cannot build a blocking-task queue. Completion is different: the
/// node already owns the ciphertext, and the parser-front in-flight gate
/// bounds these waiters, so fairness prevents fresh verification work from
/// starving authoritative receipt signing.
async fn complete_direct_relay_crypto<T, F>(work: F) -> Result<T, DirectRelayCryptoFailure>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let permit = direct_relay_cpu_admission()
        .acquire_owned()
        .await
        .map_err(|_| DirectRelayCryptoFailure::Unavailable)?;
    execute_direct_relay_crypto(permit, work).await
}

async fn execute_direct_relay_crypto<T, F>(
    permit: OwnedSemaphorePermit,
    work: F,
) -> Result<T, DirectRelayCryptoFailure>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        // [DIRECT-RECEIPT-SIGNING-COMPLETION 2026-08-31 by Codex] Keep the
        // permit in the worker after request cancellation. This domain owns
        // CPU-only cryptographic work and must never perform I/O or effects.
        let _permit = permit;
        work()
    })
    .await
    .map_err(|_| {
        warn!("[CHAT_PEER] Direct relay crypto worker failed closed");
        DirectRelayCryptoFailure::Unavailable
    })
}

/// Prepares one legacy direct request outside the asynchronous I/O runtime.
pub(crate) async fn prepare_peer_chat_relay_request_v1(
    envelope: ChatEnvelope,
) -> Result<PreparedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    prepare_direct_peer_relay_request(move || {
        encode_prepared_peer_chat_relay_request(&PeerChatRelayRequest { envelope })
    })
    .await
}

/// Prepares one authenticated v2 request outside the asynchronous I/O runtime.
pub(crate) async fn prepare_peer_chat_relay_request_v2(
    envelope: ChatEnvelope,
    node_identity: Arc<IdentityKeyPair>,
) -> Result<PreparedAuthenticatedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    prepare_direct_peer_relay_request(move || {
        let (request, request_commitment) =
            PeerChatRelayRequestV2::sign_with_commitment(envelope, node_identity.as_ref())
                .map_err(|_| DirectRelayRequestPreparationFailure::Encoding)?;
        encode_prepared_authenticated_peer_chat_relay_request(&request, request_commitment)
    })
    .await
}

/// Prepares one target-bound v3 request outside the async I/O runtime.
pub(crate) async fn prepare_peer_chat_relay_request_v3(
    envelope: ChatEnvelope,
    target_node_id: [u8; 32],
    node_identity: Arc<IdentityKeyPair>,
) -> Result<PreparedAuthenticatedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    prepare_direct_peer_relay_request(move || {
        let (request, request_commitment) = PeerChatRelayRequestV3::sign_with_commitment(
            envelope,
            target_node_id,
            node_identity.as_ref(),
        )
        .map_err(|_| DirectRelayRequestPreparationFailure::Encoding)?;
        encode_prepared_authenticated_peer_chat_relay_request(&request, request_commitment)
    })
    .await
}

/// Builds and serializes one blind request as a single bounded CPU operation.
///
/// [ATOMIC-OUTBOUND-BLIND-PREPARATION 2026-08-31 by Codex] Route planning,
/// onion KEM work, signing, and JSON encoding may be composed inside `build`
/// without returning to a Tokio I/O worker between CPU-heavy stages. The
/// context is carried beside the immutable body for receipt verification.
pub(crate) async fn prepare_peer_blind_relay_http_request_with<T, E, F>(
    build: F,
) -> Result<(PreparedPeerBlindRelayHttpRequest, T), BlindRelayRequestPreparationError<E>>
where
    T: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Result<(PeerBlindRelayRequest, T), E> + Send + 'static,
{
    let permit = blind_relay_crypto_admission()
        .try_acquire_owned()
        .map_err(|_| {
            BlindRelayRequestPreparationError::Local(
                BlindRelayRequestPreparationFailure::Backpressure,
            )
        })?;
    tokio::task::spawn_blocking(move || {
        // [OUTBOUND-BLIND-REQUEST-PREPARATION 2026-08-31 by Codex] Keep the
        // permit in the worker after caller cancellation. The composed work
        // has no I/O or effects, so abandoning the result is always retry-safe.
        let _permit = permit;
        let (request, context) = build().map_err(BlindRelayRequestPreparationError::Build)?;
        let request = encode_prepared_peer_blind_relay_request(request)
            .map_err(BlindRelayRequestPreparationError::Local)?;
        Ok((request, context))
    })
    .await
    .map_err(|_| {
        warn!("[CHAT_PEER] Outbound blind relay preparation worker failed closed");
        BlindRelayRequestPreparationError::Local(BlindRelayRequestPreparationFailure::Unavailable)
    })?
}

fn encode_prepared_peer_blind_relay_request(
    request: PeerBlindRelayRequest,
) -> Result<PreparedPeerBlindRelayHttpRequest, BlindRelayRequestPreparationFailure> {
    let route_id = request.envelope.route_id;
    let body =
        serde_json::to_vec(&request).map_err(|_| BlindRelayRequestPreparationFailure::Encoding)?;
    if body.len() > PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES {
        return Err(BlindRelayRequestPreparationFailure::BodyTooLarge);
    }
    Ok(PreparedPeerBlindRelayHttpRequest {
        body: Bytes::from(body),
        route_id,
    })
}

/// Verifies one terminal delivery receipt behind bounded CPU admission.
pub(crate) async fn verify_blind_relay_delivery_receipt(
    receipt: Option<BlindRelayDeliveryReceipt>,
    expected_route_id: [u8; 16],
    expected_payload_commitment: [u8; 32],
    expected_terminal_node_id: [u8; 32],
    observed_at: u64,
) -> Result<BlindRelayDeliveryReceipt, BlindRelayDeliveryReceiptVerificationFailure> {
    // Missing evidence is a peer/protocol outcome and must not be hidden by
    // unrelated local saturation.
    let receipt = receipt.ok_or(BlindRelayDeliveryReceiptVerificationFailure::Missing)?;
    // [BLIND-RECEIPT-FAIR-COMPLETION 2026-08-31 by Codex] The route has
    // already been exposed and cannot safely fall back to a new surface.
    // Await the fair semaphore instead of dropping authoritative evidence on
    // transient ingress pressure. Outbound fanout bounds these waiters.
    let permit = blind_relay_crypto_admission()
        .acquire_owned()
        .await
        .map_err(|_| BlindRelayDeliveryReceiptVerificationFailure::Unavailable)?;
    tokio::task::spawn_blocking(move || {
        // [OUTBOUND-BLIND-RECEIPT-VERIFICATION 2026-08-31 by Codex] Hold the
        // permit until signature verification really stops after cancellation.
        let _permit = permit;
        if blind_relay_delivery_receipt_is_valid(
            &receipt,
            &expected_route_id,
            &expected_payload_commitment,
            &expected_terminal_node_id,
            observed_at,
        ) {
            Ok(receipt)
        } else {
            Err(BlindRelayDeliveryReceiptVerificationFailure::Invalid)
        }
    })
    .await
    .map_err(|_| {
        warn!("[CHAT_PEER] Outbound blind receipt verification worker failed closed");
        BlindRelayDeliveryReceiptVerificationFailure::Unavailable
    })?
}

/// Pure receipt contract shared by the bounded worker and focused unit tests.
#[must_use]
pub(crate) fn blind_relay_delivery_receipt_is_valid(
    receipt: &BlindRelayDeliveryReceipt,
    expected_route_id: &[u8; 16],
    expected_payload_commitment: &[u8; 32],
    expected_terminal_node_id: &[u8; 32],
    observed_at: u64,
) -> bool {
    receipt.version == BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION
        && receipt.delivered_at
            <= observed_at.saturating_add(BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS)
        && observed_at.saturating_sub(receipt.delivered_at)
            <= BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS
        && receipt
            .verify_expected(
                expected_route_id,
                expected_payload_commitment,
                expected_terminal_node_id,
            )
            .is_ok()
}

fn encode_prepared_peer_chat_relay_request<T: Serialize>(
    request: &T,
) -> Result<PreparedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    let body =
        serde_json::to_vec(request).map_err(|_| DirectRelayRequestPreparationFailure::Encoding)?;
    if body.len() > PEER_CHAT_REQUEST_BODY_MAX_BYTES {
        return Err(DirectRelayRequestPreparationFailure::BodyTooLarge);
    }
    Ok(PreparedPeerChatRelayHttpRequest {
        body: Bytes::from(body),
    })
}

fn encode_prepared_authenticated_peer_chat_relay_request<T: Serialize>(
    request: &T,
    request_commitment: [u8; 32],
) -> Result<PreparedAuthenticatedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    Ok(PreparedAuthenticatedPeerChatRelayHttpRequest {
        request: encode_prepared_peer_chat_relay_request(request)?,
        request_commitment,
    })
}

/// Verifies a signed direct-custody receipt outside the async I/O runtime.
pub(crate) async fn verify_peer_chat_relay_receipt(
    receipt: PeerChatRelayReceiptV2,
    expected_request_commitment: [u8; 32],
    expected_node_id: [u8; 32],
    observed_at: u64,
) -> Result<(), DirectRelayReceiptVerificationFailure> {
    // [OUTBOUND-DIRECT-RECEIPT-VERIFICATION 2026-08-31 by Codex] A response
    // body has already been bounded before this call. Fail fast when the local
    // CPU partition is full; the caller may repeat the exact v3 request, but
    // must not attribute local saturation to the selected peer.
    let permit = direct_relay_cpu_admission()
        .try_acquire_owned()
        .map_err(|_| DirectRelayReceiptVerificationFailure::Backpressure)?;
    execute_direct_relay_crypto(permit, move || {
        receipt.verify_expected_commitment(
            &expected_request_commitment,
            &expected_node_id,
            observed_at,
        )
    })
    .await
    .map_err(|DirectRelayCryptoFailure::Unavailable| {
        DirectRelayReceiptVerificationFailure::Unavailable
    })?
    .map_err(DirectRelayReceiptVerificationFailure::Invalid)
}

async fn prepare_direct_peer_relay_request<T, F>(
    work: F,
) -> Result<T, DirectRelayRequestPreparationFailure>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, DirectRelayRequestPreparationFailure> + Send + 'static,
{
    // [OUTBOUND-DIRECT-REQUEST-PREPARATION 2026-08-31 by Codex] Outbound
    // fallback is optional and always has local durable custody behind it.
    // Fail fast instead of queueing unbounded signature work under fanout.
    let permit = direct_relay_cpu_admission()
        .try_acquire_owned()
        .map_err(|_| DirectRelayRequestPreparationFailure::Backpressure)?;
    execute_direct_relay_crypto(permit, work).await.map_err(
        |DirectRelayCryptoFailure::Unavailable| DirectRelayRequestPreparationFailure::Unavailable,
    )?
}

async fn authenticate_peer_blind_relay_request(
    request: PeerBlindRelayRequest,
) -> Result<AuthenticatedPeerBlindRelayRequest, BlindRelayError> {
    authenticate_peer_blind_relay_request_with_admission(blind_relay_crypto_admission(), request)
        .await
}

async fn authenticate_peer_blind_relay_request_with_admission(
    admission: Arc<Semaphore>,
    request: PeerBlindRelayRequest,
) -> Result<AuthenticatedPeerBlindRelayRequest, BlindRelayError> {
    // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Acquire before
    // `spawn_blocking`, then move the owned permit into the worker. This makes
    // saturation fail immediately without queue growth and prevents a
    // cancelled HTTP future from releasing capacity before CPU work stops.
    let permit = admission
        .try_acquire_owned()
        .map_err(|_| BlindRelayError::Backpressure)?;
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        authenticate_blind_relay_envelope(&request.envelope, &request.previous_hop_node_id)?;
        if let Some(onward_envelope) = request.onward_envelope.as_ref() {
            // [SIGNED-ONWARD-ENVELOPE 2026-08-24 by Codex] The forwarding
            // sender already re-signs this optional legacy frame. Verify that
            // exact previous-hop signature before the complete replay
            // commitment or any route state can trust it.
            authenticate_blind_relay_envelope(onward_envelope, &request.previous_hop_node_id)?;
        }
        let failure_request_commitment =
            BlindRelayFailureReceipt::request_commitment(&request.envelope);
        let request_commitment = blind_relay_authenticated_request_commitment(&request)
            .map_err(|_| BlindRelayError::ReplayProtectionUnavailable)?;
        Ok(AuthenticatedPeerBlindRelayRequest {
            request,
            failure_request_commitment,
            request_commitment,
        })
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            // Join failures are local runtime faults. Never expose a panic or
            // scheduler detail through the privacy protocol response.
            warn!("[CHAT_PEER] Blind relay verification worker failed closed");
            Err(BlindRelayError::Backpressure)
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

fn build_forwarded_onion_envelope(
    envelope: &BlindRelayEnvelope,
    next_hop: [u8; 32],
    inner: Vec<u8>,
    node_identity: &IdentityKeyPair,
) -> BlindRelayEnvelope {
    build_forwarded_onion_envelope_from_seed(
        BlindRelayForwardSeed::from(envelope),
        next_hop,
        inner,
        node_identity,
    )
}

fn build_forwarded_onion_envelope_from_seed(
    seed: BlindRelayForwardSeed,
    next_hop: [u8; 32],
    inner: Vec<u8>,
    node_identity: &IdentityKeyPair,
) -> BlindRelayEnvelope {
    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Every field derives
    // from authenticated ingress state. Ed25519 signing is deterministic, so
    // an exact restart retry generates the same downstream request commitment.
    BlindRelayEnvelope {
        route_id: seed.route_id,
        next_hop,
        ttl: seed.ttl.saturating_sub(1),
        encrypted_blob: inner,
        timestamp: seed.timestamp,
        signature: [0u8; 64],
    }
    .sign_with(node_identity)
}

/// Attaches one immediate-hop success proof to an already accepted response.
///
/// [BLIND-RELAY-SUCCESS-RECEIPT 2026-08-29 by Codex] This is the only helper
/// allowed to sign outbound success ACKs. It binds the exact request envelope,
/// response shape, TTL, legacy delivery evidence, and opaque response while
/// ensuring a relay never propagates a deeper hop's success signature.
async fn attach_blind_relay_success_receipt(
    envelope: Arc<BlindRelayEnvelope>,
    response: PeerBlindRelayResponse,
    accepted_at: u64,
    responder: Arc<IdentityKeyPair>,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    complete_blind_relay_crypto(move || {
        // [BLIND-SUCCESS-SIGNING-COMPLETION 2026-08-30 by Codex] The request
        // and possibly large opaque response move into the worker by ownership;
        // `Arc` avoids a ciphertext clone while durable completion waits.
        sign_blind_relay_success_receipt(
            envelope.as_ref(),
            response,
            accepted_at,
            responder.as_ref(),
        )
    })
    .await
}

/// Payload ownership required to create terminal evidence without a clone.
struct TerminalDeliveryProofInput {
    payload: Vec<u8>,
    purpose: OnionRoutePurpose,
}

async fn attach_blind_relay_terminal_success_receipts(
    envelope: Arc<BlindRelayEnvelope>,
    mut response: PeerBlindRelayResponse,
    proof: TerminalDeliveryProofInput,
    accepted_at: u64,
    responder: Arc<IdentityKeyPair>,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    complete_blind_relay_crypto(move || {
        // [BLIND-TERMINAL-PROOF-COMPLETION 2026-08-30 by Codex] One worker
        // commits the accepted payload, signs terminal evidence, and binds that
        // exact evidence into the immediate-hop ACK before durable completion.
        if response.delivery_receipt.is_some() {
            return Err(BlindRelayError::ForwardFailed);
        }
        response.delivery_receipt = Some(BlindRelayDeliveryReceipt::accepted_for_purpose(
            envelope.route_id,
            &proof.payload,
            proof.purpose,
            accepted_at,
            responder.as_ref(),
        ));
        sign_blind_relay_success_receipt(
            envelope.as_ref(),
            response,
            accepted_at,
            responder.as_ref(),
        )
    })
    .await
}

fn sign_blind_relay_success_receipt(
    envelope: &BlindRelayEnvelope,
    mut response: PeerBlindRelayResponse,
    accepted_at: u64,
    responder: &IdentityKeyPair,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    if !response.accepted || response.failure_receipt.is_some() {
        return Err(BlindRelayError::ForwardFailed);
    }
    let opaque_response = response
        .opaque_terminal_response_b64
        .as_deref()
        .map(str::as_bytes);
    let receipt = match (response.terminal, response.forwarded) {
        (true, false) => BlindRelaySuccessReceipt::terminal(
            envelope,
            response.ttl_remaining,
            response.reason.as_deref(),
            response.delivery_receipt.as_ref(),
            opaque_response,
            accepted_at,
            responder,
        ),
        (false, true) => BlindRelaySuccessReceipt::forwarded(
            envelope,
            response.ttl_remaining,
            response.reason.as_deref(),
            response.delivery_receipt.as_ref(),
            opaque_response,
            accepted_at,
            responder,
        ),
        _ => return Err(BlindRelayError::ForwardFailed),
    };
    response.success_receipt = Some(receipt);
    Ok(response)
}

#[cfg(test)]
async fn process_peer_blind_relay(
    state: ChatPeerState,
    request: PeerBlindRelayRequest,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    // [BLIND-RELAY-TEST-ADMISSION-ISOLATION 2026-08-24 by Codex] Focused
    // process tests must not race one another for the production-global CPU
    // semaphore. The dedicated admission tests still exercise that runtime
    // boundary directly; this helper keeps each unrelated route test bounded
    // to one verification worker without introducing suite-order flakiness.
    let authenticated =
        authenticate_peer_blind_relay_request_with_admission(Arc::new(Semaphore::new(1)), request)
            .await
            .map_err(|error| {
                state
                    .peer_store
                    .record_blind_relay_rejected(now_secs(), error.reason_bucket());
                error
            })?;
    process_authenticated_peer_blind_relay(state, authenticated).await
}

async fn process_authenticated_peer_blind_relay(
    state: ChatPeerState,
    authenticated: AuthenticatedPeerBlindRelayRequest,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Reaching route state
    // requires possession of the private authenticated request capability.
    let now = now_secs();
    let route_started_at = Instant::now();
    let request_commitment = authenticated.request_commitment;
    let request = authenticated.request;
    let previous_hop_node_id = request.previous_hop_node_id;
    let onward_descriptor_hint = request.onward_descriptor_hint;
    let envelope = request.envelope;

    check_blind_relay_previous_hop_allowed(&state, previous_hop_node_id, now)?;

    validate_blind_relay_metadata(&envelope, now).map_err(|error| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, error.reason_bucket());
        error
    })?;

    let self_node_id = state.node_identity.public_key_bytes();
    if previous_hop_node_id == self_node_id {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "self_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if envelope.next_hop == self_node_id {
        // Onion routing v1: if the opaque blob is an onion layer addressed to
        // this node, peel exactly one layer and either deliver locally
        // (terminal) or forward the inner layer to the revealed next hop. Legacy
        // opaque blobs (no onion magic) fall through to the existing behavior.
        if is_onion_blob(&envelope.encrypted_blob) {
            return process_onion_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                request_commitment,
                now,
                &route_started_at,
            )
            .await;
        }
        if let Some(onward_envelope) = request.onward_envelope {
            return process_onion_middle_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                onward_envelope,
                onward_descriptor_hint,
                request_commitment,
                now,
                &route_started_at,
            )
            .await;
        }
        let envelope = Arc::new(envelope);
        let route_lease = match begin_blind_relay_route(
            &state,
            envelope.route_id,
            request_commitment,
            previous_hop_node_id,
            now,
        )? {
            BlindRelayRouteStart::Acquired(lease) => lease,
            BlindRelayRouteStart::Completed(response) => {
                return attach_blind_relay_success_receipt(
                    Arc::clone(&envelope),
                    response,
                    now,
                    Arc::clone(&state.node_identity),
                )
                .await
            }
        };
        let response = attach_blind_relay_success_receipt(
            Arc::clone(&envelope),
            PeerBlindRelayResponse {
                accepted: true,
                terminal: true,
                forwarded: false,
                ttl_remaining: envelope.ttl,
                reason: Some("terminal_next_hop".to_string()),
                delivery_receipt: None,
                success_receipt: None,
                failure_receipt: None,
                opaque_terminal_response_b64: None,
            },
            now,
            Arc::clone(&state.node_identity),
        )
        .await?;
        complete_blind_relay_route(&state, route_lease, now, response.clone())?;
        record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
        state.peer_store.record_blind_relay_terminal(
            now,
            envelope.ttl,
            envelope.encrypted_blob.len(),
        );
        return Ok(response);
    }

    if envelope.next_hop == previous_hop_node_id {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if !envelope.can_forward() {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = envelope.next_hop;
    let (descriptor, used_descriptor_hint) = resolve_blind_relay_next_hop_descriptor(
        &state,
        &next_hop,
        now,
        onward_descriptor_hint.as_ref(),
    )
    .ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        BlindRelayError::NoRoute
    })?;
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        // [ROUTE-HEALTH-REMOTE-POISONING 2026-08-11 by Codex] A remote
        // previous hop can choose this target. Preflight rejection is therefore
        // not next-hop failure evidence; only a real outbound request may
        // mutate that peer's route health. The requester is rejected below.
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let original_envelope = Arc::new(envelope);
    let mut route_lease = match begin_blind_relay_route(
        &state,
        original_envelope.route_id,
        request_commitment,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => {
            return attach_blind_relay_success_receipt(
                Arc::clone(&original_envelope),
                response,
                now,
                Arc::clone(&state.node_identity),
            )
            .await
        }
    };

    let envelope_for_forwarding = Arc::clone(&original_envelope);
    let onward_envelope_for_forwarding = request.onward_envelope;
    let forwarding_identity = Arc::clone(&state.node_identity);
    let prepared_forward = run_blind_relay_crypto(move || {
        // [AUTHENTICATED-ONWARD-DOMAIN 2026-08-30 by Codex] Re-sign the
        // complete legacy forwarding unit in one bounded worker. The
        // original outer envelope stays immutable for its success receipt.
        let envelope = envelope_for_forwarding
            .decremented_ttl()
            .ok_or(BlindRelayError::TtlExhausted)?
            .sign_with(forwarding_identity.as_ref());
        let onward_envelope = onward_envelope_for_forwarding
            .map(|envelope| envelope.sign_with(forwarding_identity.as_ref()));
        Ok(PreparedLegacyBlindRelayForward {
            envelope,
            onward_envelope,
        })
    })
    .await?;
    let forwarded_onward_descriptor_hint = onward_descriptor_hint;
    let ttl_remaining = prepared_forward.envelope.ttl;

    let forward_started_at = blind_relay_response_observed_at(now, &route_started_at);
    route_lease
        .arm_effect(forward_started_at)
        .map_err(|_| record_blind_relay_replay_protection_failure(&state, forward_started_at))?;
    let observed_at = match forward_blind_relay_with_retry(
        &state,
        &url,
        &descriptor,
        PeerBlindRelayRequest {
            envelope: prepared_forward.envelope,
            previous_hop_node_id: self_node_id,
            onward_envelope: prepared_forward.onward_envelope,
            onward_descriptor_hint: forwarded_onward_descriptor_hint,
        },
        forward_started_at,
    )
    .await
    {
        Ok(outcome) => outcome.observed_at,
        Err(error) => return Err(error),
    };

    let response = attach_blind_relay_success_receipt(
        original_envelope,
        PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining,
            reason: Some("forwarded".to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        },
        observed_at,
        Arc::clone(&state.node_identity),
    )
    .await?;
    complete_blind_relay_route(&state, route_lease, observed_at, response.clone())?;
    let _ = state
        .peer_store
        .record_route_forward_success_for_descriptor(&descriptor, observed_at);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
    state
        .peer_store
        .record_blind_relay_forwarded(observed_at, ttl_remaining);

    Ok(response)
}

/// Onion routing v1 — this node is the addressed hop and the opaque blob is an
/// onion layer. Peel exactly one layer with the node's rotating onion key(s),
/// then either deliver locally (terminal hop) or forward the revealed inner
/// layer to the next hop (entry/middle hop).
///
/// Privacy invariant: a relay learns only the previous hop (transport auth) and
/// the immediate next hop (from its own peeled layer). It never sees the
/// original source, the final destination, or the payload. The onion secret
/// keys (current + previous within the rotation grace window, see
/// services::onion_keys) are never logged.
async fn process_onion_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    envelope: BlindRelayEnvelope,
    request_commitment: [u8; 32],
    now: u64,
    route_started_at: &Instant,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let self_node_id = state.node_identity.public_key_bytes();
    let envelope = Arc::new(envelope);

    // Per-route replay/dedup, identical to the opaque terminal/forward paths.
    let mut route_lease = match begin_blind_relay_route(
        &state,
        envelope.route_id,
        request_commitment,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => {
            return attach_blind_relay_success_receipt(
                Arc::clone(&envelope),
                response,
                now,
                Arc::clone(&state.node_identity),
            )
            .await
        }
    };

    // Peel exactly one onion layer with the node's rotating onion key(s): the
    // current key, plus the previous key while it is within the rotation grace
    // window (forward secrecy — see services::onion_keys). A failure yields a
    // coarse bucket only, never a payload leak.
    let onion_secrets = crate::services::onion_keys::peel_secrets(now);
    let encrypted_envelope = Arc::clone(&envelope);
    let peel = match run_blind_relay_crypto(move || {
        try_open_onion_layer(&encrypted_envelope.encrypted_blob, &onion_secrets)
            .map_err(|_| BlindRelayError::OnionPeelFailed)
    })
    .await
    {
        Ok(peel) => peel,
        Err(BlindRelayError::OnionPeelFailed) => {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "onion_peel_failed");
            return Err(BlindRelayError::OnionPeelFailed);
        }
        Err(error) => return Err(error),
    };

    match peel.next_hop {
        // Terminal hop: `inner` is a ChatEnvelope, legacy signed Blind Vault
        // Put, or reply-capable Blind Vault request. Fixed protocol magic
        // selects the parser; malformed declared frames never fall back.
        None => {
            // [PREPARED-TERMINAL-EFFECT 2026-08-30 by Codex] Parsing, response
            // negotiation, and chat sender authentication must finish before
            // mutation recovery is armed. Read-only vault observations stay
            // unarmed, so cancellation can release and safely retry the route.
            let prepared = prepare_onion_terminal_payload(&state, &peel.inner, now).await?;
            if prepared.requires_durable_guard() {
                route_lease
                    .arm_effect(now)
                    .map_err(|_| record_blind_relay_replay_protection_failure(&state, now))?;
            }
            let terminal_delivery = match execute_onion_terminal_payload(
                &state,
                envelope.route_id,
                prepared,
                now,
            )
            .await
            {
                Ok(delivery) => delivery,
                Err(error) => {
                    let failed_at = blind_relay_response_observed_at(now, route_started_at);
                    debug!(
                        reason = error.reason_bucket(),
                        "[BLIND_RELAY] Onion terminal delivery failed"
                    );
                    reject_blind_relay_previous_hop(
                        &state,
                        previous_hop_node_id,
                        failed_at,
                        "onion_terminal_delivery_failed",
                    );
                    return Err(error);
                }
            };
            let accepted_at = blind_relay_response_observed_at(now, route_started_at);
            // [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Sign v2 only after
            // the selected terminal workload has crossed its durable acceptance
            // boundary. The purpose is committed with the opaque payload hash,
            // not returned as relay-visible metadata.
            let response = PeerBlindRelayResponse {
                accepted: true,
                terminal: true,
                forwarded: false,
                ttl_remaining: envelope.ttl,
                reason: Some("onion_terminal_delivered".to_string()),
                delivery_receipt: None,
                success_receipt: None,
                failure_receipt: None,
                opaque_terminal_response_b64: terminal_delivery.opaque_response_b64,
            };
            let response = match terminal_delivery.proof_mode {
                OnionReplyProofMode::RelayVisibleTerminalReceipt => {
                    attach_blind_relay_terminal_success_receipts(
                        Arc::clone(&envelope),
                        response,
                        TerminalDeliveryProofInput {
                            payload: peel.inner,
                            purpose: terminal_delivery.purpose,
                        },
                        accepted_at,
                        Arc::clone(&state.node_identity),
                    )
                    .await?
                }
                // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] The
                // terminal identity and signed workload result are already
                // authenticated inside this fixed-size ciphertext. Omitting
                // the clear terminal receipt prevents every middle hop from
                // reconstructing the final route endpoint.
                OnionReplyProofMode::SourceSealedTerminalProof => {
                    attach_blind_relay_success_receipt(
                        Arc::clone(&envelope),
                        response,
                        accepted_at,
                        Arc::clone(&state.node_identity),
                    )
                    .await?
                }
            };
            complete_blind_relay_route(&state, route_lease, accepted_at, response.clone())?;
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
            state.peer_store.record_blind_relay_terminal(
                accepted_at,
                envelope.ttl,
                envelope.encrypted_blob.len(),
            );
            Ok(response)
        }
        // Entry/middle hop: forward the inner layer to the revealed next hop.
        Some(next_hop) => {
            if next_hop == self_node_id || next_hop == previous_hop_node_id {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
                return Err(BlindRelayError::RouteLoop);
            }
            if !is_onion_blob(&peel.inner) {
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "onion_inner_not_layer",
                );
                return Err(BlindRelayError::OnionPeelFailed);
            }
            if !envelope.can_forward() {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
                return Err(BlindRelayError::TtlExhausted);
            }

            let descriptor = state.peer_store.get_valid(&next_hop, now).ok_or_else(|| {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                BlindRelayError::NoRoute
            })?;
            if !descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }
            // True onion routes are the recovery/proof path for a small mesh:
            // after restart a fresh signed peer may not yet have routeability
            // evidence, and refusing the proof attempt creates a deadlock. Keep
            // the hard stop for peers under local route quarantine; forward
            // errors below will still feed route health without logging payloads.
            if state.peer_store.is_route_quarantined_now(&next_hop, now) {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }

            let endpoint = descriptor
                .descriptor
                .public_endpoint
                .as_deref()
                .ok_or_else(|| {
                    reject_blind_relay_previous_hop(
                        &state,
                        previous_hop_node_id,
                        now,
                        "missing_endpoint",
                    );
                    BlindRelayError::InvalidEndpoint
                })?;
            let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "invalid_endpoint",
                );
                BlindRelayError::InvalidEndpoint
            })?;

            // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Preserve the
            // authenticated ingress timestamp when reconstructing this exact
            // hop. A restart retry must produce byte-identical signed onward
            // input so the downstream node can replay its durable ACK without
            // repeating terminal storage or another network effect.
            let forward_seed = BlindRelayForwardSeed::from(envelope.as_ref());
            let forwarding_identity = Arc::clone(&state.node_identity);
            let forwarded_envelope = run_blind_relay_crypto(move || {
                Ok(build_forwarded_onion_envelope_from_seed(
                    forward_seed,
                    next_hop,
                    peel.inner,
                    forwarding_identity.as_ref(),
                ))
            })
            .await?;
            let ttl_remaining = forwarded_envelope.ttl;

            let forward_started_at = blind_relay_response_observed_at(now, route_started_at);
            route_lease.arm_effect(forward_started_at).map_err(|_| {
                record_blind_relay_replay_protection_failure(&state, forward_started_at)
            })?;
            let next_hop_forward = match forward_blind_relay_with_retry(
                &state,
                &url,
                &descriptor,
                PeerBlindRelayRequest {
                    envelope: forwarded_envelope,
                    previous_hop_node_id: self_node_id,
                    onward_envelope: None,
                    onward_descriptor_hint: None,
                },
                forward_started_at,
            )
            .await
            {
                Ok(ack) => ack,
                Err(error) => return Err(error),
            };
            let observed_at = next_hop_forward.observed_at;
            let next_hop_ack = next_hop_forward.response;

            let response = attach_blind_relay_success_receipt(
                Arc::clone(&envelope),
                PeerBlindRelayResponse {
                    accepted: true,
                    terminal: false,
                    forwarded: true,
                    ttl_remaining,
                    reason: Some("onion_forwarded".to_string()),
                    delivery_receipt: next_hop_ack.delivery_receipt,
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: next_hop_ack.opaque_terminal_response_b64,
                },
                observed_at,
                Arc::clone(&state.node_identity),
            )
            .await?;
            complete_blind_relay_route(&state, route_lease, observed_at, response.clone())?;
            let _ = state
                .peer_store
                .record_route_forward_success_for_descriptor(&descriptor, observed_at);
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
            state
                .peer_store
                .record_blind_relay_forwarded(observed_at, ttl_remaining);

            Ok(response)
        }
    }
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

impl PreparedOnionTerminalPayload {
    const fn requires_durable_guard(&self) -> bool {
        match self {
            Self::BlindVaultReply { reply, .. } => reply.effect().requires_durable_guard(),
            Self::LegacyBlindVaultPut { .. } | Self::Message { .. } => true,
        }
    }
}

/// Performs terminal wire parsing and sender authentication without touching
/// Blind Vault or pending-message storage.
async fn prepare_onion_terminal_payload(
    state: &ChatPeerState,
    payload: &[u8],
    now_secs: u64,
) -> Result<PreparedOnionTerminalPayload, BlindRelayError> {
    if is_onion_reply_request(payload) {
        state
            .blind_vault
            .as_ref()
            .ok_or(BlindRelayError::ForwardFailed)?;
        let reply =
            prepare_blind_vault_inline_reply(payload).map_err(map_terminal_reply_failure)?;
        let execution_permit = blind_vault_terminal_admission()
            .try_acquire_owned()
            .map_err(|_| BlindRelayError::Backpressure)?;
        return Ok(PreparedOnionTerminalPayload::BlindVaultReply {
            reply,
            execution_permit,
        });
    }

    if is_blind_vault_frame(payload) {
        let frame = decode_blind_vault_frame(payload)
            .map_err(|_| BlindRelayError::OnionTerminalPayloadRejected)?;
        let BlindVaultFrame::Put(request) = frame else {
            // Lease admission, pull, delete, issuer, and response frames retain
            // their dedicated bounded client API. The legacy path is Put-only.
            return Err(BlindRelayError::OnionTerminalPayloadRejected);
        };
        state
            .blind_vault
            .as_ref()
            .ok_or(BlindRelayError::ForwardFailed)?;
        let execution_permit = blind_vault_terminal_admission()
            .try_acquire_owned()
            .map_err(|_| BlindRelayError::Backpressure)?;
        return Ok(PreparedOnionTerminalPayload::LegacyBlindVaultPut {
            request,
            execution_permit,
        });
    }

    let envelope =
        decode_envelope(payload).map_err(|_| BlindRelayError::OnionTerminalPayloadRejected)?;
    let envelope = validate_peer_envelope_for_relay(state, envelope, now_secs)
        .await
        .map_err(map_terminal_chat_preparation_error)?;
    let storage_permit =
        acquire_chat_relay_storage(state, now_secs).map_err(map_terminal_chat_preparation_error)?;
    Ok(PreparedOnionTerminalPayload::Message {
        envelope,
        storage_permit,
    })
}

async fn execute_onion_terminal_payload(
    state: &ChatPeerState,
    route_id: [u8; 16],
    prepared: PreparedOnionTerminalPayload,
    now_secs: u64,
) -> Result<OnionTerminalDelivery, BlindRelayError> {
    match prepared {
        PreparedOnionTerminalPayload::BlindVaultReply {
            reply,
            execution_permit,
        } => {
            let vault = Arc::clone(
                state
                    .blind_vault
                    .as_ref()
                    .ok_or(BlindRelayError::ForwardFailed)?,
            );
            let terminal_identity = Arc::clone(&state.node_identity);
            let now_ms = now_secs.saturating_mul(1_000);
            let reply = tokio::task::spawn_blocking(move || {
                let _execution_permit = execution_permit;
                reply.execute(vault.as_ref(), terminal_identity.as_ref(), route_id, now_ms)
            })
            .await
            .map_err(|_| BlindRelayError::ForwardFailed)?
            .map_err(map_terminal_reply_failure)?;
            Ok(OnionTerminalDelivery {
                purpose: reply.purpose,
                proof_mode: reply.proof_mode,
                opaque_response_b64: Some(reply.opaque_response_b64),
            })
        }
        PreparedOnionTerminalPayload::LegacyBlindVaultPut {
            request,
            execution_permit,
        } => {
            let vault = Arc::clone(
                state
                    .blind_vault
                    .as_ref()
                    .ok_or(BlindRelayError::ForwardFailed)?,
            );
            let now_ms = now_secs.saturating_mul(1_000);
            tokio::task::spawn_blocking(move || {
                let _execution_permit = execution_permit;
                vault.put(&request, now_ms)
            })
            .await
            .map_err(|_| BlindRelayError::ForwardFailed)?
            .map_err(|error| map_blind_vault_put_error(&error))?;
            Ok(OnionTerminalDelivery {
                purpose: OnionRoutePurpose::BlindVaultPut,
                proof_mode: OnionReplyProofMode::RelayVisibleTerminalReceipt,
                opaque_response_b64: None,
            })
        }
        PreparedOnionTerminalPayload::Message {
            envelope,
            storage_permit,
        } => process_authenticated_peer_relay_with_storage_permit(
            state.clone(),
            envelope,
            now_secs,
            storage_permit,
        )
        .await
        .map(|_| OnionTerminalDelivery {
            purpose: OnionRoutePurpose::MessageRelay,
            proof_mode: OnionReplyProofMode::RelayVisibleTerminalReceipt,
            opaque_response_b64: None,
        })
        .map_err(|_| BlindRelayError::ForwardFailed),
    }
}

fn map_terminal_reply_failure(failure: TerminalReplyFailure) -> BlindRelayError {
    match failure {
        TerminalReplyFailure::Rejected | TerminalReplyFailure::ResponseTooLarge => {
            BlindRelayError::OnionTerminalPayloadRejected
        }
        // [BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] Valid workload
        // failures are sealed inside opaque replies. This remains fail-closed
        // for any capacity failure that occurs before response sealing.
        TerminalReplyFailure::Capacity => BlindRelayError::OnionTerminalCapacityExhausted,
        TerminalReplyFailure::Unavailable => BlindRelayError::ForwardFailed,
    }
}

fn map_terminal_chat_preparation_error(error: ChatPeerRelayError) -> BlindRelayError {
    match error {
        ChatPeerRelayError::VerificationBackpressure | ChatPeerRelayError::StorageBackpressure => {
            BlindRelayError::Backpressure
        }
        _ => BlindRelayError::ForwardFailed,
    }
}

fn map_blind_vault_put_error(error: &BlindVaultServiceError) -> BlindRelayError {
    // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] The relay must make a
    // useful retry decision without forwarding replica-local state. Every
    // authorization, signature, lease, and object conflict shares one bucket.
    match error.put_failure_class() {
        BlindVaultPutFailureClass::Rejected => BlindRelayError::OnionTerminalPayloadRejected,
        BlindVaultPutFailureClass::Capacity => BlindRelayError::OnionTerminalCapacityExhausted,
        BlindVaultPutFailureClass::Unavailable => BlindRelayError::ForwardFailed,
    }
}

async fn process_onion_middle_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    outer_envelope: BlindRelayEnvelope,
    onward_envelope: BlindRelayEnvelope,
    onward_descriptor_hint: Option<SignedNodeDescriptor>,
    request_commitment: [u8; 32],
    now: u64,
    route_started_at: &Instant,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let outer_envelope = Arc::new(outer_envelope);
    // [AUTHENTICATED-ONWARD-DOMAIN 2026-08-30 by Codex] The private
    // `AuthenticatedPeerBlindRelayRequest` constructor already verified the
    // onward signature inside bounded blocking admission. Repeating that work
    // here would let valid requests consume Ed25519 verification on Tokio.
    validate_blind_relay_metadata(&onward_envelope, now).map_err(|error| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, error.reason_bucket());
        error
    })?;

    let self_node_id = state.node_identity.public_key_bytes();
    if onward_envelope.next_hop == self_node_id || onward_envelope.next_hop == previous_hop_node_id
    {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if !onward_envelope.can_forward() {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = onward_envelope.next_hop;
    let (descriptor, used_descriptor_hint) = resolve_blind_relay_next_hop_descriptor(
        &state,
        &next_hop,
        now,
        onward_descriptor_hint.as_ref(),
    )
    .ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        BlindRelayError::NoRoute
    })?;
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let mut route_lease = match begin_blind_relay_route(
        &state,
        outer_envelope.route_id,
        request_commitment,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => {
            return attach_blind_relay_success_receipt(
                Arc::clone(&outer_envelope),
                response,
                now,
                Arc::clone(&state.node_identity),
            )
            .await
        }
    };

    let forwarding_identity = Arc::clone(&state.node_identity);
    let forwarded_envelope = run_blind_relay_crypto(move || {
        onward_envelope
            .decremented_ttl()
            .ok_or(BlindRelayError::TtlExhausted)
            .map(|envelope| envelope.sign_with(forwarding_identity.as_ref()))
    })
    .await?;
    let ttl_remaining = forwarded_envelope.ttl;

    let forward_started_at = blind_relay_response_observed_at(now, route_started_at);
    route_lease
        .arm_effect(forward_started_at)
        .map_err(|_| record_blind_relay_replay_protection_failure(&state, forward_started_at))?;
    let next_hop_forward = match forward_blind_relay_with_retry(
        &state,
        &url,
        &descriptor,
        PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: self_node_id,
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
        forward_started_at,
    )
    .await
    {
        Ok(ack) => ack,
        Err(error) => return Err(error),
    };
    let observed_at = next_hop_forward.observed_at;
    let next_hop_ack = next_hop_forward.response;

    let response = attach_blind_relay_success_receipt(
        outer_envelope,
        PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining,
            reason: Some("onion_middle_forwarded".to_string()),
            delivery_receipt: next_hop_ack.delivery_receipt,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: next_hop_ack.opaque_terminal_response_b64,
        },
        observed_at,
        Arc::clone(&state.node_identity),
    )
    .await?;
    complete_blind_relay_route(&state, route_lease, observed_at, response.clone())?;
    let _ = state
        .peer_store
        .record_route_forward_success_for_descriptor(&descriptor, observed_at);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
    state
        .peer_store
        .record_blind_relay_forwarded(observed_at, ttl_remaining);

    Ok(response)
}

fn resolve_blind_relay_next_hop_descriptor(
    state: &ChatPeerState,
    next_hop: &[u8; 32],
    now: u64,
    descriptor_hint: Option<&SignedNodeDescriptor>,
) -> Option<(SignedNodeDescriptor, bool)> {
    if let Some(descriptor) = state.peer_store.get_valid(next_hop, now) {
        return Some((descriptor, false));
    }

    let descriptor = descriptor_hint?;
    if descriptor.node_id() != *next_hop {
        return None;
    }
    if descriptor.verify_at(now).is_err() {
        return None;
    }
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        return None;
    }
    Some((descriptor.clone(), true))
}

fn begin_blind_relay_route(
    state: &ChatPeerState,
    route_id: [u8; 16],
    request_commitment: [u8; 32],
    previous_hop: [u8; 32],
    now: u64,
) -> Result<BlindRelayRouteStart, BlindRelayError> {
    if let Some(relay) = state.chat_relay.as_ref() {
        let admission = relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .map_err(|_| record_blind_relay_replay_protection_failure(state, now))?;
        return match admission {
            BlindRelayRouteAdmission::Reserved => Ok(BlindRelayRouteStart::Acquired(
                BlindRelayRouteLease::durable(
                    Arc::clone(relay),
                    route_id,
                    request_commitment,
                    false,
                ),
            )),
            BlindRelayRouteAdmission::ReservedForRecovery => Ok(BlindRelayRouteStart::Acquired(
                BlindRelayRouteLease::durable(
                    Arc::clone(relay),
                    route_id,
                    request_commitment,
                    true,
                ),
            )),
            BlindRelayRouteAdmission::Pending => {
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "route_in_flight");
                Err(BlindRelayError::RouteInFlight)
            }
            BlindRelayRouteAdmission::Conflict => {
                reject_blind_relay_previous_hop(state, previous_hop, now, "replay_conflict");
                Err(BlindRelayError::ReplayConflict)
            }
            BlindRelayRouteAdmission::CapacityExhausted => {
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "replay_capacity");
                Err(BlindRelayError::ReplayCapacity)
            }
            BlindRelayRouteAdmission::Completed {
                response,
                completed_at,
            } => {
                // Signed receipts are online evidence. Do not replay one after
                // its verifier freshness window, but retain the route row for
                // the full envelope horizon so stale retries cannot re-execute.
                if completed_at > now
                    || now.saturating_sub(completed_at) > BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS
                {
                    state
                        .peer_store
                        .record_blind_relay_rejected(now, "replay_response_expired");
                    return Err(BlindRelayError::ReplayResponseExpired);
                }
                let response = decode_durable_blind_relay_response(&response)
                    .map_err(|_| record_blind_relay_replay_protection_failure(state, now))?;
                validate_completed_blind_relay_response(&response)
                    .map_err(|_| record_blind_relay_replay_protection_failure(state, now))?;
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "duplicate_route");
                record_blind_relay_previous_hop_success(state, previous_hop);
                Ok(BlindRelayRouteStart::Completed(response))
            }
        };
    }

    let decision = state
        .blind_relay_replay_registry
        .observe(route_id, request_commitment, now);

    match decision {
        BlindRelayRouteReplayDecision::New { generation } => {
            Ok(BlindRelayRouteStart::Acquired(BlindRelayRouteLease::local(
                Arc::clone(&state.blind_relay_replay_registry),
                route_id,
                request_commitment,
                generation,
            )))
        }
        BlindRelayRouteReplayDecision::InFlight => {
            // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] An unresolved first
            // attempt is not proof of acceptance. Return a retryable status and
            // leave previous-hop health unchanged; a later retry will either
            // replay the durable result or own a fresh attempt after failure.
            state
                .peer_store
                .record_blind_relay_rejected(now, "route_in_flight");
            Err(BlindRelayError::RouteInFlight)
        }
        BlindRelayRouteReplayDecision::Saturated => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "replay_capacity");
            Err(BlindRelayError::ReplayCapacity)
        }
        BlindRelayRouteReplayDecision::Conflict => {
            reject_blind_relay_previous_hop(state, previous_hop, now, "replay_conflict");
            Err(BlindRelayError::ReplayConflict)
        }
        BlindRelayRouteReplayDecision::Completed(response) => {
            // ACK-loss retries receive the exact bounded success response,
            // including any terminal-signed receipt. No payload is retained.
            state
                .peer_store
                .record_blind_relay_rejected(now, "duplicate_route");
            record_blind_relay_previous_hop_success(state, previous_hop);
            Ok(BlindRelayRouteStart::Completed(*response))
        }
    }
}

fn complete_blind_relay_route(
    state: &ChatPeerState,
    route_lease: BlindRelayRouteLease,
    now: u64,
    response: PeerBlindRelayResponse,
) -> Result<(), BlindRelayError> {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] One completion boundary
    // keeps every terminal/forward path on the same durable failure telemetry.
    // The fixed bucket exposes no route, peer, endpoint, receipt, or payload.
    route_lease
        .complete(now, response)
        .map_err(|_| record_blind_relay_replay_protection_failure(state, now))
}

fn record_blind_relay_replay_protection_failure(
    state: &ChatPeerState,
    now: u64,
) -> BlindRelayError {
    state
        .peer_store
        .record_blind_relay_rejected(now, "replay_protection_unavailable");
    BlindRelayError::ReplayProtectionUnavailable
}

fn check_blind_relay_previous_hop_allowed(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
) -> Result<(), BlindRelayError> {
    let decision = state
        .blind_relay_abuse_guard
        .observe_request(previous_hop, now);

    match decision {
        BlindRelayAbuseDecision::Allowed => Ok(()),
        BlindRelayAbuseDecision::CapacityLimited => {
            // [BLIND-RELAY-BUCKET-FAIRNESS 2026-08-21 by Codex] Capacity
            // pressure is aggregate node protection, not evidence that this
            // authenticated peer misbehaved. Do not mutate peer reputation or
            // quarantine state while every retained bucket is still protected.
            state
                .peer_store
                .record_blind_relay_rejected(now, "rate_limited");
            Err(BlindRelayError::RateLimited)
        }
        BlindRelayAbuseDecision::RateLimited { quarantine_until } => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "rate_limited");
            state
                .peer_store
                .record_blind_relay_quarantine_started(now, "rate_limit");
            state
                .peer_store
                .record_peer_relay_rejection(&previous_hop, now, "rate_limited");
            state.peer_store.record_peer_relay_quarantine_started(
                &previous_hop,
                now,
                quarantine_until,
                "rate_limit",
            );
            Err(BlindRelayError::RateLimited)
        }
        BlindRelayAbuseDecision::Quarantined { quarantine_until } => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "quarantined");
            state
                .peer_store
                .record_peer_relay_rejection(&previous_hop, now, "quarantined");
            state.peer_store.record_peer_relay_quarantine_started(
                &previous_hop,
                now,
                quarantine_until,
                "still_quarantined",
            );
            Err(BlindRelayError::Quarantined)
        }
    }
}

fn reject_blind_relay_previous_hop(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
    reason: &'static str,
) {
    state.peer_store.record_blind_relay_rejected(now, reason);
    state
        .peer_store
        .record_peer_relay_rejection(&previous_hop, now, reason);
    if !blind_relay_reason_counts_toward_quarantine(reason) {
        return;
    }

    let quarantine_until = state
        .blind_relay_abuse_guard
        .record_failure(previous_hop, now);
    if let Some(quarantine_until) = quarantine_until {
        state
            .peer_store
            .record_blind_relay_quarantine_started(now, "failure_threshold");
        state.peer_store.record_peer_relay_quarantine_started(
            &previous_hop,
            now,
            quarantine_until,
            "failure_threshold",
        );
    }
}

fn record_blind_relay_previous_hop_success(state: &ChatPeerState, previous_hop: [u8; 32]) {
    state.blind_relay_abuse_guard.record_success(previous_hop);
}

fn blind_relay_reason_counts_toward_quarantine(reason: &str) -> bool {
    matches!(
        reason,
        "invalid_previous_hop" | "invalid_signature" | "self_loop" | "route_loop" | "ttl_exhausted"
    )
}

/// Projects a monotonic request duration onto the caller's Unix timestamp.
///
/// [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Tests and recovery
/// probes inject a stable Unix base. `Instant` supplies elapsed time without
/// depending on wall-clock jumps, while saturation keeps failure handling
/// total even near the integer boundary.
fn blind_relay_response_observed_at(started_at: u64, started: &Instant) -> u64 {
    started_at.saturating_add(started.elapsed().as_secs())
}

/// Replaceable capabilities composed for one forwarding operation.
struct BlindRelayForwardComponents<'a> {
    retry_policy: Arc<dyn BlindRelayRetryPolicy>,
    response_policy: Arc<dyn BlindRelayResponsePolicy>,
    transport: &'a dyn BlindRelayTransport,
    observer: &'a dyn BlindRelayForwardObserver,
}

async fn forward_blind_relay_with_retry(
    state: &ChatPeerState,
    url: &str,
    descriptor: &SignedNodeDescriptor,
    request: PeerBlindRelayRequest,
    now: u64,
) -> Result<BlindRelayForwardOutcome, BlindRelayError> {
    let transport = ReqwestBlindRelayTransport::new(Arc::clone(&state.http_client));
    let observer = PeerStoreBlindRelayForwardObserver::new(state.peer_store.as_ref());
    forward_blind_relay_with_components(
        url,
        descriptor,
        request,
        now,
        BlindRelayForwardComponents {
            retry_policy: Arc::new(BlindRelayRetryDomain::default()),
            response_policy: Arc::new(BlindRelayResponseDomain),
            transport: &transport,
            observer: &observer,
        },
    )
    .await
}

/// [ROUTE-FAILURE-SURFACE-BINDING 2026-08-11 by Codex] Keeps the exact signed
/// descriptor that selected `url` through retries, so delayed observations can
/// update health only when the selected route surface remains current.
async fn forward_blind_relay_with_components(
    url: &str,
    descriptor: &SignedNodeDescriptor,
    request: PeerBlindRelayRequest,
    now: u64,
    components: BlindRelayForwardComponents<'_>,
) -> Result<BlindRelayForwardOutcome, BlindRelayError> {
    let request = Arc::new(request);
    let next_hop = descriptor.node_id();
    let failure_receipt_required = blind_relay_failure_receipt_required(descriptor);
    let success_receipt_required = blind_relay_success_receipt_required(descriptor);
    let source_sealed_terminal_proof_allowed =
        onion_source_sealed_terminal_proof_allowed(descriptor);
    let large_pull_response_allowed = onion_large_pull_response_allowed(descriptor);
    let request_started_at = Instant::now();
    for attempt in 1..=components.retry_policy.max_attempts().get() {
        let retry_context = blind_relay_retry_context(request.as_ref(), next_hop, attempt)?;
        let transport_outcome = components.transport.send(url, request.as_ref()).await;
        let observed_at = blind_relay_response_observed_at(now, &request_started_at);
        let request_for_validation = Arc::clone(&request);
        let response_policy = Arc::clone(&components.response_policy);
        let retry_policy = Arc::clone(&components.retry_policy);
        let decision = complete_blind_relay_crypto(move || {
            // [BLIND-RESPONSE-CRYPTO-COMPLETION 2026-08-30 by Codex] Response
            // decoding is already byte-bounded by the transport. Keep policy
            // evaluation pure while moving receipt verification and payload
            // commitment hashing away from the asynchronous I/O runtime.
            Ok(response_policy.evaluate(
                transport_outcome,
                BlindRelayResponseContext {
                    request: request_for_validation.as_ref(),
                    next_hop,
                    observed_at,
                    failure_receipt_required,
                    success_receipt_required,
                    source_sealed_terminal_proof_allowed,
                    large_pull_response_allowed,
                    retry_context,
                    retry_policy: retry_policy.as_ref(),
                },
            ))
        })
        .await?;
        match decision {
            BlindRelayResponseDecision::Accepted(response) => {
                return Ok(complete_blind_relay_forward(
                    components.observer,
                    *response,
                    observed_at,
                    attempt,
                ))
            }
            BlindRelayResponseDecision::PeerDeclaredFailure {
                failure,
                status,
                receipt_authenticated,
            } => {
                // [DOWNSTREAM-FAILURE-ATTRIBUTION 2026-08-11 by Codex] A
                // bounded error ACK cannot identify which deeper hop failed.
                debug!(
                    attempt,
                    status = %status,
                    receipt_authenticated,
                    "[BLIND_RELAY] Peer-declared downstream failure left unattributed"
                );
                let error = BlindRelayError::from(failure);
                components
                    .observer
                    .rejected(observed_at, error.reason_bucket());
                return Err(error);
            }
            BlindRelayResponseDecision::RetryAfter {
                delay,
                reason,
                source,
            } => {
                components.observer.retry_attempted(observed_at, &reason);
                log_blind_relay_retry(attempt, source, &reason);
                sleep(delay).await;
            }
            BlindRelayResponseDecision::Reject(failure) => {
                let error = BlindRelayError::from(failure);
                components
                    .observer
                    .route_failed(descriptor, observed_at, error.reason_bucket());
                return Err(error);
            }
            BlindRelayResponseDecision::InvalidResponse {
                kind,
                diagnostic,
                health_reason,
                counts_as_retry_exhaustion,
            } => {
                log_invalid_blind_relay_response(attempt, kind, diagnostic);
                if counts_as_retry_exhaustion && attempt > 1 {
                    components
                        .observer
                        .retry_exhausted(observed_at, attempt, health_reason);
                }
                components
                    .observer
                    .route_failed(descriptor, observed_at, health_reason);
                return Err(BlindRelayError::ForwardFailed);
            }
            BlindRelayResponseDecision::Exhausted { reason, source } => {
                log_blind_relay_exhausted(attempt, source, &reason);
                if attempt > 1 {
                    components
                        .observer
                        .retry_exhausted(observed_at, attempt, &reason);
                }
                components
                    .observer
                    .route_failed(descriptor, observed_at, &reason);
                return Err(BlindRelayError::ForwardFailed);
            }
        }
    }

    Err(BlindRelayError::ForwardFailed)
}

fn blind_relay_retry_context(
    request: &PeerBlindRelayRequest,
    next_hop: [u8; 32],
    attempt: usize,
) -> Result<BlindRelayRetryContext, BlindRelayError> {
    BlindRelayRetryContext::new(request.envelope.route_id, next_hop, attempt)
        .ok_or(BlindRelayError::ForwardFailed)
}

fn blind_relay_failure_receipt_required(descriptor: &SignedNodeDescriptor) -> bool {
    // [FAILURE-RECEIPT-ANTI-DOWNGRADE 2026-08-11 by Codex] Negotiate from the
    // exact signed descriptor that selected this URL. An attacker cannot strip
    // this token without invalidating the descriptor signature.
    descriptor
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::BlindRelayFailureReceiptV1)
}

fn blind_relay_success_receipt_required(descriptor: &SignedNodeDescriptor) -> bool {
    descriptor
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::BlindRelaySuccessReceiptV1)
}

fn onion_source_sealed_terminal_proof_allowed(descriptor: &SignedNodeDescriptor) -> bool {
    // Both tokens are required because an opaque-only response is safe to
    // accept only when the immediate peer authenticates the exact bytes it
    // returned. Signed descriptor negotiation prevents downgrade by gossip or
    // an endpoint that does not own the advertised identity.
    blind_relay_success_receipt_required(descriptor)
        && descriptor
            .descriptor
            .advertises_protocol_feature(NodeProtocolFeature::OnionSourceSealedTerminalProofV1)
}

fn onion_large_pull_response_allowed(descriptor: &SignedNodeDescriptor) -> bool {
    // [BLIND-VAULT-LARGE-PULL-VALIDATION 2026-08-30 by Codex] A bounded large
    // ACK is accepted only from the exact signed descriptor selected for this
    // hop. The response remains opaque, so no relay learns the workload type.
    descriptor
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultLargePullV1)
}

fn complete_blind_relay_forward(
    observer: &dyn BlindRelayForwardObserver,
    response: PeerBlindRelayResponse,
    observed_at: u64,
    attempt: usize,
) -> BlindRelayForwardOutcome {
    if attempt > 1 {
        observer.retry_succeeded(observed_at, attempt);
    }
    BlindRelayForwardOutcome {
        response,
        observed_at,
    }
}

fn log_blind_relay_retry(attempt: usize, source: BlindRelayResponseSource, reason: &str) {
    match source {
        BlindRelayResponseSource::HttpStatus(status) => debug!(
            attempt,
            status = %status,
            "[BLIND_RELAY] Next-hop returned retryable status"
        ),
        BlindRelayResponseSource::Transport => debug!(
            attempt,
            reason, "[BLIND_RELAY] Next-hop forward failed; retrying"
        ),
    }
}

fn log_blind_relay_exhausted(attempt: usize, source: BlindRelayResponseSource, reason: &str) {
    match source {
        BlindRelayResponseSource::HttpStatus(status) => debug!(
            attempt,
            status = %status,
            "[BLIND_RELAY] Next-hop returned non-success"
        ),
        BlindRelayResponseSource::Transport => {
            debug!(attempt, reason, "[BLIND_RELAY] Next-hop forward failed")
        }
    }
}

fn log_invalid_blind_relay_response(
    attempt: usize,
    kind: BlindRelayInvalidResponseKind,
    diagnostic: &'static str,
) {
    match kind {
        BlindRelayInvalidResponseKind::SuccessAck => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop ACK invalid"
        ),
        BlindRelayInvalidResponseKind::SuccessReceipt => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop hop-local success receipt verification failed"
        ),
        BlindRelayInvalidResponseKind::DeliveryReceipt => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop delivery receipt verification failed"
        ),
        BlindRelayInvalidResponseKind::OpaqueTerminalResponse => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop opaque terminal response validation failed"
        ),
        BlindRelayInvalidResponseKind::FailureReceipt => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop failure receipt verification failed"
        ),
    }
}

#[cfg(test)]
fn validate_blind_relay_envelope(
    envelope: &BlindRelayEnvelope,
    previous_hop_node_id: &[u8; 32],
    now: u64,
) -> Result<(), BlindRelayError> {
    authenticate_blind_relay_envelope(envelope, previous_hop_node_id)?;
    validate_blind_relay_metadata(envelope, now)
}

fn authenticate_blind_relay_envelope(
    envelope: &BlindRelayEnvelope,
    previous_hop_node_id: &[u8; 32],
) -> Result<(), BlindRelayError> {
    let previous_hop = IdentityPublicKey::from_bytes(previous_hop_node_id)
        .map_err(|_| BlindRelayError::InvalidPreviousHop)?;
    envelope
        .verify_signature_from(&previous_hop)
        .map_err(|_| BlindRelayError::InvalidSignature)?;
    Ok(())
}

fn validate_blind_relay_metadata(
    envelope: &BlindRelayEnvelope,
    now: u64,
) -> Result<(), BlindRelayError> {
    validate_blind_relay_timestamp(envelope.timestamp, now)?;
    validate_blind_relay_envelope_size(envelope).map_err(|_| BlindRelayError::EnvelopeTooLarge)?;
    Ok(())
}

fn validate_blind_relay_timestamp(timestamp: u64, now: u64) -> Result<(), BlindRelayError> {
    validate_relay_timestamp(
        timestamp,
        now,
        BLIND_RELAY_MAX_ENVELOPE_AGE_SECS,
        BLIND_RELAY_MAX_FUTURE_SKEW_SECS,
    )
    .map_err(|error| match error {
        RelayTimestampError::Expired => BlindRelayError::TimestampExpired,
        RelayTimestampError::InFuture => BlindRelayError::TimestampInFuture,
    })
}

fn blind_peer_relay_url(endpoint: &str) -> Option<String> {
    // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] A next-hop descriptor is
    // permissionless input. Its signature cannot authorize localhost, private
    // networks, metadata services, DNS rebinding, or URL-controlled paths.
    if !peer_endpoint_is_public_ip(endpoint) {
        #[cfg(not(test))]
        return None;
        #[cfg(test)]
        if !crate::api::peer_endpoint_is_loopback_ip(endpoint) {
            return None;
        }
    }
    canonical_peer_http_url(endpoint, "/api/chat/peer/blind-relay")
        .ok()
        .map(|url| url.to_string())
}

fn validate_peer_envelope(envelope: &ChatEnvelope, now: u64) -> Result<(), ChatPeerRelayError> {
    envelope
        .verify_signature()
        .map_err(|_| ChatPeerRelayError::InvalidSignature)?;

    // [PEER-RELAY-REPLAY-WINDOW 2026-08-15 by Codex] Direct compatibility
    // relay is immediate node-to-node work, not a durable replay token. Use
    // the same bounded freshness policy as blind routing so an observed signed
    // ciphertext cannot be admitted to fresh node mailboxes indefinitely.
    validate_relay_timestamp(
        envelope.timestamp,
        now,
        BLIND_RELAY_MAX_ENVELOPE_AGE_SECS,
        BLIND_RELAY_MAX_FUTURE_SKEW_SECS,
    )
    .map_err(|error| match error {
        RelayTimestampError::Expired => ChatPeerRelayError::TimestampExpired,
        RelayTimestampError::InFuture => ChatPeerRelayError::TimestampInFuture,
    })?;

    let encoded = encode_envelope(envelope).map_err(|_| ChatPeerRelayError::Serialization)?;
    if encoded.len() > MAX_PEER_CHAT_ENVELOPE_BYTES {
        return Err(ChatPeerRelayError::EnvelopeTooLarge {
            size: encoded.len(),
        });
    }

    Ok(())
}

fn validate_relay_timestamp(
    timestamp: u64,
    now: u64,
    max_age_secs: u64,
    max_future_skew_secs: u64,
) -> Result<(), RelayTimestampError> {
    if timestamp > now.saturating_add(max_future_skew_secs) {
        return Err(RelayTimestampError::InFuture);
    }
    if now.saturating_sub(timestamp) > max_age_secs {
        return Err(RelayTimestampError::Expired);
    }
    Ok(())
}

async fn send_envelope_to_session(
    envelope: &ChatEnvelope,
    session: &Arc<Session>,
    udp: &Arc<UdpTransport>,
) -> bool {
    let msg = MemChainMessage::ChatRelay(envelope.clone());
    let plaintext = match encode_memchain(&msg) {
        Ok(plaintext) => plaintext,
        Err(_error) => {
            warn!(
                reason = "encode_client_relay_message_failed",
                "[CHAT_PEER] Failed to encode client relay message"
            );
            return false;
        }
    };

    let crypto = DefaultTransportCrypto::new();
    let counter = session.next_tx_counter();
    let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
    let len = match crypto.encrypt(
        &session.session_key,
        counter,
        session.id.as_bytes(),
        &plaintext,
        &mut encrypted,
    ) {
        Ok(len) => len,
        Err(_error) => {
            warn!(
                reason = "encrypt_client_relay_message_failed",
                "[CHAT_PEER] Failed to encrypt client relay message"
            );
            return false;
        }
    };
    encrypted.truncate(len);

    let packet = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
    let bytes = encode_data_packet(&packet).to_vec();
    udp.send(&bytes, &session.client_endpoint).await.is_ok()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultAdmissionTicket, BlindVaultLeaseAdmissionRequest,
    };
    use aeronyx_core::protocol::chat::ChatContentType;
    use aeronyx_core::protocol::{
        encode_blind_vault_frame, BlindVaultFrame, BlindVaultLeaseCreateRequest,
        BlindVaultPutRequest, NodeCapability, NodeCapacity, NodeDescriptor, SignedNodeDescriptor,
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

    #[test]
    fn direct_peer_relay_receipt_binds_request_target_signature_and_freshness() {
        // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Every receipt is
        // usable only for the exact authenticated request and selected node,
        // within the short online acknowledgement window.
        let previous_hop = IdentityKeyPair::generate();
        let target = IdentityKeyPair::generate();
        let observed_at = 1_800_000_000u64;
        let request =
            PeerChatRelayRequestV2::sign(signed_envelope_at(observed_at), &previous_hop).unwrap();
        let receipt = PeerChatRelayReceiptV2::accepted(
            request.request_commitment().unwrap(),
            observed_at,
            &target,
        );

        assert_eq!(
            receipt.verify_expected(&request, &target.public_key_bytes(), observed_at),
            Ok(())
        );

        let wrong_target = IdentityKeyPair::generate();
        assert_eq!(
            receipt.verify_expected(&request, &wrong_target.public_key_bytes(), observed_at),
            Err("receipt_binding_invalid")
        );

        let other_request = PeerChatRelayRequestV2::sign(
            signed_envelope_at(observed_at.saturating_add(1)),
            &previous_hop,
        )
        .unwrap();
        assert_eq!(
            receipt.verify_expected(&other_request, &target.public_key_bytes(), observed_at),
            Err("receipt_binding_invalid")
        );

        let mut forged = receipt.clone();
        forged.signature[0] ^= 0x01;
        assert_eq!(
            forged.verify_expected(&request, &target.public_key_bytes(), observed_at),
            Err("receipt_signature_invalid")
        );

        let expired = PeerChatRelayReceiptV2::accepted(
            request.request_commitment().unwrap(),
            observed_at.saturating_sub(PEER_CHAT_RELAY_RECEIPT_MAX_AGE_SECS + 1),
            &target,
        );
        assert_eq!(
            expired.verify_expected(&request, &target.public_key_bytes(), observed_at),
            Err("receipt_timestamp_expired")
        );

        let future = PeerChatRelayReceiptV2::accepted(
            request.request_commitment().unwrap(),
            observed_at.saturating_add(PEER_CHAT_RELAY_RECEIPT_MAX_FUTURE_SKEW_SECS + 1),
            &target,
        );
        assert_eq!(
            future.verify_expected(&request, &target.public_key_bytes(), observed_at),
            Err("receipt_timestamp_in_future")
        );

        let encoded = serde_json::to_value(PeerChatRelayResponseV2 {
            relay: durable_peer_acceptance_response(),
            receipt: Some(receipt),
        })
        .unwrap();
        let object = encoded.as_object().unwrap();
        assert_eq!(object.len(), 5);
        assert!(object.contains_key("receipt"));
        for forbidden in [
            "receiver",
            "message_id",
            "online",
            "endpoint",
            "payload_size",
        ] {
            assert!(!object.contains_key(forbidden));
        }
    }

    #[test]
    fn delayed_receipt_validation_uses_response_observation_time() {
        // [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] A valid ACK
        // produced after a long request must be compared with response time,
        // not the stale ingress timestamp. Backdating only the monotonic test
        // clock keeps this regression deterministic and fast.
        let started_at = 1_800_000_000u64;
        let request_started_at = Instant::now() - Duration::from_secs(31);
        let observed_at = blind_relay_response_observed_at(started_at, &request_started_at);
        assert!(observed_at >= started_at + 31);

        let terminal = IdentityKeyPair::generate();
        let terminal_node_id = terminal.public_key_bytes();
        let route_id = [0x81u8; 16];
        let payload = b"opaque delayed message relay payload";
        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted_for_purpose(
                route_id,
                payload,
                OnionRoutePurpose::MessageRelay,
                started_at + 31,
                &terminal,
            )),
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };

        assert_eq!(
            validate_downstream_delivery_receipt(&ack, &route_id, &terminal_node_id, started_at,),
            Err("receipt_timestamp_in_future")
        );
        assert!(validate_downstream_delivery_receipt(
            &ack,
            &route_id,
            &terminal_node_id,
            observed_at,
        )
        .is_ok());
    }

    #[test]
    fn three_hop_forwarded_ack_accepts_downstream_terminal_receipt() {
        let now = 1_800_000_100;
        let route_id = [0xa1; 16];
        let immediate_middle = IdentityKeyPair::generate();
        let downstream_terminal = IdentityKeyPair::generate();
        assert_ne!(
            immediate_middle.public_key_bytes(),
            downstream_terminal.public_key_bytes()
        );

        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("onion_forwarded".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
                route_id,
                [0xb2; 32],
                now,
                &downstream_terminal,
            )),
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };

        validate_downstream_delivery_receipt(
            &ack,
            &route_id,
            &immediate_middle.public_key_bytes(),
            now,
        )
        .expect("an intermediate ACK may propagate a deeper terminal receipt");
    }

    #[test]
    fn direct_terminal_ack_requires_immediate_next_hop_receipt_signer() {
        let now = 1_800_000_100;
        let route_id = [0xc3; 16];
        let immediate_terminal = IdentityKeyPair::generate();
        let wrong_terminal = IdentityKeyPair::generate();
        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
                route_id,
                [0xd4; 32],
                now,
                &wrong_terminal,
            )),
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };

        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack,
                &route_id,
                &immediate_terminal.public_key_bytes(),
                now,
            ),
            Err("terminal_receipt_signer_mismatch")
        );
    }

    #[test]
    fn downstream_success_ack_requires_exactly_one_delivery_disposition() {
        // [RELAY-ACK-STATE-MACHINE 2026-08-11 by Codex] Receipt-less ACKs stay
        // valid for mixed-version peers only when their state proves one real
        // terminal or forwarding action. No-op and contradictory success
        // shapes must fail before they can become route-health evidence.
        let now = 1_800_000_100;
        let route_id = [0xd5; 16];
        let immediate_next_hop = IdentityKeyPair::generate().public_key_bytes();
        let ack = |accepted, terminal, forwarded| PeerBlindRelayResponse {
            accepted,
            terminal,
            forwarded,
            ttl_remaining: 1,
            reason: None,
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };

        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack(true, false, false),
                &route_id,
                &immediate_next_hop,
                now,
            ),
            Err("invalid_ack_shape")
        );
        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack(true, true, true),
                &route_id,
                &immediate_next_hop,
                now,
            ),
            Err("invalid_ack_shape")
        );
        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack(false, false, false),
                &route_id,
                &immediate_next_hop,
                now,
            ),
            Err("ack_not_accepted")
        );
        assert!(validate_downstream_delivery_receipt(
            &ack(true, true, false),
            &route_id,
            &immediate_next_hop,
            now,
        )
        .is_ok());
        assert!(validate_downstream_delivery_receipt(
            &ack(true, false, true),
            &route_id,
            &immediate_next_hop,
            now,
        )
        .is_ok());
    }

    #[test]
    fn signed_failure_receipt_is_exact_fresh_and_legacy_compatible() {
        let now = 1_800_000_200;
        let previous_hop = IdentityKeyPair::generate();
        let responder = IdentityKeyPair::generate();
        let other_responder = IdentityKeyPair::generate();
        let envelope = BlindRelayEnvelope {
            route_id: [0xb1; 16],
            next_hop: responder.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque failure receipt request".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);
        let request = PeerBlindRelayRequest {
            envelope: envelope.clone(),
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let signed_receipt = |failed_at, signer: &IdentityKeyPair| {
            BlindRelayFailureReceipt::failed(
                envelope.route_id,
                BlindRelayFailureReceipt::request_commitment(&envelope),
                "forward_failed",
                failed_at,
                signer,
            )
        };
        let failure_ack = |receipt| PeerBlindRelayResponse {
            accepted: false,
            terminal: false,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some("forward_failed".to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: receipt,
            opaque_terminal_response_b64: None,
        };

        let authenticated = failure_ack(Some(signed_receipt(now, &responder)));
        assert_eq!(
            validate_downstream_failure_receipt(
                &authenticated,
                &request,
                &responder.public_key_bytes(),
                now,
                false,
            ),
            Ok(true)
        );
        assert_eq!(
            validate_downstream_failure_receipt(
                &failure_ack(None),
                &request,
                &responder.public_key_bytes(),
                now,
                false,
            ),
            Ok(false),
            "missing receipt remains the explicit mixed-version path"
        );
        assert_eq!(
            validate_downstream_failure_receipt(
                &failure_ack(None),
                &request,
                &responder.public_key_bytes(),
                now,
                true,
            ),
            Err("failure_receipt_required"),
            "an advertised receipt cannot be silently downgraded"
        );
        let legacy_ack: PeerBlindRelayResponse = serde_json::from_value(serde_json::json!({
            "accepted": false,
            "terminal": false,
            "forwarded": false,
            "ttl_remaining": 0,
            "reason": "forward_failed"
        }))
        .expect("legacy ACK without receipt fields must remain decodable");
        assert!(legacy_ack.delivery_receipt.is_none());
        assert!(legacy_ack.failure_receipt.is_none());

        let mut reason_substitution = authenticated.clone();
        reason_substitution.reason = Some("no_route".to_string());
        assert_eq!(
            validate_downstream_failure_receipt(
                &reason_substitution,
                &request,
                &responder.public_key_bytes(),
                now,
                false,
            ),
            Err("failure_receipt_binding_invalid")
        );
        assert_eq!(
            validate_downstream_failure_receipt(
                &failure_ack(Some(signed_receipt(now, &other_responder))),
                &request,
                &responder.public_key_bytes(),
                now,
                false,
            ),
            Err("failure_receipt_binding_invalid")
        );
        assert_eq!(
            validate_downstream_failure_receipt(
                &failure_ack(Some(signed_receipt(
                    now - BLIND_RELAY_FAILURE_RECEIPT_MAX_AGE_SECS - 1,
                    &responder,
                ))),
                &request,
                &responder.public_key_bytes(),
                now,
                false,
            ),
            Err("failure_receipt_timestamp_expired")
        );
        assert_eq!(
            validate_downstream_failure_receipt(
                &failure_ack(Some(signed_receipt(
                    now + BLIND_RELAY_FAILURE_RECEIPT_MAX_FUTURE_SKEW_SECS + 1,
                    &responder,
                ))),
                &request,
                &responder.public_key_bytes(),
                now,
                false,
            ),
            Err("failure_receipt_timestamp_in_future")
        );

        let mut contradictory_success = authenticated;
        contradictory_success.accepted = true;
        contradictory_success.terminal = true;
        assert_eq!(
            validate_downstream_delivery_receipt(
                &contradictory_success,
                &envelope.route_id,
                &responder.public_key_bytes(),
                now,
            ),
            Err("unexpected_failure_receipt")
        );
    }

    #[test]
    fn forwarded_ack_rejects_tampered_downstream_receipt() {
        let now = 1_800_000_100;
        let route_id = [0xe5; 16];
        let immediate_middle = IdentityKeyPair::generate();
        let downstream_terminal = IdentityKeyPair::generate();
        let mut receipt =
            BlindRelayDeliveryReceipt::accepted(route_id, [0xf6; 32], now, &downstream_terminal);
        receipt.payload_commitment[0] ^= 0xff;
        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("onion_forwarded".to_string()),
            delivery_receipt: Some(receipt),
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };

        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack,
                &route_id,
                &immediate_middle.public_key_bytes(),
                now,
            ),
            Err("receipt_signature_invalid")
        );
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
    fn validate_peer_envelope_rejects_tampered_ciphertext() {
        let mut envelope = signed_envelope();
        envelope.ciphertext.push(0x44);

        assert!(matches!(
            validate_peer_envelope(&envelope, now_secs()),
            Err(ChatPeerRelayError::InvalidSignature)
        ));
    }

    #[test]
    fn validate_peer_envelope_enforces_bounded_replay_window() {
        let now = 1_800_000_000u64;
        assert!(validate_peer_envelope(
            &signed_envelope_at(now.saturating_sub(BLIND_RELAY_MAX_ENVELOPE_AGE_SECS)),
            now,
        )
        .is_ok());
        assert!(matches!(
            validate_peer_envelope(
                &signed_envelope_at(
                    now.saturating_sub(BLIND_RELAY_MAX_ENVELOPE_AGE_SECS)
                        .saturating_sub(1)
                ),
                now,
            ),
            Err(ChatPeerRelayError::TimestampExpired)
        ));
        assert!(validate_peer_envelope(
            &signed_envelope_at(now.saturating_add(BLIND_RELAY_MAX_FUTURE_SKEW_SECS)),
            now,
        )
        .is_ok());
        assert!(matches!(
            validate_peer_envelope(
                &signed_envelope_at(
                    now.saturating_add(BLIND_RELAY_MAX_FUTURE_SKEW_SECS)
                        .saturating_add(1)
                ),
                now,
            ),
            Err(ChatPeerRelayError::TimestampInFuture)
        ));
    }

    #[test]
    fn chat_peer_logs_stay_route_safe() {
        let source = include_str!("chat_peer.rs");
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
    fn pending_store_capacity_maps_to_retryable_service_unavailable() {
        let capacity = ChatRelayError::PendingMessageQueueFull {
            current: 100,
            limit: 100,
        };
        let mapped = map_pending_store_error(&capacity);

        assert!(matches!(mapped, ChatPeerRelayError::PendingCapacity));
        assert_eq!(mapped.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(mapped.reason_bucket(), "pending_capacity_exhausted");

        let storage = ChatRelayError::Sqlite(rusqlite::Error::InvalidQuery);
        assert!(matches!(
            map_pending_store_error(&storage),
            ChatPeerRelayError::StoreFailed
        ));

        let oversized = ChatRelayError::MessageTooLarge {
            size: 65_537,
            limit: 65_536,
        };
        assert!(matches!(
            map_pending_store_error(&oversized),
            ChatPeerRelayError::EnvelopeTooLarge { size: 65_537 }
        ));
    }

    #[test]
    fn downstream_domain_errors_preserve_public_status_mapping() {
        assert!(matches!(
            map_blind_vault_put_error(&BlindVaultServiceError::LeaseNotFound),
            BlindRelayError::OnionTerminalPayloadRejected
        ));
        assert!(matches!(
            map_blind_vault_put_error(&BlindVaultServiceError::QuotaExceeded),
            BlindRelayError::OnionTerminalCapacityExhausted
        ));
        assert!(matches!(
            map_blind_vault_put_error(&BlindVaultServiceError::Disabled),
            BlindRelayError::ForwardFailed
        ));
        assert_eq!(
            BlindRelayError::OnionTerminalCapacityExhausted.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            BlindRelayError::RouteInFlight.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            BlindRelayError::RouteInFlight.reason_bucket(),
            "route_in_flight"
        );
        assert_eq!(
            BlindRelayError::ReplayCapacity.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            BlindRelayError::ReplayCapacity.reason_bucket(),
            "replay_capacity"
        );
        assert_eq!(
            BlindRelayError::Backpressure.status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            BlindRelayError::Backpressure.reason_bucket(),
            "backpressure"
        );
        assert!(matches!(
            BlindRelayError::from(BlindRelayDownstreamFailure::OnionTerminalCapacityExhausted),
            BlindRelayError::OnionTerminalCapacityExhausted
        ));
        assert!(matches!(
            BlindRelayError::from(BlindRelayDownstreamFailure::ForwardFailed),
            BlindRelayError::ForwardFailed
        ));
        assert!(matches!(
            BlindRelayError::from(BlindRelayDownstreamFailure::DownstreamRejected),
            BlindRelayError::DownstreamRejected
        ));
    }

    #[tokio::test]
    async fn blind_relay_signature_admission_rejects_without_queueing() {
        // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] A saturated
        // verifier must reject before spawning more blocking work. Once the
        // permit is released, the same authenticated request remains valid.
        let admission = Arc::new(Semaphore::new(1));
        let held_permit = Arc::clone(&admission)
            .try_acquire_owned()
            .expect("reserve the only verification permit");
        let previous_hop = IdentityKeyPair::generate();
        let envelope = BlindRelayEnvelope {
            route_id: [0x3du8; 16],
            next_hop: IdentityKeyPair::generate().public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque bounded verification test".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);
        let request = PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };

        let rejected = authenticate_peer_blind_relay_request_with_admission(
            Arc::clone(&admission),
            request.clone(),
        )
        .await;
        assert!(matches!(rejected, Err(BlindRelayError::Backpressure)));

        drop(held_permit);
        let authenticated =
            authenticate_peer_blind_relay_request_with_admission(admission, request.clone())
                .await
                .expect("released verifier must accept valid signed work");
        assert_eq!(authenticated.request.envelope.route_id, [0x3du8; 16]);
        assert_eq!(
            authenticated.failure_request_commitment,
            BlindRelayFailureReceipt::request_commitment(&request.envelope)
        );
    }

    #[test]
    fn blind_relay_replay_commitment_binds_optional_onward_envelope() {
        // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] The route id alone
        // is not an idempotency key: a substituted onward frame under the same
        // authenticated outer envelope must become a conflict, never an exact
        // replay or a second relay side effect.
        let previous_hop = IdentityKeyPair::generate();
        let middle = IdentityKeyPair::generate();
        let terminal = IdentityKeyPair::generate();
        let outer = BlindRelayEnvelope {
            route_id: [0x3Eu8; 16],
            next_hop: middle.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque outer layer".to_vec(),
            timestamp: 1_800_000_000,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);
        let base = PeerBlindRelayRequest {
            envelope: outer,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let mut with_onward = base.clone();
        with_onward.onward_envelope = Some(
            BlindRelayEnvelope {
                route_id: [0x3Fu8; 16],
                next_hop: terminal.public_key_bytes(),
                ttl: 1,
                encrypted_blob: b"opaque inner layer".to_vec(),
                timestamp: 1_800_000_000,
                signature: [0u8; 64],
            }
            .sign_with(&previous_hop),
        );

        let base_commitment = blind_relay_authenticated_request_commitment(&base)
            .expect("commit base blind-relay request");
        assert_eq!(
            blind_relay_authenticated_request_commitment(&base)
                .expect("repeat base blind-relay commitment"),
            base_commitment
        );
        assert_ne!(
            blind_relay_authenticated_request_commitment(&with_onward)
                .expect("commit blind-relay request with onward envelope"),
            base_commitment
        );
    }

    #[tokio::test]
    async fn blind_relay_authentication_rejects_substituted_onward_envelope() {
        // [SIGNED-ONWARD-ENVELOPE 2026-08-24 by Codex] A transport intermediary
        // cannot alter the optional legacy onward ciphertext and make this node
        // sign the substituted frame for the terminal hop.
        let previous_hop = IdentityKeyPair::generate();
        let middle = IdentityKeyPair::generate();
        let terminal = IdentityKeyPair::generate();
        let outer = BlindRelayEnvelope {
            route_id: [0x40; 16],
            next_hop: middle.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque outer carrier".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);
        let mut onward = BlindRelayEnvelope {
            route_id: [0x41; 16],
            next_hop: terminal.public_key_bytes(),
            ttl: 1,
            encrypted_blob: b"signed opaque onward frame".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);
        onward.encrypted_blob = b"substituted opaque onward frame".to_vec();

        let result = authenticate_peer_blind_relay_request_with_admission(
            Arc::new(Semaphore::new(1)),
            PeerBlindRelayRequest {
                envelope: outer,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: Some(onward),
                onward_descriptor_hint: None,
            },
        )
        .await;
        assert!(matches!(result, Err(BlindRelayError::InvalidSignature)));
    }

    #[tokio::test]
    async fn peer_relay_endpoint_stores_offline_receiver_message() {
        let (relay, path) = temp_chat_relay("chat-peer");
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let http_client = Arc::new(reqwest::Client::new());
        let envelope = signed_envelope();
        let receiver = envelope.receiver;

        let app = build_chat_peer_router(
            Some(Arc::clone(&relay)),
            sessions,
            udp,
            peer_store,
            node_identity,
            http_client,
            None,
        );
        let body = serde_json::to_vec(&PeerChatRelayRequest { envelope }).unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let retry = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(retry.status(), StatusCode::OK);
        let response: PeerChatRelayResponse = serde_json::from_slice(
            &to_bytes(response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        let retry: PeerChatRelayResponse = serde_json::from_slice(
            &to_bytes(retry.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        let privacy_safe_acceptance = durable_peer_acceptance_response();
        assert_eq!(response, privacy_safe_acceptance);
        assert_eq!(retry, privacy_safe_acceptance);
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("pending message should be readable");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        let status = relay.peer_status();
        assert_eq!(status.inbound_accepted_total, 2);
        assert_eq!(status.inbound_duplicate_total, 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn peer_relay_v2_rejects_tampering_before_durable_storage() {
        // [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] A forged previous-hop
        // claim must not reach the durable queue, while the exact signed
        // request remains accepted through the same relay processing path.
        let (relay, path) = temp_chat_relay_with_rates(
            "chat-peer-auth-v2",
            DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE,
            1,
        );
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let previous_hop = IdentityKeyPair::generate();
        let envelope = signed_envelope();
        let receiver = envelope.receiver;
        let request = PeerChatRelayRequestV2::sign(envelope, &previous_hop).unwrap();
        let mut tampered = request.clone();
        tampered.envelope.ciphertext[0] ^= 0x01;

        let target_identity = Arc::new(IdentityKeyPair::generate());
        let target_node_id = target_identity.public_key_bytes();
        let app = build_chat_peer_router(
            Some(Arc::clone(&relay)),
            sessions,
            udp,
            Arc::new(PeerStore::new()),
            Arc::clone(&target_identity),
            Arc::new(reqwest::Client::new()),
            None,
        );
        let tampered_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay-v2")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&tampered).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(tampered_response.status(), StatusCode::UNAUTHORIZED);
        assert!(relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .unwrap()
            .0
            .is_empty());
        let rejected_status = relay.peer_status();
        assert_eq!(rejected_status.inbound_rejected_total, 1);
        assert_eq!(
            rejected_status.last_inbound_failure_reason.as_deref(),
            Some("peer_auth_invalid")
        );

        let accepted_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay-v2")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(accepted_response.status(), StatusCode::OK);
        let accepted_response: PeerChatRelayResponseV2 = serde_json::from_slice(
            &to_bytes(accepted_response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(accepted_response.relay, durable_peer_acceptance_response());
        accepted_response
            .receipt
            .as_ref()
            .expect("authenticated direct relay should return signed custody evidence")
            .verify_expected(&request, &target_node_id, now_secs())
            .expect("receipt should bind the exact request to the target node");
        assert_eq!(
            relay
                .pull_pending(&receiver, 0, &[0u8; 16], 10)
                .unwrap()
                .0
                .len(),
            1
        );
        let replayed_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay-v2")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replayed_response.status(), StatusCode::OK);
        let replayed_response: PeerChatRelayResponseV2 = serde_json::from_slice(
            &to_bytes(replayed_response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(replayed_response, accepted_response);
        let status = relay.peer_status();
        assert_eq!(status.inbound_rejected_total, 1);
        assert_eq!(status.inbound_accepted_total, 2);
        assert_eq!(status.inbound_duplicate_total, 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn peer_relay_v3_rejects_request_signed_for_another_target() {
        // [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] A request that
        // is fully valid for target A must not become authenticated work for
        // target B. The rejected attempt also must not consume B's per-node
        // authenticated quota or reach its durable pending store.
        let (relay, path) = temp_chat_relay_with_rates(
            "chat-peer-target-binding-v3",
            DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE,
            1,
        );
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let previous_hop = IdentityKeyPair::generate();
        let target_identity = Arc::new(IdentityKeyPair::generate());
        let target_node_id = target_identity.public_key_bytes();
        let other_target_node_id = IdentityKeyPair::generate().public_key_bytes();
        let envelope = signed_envelope();
        let receiver = envelope.receiver;
        let wrong_target_request =
            PeerChatRelayRequestV3::sign(envelope.clone(), other_target_node_id, &previous_hop)
                .unwrap();
        assert!(wrong_target_request.verify_for_target(&other_target_node_id));

        let app = build_chat_peer_router(
            Some(Arc::clone(&relay)),
            sessions,
            udp,
            Arc::new(PeerStore::new()),
            Arc::clone(&target_identity),
            Arc::new(reqwest::Client::new()),
            None,
        );
        let rejected = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay-v3")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&wrong_target_request).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
        assert!(relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .unwrap()
            .0
            .is_empty());

        let accepted_request =
            PeerChatRelayRequestV3::sign(envelope, target_node_id, &previous_hop).unwrap();
        let expected_commitment = accepted_request.request_commitment().unwrap();
        let accepted = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay-v3")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&accepted_request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let accepted: PeerChatRelayResponseV2 = serde_json::from_slice(
            &to_bytes(accepted.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        accepted
            .receipt
            .as_ref()
            .expect("target-bound durable acceptance should be signed")
            .verify_expected_commitment(&expected_commitment, &target_node_id, now_secs())
            .expect("receipt should bind the exact v3 request to the target");
        assert_eq!(
            relay
                .pull_pending(&receiver, 0, &[0u8; 16], 10)
                .unwrap()
                .0
                .len(),
            1
        );

        // [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Simulate an
        // ACK lost after durable custody. The byte-identical retry bypasses
        // the already-consumed per-node quota and returns the exact signed ACK
        // without inserting or delivering the encrypted envelope twice.
        let replayed = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay-v3")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&accepted_request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replayed.status(), StatusCode::OK);
        let replayed: PeerChatRelayResponseV2 = serde_json::from_slice(
            &to_bytes(replayed.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(replayed, accepted);
        assert_eq!(
            relay
                .pull_pending(&receiver, 0, &[0u8; 16], 10)
                .unwrap()
                .0
                .len(),
            1
        );
        let status = relay.peer_status();
        assert_eq!(status.inbound_rejected_total, 1);
        assert_eq!(status.inbound_accepted_total, 2);
        assert_eq!(status.inbound_duplicate_total, 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn peer_relay_rate_limit_rejects_before_duplicate_processing() {
        let (relay, path) = temp_chat_relay_with_peer_rate("chat-peer-rate-limit", 1);
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let envelope = signed_envelope();
        let body = serde_json::to_vec(&PeerChatRelayRequest { envelope }).unwrap();
        let app = build_chat_peer_router(
            Some(Arc::clone(&relay)),
            sessions,
            udp,
            Arc::new(PeerStore::new()),
            Arc::new(IdentityKeyPair::generate()),
            Arc::new(reqwest::Client::new()),
            None,
        );

        let first = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let status = relay.peer_status();
        assert_eq!(status.inbound_accepted_total, 1);
        assert_eq!(status.inbound_duplicate_total, 0);
        assert_eq!(status.inbound_rejected_total, 1);
        assert_eq!(
            status.last_inbound_failure_reason.as_deref(),
            Some("rate_limited")
        );

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_global_rate_limit_rejects_before_json_deserialization() {
        let (relay, path) = temp_chat_relay_with_peer_rate("blind-relay-global-rate-limit", 1);
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let app = build_chat_peer_router(
            Some(relay),
            sessions,
            udp,
            Arc::clone(&peer_store),
            Arc::new(IdentityKeyPair::generate()),
            Arc::new(reqwest::Client::new()),
            None,
        );

        // [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] Malformed JSON
        // proves the second attempt is rejected by middleware before Axum can
        // parse an identity, route, next hop, or opaque encrypted body.
        let request = || {
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .body(Body::from("not-json"))
                .unwrap()
        };
        let first = app.clone().oneshot(request()).await.unwrap();
        let second = app.oneshot(request()).await.unwrap();

        assert_ne!(first.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = axum::body::to_bytes(second.into_body(), usize::MAX)
            .await
            .unwrap();
        let response: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(response.reason.as_deref(), Some("rate_limited"));
        let stats = peer_store.status(now_secs()).runtime.blind_relay;
        assert_eq!(stats.received, 1);
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.rate_limited, 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn peer_routes_reject_oversized_bodies_before_json_deserialization() {
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let app = build_chat_peer_router(
            None,
            sessions,
            udp,
            Arc::clone(&peer_store),
            Arc::new(IdentityKeyPair::generate()),
            Arc::new(reqwest::Client::new()),
            None,
        );

        let peer_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(vec![b' '; PEER_CHAT_REQUEST_BODY_MAX_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        let blind_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(vec![
                        b' ';
                        PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES + 1
                    ]))
                    .unwrap(),
            )
            .await
            .unwrap();
        let declared_blind_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .header(
                        axum::http::header::CONTENT_LENGTH,
                        (PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES + 1).to_string(),
                    )
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(peer_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(blind_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            declared_blind_response.status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        let blind_stats = peer_store.status(now_secs()).runtime.blind_relay;
        assert_eq!(blind_stats.received, 0, "oversized body reached handler");
        assert_eq!(blind_stats.rejected, 0, "oversized body reached handler");
    }

    #[tokio::test]
    async fn blind_relay_endpoint_terminal_accepts_opaque_blob_without_parsing() {
        let (relay, path) = temp_chat_relay("blind-relay-terminal-http");
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let http_client = Arc::new(reqwest::Client::new());
        let opaque_blob = br#"{"looks_like":"json","must_not_be_parsed":true}"#.to_vec();
        let now = now_secs();
        let envelope = BlindRelayEnvelope {
            route_id: [0x41u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: opaque_blob,
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let app = build_chat_peer_router(
            Some(relay),
            sessions,
            udp,
            Arc::clone(&peer_store),
            node_identity,
            http_client,
            None,
        );
        let body = serde_json::to_vec(&PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        })
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();

        assert!(parsed.accepted);
        assert!(parsed.terminal);
        assert!(!parsed.forwarded);
        assert_eq!(parsed.ttl_remaining, 2);
        let blind_stats = peer_store.status(now + 10).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 0);
        assert!(peer_store
            .recent_audit_events()
            .iter()
            .any(|event| event.action == "blind_relay_terminal"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_handler_signs_exact_failure_response() {
        let (relay, path) = temp_chat_relay("blind-relay-signed-failure-http");
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let node_id = node_identity.public_key_bytes();
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let now = now_secs();
        let request = PeerBlindRelayRequest {
            envelope: BlindRelayEnvelope {
                route_id: [0x42u8; 16],
                next_hop: node_id,
                ttl: 2,
                encrypted_blob: b"opaque expired failure request".to_vec(),
                timestamp: now - BLIND_RELAY_MAX_ENVELOPE_AGE_SECS - 1,
                signature: [0u8; 64],
            }
            .sign_with(&previous_hop),
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let app = build_chat_peer_router(
            Some(relay),
            sessions,
            udp,
            Arc::new(PeerStore::new()),
            node_identity,
            Arc::new(reqwest::Client::new()),
            None,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.reason.as_deref(), Some("timestamp_expired"));
        assert!(parsed.failure_receipt.is_some());
        assert_eq!(
            validate_downstream_failure_receipt(&parsed, &request, &node_id, now_secs(), true),
            Ok(true)
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_handler_never_signs_unauthenticated_failure() {
        // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] An attacker may
        // choose both the ciphertext and claimed node id. Invalid work gets a
        // coarse retry/error bucket, but no node-authored receipt oracle.
        let (relay, path) = temp_chat_relay("blind-relay-unsigned-failure-http");
        let claimed_previous_hop = IdentityKeyPair::generate();
        let attacker = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let request = PeerBlindRelayRequest {
            envelope: BlindRelayEnvelope {
                route_id: [0x4au8; 16],
                next_hop: node_identity.public_key_bytes(),
                ttl: 2,
                encrypted_blob: b"opaque unauthenticated failure".to_vec(),
                timestamp: now_secs(),
                signature: [0u8; 64],
            }
            .sign_with(&attacker),
            previous_hop_node_id: claimed_previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let app = build_chat_peer_router(
            Some(relay),
            Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            Arc::new(PeerStore::new()),
            node_identity,
            Arc::new(reqwest::Client::new()),
            None,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.reason.as_deref(), Some("invalid_signature"));
        assert!(parsed.failure_receipt.is_none());
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_rejects_stale_timestamp_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let now = now_secs();
        let envelope = BlindRelayEnvelope {
            route_id: [0x42u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: br#"{"opaque":"old route frame"}"#.to_vec(),
            timestamp: now - BLIND_RELAY_MAX_ENVELOPE_AGE_SECS - 1,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::TimestampExpired)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "timestamp_expired"
        }));
    }

    #[tokio::test]
    async fn blind_relay_rejects_future_timestamp_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let now = now_secs();
        let envelope = BlindRelayEnvelope {
            route_id: [0x43u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: br#"{"opaque":"future route frame"}"#.to_vec(),
            timestamp: now + BLIND_RELAY_MAX_FUTURE_SKEW_SECS + 1,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::TimestampInFuture)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "timestamp_in_future"
        }));
    }

    #[tokio::test]
    async fn onion_terminal_layer_is_peeled_and_delivered() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let (relay, path) = temp_chat_relay("onion-terminal");
        let state = ChatPeerState {
            chat_relay: Some(Arc::clone(&relay)),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let now = now_secs();

        // Single-hop onion addressed to this node; inner payload is a ChatEnvelope.
        let delivered_envelope = signed_envelope();
        let receiver = delivered_envelope.receiver;
        let inner = encode_envelope(&delivered_envelope).unwrap();
        let hop = OnionHop {
            node_id: node_identity.public_key_bytes(),
            // Build to the node's CURRENT rotating onion key — what the handler
            // peels with (not the identity-derived key).
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let envelope = build_onion_envelope(&[hop], &inner, [0x55u8; 16], 4, now, &source).unwrap();
        assert!(is_onion_blob(&envelope.encrypted_blob));

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        assert!(result.terminal);
        assert!(!result.forwarded);
        assert_eq!(result.reason.as_deref(), Some("onion_terminal_delivered"));
        result
            .delivery_receipt
            .as_ref()
            .expect("terminal onion delivery must return a signed receipt")
            .verify_expected_for_purpose(
                &[0x55u8; 16],
                &inner,
                OnionRoutePurpose::MessageRelay,
                &node_identity.public_key_bytes(),
            )
            .expect("receipt must bind the exact terminal payload, purpose, and node");
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.rejected, 0);
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("terminal onion delivery should enter pending relay queue");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn onion_forward_reconstruction_is_byte_stable() {
        // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] A middle hop that
        // crashes after sending must reconstruct exactly the same signed frame;
        // otherwise downstream durable replay sees a conflicting request.
        let previous_hop = IdentityKeyPair::generate();
        let middle = IdentityKeyPair::generate();
        let outer = BlindRelayEnvelope {
            route_id: [0x61; 16],
            next_hop: middle.public_key_bytes(),
            ttl: 4,
            encrypted_blob: b"opaque outer onion layer".to_vec(),
            timestamp: 1_800_000_123,
            signature: [0; 64],
        }
        .sign_with(&previous_hop);
        let next_hop = IdentityKeyPair::generate().public_key_bytes();
        let inner = b"opaque inner onion layer".to_vec();

        let first = build_forwarded_onion_envelope(&outer, next_hop, inner.clone(), &middle);
        let after_restart = build_forwarded_onion_envelope(&outer, next_hop, inner, &middle);

        assert_eq!(after_restart, first);
        assert_eq!(first.timestamp, outer.timestamp);
        assert_eq!(first.ttl, outer.ttl - 1);
    }

    #[tokio::test]
    async fn onion_terminal_armed_claim_recovers_without_duplicate_storage() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Model a crash after
        // terminal custody succeeds but before the route ACK is sealed. Exact
        // retry takes over the armed claim and reuses idempotent store_pending.
        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let (old_relay, path) = temp_chat_relay("onion-terminal-armed-recovery");
        let now = now_secs();
        let delivered_envelope = signed_envelope_at(now);
        let receiver = delivered_envelope.receiver;
        let inner = encode_envelope(&delivered_envelope).expect("encode terminal payload");
        let route_id = [0x62; 16];
        let request = PeerBlindRelayRequest {
            envelope: build_onion_envelope(
                &[OnionHop {
                    node_id: node_identity.public_key_bytes(),
                    kem_pub: crate::services::onion_keys::current_public_key(),
                }],
                &inner,
                route_id,
                4,
                now,
                &source,
            )
            .expect("build recoverable terminal onion"),
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let request_commitment = blind_relay_authenticated_request_commitment(&request)
            .expect("commit recoverable request");
        assert_eq!(
            old_relay
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("reserve pre-crash route"),
            BlindRelayRouteAdmission::Reserved
        );
        old_relay
            .arm_blind_relay_route_effect(&route_id, &request_commitment, now)
            .expect("arm pre-crash route");
        old_relay
            .store_pending(&delivered_envelope)
            .expect("complete terminal custody before crash");
        drop(old_relay);

        let aged_at = now
            .saturating_sub(crate::services::chat_relay::BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1);
        Connection::open(&path)
            .expect("open crashed relay database")
            .execute(
                "UPDATE relay_blind_route_reservations
                 SET reserved_at = ?1, owner_acquired_at = ?1",
                [i64::try_from(aged_at).expect("fit test timestamp")],
            )
            .expect("age crashed owner lease");
        let recovered_relay = Arc::new(
            ChatRelayService::new(
                test_chat_config(path.to_string_lossy().into_owned()),
                [7u8; 32],
            )
            .expect("restart relay for armed reconciliation"),
        );
        let state = ChatPeerState {
            chat_relay: Some(Arc::clone(&recovered_relay)),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };

        let recovered = process_peer_blind_relay(state, request)
            .await
            .expect("reconcile armed terminal route");
        assert!(recovered.terminal);
        assert_eq!(
            recovered.reason.as_deref(),
            Some("onion_terminal_delivered")
        );
        let (messages, has_more) = recovered_relay
            .pull_pending(&receiver, 0, &[0; 16], 10)
            .expect("pull reconciled terminal custody");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);
        assert_eq!(
            encode_envelope(&messages[0].envelope).expect("encode recovered message"),
            encode_envelope(&delivered_envelope).expect("encode expected message"),
        );

        drop(recovered_relay);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn onion_middle_armed_claim_recovers_through_terminal_durable_replay() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        // [MIDDLE-HOP-ARMED-RECOVERY 2026-08-25 by Codex] Exercise the real
        // HTTP boundary on both sides of a crashed middle hop. The terminal
        // has already accepted durable custody, but the middle has not sealed
        // its upstream ACK. Restart recovery must reconstruct byte-identical
        // downstream work, receive the terminal's durable replay, and avoid a
        // second pending message.
        let source = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = Arc::new(IdentityKeyPair::generate());
        let now = now_secs();
        let delivered_envelope = signed_envelope_at(now);
        let receiver = delivered_envelope.receiver;
        let terminal_payload =
            encode_envelope(&delivered_envelope).expect("encode terminal message payload");

        let (terminal_relay, terminal_path) = temp_chat_relay("onion-terminal-replay-target");
        let terminal_peer_store = Arc::new(PeerStore::new());
        let terminal_app = build_chat_peer_router(
            Some(Arc::clone(&terminal_relay)),
            Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            Arc::clone(&terminal_peer_store),
            Arc::clone(&terminal_identity),
            Arc::new(reqwest::Client::new()),
            None,
        );
        let terminal_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", terminal_listener.local_addr().unwrap());
        let terminal_server = tokio::spawn(async move {
            axum::serve(terminal_listener, terminal_app).await.unwrap();
        });

        let terminal_descriptor = signed_chat_relay_peer_descriptor_for(
            terminal_identity.as_ref(),
            terminal_endpoint.clone(),
            now,
            now + 300,
        );
        let middle_peer_store = Arc::new(PeerStore::new());
        middle_peer_store
            .upsert_verified_from_source(terminal_descriptor.clone(), now, "gossip_snapshot")
            .expect("install terminal descriptor at middle hop");

        let (old_middle_relay, middle_path) = temp_chat_relay("onion-middle-armed-recovery");
        let old_middle_state = ChatPeerState {
            chat_relay: Some(Arc::clone(&old_middle_relay)),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&middle_peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let route_id = [0x69; 16];
        let request = PeerBlindRelayRequest {
            envelope: build_onion_envelope(
                &[
                    OnionHop {
                        node_id: middle_identity.public_key_bytes(),
                        kem_pub: crate::services::onion_keys::current_public_key(),
                    },
                    OnionHop {
                        node_id: terminal_identity.public_key_bytes(),
                        kem_pub: crate::services::onion_keys::current_public_key(),
                    },
                ],
                &terminal_payload,
                route_id,
                4,
                now,
                &source,
            )
            .expect("build recoverable two-hop onion"),
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let request_commitment = blind_relay_authenticated_request_commitment(&request)
            .expect("commit recoverable middle-hop request");
        assert_eq!(
            old_middle_relay
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("reserve pre-crash middle route"),
            BlindRelayRouteAdmission::Reserved
        );
        old_middle_relay
            .arm_blind_relay_route_effect(&route_id, &request_commitment, now)
            .expect("arm pre-crash middle route");

        let peel = try_open_onion_layer(
            &request.envelope.encrypted_blob,
            &crate::services::onion_keys::peel_secrets(now),
        )
        .expect("peel pre-crash middle layer");
        assert_eq!(peel.next_hop, Some(terminal_identity.public_key_bytes()));
        let forwarded_envelope = build_forwarded_onion_envelope(
            &request.envelope,
            terminal_identity.public_key_bytes(),
            peel.inner,
            middle_identity.as_ref(),
        );
        let terminal_url = blind_peer_relay_url(&terminal_endpoint)
            .expect("construct canonical terminal relay URL");
        let first_forwarded_request = PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: middle_identity.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let first_terminal_ack = forward_blind_relay_with_retry(
            &old_middle_state,
            &terminal_url,
            &terminal_descriptor,
            first_forwarded_request.clone(),
            now,
        )
        .await
        .expect("terminal accepts custody before middle crash");
        assert!(first_terminal_ack.response.terminal);
        assert!(first_terminal_ack.response.delivery_receipt.is_some());
        let terminal_request_commitment =
            blind_relay_authenticated_request_commitment(&first_forwarded_request)
                .expect("commit terminal replay request");
        let sealed_terminal_response = match terminal_relay
            .reserve_blind_relay_route(&route_id, &terminal_request_commitment)
            .expect("read terminal sealed route")
        {
            BlindRelayRouteAdmission::Completed { response, .. } => response,
            admission => panic!("terminal route was not sealed: {admission:?}"),
        };
        let sealed_terminal_response =
            decode_durable_blind_relay_response(&sealed_terminal_response)
                .expect("decode terminal sealed response");
        validate_completed_blind_relay_response(&sealed_terminal_response)
            .expect("validate terminal sealed response");
        assert_eq!(sealed_terminal_response, first_terminal_ack.response);

        drop(old_middle_state);
        drop(old_middle_relay);
        let aged_at = now
            .saturating_sub(crate::services::chat_relay::BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1);
        Connection::open(&middle_path)
            .expect("open crashed middle database")
            .execute(
                "UPDATE relay_blind_route_reservations
                 SET reserved_at = ?1, owner_acquired_at = ?1",
                [i64::try_from(aged_at).expect("fit middle lease timestamp")],
            )
            .expect("age crashed middle owner lease");

        let recovered_middle_relay = Arc::new(
            ChatRelayService::new(
                test_chat_config(middle_path.to_string_lossy().into_owned()),
                [7u8; 32],
            )
            .expect("restart middle relay for armed reconciliation"),
        );
        let recovered_middle_state = ChatPeerState {
            chat_relay: Some(Arc::clone(&recovered_middle_relay)),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&middle_peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let recovered_peel = try_open_onion_layer(
            &request.envelope.encrypted_blob,
            &crate::services::onion_keys::peel_secrets(now),
        )
        .expect("peel recovered middle layer");
        let recovered_forwarded_request = PeerBlindRelayRequest {
            envelope: build_forwarded_onion_envelope(
                &request.envelope,
                terminal_identity.public_key_bytes(),
                recovered_peel.inner,
                middle_identity.as_ref(),
            ),
            previous_hop_node_id: middle_identity.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        assert_eq!(
            serde_json::to_vec(&recovered_forwarded_request)
                .expect("encode recovered downstream request"),
            serde_json::to_vec(&first_forwarded_request)
                .expect("encode original downstream request"),
            "middle restart changed authenticated downstream request bytes"
        );
        let recovered = process_peer_blind_relay(recovered_middle_state, request)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "recover middle route through terminal durable replay: {error:?}; middle={:?}; terminal={:?}; terminal_events={:?}",
                    middle_peer_store.status(now + 1).runtime.blind_relay,
                    terminal_peer_store.status(now + 1).runtime.blind_relay,
                    terminal_peer_store.recent_audit_events(),
                )
            });
        assert!(recovered.accepted);
        assert!(recovered.forwarded);
        assert!(!recovered.terminal);
        assert_eq!(recovered.reason.as_deref(), Some("onion_forwarded"));
        assert_eq!(
            recovered.delivery_receipt,
            first_terminal_ack.response.delivery_receipt
        );

        let (messages, has_more) = terminal_relay
            .pull_pending(&receiver, 0, &[0; 16], 10)
            .expect("pull terminal custody after middle recovery");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);
        let terminal_stats = terminal_peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(terminal_stats.terminal, 1);
        assert_eq!(terminal_stats.replay_dropped, 1);
        let recovery_status = recovered_middle_relay.peer_status().blind_route_recovery;
        assert_eq!(recovery_status.attempted_total, 1);
        assert_eq!(recovery_status.completed_total, 1);
        assert_eq!(recovery_status.deferred_total, 0);
        assert_eq!(recovery_status.last_outcome.as_deref(), Some("completed"));

        terminal_server.abort();
        let _ = terminal_server.await;
        drop(recovered_middle_relay);
        drop(terminal_relay);
        let _ = std::fs::remove_file(middle_path);
        let _ = std::fs::remove_file(terminal_path);
    }

    #[tokio::test]
    async fn onion_terminal_rejects_same_message_id_with_different_ciphertext() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let chat_sender = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let (relay, path) = temp_chat_relay("onion-terminal-id-conflict");
        let state = ChatPeerState {
            chat_relay: Some(Arc::clone(&relay)),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let now = now_secs();
        let receiver = [0xA2; 32];
        let message_id = [0xA3; 16];
        let make_chat_envelope = |ciphertext: &[u8]| {
            let mut envelope = ChatEnvelope {
                message_id,
                sender: chat_sender.public_key_bytes(),
                receiver,
                timestamp: now,
                ciphertext: ciphertext.to_vec(),
                nonce: [0xA4; 24],
                content_type: ChatContentType::Text,
                signature: [0u8; 64],
            };
            envelope.signature = chat_sender.sign(&envelope.sign_data());
            envelope
        };
        let hop = OnionHop {
            node_id: node_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let make_request = |route_id, chat_envelope: &ChatEnvelope| {
            let payload = encode_envelope(chat_envelope).expect("encode terminal chat envelope");
            PeerBlindRelayRequest {
                envelope: build_onion_envelope(
                    std::slice::from_ref(&hop),
                    &payload,
                    route_id,
                    4,
                    now,
                    &source,
                )
                .expect("build terminal onion envelope"),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            }
        };

        let original = make_chat_envelope(b"first opaque ciphertext");
        let first = process_peer_blind_relay(state.clone(), make_request([0xA5; 16], &original))
            .await
            .expect("first terminal envelope should be durably accepted");
        assert!(first.delivery_receipt.is_some());

        // [DURABLE-RECEIPT-BOUNDARY 2026-08-15 by Codex] A fresh route can
        // legitimately retry the same exact envelope, but reusing its message
        // ID for different signed bytes must never produce a terminal receipt.
        let conflict = make_chat_envelope(b"different opaque ciphertext");
        let rejected = process_peer_blind_relay(state, make_request([0xA6; 16], &conflict)).await;
        assert!(matches!(rejected, Err(BlindRelayError::ForwardFailed)));

        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("original durable envelope should remain readable");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(
            encode_envelope(&messages[0].envelope).expect("re-encode stored envelope"),
            encode_envelope(&original).expect("re-encode original envelope")
        );
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.rejected, 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn onion_terminal_persists_anonymous_blind_vault_put_idempotently() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let now = now_secs();
        let now_ms = now.saturating_mul(1_000);
        let (_directory, vault, put) = temp_blind_vault_with_put(node_identity.as_ref(), now_ms);
        let encoded_put =
            encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode vault put");
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: Some(Arc::clone(&vault)),
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };

        let make_envelope = |route_id| {
            build_onion_envelope(
                &[OnionHop {
                    node_id: node_identity.public_key_bytes(),
                    kem_pub: crate::services::onion_keys::current_public_key(),
                }],
                &encoded_put,
                route_id,
                4,
                now,
                &source,
            )
            .expect("build vault terminal onion")
        };
        let first_route = [0x67; 16];
        let result = process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: make_envelope(first_route),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("signed anonymous put should reach Blind Vault");

        assert!(result.accepted);
        assert!(result.terminal);
        assert!(!result.forwarded);
        assert_eq!(result.reason.as_deref(), Some("onion_terminal_delivered"));
        result
            .delivery_receipt
            .as_ref()
            .expect("vault acceptance must return a route-safe terminal receipt")
            .verify_expected_for_purpose(
                &first_route,
                &encoded_put,
                OnionRoutePurpose::BlindVaultPut,
                &node_identity.public_key_bytes(),
            )
            .expect("receipt must bind exact encoded put and purpose without exposing metadata");
        let public_ack = serde_json::to_string(&result).expect("serialize terminal ACK");
        assert!(!public_ack.contains("lease_id"));
        assert!(!public_ack.contains("object_id"));
        assert!(!public_ack.contains("ciphertext"));

        let status = vault.status(now_ms + 1).expect("vault status after put");
        assert_eq!(status.live_objects, 1);
        assert_eq!(status.live_ciphertext_bytes, 4 * 1024);

        // A source may rebuild a route after losing the first ACK. Blind Vault
        // handles the exact Put idempotently even when the relay route differs.
        process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: make_envelope([0x68; 16]),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("same immutable put through a fresh route should be idempotent");
        assert_eq!(
            vault
                .status(now_ms + 2)
                .expect("vault status after retry")
                .live_objects,
            1
        );

        assert!(matches!(
            prepare_onion_terminal_payload(&state, b"ANBV", now).await,
            Err(BlindRelayError::OnionTerminalPayloadRejected)
        ));
    }

    #[tokio::test]
    async fn onion_terminal_requires_chat_relay_delivery_before_ack() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let replay_registry: Arc<dyn BlindRelayReplayRegistry> =
            Arc::new(BlindRelayReplayDomain::default());
        let abuse_guard: Arc<dyn BlindRelayAbusePolicy> =
            Arc::new(BlindRelayAbuseDomain::default());
        let failed_state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::clone(&replay_registry),
            blind_relay_abuse_guard: Arc::clone(&abuse_guard),
        };
        let now = now_secs();

        let delivered_envelope = signed_envelope();
        let receiver = delivered_envelope.receiver;
        let inner = encode_envelope(&delivered_envelope).unwrap();
        let hop = OnionHop {
            node_id: node_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let envelope = build_onion_envelope(&[hop], &inner, [0x56u8; 16], 4, now, &source).unwrap();

        let result = process_peer_blind_relay(
            failed_state,
            PeerBlindRelayRequest {
                envelope: envelope.clone(),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.rejected, 1);

        let (relay, path) = temp_chat_relay("onion-terminal-retry-after-relay-failure");
        let retry_state = ChatPeerState {
            chat_relay: Some(Arc::clone(&relay)),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::clone(&replay_registry),
            blind_relay_abuse_guard: Arc::clone(&abuse_guard),
        };

        // A terminal delivery failure must release the route id from the
        // replay cache. Otherwise a transient ChatRelay outage would make the
        // sender's retry look like a duplicate replay and permanently strand
        // the E2E-encrypted message.
        let retry = process_peer_blind_relay(
            retry_state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("terminal delivery retry should not be blocked by replay cache");

        assert!(retry.terminal);
        assert_eq!(retry.reason.as_deref(), Some("onion_terminal_delivered"));
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("retry should store the terminal onion payload");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn onion_layer_with_wrong_node_key_is_rejected() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let wrong_target = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        let (relay, path) = temp_chat_relay("onion-wrong-node-key-retry");
        let state = ChatPeerState {
            chat_relay: Some(relay),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let now = now_secs();

        // Layer is sealed to a different node's KEM key, but addressed (next_hop)
        // to this node — peel must fail without leaking anything.
        let inner = encode_envelope(&signed_envelope()).unwrap();
        let sealed_for_wrong = OnionHop {
            node_id: node_identity.public_key_bytes(),
            kem_pub: wrong_target.x25519_public_key_bytes(),
        };
        let envelope =
            build_onion_envelope(&[sealed_for_wrong], &inner, [0x56u8; 16], 4, now, &source)
                .unwrap();

        // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Peeling is a
        // pure preflight step. Its failure must release the durable unarmed
        // claim so an identical retry is classified by payload validation,
        // never stranded as an in-flight side effect.
        for _ in 0..2 {
            let result = process_peer_blind_relay(
                state.clone(),
                PeerBlindRelayRequest {
                    envelope: envelope.clone(),
                    previous_hop_node_id: source.public_key_bytes(),
                    onward_envelope: None,
                    onward_descriptor_hint: None,
                },
            )
            .await;

            assert!(matches!(result, Err(BlindRelayError::OnionPeelFailed)));
        }
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.rejected, 2);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn onion_middle_allows_fresh_signed_peer_before_routeability_probe() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, is_onion_blob, OnionHop};

        let terminal_requests: Arc<Mutex<Vec<PeerBlindRelayRequest>>> =
            Arc::new(Mutex::new(Vec::new()));
        let terminal_requests_for_route = Arc::clone(&terminal_requests);
        let terminal_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let terminal_requests_for_request = Arc::clone(&terminal_requests_for_route);
                async move {
                    terminal_requests_for_request.lock().unwrap().push(request);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 2,
                        reason: Some("terminal_next_hop".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, terminal_app).await.unwrap();
        });

        let now = now_secs();
        let source = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = IdentityKeyPair::generate();
        let terminal_node_id = terminal_identity.public_key_bytes();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    &terminal_identity,
                    terminal_endpoint,
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };

        // Build a true two-layer onion. The middle hop can peel only the outer
        // layer and must forward the remaining opaque onion blob without knowing
        // the final payload or the terminal's user-level receiver.
        let inner = encode_envelope(&signed_envelope()).unwrap();
        let middle_hop = OnionHop {
            node_id: middle_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let terminal_hop = OnionHop {
            node_id: terminal_node_id,
            kem_pub: terminal_identity.x25519_public_key_bytes(),
        };
        let envelope = build_onion_envelope(
            &[middle_hop, terminal_hop],
            &inner,
            [0x66u8; 16],
            4,
            now,
            &source,
        )
        .unwrap();

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(response.reason.as_deref(), Some("onion_forwarded"));

        let terminal_requests = terminal_requests.lock().unwrap();
        assert_eq!(terminal_requests.len(), 1);
        let terminal_request = &terminal_requests[0];
        assert_eq!(
            terminal_request.previous_hop_node_id,
            middle_identity.public_key_bytes()
        );
        assert_eq!(terminal_request.envelope.next_hop, terminal_node_id);
        assert!(is_onion_blob(&terminal_request.envelope.encrypted_blob));
        assert!(terminal_request.onward_envelope.is_none());

        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&terminal_node_id[..4]))
            .expect("terminal route row should remain visible");
        assert!(route_row.routeability_ready);
        assert_eq!(route_row.routeability_state, "reachable");
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);
    }

    #[tokio::test]
    async fn two_hop_onion_relay_delivers_real_ciphertext_payload_to_terminal_store() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, open_onion_layer, OnionHop};

        let now = now_secs();
        let source = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = IdentityKeyPair::generate();
        let terminal_receipt_identity = terminal_identity.clone();
        let terminal_node_id = terminal_identity.public_key_bytes();
        let terminal_secret = terminal_identity.to_x25519().0;

        let chat_sender = IdentityKeyPair::generate();
        let route_id = [0x7au8; 16];
        let receiver = [0x8bu8; 32];
        let mut delivered_envelope = ChatEnvelope {
            message_id: route_id,
            sender: chat_sender.public_key_bytes(),
            receiver,
            timestamp: now,
            ciphertext: b"real e2e ciphertext payload carried through two hops".to_vec(),
            nonce: [0x9cu8; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        delivered_envelope.signature = chat_sender.sign(&delivered_envelope.sign_data());
        let encoded_chat = encode_envelope(&delivered_envelope).unwrap();

        let (terminal_relay, terminal_db_path) = temp_chat_relay("two-hop-onion-terminal-store");
        let terminal_relay_for_route = Arc::clone(&terminal_relay);
        let terminal_previous_hops: Arc<Mutex<Vec<[u8; 32]>>> = Arc::new(Mutex::new(Vec::new()));
        let terminal_previous_hops_for_route = Arc::clone(&terminal_previous_hops);

        let terminal_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let terminal_relay_for_request = Arc::clone(&terminal_relay_for_route);
                let terminal_receipt_identity = terminal_receipt_identity.clone();
                let terminal_previous_hops_for_request =
                    Arc::clone(&terminal_previous_hops_for_route);
                async move {
                    if validate_blind_relay_envelope(
                        &request.envelope,
                        &request.previous_hop_node_id,
                        now_secs(),
                    )
                    .is_err()
                    {
                        return StatusCode::BAD_REQUEST.into_response();
                    }

                    let peel = match open_onion_layer(
                        &request.envelope.encrypted_blob,
                        &terminal_secret,
                    ) {
                        Ok(peel) => peel,
                        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                    };
                    if peel.next_hop.is_some() {
                        return StatusCode::BAD_REQUEST.into_response();
                    }

                    let inner = match decode_envelope(&peel.inner) {
                        Ok(envelope) => envelope,
                        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                    };
                    if validate_peer_envelope(&inner, now_secs()).is_err() {
                        return StatusCode::BAD_REQUEST.into_response();
                    }
                    if terminal_relay_for_request.store_pending(&inner).is_err() {
                        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                    }
                    let delivery_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
                        request.envelope.route_id,
                        &peel.inner,
                        OnionRoutePurpose::MessageRelay,
                        now_secs(),
                        &terminal_receipt_identity,
                    );
                    terminal_previous_hops_for_request
                        .lock()
                        .unwrap()
                        .push(request.previous_hop_node_id);

                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: request.envelope.ttl,
                        reason: Some("onion_terminal_delivered".to_string()),
                        delivery_receipt: Some(delivery_receipt),
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, terminal_app).await.unwrap();
        });

        let mut terminal_descriptor = NodeDescriptor::new(
            terminal_node_id,
            now,
            now,
            now + 300,
            "test-terminal-onion-peer",
        )
        .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
        terminal_descriptor.public_endpoint = Some(terminal_endpoint);
        terminal_descriptor.capabilities = vec![NodeCapability::ChatRelay];
        terminal_descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        let terminal_descriptor =
            SignedNodeDescriptor::sign(terminal_descriptor, &terminal_identity).unwrap();

        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(terminal_descriptor, now, "gossip_snapshot")
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };

        let middle_hop = OnionHop {
            node_id: middle_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let terminal_hop = OnionHop {
            node_id: terminal_node_id,
            kem_pub: terminal_identity.x25519_public_key_bytes(),
        };
        let envelope = build_onion_envelope(
            &[middle_hop, terminal_hop],
            &encoded_chat,
            route_id,
            2,
            now,
            &source,
        )
        .unwrap();

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("middle hop should forward real onion payload to terminal");

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(response.ttl_remaining, 1);
        assert_eq!(response.reason.as_deref(), Some("onion_forwarded"));
        response
            .delivery_receipt
            .as_ref()
            .expect("middle hop must propagate the terminal receipt")
            .verify_expected_for_purpose(
                &route_id,
                &encoded_chat,
                OnionRoutePurpose::MessageRelay,
                &terminal_node_id,
            )
            .expect("propagated receipt must retain terminal purpose binding");

        let previous_hops = terminal_previous_hops.lock().unwrap();
        assert_eq!(
            previous_hops.as_slice(),
            &[middle_identity.public_key_bytes()]
        );
        drop(previous_hops);

        let (messages, has_more) = terminal_relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("terminal should store the delivered E2E envelope");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);
        assert_eq!(
            messages[0].envelope.ciphertext,
            delivered_envelope.ciphertext
        );
        assert_eq!(messages[0].envelope.nonce, delivered_envelope.nonce);
        assert_eq!(messages[0].envelope.sender, delivered_envelope.sender);
        assert_eq!(messages[0].envelope.receiver, delivered_envelope.receiver);

        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);

        let _ = std::fs::remove_file(terminal_db_path);
    }

    #[tokio::test]
    async fn blind_relay_http_gate_requires_durable_replay_before_body_parse() {
        // [DURABLE-BLIND-RELAY-ADMISSION 2026-08-24 by Codex] Invalid JSON is
        // deliberate: a missing replay store must stop the request before the
        // extractor can parse or allocate for an attacker-controlled envelope.
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::new(IdentityKeyPair::generate()),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let app = Router::new()
            .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                peer_blind_relay_request_gate,
            ))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from("not-json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap();
        let rejection: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            rejection.reason.as_deref(),
            Some("replay_protection_unavailable")
        );
        let stats = peer_store.status(now_secs()).runtime.blind_relay;
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.terminal, 0);
        assert_eq!(stats.forwarded, 0);
        assert_eq!(
            rejection.delivery_receipt, None,
            "unavailable admission must not manufacture delivery evidence"
        );
        assert_eq!(
            rejection.failure_receipt, None,
            "unavailable admission must not sign failure evidence"
        );
    }

    #[tokio::test]
    async fn peer_request_in_flight_guard_enforces_backpressure_limit() {
        let (relay, path) = temp_chat_relay("blind-relay-backpressure");
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: Some(relay),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store,
            node_identity: Arc::new(IdentityKeyPair::generate()),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };

        assert!(InFlightRequestGuard::try_acquire(
            &state.blind_relay_in_flight,
            MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS,
        )
        .is_none());

        let app = Router::new()
            .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                peer_blind_relay_request_gate,
            ))
            .with_state(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from("not-json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_rejects_immediate_previous_hop_loop_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x44u8; 16],
            next_hop: previous_hop.public_key_bytes(),
            ttl: 2,
            encrypted_blob: br#"{"opaque":"must_not_be_parsed"}"#.to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::RouteLoop)));
        let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.loop_detected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "route_loop"
        }));
    }

    #[test]
    fn blind_relay_route_lease_releases_cancellation_and_commits_exact_ack() {
        // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Cancellation
        // releases an unarmed claim, but preserves an armed claim whose effect
        // may have happened. Completion still publishes the exact bounded ACK.
        let replay_registry: Arc<dyn BlindRelayReplayRegistry> =
            Arc::new(BlindRelayReplayDomain::default());
        let route_id = [0x48u8; 16];
        let request_commitment = [0xA8u8; 32];
        let started_at = 1_800_000_000;

        let first_generation =
            match replay_registry.observe(route_id, request_commitment, started_at) {
                BlindRelayRouteReplayDecision::New { generation } => generation,
                decision => panic!("unexpected replay decision: {decision:?}"),
            };
        drop(BlindRelayRouteLease::local(
            Arc::clone(&replay_registry),
            route_id,
            request_commitment,
            first_generation,
        ));
        let armed_generation =
            match replay_registry.observe(route_id, request_commitment, started_at + 1) {
                BlindRelayRouteReplayDecision::New { generation } => generation,
                decision => panic!("unexpected replay decision: {decision:?}"),
            };

        let mut armed_lease = BlindRelayRouteLease::local(
            Arc::clone(&replay_registry),
            route_id,
            request_commitment,
            armed_generation,
        );
        armed_lease.arm_effect(started_at + 2).unwrap();
        drop(armed_lease);
        assert_eq!(
            replay_registry.observe(route_id, request_commitment, started_at + 3),
            BlindRelayRouteReplayDecision::InFlight
        );
        assert_eq!(
            replay_registry.release(route_id, request_commitment, armed_generation),
            BlindRelayReplayMutation::Applied
        );
        let completion_generation =
            match replay_registry.observe(route_id, request_commitment, started_at + 4) {
                BlindRelayRouteReplayDecision::New { generation } => generation,
                decision => panic!("unexpected replay decision: {decision:?}"),
            };

        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("forwarded".to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        };
        BlindRelayRouteLease::local(
            Arc::clone(&replay_registry),
            route_id,
            request_commitment,
            completion_generation,
        )
        .complete(started_at + 5, response.clone())
        .unwrap();
        assert_eq!(
            replay_registry.observe(route_id, request_commitment, started_at + 6),
            BlindRelayRouteReplayDecision::Completed(Box::new(response))
        );
    }

    #[test]
    fn recovered_blind_route_lease_reports_deferred_without_route_dimensions() {
        // [BLIND-ROUTE-RECOVERY-STATUS 2026-08-25 by Codex] A cancelled
        // takeover remains durably armed and emits only one aggregate deferred
        // transition. No route, request, peer, endpoint, or reason is retained.
        let (old_relay, path) = temp_chat_relay("blind-route-recovery-deferred");
        let route_id = [0x49; 16];
        let request_commitment = [0xA9; 32];
        let now = now_secs();
        assert_eq!(
            old_relay
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("reserve old process route"),
            BlindRelayRouteAdmission::Reserved
        );
        old_relay
            .arm_blind_relay_route_effect(&route_id, &request_commitment, now)
            .expect("arm old process route");
        drop(old_relay);

        let aged_at = now
            .saturating_sub(crate::services::chat_relay::BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1);
        Connection::open(&path)
            .expect("open deferred recovery database")
            .execute(
                "UPDATE relay_blind_route_reservations
                 SET reserved_at = ?1, owner_acquired_at = ?1",
                [i64::try_from(aged_at).expect("fit deferred lease timestamp")],
            )
            .expect("age prior process lease");

        let recovered_relay = Arc::new(
            ChatRelayService::new(
                test_chat_config(path.to_string_lossy().into_owned()),
                [7; 32],
            )
            .expect("restart deferred recovery relay"),
        );
        assert_eq!(
            recovered_relay
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("take over armed route"),
            BlindRelayRouteAdmission::ReservedForRecovery
        );
        drop(BlindRelayRouteLease::durable(
            Arc::clone(&recovered_relay),
            route_id,
            request_commitment,
            true,
        ));

        let status = recovered_relay.peer_status().blind_route_recovery;
        assert_eq!(status.attempted_total, 1);
        assert_eq!(status.completed_total, 0);
        assert_eq!(status.deferred_total, 1);
        assert_eq!(status.last_outcome.as_deref(), Some("deferred"));
        assert!(status.last_event_at.is_some());

        drop(recovered_relay);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn forged_previous_hop_signatures_cannot_poison_node_quarantine() {
        let claimed_previous_hop = IdentityKeyPair::generate();
        let attacker = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let now = now_secs();
        let claimed_node_id = claimed_previous_hop.public_key_bytes();

        for attempt in 0..=BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD {
            let mut route_id = [0x91u8; 16];
            route_id[0] = u8::try_from(attempt).unwrap_or(u8::MAX);
            let forged = BlindRelayEnvelope {
                route_id,
                next_hop: node_identity.public_key_bytes(),
                ttl: 2,
                encrypted_blob: b"opaque forged-attribution candidate".to_vec(),
                timestamp: now,
                signature: [0u8; 64],
            }
            .sign_with(&attacker);

            assert!(matches!(
                process_peer_blind_relay(
                    state.clone(),
                    PeerBlindRelayRequest {
                        envelope: forged,
                        previous_hop_node_id: claimed_node_id,
                        onward_envelope: None,
                        onward_descriptor_hint: None,
                    },
                )
                .await,
                Err(BlindRelayError::InvalidSignature)
            ));
        }

        let decision = state
            .blind_relay_abuse_guard
            .observe_request(claimed_node_id, now);
        assert_eq!(decision, BlindRelayAbuseDecision::Allowed);

        let valid = BlindRelayEnvelope {
            route_id: [0xa2u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque authenticated previous-hop payload".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&claimed_previous_hop);
        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope: valid,
                previous_hop_node_id: claimed_node_id,
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("valid claimed previous hop must remain admissible");

        assert!(response.accepted);
        assert!(response.terminal);
        let blind_status = peer_store.status(now).runtime.blind_relay;
        assert_eq!(
            blind_status.rejected,
            u64::from(BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD) + 1
        );
        assert_eq!(blind_status.quarantine_started, 0);
    }

    #[tokio::test]
    async fn blind_relay_replays_completed_response_without_forwarding_again() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x45u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted replay candidate".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let first = process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: envelope.clone(),
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();
        let duplicate = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        assert!(first.terminal);
        assert_eq!(duplicate, first);
        let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 2);
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.replay_dropped, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "duplicate_route"
        }));
        assert!(!blind_relay_reason_counts_toward_quarantine(
            "duplicate_route"
        ));
    }

    #[tokio::test]
    async fn blind_relay_in_flight_duplicate_never_returns_false_acceptance() {
        // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] A concurrent retry must
        // remain retryable until the owner attempt publishes a durable result.
        // Returning an accepted replay here could lose the route if that owner
        // subsequently fails.
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let route_id = [0x47u8; 16];
        let envelope = BlindRelayEnvelope {
            route_id,
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque concurrent replay candidate".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);
        let request = PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let request_commitment = blind_relay_authenticated_request_commitment(&request).unwrap();
        let (relay, path) = temp_chat_relay("blind-relay-in-flight");
        assert_eq!(
            relay
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .unwrap(),
            BlindRelayRouteAdmission::Reserved
        );
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: Some(relay),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let error = process_peer_blind_relay(state, request).await.unwrap_err();

        assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(matches!(error, BlindRelayError::RouteInFlight));
        let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "route_in_flight"
        }));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_capacity_rejects_before_terminal_or_forward_effects() {
        // [BLIND-RELAY-NO-EVICTION-ADMISSION 2026-08-24 by Codex] Exercise the
        // real authenticated handler with a full replay map. The new route must
        // fail before terminal accounting, forwarding, or receipt creation.
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let now = now_secs();
        let request = PeerBlindRelayRequest {
            envelope: BlindRelayEnvelope {
                route_id: [0x4Au8; 16],
                next_hop: node_identity.public_key_bytes(),
                ttl: 2,
                encrypted_blob: b"opaque capacity-gated relay candidate".to_vec(),
                timestamp: now,
                signature: [0u8; 64],
            }
            .sign_with(&previous_hop),
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let request_commitment = blind_relay_authenticated_request_commitment(&request).unwrap();
        let replay_registry = BlindRelayReplayDomain::default();
        for sequence in 0..MAX_BLIND_RELAY_SEEN_ROUTES {
            let mut retained_route = [0x49u8; 16];
            retained_route[..8].copy_from_slice(&(sequence as u64).to_be_bytes());
            assert!(matches!(
                replay_registry.observe(retained_route, request_commitment, now),
                BlindRelayRouteReplayDecision::New { .. }
            ));
        }
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(replay_registry),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let error = process_peer_blind_relay(state, request).await.unwrap_err();

        assert!(matches!(error, BlindRelayError::ReplayCapacity));
        assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "replay_capacity"
        }));
    }

    #[tokio::test]
    async fn blind_relay_forward_retries_transient_next_hop_failure() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    let attempt = attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    if attempt == 0 {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    } else {
                        Json(PeerBlindRelayResponse {
                            accepted: true,
                            terminal: true,
                            forwarded: false,
                            ttl_remaining: 1,
                            reason: Some("terminal_next_hop".to_string()),
                            delivery_receipt: None,
                            success_receipt: None,
                            failure_receipt: None,
                            opaque_terminal_response_b64: None,
                        })
                        .into_response()
                    }
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x42u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);
        assert_eq!(blind_stats.retry_attempted, 1);
        assert_eq!(blind_stats.retry_succeeded, 1);
        assert_eq!(blind_stats.retry_exhausted, 0);
        assert!(peer_store
            .recent_audit_events()
            .iter()
            .any(|event| { event.action == "blind_relay_retry" && event.outcome == "accepted" }));
    }

    #[tokio::test]
    async fn blind_relay_middle_hop_forwards_onward_envelope_without_payload_inspection() {
        let terminal_requests: Arc<Mutex<Vec<PeerBlindRelayRequest>>> =
            Arc::new(Mutex::new(Vec::new()));
        let terminal_requests_for_route = Arc::clone(&terminal_requests);
        let terminal_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let terminal_requests_for_request = Arc::clone(&terminal_requests_for_route);
                async move {
                    terminal_requests_for_request.lock().unwrap().push(request);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 0,
                        reason: Some("terminal_next_hop".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, terminal_app).await.unwrap();
        });

        let now = now_secs();
        let entry_identity = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    &terminal_identity,
                    terminal_endpoint,
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&terminal_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let outer_envelope = BlindRelayEnvelope {
            route_id: [0x62u8; 16],
            next_hop: middle_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque middle-hop carrier; do not parse".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&entry_identity);
        let onward_envelope = BlindRelayEnvelope {
            route_id: [0x63u8; 16],
            next_hop: terminal_identity.public_key_bytes(),
            ttl: 1,
            encrypted_blob: b"opaque terminal relay blob; do not parse".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&entry_identity);

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope: outer_envelope,
                previous_hop_node_id: entry_identity.public_key_bytes(),
                onward_envelope: Some(onward_envelope),
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(response.reason.as_deref(), Some("onion_middle_forwarded"));

        let terminal_requests = terminal_requests.lock().unwrap();
        assert_eq!(terminal_requests.len(), 1);
        let terminal_request = &terminal_requests[0];
        assert_eq!(
            terminal_request.previous_hop_node_id,
            middle_identity.public_key_bytes()
        );
        assert_eq!(
            terminal_request.envelope.next_hop,
            terminal_identity.public_key_bytes()
        );
        assert_eq!(terminal_request.envelope.ttl, 0);
        assert!(terminal_request.onward_envelope.is_none());
        let middle_public =
            IdentityPublicKey::from_bytes(&middle_identity.public_key_bytes()).unwrap();
        assert!(terminal_request
            .envelope
            .verify_signature_from(&middle_public)
            .is_ok());
        assert_eq!(
            terminal_request.envelope.encrypted_blob,
            b"opaque terminal relay blob; do not parse".to_vec()
        );

        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);
    }

    #[tokio::test]
    async fn blind_relay_forward_requires_accepted_next_hop_ack() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerBlindRelayResponse {
                        accepted: false,
                        terminal: false,
                        forwarded: false,
                        ttl_remaining: 1,
                        reason: Some("relay_unavailable".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x56u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "forward_failed"
        }));
    }

    #[tokio::test]
    async fn blind_relay_forward_rejects_malformed_success_ack() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    (StatusCode::OK, "not-a-peer-blind-relay-ack").into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x57u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert_eq!(blind_stats.retry_attempted, 0);
        assert_eq!(blind_stats.retry_exhausted, 0);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "forward_failed"
                && !event.detail.contains("not-a-peer-blind-relay-ack")
        }));
    }

    #[tokio::test]
    async fn blind_relay_requires_next_hop_chat_relay_capability() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::OK.into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_peer_descriptor_for(
                    &next_hop_identity,
                    endpoint,
                    now,
                    now + 300,
                    vec![NodeCapability::PrivacyRelay],
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x54u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::NoRoute)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 0);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.no_route, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "no_route"
        }));
    }

    #[tokio::test]
    async fn blind_relay_requires_routeability_evidence_before_forwarding() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 1,
                        reason: Some("terminal_next_hop".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let next_hop_node_id = next_hop_identity.public_key_bytes();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x59u8; 16],
            next_hop: next_hop_node_id,
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::NoRoute)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 0);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.no_route, 1);
        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
            .expect("chat relay row should remain visible");
        // [ROUTE-HEALTH-REMOTE-POISONING 2026-08-11 by Codex] This request was
        // rejected before any outbound attempt. It must not let an untrusted
        // previous hop manufacture failure or quarantine evidence for a peer.
        assert_eq!(route_row.routeability_state, "unknown");
        assert!(!route_row.routeability_ready);
        assert_eq!(route_row.route_failure_count, 0);
        assert_eq!(route_row.route_consecutive_failures, 0);
        assert!(route_row.last_route_failure_reason.is_none());
        assert!(!route_row.route_quarantined);
        assert!(peer_store.recent_audit_events().iter().all(|event| {
            event.action != "blind_relay_route_health"
                || !event.detail.contains("reason=routeability_not_ready")
        }));
    }

    #[tokio::test]
    async fn peer_declared_downstream_failure_does_not_poison_next_hop_reputation() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity_for_route = Arc::clone(&next_hop_identity);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                let next_hop_identity = Arc::clone(&next_hop_identity_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    let failure_receipt = BlindRelayFailureReceipt::failed(
                        request.envelope.route_id,
                        BlindRelayFailureReceipt::request_commitment(&request.envelope),
                        "forward_failed",
                        now_secs(),
                        next_hop_identity.as_ref(),
                    );
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(PeerBlindRelayResponse {
                            accepted: false,
                            terminal: false,
                            forwarded: false,
                            ttl_remaining: 0,
                            reason: Some("forward_failed".to_string()),
                            delivery_receipt: None,
                            success_receipt: None,
                            failure_receipt: Some(failure_receipt),
                            opaque_terminal_response_b64: None,
                        }),
                    )
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_node_id = next_hop_identity.public_key_bytes();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    next_hop_identity.as_ref(),
                    endpoint,
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_node_id, now);

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x59u8; 16],
            next_hop: next_hop_node_id,
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);

        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
            .expect("chat relay row should remain visible");
        // [DOWNSTREAM-FAILURE-ATTRIBUTION 2026-08-11 by Codex] The peer was
        // reachable and returned a valid bounded error ACK. That is an
        // end-to-end failure signal, not proof that this route surface failed.
        assert!(route_row.routeability_ready);
        assert_eq!(route_row.route_failure_count, 0);
        assert_eq!(route_row.route_consecutive_failures, 0);
        assert!(route_row.last_route_failure_reason.is_none());
        assert!(!route_row.route_quarantined);
        assert!(peer_store.recent_audit_events().iter().all(|event| {
            event.action != "blind_relay_route_health" || event.outcome != "rejected"
        }));
    }

    #[tokio::test]
    async fn advertised_failure_receipt_omission_penalizes_exact_next_hop_surface() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(PeerBlindRelayResponse {
                            accepted: false,
                            terminal: false,
                            forwarded: false,
                            ttl_remaining: 0,
                            reason: Some("forward_failed".to_string()),
                            delivery_receipt: None,
                            success_receipt: None,
                            failure_receipt: None,
                            opaque_terminal_response_b64: None,
                        }),
                    )
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let current_node = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let next_hop_node_id = next_hop_identity.public_key_bytes();
        let legacy_descriptor = signed_chat_relay_peer_descriptor_for(
            &next_hop_identity,
            endpoint.clone(),
            now,
            now + 300,
        );
        let advertised_descriptor = SignedNodeDescriptor::sign(
            legacy_descriptor
                .descriptor
                .with_protocol_features([NodeProtocolFeature::BlindRelayFailureReceiptV1]),
            &next_hop_identity,
        )
        .unwrap();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(advertised_descriptor.clone(), now, "gossip_snapshot")
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_node_id, now);
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&current_node),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let request = PeerBlindRelayRequest {
            envelope: BlindRelayEnvelope {
                route_id: [0x6bu8; 16],
                next_hop: next_hop_node_id,
                ttl: 1,
                encrypted_blob: b"opaque downgraded failure receipt request".to_vec(),
                timestamp: now,
                signature: [0u8; 64],
            }
            .sign_with(current_node.as_ref()),
            previous_hop_node_id: current_node.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };

        let result = forward_blind_relay_with_retry(
            &state,
            &blind_peer_relay_url(&endpoint).unwrap(),
            &advertised_descriptor,
            request,
            now,
        )
        .await;
        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
            .expect("chat relay row should remain visible");
        assert_eq!(route_row.route_failure_count, 1);
        assert_eq!(route_row.route_consecutive_failures, 1);
        assert_eq!(
            route_row.last_route_failure_reason.as_deref(),
            Some("failure_receipt_downgrade")
        );
    }

    #[tokio::test]
    async fn invalid_failure_receipt_penalizes_immediate_next_hop_protocol() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let wrong_signer = Arc::new(IdentityKeyPair::generate());
        let wrong_signer_for_route = Arc::clone(&wrong_signer);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                let wrong_signer = Arc::clone(&wrong_signer_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    let receipt = BlindRelayFailureReceipt::failed(
                        request.envelope.route_id,
                        BlindRelayFailureReceipt::request_commitment(&request.envelope),
                        "forward_failed",
                        now_secs(),
                        wrong_signer.as_ref(),
                    );
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(PeerBlindRelayResponse {
                            accepted: false,
                            terminal: false,
                            forwarded: false,
                            ttl_remaining: 0,
                            reason: Some("forward_failed".to_string()),
                            delivery_receipt: None,
                            success_receipt: None,
                            failure_receipt: Some(receipt),
                            opaque_terminal_response_b64: None,
                        }),
                    )
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let current_node = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let next_hop_node_id = next_hop_identity.public_key_bytes();
        let descriptor = signed_chat_relay_peer_descriptor_for(
            &next_hop_identity,
            endpoint.clone(),
            now,
            now + 300,
        );
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(descriptor.clone(), now, "gossip_snapshot")
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_node_id, now);
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&current_node),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let request = PeerBlindRelayRequest {
            envelope: BlindRelayEnvelope {
                route_id: [0x6au8; 16],
                next_hop: next_hop_node_id,
                ttl: 1,
                encrypted_blob: b"opaque invalid failure receipt request".to_vec(),
                timestamp: now,
                signature: [0u8; 64],
            }
            .sign_with(current_node.as_ref()),
            previous_hop_node_id: current_node.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };

        let result = forward_blind_relay_with_retry(
            &state,
            &blind_peer_relay_url(&endpoint).unwrap(),
            &descriptor,
            request,
            now,
        )
        .await;
        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
            .expect("chat relay row should remain visible");
        assert_eq!(route_row.route_failure_count, 1);
        assert_eq!(route_row.route_consecutive_failures, 1);
        assert_eq!(
            route_row.last_route_failure_reason.as_deref(),
            Some("failure_receipt_invalid")
        );
    }

    #[tokio::test]
    async fn blind_relay_forward_reports_retry_exhaustion_without_payload_data() {
        let (relay, path) = temp_chat_relay("blind-relay-retry-exhaustion");
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::SERVICE_UNAVAILABLE.into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: Some(relay),
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x43u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(
            attempts.load(AtomicOrdering::SeqCst),
            MAX_BLIND_RELAY_FORWARD_ATTEMPTS
        );
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert_eq!(
            blind_stats.retry_attempted,
            (MAX_BLIND_RELAY_FORWARD_ATTEMPTS - 1) as u64
        );
        assert_eq!(blind_stats.retry_succeeded, 0);
        assert_eq!(blind_stats.retry_exhausted, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_retry"
                && event.outcome == "rejected"
                && !event.detail.contains("opaque encrypted relay bytes")
        }));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_forward_retries_timeout_without_endpoint_leak() {
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(
                move |Json(_request): Json<PeerBlindRelayRequest>| async move {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 1,
                        reason: Some("terminal_next_hop".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                },
            ),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    &next_hop_identity,
                    endpoint.clone(),
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(
                reqwest::Client::builder()
                    .timeout(std::time::Duration::from_millis(30))
                    .build()
                    .unwrap(),
            ),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
            blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x58u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert_eq!(
            blind_stats.retry_attempted,
            (MAX_BLIND_RELAY_FORWARD_ATTEMPTS - 1) as u64
        );
        assert_eq!(blind_stats.retry_succeeded, 0);
        assert_eq!(blind_stats.retry_exhausted, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_retry"
                && event.outcome == "scheduled"
                && event
                    .detail
                    .contains("reason_bucket=blind_relay_request_timeout")
                && !event.detail.contains(&endpoint)
        }));
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_retry"
                && event.outcome == "rejected"
                && event
                    .detail
                    .contains("reason_bucket=blind_relay_request_timeout")
                && !event.detail.contains(&endpoint)
                && !event.detail.contains("opaque encrypted relay bytes")
        }));
    }
}
