# AeroNyx Node Discovery and Encrypted Relay Development Plan

## File Creation / Modification Notes

Creation Reason: Define the long-term Rust protocol plan for node-to-node discovery, signed node descriptors, encrypted envelope relay, Memory Chain coordination, and a future Directory Chain without smart contracts.

Modification Reason: v0.24.0 - Configurable pinned-witness corroboration targets with retryable receipt collection.

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

Last Modified: v0.24.0 - Distinguished independent multi-node corroboration from one-receipt evidence without claiming consensus.
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

AeroNyx currently has several important building blocks:

- Rust privacy protocol node runtime.
- Aggregate health, capacity, and event reporting to backend/nodeboard.
- `nodeboard` for node registration, health review, capacity decision, and incident closure.
- Memory Chain primitives and append-only ledger structures.
- Chat relay and wallet route cache concepts.
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
- A dedicated `discovery.directory_chain_sync_peer_node_ids` pin list is
  required in addition to a current valid signed PeerStore descriptor. Empty
  configuration remains fail-closed and backward compatible.
- Requests enforce timestamp freshness, replay rejection, per-peer rate
  limits, strict body/page/object limits, canonical decoding, and chain id.
- Every response is gated by a complete persisted-chain audit. Object batches
  are all-or-nothing and preserve the requested hash order.

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

Still pending before Directory Chain can be described as live:

- Portable co-signature certificate exchange and deterministic fork selection.
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

Initial entry:

```text
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
