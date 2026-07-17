# AeroNyx Node Discovery and Encrypted Relay Development Plan

## File Creation / Modification Notes

Creation Reason: Define the long-term Rust protocol plan for node-to-node discovery, signed node descriptors, encrypted envelope relay, Memory Chain coordination, and a future Directory Chain without smart contracts.

Modification Reason: v0.5.0 - Added a monotonic signed local anchor that rejects older valid delivery-evidence caches without persisting correlatable relay data.

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

Last Modified: v0.5.0 - Added local signed rollback protection for verified-client delivery evidence.
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

Directory Chain is a future append-only descriptor ledger.

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

Potential new files:

```text
crates/aeronyx-core/src/protocol/directory_chain.rs
crates/aeronyx-server/src/services/discovery/directory_chain.rs
crates/aeronyx-server/src/services/discovery/witness.rs
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

Status: Future.

Goals:

- Pack descriptor events into hash-linked blocks.
- Add Merkle root by epoch.
- Add witness signatures.
- Add snapshot validation.

Files likely changed:

```text
crates/aeronyx-core/src/protocol/directory_chain.rs
crates/aeronyx-server/src/services/discovery/directory_chain.rs
crates/aeronyx-server/src/services/discovery/witness.rs
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
