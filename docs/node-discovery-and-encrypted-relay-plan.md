# AeroNyx Node Discovery and Encrypted Relay Development Plan

## File Creation / Modification Notes

Creation Reason: Define the long-term Rust protocol plan for node-to-node discovery, signed node descriptors, encrypted envelope relay, and a future Directory Chain without smart contracts.

Modification Reason: Initial planning document requested during product architecture discussion. This file should be maintained as implementation progresses.

Main Functionality:

- Record the product boundary for AeroNyx as an open privacy protocol, not a node operator.
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
- Do not add smart contracts to this design. The proposed Directory Chain is a signed, append-only node directory ledger only.
- Do not store or sync packet payloads, DNS contents, destinations, domains, URLs, browsing history, voucher secrets, client public IPs, chat plaintext, private keys, or wallet-level traffic.
- Default routing policy must be no-exit unless an operator explicitly enables a future exit capability.

Last Modified: v0.1.0 - Initial node discovery and encrypted relay architecture plan.

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

## 5. Node Identity

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

## 6. Signed Node Descriptor

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

## 7. Bootstrap Directory

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

## 8. Peer Store

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

## 9. Gossip Sync

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

## 10. Encrypted Envelope Relay

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

## 11. Store-and-Forward Queue

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

## 12. Directory Chain Without Smart Contracts

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

## 13. Onion Routing Relationship

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

## 14. Client Product Implications

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

## 15. nodeboard Product Implications

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

## 16. Backend Product Implications

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

## 17. Development Phases

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

## 18. Open Questions

- Should node signing keys be generated during registration or first local startup?
- Should operator key be wallet-based, nodeboard-account-based, or both?
- What is the minimum descriptor expiry window for reliable mobile clients?
- Should bootstrap snapshots include only public nodes or also private invite-only nodes?
- What descriptor fields must be visible to clients versus only to operators?
- How should a node rotate keys without losing reputation/history?
- What is the first client use case: chat relay, privacy relay, storage, or agent relay?

## 19. Maintenance Log

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
```
