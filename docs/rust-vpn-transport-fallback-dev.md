<!--
============================================
File Creation/Modification Notes
============================================
Creation Reason:
  Document the current AeroNyx Rust VPN transport status and the production
  development path for adding TCP/TLS/HTTPS/WebSocket fallback transports.

Modification Reason:
  The current Rust VPN data plane only supports UDP. Commercial VPN deployment
  needs a clear multi-end development plan before changing server, client,
  backend, and nodeboard behavior.

Main Functionality:
  1. Clarifies the existing UDP-only VPN tunnel implementation.
  2. Lists affected Rust node, Flutter client, backend, and nodeboard files.
  3. Defines a phased, backward-compatible development plan.
  4. Documents compatibility, privacy, performance, and testing requirements.

Dependencies:
  - Rust node repository: AeroNyxNetwork/AeroNyx
  - Flutter client repository: current app workspace
  - Backend API server: privacy network node metadata and observability APIs
  - Nodeboard: operator UI for node transport capability and health

Main Logical Flow:
  1. Keep UDP as the primary high-performance transport.
  2. Add a transport abstraction instead of replacing UDP code.
  3. Add TCP/TLS 443 and WebSocket-over-HTTPS fallback as optional transports.
  4. Report transport capabilities and live health to backend.
  5. Let clients auto-select the best reachable transport.
  6. Let nodeboard expose operator configuration and diagnostics.

Important Note for Next Developer:
  - Do not break the existing UDP 51820 protocol or client behavior.
  - Do not change FFI function signatures until mobile/native callers are
    migrated through a compatibility layer.
  - Any fallback transport must carry the same encrypted AeroNyx packet format;
    it must not expose payload, destination, DNS content, URL, wallet traffic,
    voucher secret, or client public IP beyond existing aggregate metadata.
  - HTTP/HTTPS management APIs are not VPN tunnels today. Do not present them as
    user-data tunnel support until data-plane fallback is implemented.

Last Modified:
  v0.1.0 - Initial Rust VPN multi-transport fallback development plan.
============================================
-->

# AeroNyx Rust VPN Multi-Transport Fallback Development Plan

## 1. Current Status

The current AeroNyx Rust VPN data plane is UDP-only.

The Rust node binds the VPN transport with `UdpTransport::bind_addr(self.config.listen_addr())`. The default listen address is `0.0.0.0:51820`.

HTTP/TCP exists in the Rust node, but it is used for local APIs, health checks, MemChain, Voice, management reporting, or backend communication. It is not currently used as the VPN tunnel data plane.

Current practical meaning:

- Normal VPN packets: UDP `51820`.
- Rust node health/API: HTTP on local/internal ports such as `127.0.0.1:8421`.
- Node-to-backend reporting: HTTPS to the centralized API.
- No implemented TCP VPN, TLS VPN, HTTPS VPN, WebSocket VPN, or HTTP CONNECT VPN tunnel yet.

## 2. Why This Matters For Commercial VPN

UDP is the right default for performance, latency, and packet behavior. However, commercial VPN users often connect from networks where UDP is blocked or degraded:

- corporate networks
- school networks
- hotel Wi-Fi
- airport Wi-Fi
- mobile carrier NAT
- public Wi-Fi with captive portals
- regions with strict firewall policies

For commercial reliability, AeroNyx should keep UDP as the primary transport and add fallback transports:

1. UDP primary transport.
2. TCP/TLS 443 fallback.
3. WebSocket-over-HTTPS 443 fallback.
4. Optional future HTTP/2 or QUIC strategy after measurement.

The fallback transports should carry the same encrypted AeroNyx packet format. They are transport wrappers, not a new privacy protocol.

## 3. Existing File Map

### 3.1 Rust Node: AeroNyxNetwork/AeroNyx

Production server paths:

- US1 source: `/root/open/AeroNyx`
- Korean1 runtime source: `/root/a/AeroNyx`
- Korean1 clean build source: `/root/a/AeroNyx-main-build`

Important Rust files:

- `crates/aeronyx-server/src/server.rs`
  - Starts the server.
  - Currently creates the VPN data transport through `UdpTransport::bind_addr(...)`.
  - Wires packet handler, TUN, management reporter, DNS proxy, health API, MemChain, Voice, and miner.

- `crates/aeronyx-server/src/config_infra.rs`
  - Defines default VPN listen address.
  - Current default is `0.0.0.0:51820`.
  - Needs new config fields for enabled transports and fallback ports.

- `crates/aeronyx-server/src/config.rs`
  - Loads server config and exposes accessors.
  - Needs compatibility-preserving accessors for `vpn.transports`.

- `crates/aeronyx-transport/src/udp.rs`
  - Current UDP transport implementation.
  - Implements send/recv over Tokio `UdpSocket`.
  - Should remain unchanged as the stable primary transport where possible.

- `crates/aeronyx-transport/src/lib.rs`
  - Exports transport modules.
  - Should export new TCP/TLS/WebSocket transport modules when added.

- `crates/aeronyx-transport/src/traits.rs`
  - Existing common transport traits.
  - Needs review before adding stream-based transport because TCP/WebSocket are connection-oriented while UDP is packet-oriented.

- `crates/aeronyx-server/src/handlers/packet.rs`
  - Decrypts and handles AeroNyx packets.
  - Should receive packets from a transport-neutral input path.

- `crates/aeronyx-server/src/api/vpn_health.rs`
  - Current health endpoint.
  - Should expose transport capability and live health:
    - `udp_enabled`
    - `udp_listen_addr`
    - `tcp_tls_enabled`
    - `tcp_tls_listen_addr`
    - `websocket_enabled`
    - `websocket_url`
    - `preferred_transport`
    - `fallback_available`
    - per-transport packet counters, drops, errors, last successful probe

- `crates/aeronyx-server/src/management/reporter.rs`
  - Reports node heartbeat to backend.
  - Should report transport capability and aggregate health.

- `crates/aeronyx-server/server.example.toml`
  - Should document all transport options with safe defaults.

- `docs/deploy/README.md` or deployment docs in the Rust repo
  - Should describe firewall, systemd, TLS cert, reverse proxy, and port requirements.

### 3.2 Flutter Client: Current Workspace

Important local client files:

- `rust/src/udp_client.rs`
  - Current mobile/native Rust client.
  - Implements `UdpVpnClient`.
  - Performs UDP socket connect, handshake, encrypted packet send/receive, keepalive, replay protection, and zero-allocation receive path.

- `rust/src/vpn_ffi.rs`
  - C-compatible FFI used by iOS/macOS Network Extension and native callers.
  - Currently stores `client: Option<Arc<UdpVpnClient>>`.
  - Function signatures should not be broken. Add transport selection behind the existing API first.

- `rust/src/lib.rs`
  - Exports `UdpVpnClient`, `VpnSession`, and FFI-facing types.
  - Should later export a transport-neutral client facade.

- `lib/models/vpn_models.dart`
  - `VpnServer.defaultPort = 51820`.
  - Comments currently state the port is UDP.
  - Needs transport fields such as `supportedTransports`, `preferredTransport`, `udpPort`, `tlsPort`, `websocketUrl`.

- `lib/services/vpn_node_service.dart`
  - Parses public VPN nodes from backend.
  - Currently normalizes stale management port values back to UDP `51820`.
  - Needs parsing for transport capability from backend.

- `lib/services/aeronyx_vpn_service.dart`
  - Orchestrates VPN connection from Flutter side.
  - Should decide selected transport and pass it to Rust FFI once Rust supports it.

- `lib/providers/vpn_provider.dart`
  - User-facing connection state.
  - Should surface fallback attempts and final selected transport only as useful diagnostics, not noisy UI.

- `lib/services/vpn_node_health_service.dart`
  - Read-only health checks for connected VPN node.
  - Should include transport reachability diagnostics.

- `docs/windows_vpn_full_tunnel.md`
  - Windows currently documents outbound UDP allowlisting.
  - Must be updated when TCP/TLS/WebSocket fallback is implemented.

### 3.3 Backend API Server

Known production path:

- `/root/aeronyx`

Relevant areas:

- Privacy network node model and heartbeat ingestion.
- Public node listing API used by the app.
- VPN observability APIs used by nodeboard.
- Stats and health aggregation.

Backend must add fields without breaking existing clients:

- `supported_transports`
- `preferred_transport`
- `udp_port`
- `tcp_tls_port`
- `websocket_url`
- `transport_health`
- `transport_last_reported_at`

The old `vpn_port`, `udp_port`, or `port` fields must remain compatible until all clients migrate.

### 3.4 Nodeboard

Known production path:

- `/root/open/nodeboard`

Relevant areas:

- `app/dashboard/services/page.tsx`
  - Existing service overview and detail panels.
  - Should add a transport capability panel or extend the capacity/network panel.

- `types/index.ts`
  - Should type new backend transport fields.

- `lib/i18n/index.ts`
  - Must add multi-language labels for transport capability and fallback status.

Suggested UI fields:

- UDP status
- TCP/TLS status
- WebSocket HTTPS status
- Preferred transport
- Last successful fallback probe
- Failed transport reason
- Public endpoint and operator firewall hints

## 4. Target Architecture

### 4.1 Principle

Do not replace UDP. Add a transport layer that lets the same encrypted AeroNyx packet protocol run over multiple carriers.

Recommended hierarchy:

```text
Encrypted AeroNyx Packet Protocol
  |
  +-- UDP carrier, primary, low latency
  +-- TCP/TLS carrier, fallback for UDP-blocked networks
  +-- WebSocket HTTPS carrier, fallback for strict proxy networks
```

### 4.2 Server Runtime Model

The server should be able to start any combination:

```toml
[vpn]
listen_addr = "0.0.0.0:51820"

[vpn.transports]
udp_enabled = true
udp_listen_addr = "0.0.0.0:51820"

tcp_tls_enabled = false
tcp_tls_listen_addr = "0.0.0.0:443"
tcp_tls_cert_path = "/etc/aeronyx/tls/fullchain.pem"
tcp_tls_key_path = "/etc/aeronyx/tls/privkey.pem"

websocket_enabled = false
websocket_listen_addr = "127.0.0.1:8443"
websocket_public_url = "wss://node.example.com/aeronyx/vpn"

preferred_transport = "udp"
```

Backward compatibility:

- If `[vpn.transports]` is missing, behave exactly like today.
- `listen_addr` continues to mean UDP listen address.
- Existing node configs should not require edits.

### 4.3 Client Selection Model

The client should try transports in this order unless backend says otherwise:

1. UDP.
2. TCP/TLS 443.
3. WebSocket HTTPS 443.

The client should cache recent success per node/network type:

- Wi-Fi SSID hash or network category, privacy-safe only.
- Mobile vs Wi-Fi.
- Last successful transport.
- Last failure reason category, not raw destination or user data.

Do not upload client network names, SSIDs, domains, DNS queries, packet payloads, or browsing data.

## 5. Development Phases

### Phase 1: Capability Schema And Reporting

Goal: expose capability metadata before changing tunnel behavior.

Rust node:

- Add transport config structs.
- Add health JSON fields.
- Add heartbeat fields.
- Keep UDP-only runtime.

Backend:

- Accept and store reported transport capabilities.
- Return fields in public node list and nodeboard observability APIs.

Nodeboard:

- Display transport capability and missing configuration.

Client:

- Parse fields but continue using UDP only.

Success criteria:

- Existing clients still connect by UDP.
- Nodeboard can show that all current nodes are UDP-only.
- No behavior change for production traffic.

### Phase 2: Server TCP/TLS Carrier

Goal: add TCP/TLS as a server-side optional carrier.

Rust node:

- Add `TcpTlsTransport` or `StreamTransport`.
- Define packet framing over stream:
  - 4-byte big-endian frame length.
  - encrypted AeroNyx packet bytes.
  - max frame size enforced.
- Add TLS acceptor.
- Route decoded frames into the same packet handler.
- Add per-transport metrics.

Security:

- TLS is only an outer carrier.
- Inner AeroNyx packet encryption remains mandatory.
- Rate limit handshake attempts.
- Enforce max frame size and read timeouts.

Success criteria:

- UDP still works.
- TCP/TLS can complete handshake and pass test traffic.
- Broken TCP connections clean up sessions.

### Phase 3: Client TCP/TLS Fallback

Goal: mobile/desktop clients can use TCP/TLS if UDP fails.

Client Rust:

- Add a transport-neutral client facade, for example `VpnClient`.
- Keep `UdpVpnClient` for compatibility.
- Add `TcpTlsVpnClient`.
- Keep FFI signatures stable initially.
- Add optional transport selection setter or extended init function only after compatibility layer is ready.

Flutter:

- Parse transport capability.
- Try UDP first.
- If UDP times out, try TCP/TLS if node supports it.
- Show one clean connection state, not multiple confusing states.

Success criteria:

- Existing app versions keep working.
- New app can connect on UDP-blocked test network through TCP/TLS.

### Phase 4: WebSocket HTTPS Carrier

Goal: support strict proxy environments.

Rust node:

- Add WebSocket route such as `/aeronyx/vpn`.
- Use binary WebSocket frames only.
- Carry encrypted AeroNyx packet bytes.
- Add backpressure and frame size limits.

Deployment:

- Prefer reverse proxy with Nginx/Caddy for public TLS.
- Keep Rust internal listener private if reverse proxy terminates TLS.

Client:

- Add `WebSocketVpnClient`.
- Try it after TCP/TLS.

Success criteria:

- Can connect from network allowing only HTTPS/WebSocket.
- Packet latency and throughput are measured and visible.

### Phase 5: Commercial Observability

Goal: node operators and the official website can explain network reliability clearly.

Backend and nodeboard:

- Aggregate transport capability:
  - total UDP nodes
  - total TCP/TLS nodes
  - total WebSocket nodes
  - fallback-ready percentage
  - per-region fallback readiness

Website:

- Display privacy-safe aggregate transport reliability.
- Do not expose sensitive node or client details.

## 6. Required Tests

Rust node:

- UDP existing tests remain passing.
- TCP frame codec tests.
- TLS handshake tests.
- WebSocket binary frame tests.
- Max frame size rejection.
- Read timeout cleanup.
- Packet handler compatibility.
- Health JSON serialization.

Client:

- UDP existing tests remain passing.
- Transport selection order.
- UDP timeout fallback to TCP/TLS.
- TCP/TLS timeout fallback to WebSocket.
- FFI backward compatibility.
- iOS/macOS Network Extension memory stability.
- Android and Windows build compatibility.

Backend:

- Heartbeat accepts old nodes without transport fields.
- Heartbeat accepts new transport fields.
- Public node list remains backward compatible.
- Nodeboard API returns typed transport health.

Nodeboard:

- Type-check and production build.
- Multi-language labels.
- Responsive layout.
- Empty, unknown, degraded, healthy states.

Manual network tests:

- UDP open.
- UDP blocked, TCP 443 open.
- UDP and raw TCP blocked, WebSocket HTTPS open.
- Captive portal.
- Mobile network NAT.
- Router reboot during session.

## 7. Compatibility Requirements

Do not break:

- Existing UDP `51820` nodes.
- Existing mobile clients.
- Existing `vpn_port`, `udp_port`, and `port` parsing.
- Existing FFI function names and C ABI.
- Existing node configs.
- Existing backend public node APIs.

Migration rule:

- Add fields first.
- Deploy backend acceptance.
- Deploy Rust reporting.
- Deploy nodeboard display.
- Deploy client parsing.
- Only then enable fallback transports node by node.

## 8. Privacy Boundary

Allowed to report:

- transport capability
- aggregate packet counts
- aggregate byte counts
- aggregate failures
- selected transport type
- node-level availability
- per-transport health status

Not allowed to report:

- packet payloads
- DNS query contents
- domains
- URLs
- destination IPs per user
- client public IPs
- wallet-level browsing traffic
- voucher secrets
- raw connection logs tied to users

## 9. Recommended Next Development Task

Start with Phase 1.

Reason:

- It is low risk.
- It does not change live VPN traffic.
- It makes nodeboard and backend ready for commercial rollout.
- It gives operators visibility before enabling TCP/TLS or WebSocket.

First implementation target:

1. Rust node config:
   - add `vpn.transports`
   - default to UDP-only
2. Rust health:
   - expose `supported_transports`
   - expose `preferred_transport`
3. Rust heartbeat:
   - report the same metadata
4. Backend:
   - store and pass through fields
5. Nodeboard:
   - add transport capability display
6. Client:
   - parse but continue UDP-only

This keeps the system stable while preparing the full commercial fallback path.
