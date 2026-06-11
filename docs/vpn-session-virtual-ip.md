# VPN Session Virtual IP Reporting

## Goal

Expose the tunnel-local VPN virtual IP assigned by the Rust node so the API and
nodeboard can diagnose active sessions without asking operators to SSH into the
node.

This field is operational metadata only. It is the internal VPN tunnel address
assigned to a client session, for example `100.64.0.2`. It is not the client's
public IP, DNS query, destination IP, visited website, payload, or browsing
history.

## Source Paths

- `crates/aeronyx-server/src/services/session.rs`
  - `Session.virtual_ip` is the authoritative tunnel IP allocated by the node.
- `crates/aeronyx-server/src/management/models.rs`
  - `SessionEventReport.virtual_ip` serializes the optional tunnel IP for the
    centralized API.
- `crates/aeronyx-server/src/management/reporter.rs`
  - `SessionEvent` carries `virtual_ip`.
  - `SessionEvent::created`, `SessionEvent::snapshot`, and
    `SessionEvent::ended` include `virtual_ip`.
  - `SessionEventSender::session_created`,
    `SessionEventSender::session_traffic_snapshot`, and
    `SessionEventSender::session_ended` forward it to the report payload.
- `crates/aeronyx-server/src/server.rs`
  - Handshake success reports `result.session.virtual_ip`.
  - Periodic traffic snapshots report `session.virtual_ip`.
  - Expired-session cleanup reports the released session virtual IP.
- `crates/aeronyx-server/src/management/command_handler.rs`
  - Remote kick-session and ban-wallet final session reports include the removed
    session virtual IP.

## API Contract

The node may include this optional field in all session report types:

```json
{
  "type": "session_traffic_snapshot",
  "session_id": "vpn-session-id",
  "client_wallet": "wallet-or-voucher-owner",
  "virtual_ip": "100.64.0.2",
  "bytes_in": 1048576,
  "bytes_out": 524288,
  "timestamp": 1781200000
}
```

If a session is reported before a virtual IP is available, the node omits the
field and the API keeps it empty until a later report updates it.

## Operator Value

Node operators can now identify which tunnel IP maps to a wallet/session in
nodeboard, correlate traffic and billing rows, and act on the session via
centralized commands without exposing private browsing data.
