# AeroNyx Blind Issuer Deployment

<!--
============================================
File Creation/Modification Notes
============================================
[BLIND-ISSUER-DEPLOY 2026-07-23 by Codex]

Creation Reason:
- Define the production boundary between entitlement decisions, blind RSA
  signing, and decentralized Blind Vault redemption.

Main Functionality:
- Builds and installs the isolated Rust signer.
- Provisions an unprivileged custody account and owner-only secret files.
- Documents authenticated backend frames and safe key rotation.
- Documents bounded HSM/KMS response deadlines and retry semantics.
- Documents aggregate status and circuit-breaker recovery.
- Keeps all private issuer material away from decentralized storage nodes.

Dependencies:
- crates/aeronyx-blind-issuer: runtime and provisioning commands.
- aeronyx-blind-issuer.service: systemd sandbox.
- blind-issuer.example.toml: fail-closed configuration shape.
- deploy/node/server.example.toml: public keys only on storage nodes.

Main Logical Flow:
1. An entitlement backend approves an anonymous issuance request.
2. The client blinds a domain-separated admission token.
3. The backend sends only key ID + blinded bytes to this localhost signer.
4. The client finalizes the signature and later redeems it at any configured
   Blind Vault node; issuance and redemption cannot be joined by the signer.

Important Note for Next Developer:
- This process is not a decentralized node and must not be part of nodeboard's
  node installer.
- Never add wallet, account, device, lease, IP, request-body logs, or storage
  node state to the signer API.
- Production private keys should ultimately move behind the existing signer
  boundary into an HSM/KMS without changing the wire contract.

Last Modified: v1.6.0-BlindIssuerDeploy - Documented monotonic token-bucket
admission and its bounded burst behavior.
============================================
-->

The blind issuer is a deliberately small custody process. It performs RFC 9474
private RSA operations, but it never sees an account, wallet, device, lease,
storage node, client IP, or redemption event. Decentralized nodes receive only
public RSA epochs and finalized unlinkable credentials.

Do **not** install this service on US1, Korean1, Noway1, or any independent
storage/relay node. Run it beside the trusted entitlement backend or behind an
HSM/KMS adapter on a separately audited host.

## Build

```bash
cargo test -p aeronyx-blind-issuer
cargo clippy -p aeronyx-blind-issuer --all-targets --no-deps -- -D warnings
cargo build --release -p aeronyx-blind-issuer
sudo install -o root -g root -m 0755 \
  target/release/aeronyx-blind-issuer \
  /usr/local/bin/aeronyx-blind-issuer
```

## Provision Custody

Create a stable unprivileged account. The Rust process verifies that every
secret is a regular, single-link, owner-only file owned by this exact account.

```bash
sudo useradd --system --user-group --home-dir /nonexistent \
  --shell /usr/sbin/nologin aeronyx-issuer
sudo install -d -o aeronyx-issuer -g aeronyx-issuer -m 0700 \
  /var/lib/aeronyx-blind-issuer

sudo runuser -u aeronyx-issuer -- \
  /usr/local/bin/aeronyx-blind-issuer generate-auth-token \
  --output /var/lib/aeronyx-blind-issuer/backend.token

sudo runuser -u aeronyx-issuer -- \
  /usr/local/bin/aeronyx-blind-issuer generate-key \
  --bits 3072 \
  --output /var/lib/aeronyx-blind-issuer/issuer-epoch-1.der
```

`generate-key` prints only the public key fingerprint and public DER. Capture
those public values in the audited deployment record; stdout never contains the
private key.

The entitlement backend needs the same bearer-token bytes, but should not read
the issuer-owned file directly. Provision a second owner-only copy through the
deployment secret manager, owned by the backend service account. Never make one
shared token file group-readable.

## Configure And Start

Copy `blind-issuer.example.toml` to `/etc/aeronyx/blind-issuer.toml`, replace
the key path and Unix-second activation/expiry values, and keep every epoch at
31 days or less. The example timestamps are intentionally not production-ready.

```bash
sudo install -d -o root -g aeronyx-issuer -m 0750 /etc/aeronyx
sudo install -o root -g aeronyx-issuer -m 0640 \
  deploy/blind-issuer/blind-issuer.example.toml \
  /etc/aeronyx/blind-issuer.toml
sudo install -o root -g root -m 0644 \
  deploy/blind-issuer/aeronyx-blind-issuer.service \
  /etc/systemd/system/aeronyx-blind-issuer.service
sudo systemctl daemon-reload
sudo systemctl enable --now aeronyx-blind-issuer.service
sudo systemctl status aeronyx-blind-issuer.service
```

The service binds only to `127.0.0.1`, and systemd independently denies all
non-local network traffic. It has no capabilities, devices, writable paths,
home access, or access to AeroNyx node databases.

`max_requests_per_second` is enforced by a monotonic, constant-memory token
bucket before request-body extraction. Its burst capacity is exactly one second
of the configured rate, and capacity refills continuously at that rate. Wall
clock/NTP corrections cannot reset or bypass the limiter. `max_in_flight`
remains the separate hard ceiling for concurrent private-key operations.

`signing_timeout_ms` controls how long the backend waits for an HTTP response
(default 10,000ms; allowed 100–120,000ms). A timeout returns `503`, but it does
not cancel a native software/HSM/KMS operation. That operation retains its
`max_in_flight` permit until completion, so disconnected or retrying callers
cannot exceed the configured custody capacity. Upstream retries must use
bounded exponential backoff; blind signing is deterministic for the same
validated request, so a retry does not create a linkable redemption identity.

`circuit_failure_threshold` (default 5; allowed 1–100) opens the process-wide
custody circuit after consecutive backend failures or caller timeouts.
`circuit_cooldown_ms` (default 30,000ms; allowed 1,000–300,000ms) uses a
monotonic clock for enforcement. While open, signing returns `503` before body
extraction. Cooldown expiry does not declare the signer healthy: it admits one
half-open signing probe while concurrent requests continue to receive `503`.
Only a successful custody operation closes the circuit and resets the
consecutive-failure count. A failed or timed-out probe opens a fresh cooldown;
a malformed/policy-rejected request releases the probe without claiming that
the private-key backend recovered.

## Internal API

All responses use `Cache-Control: no-store`. No request-body tracing middleware
is installed.

| Endpoint | Authentication | Result |
|---|---|---|
| `GET /internal/v1/health` | none, loopback only | `204` with an active key and verified closed circuit; otherwise `503` |
| `GET /internal/v1/issuer-epochs` | Bearer token | bounded `ANBE` public epoch snapshot |
| `GET /internal/v1/status` | Bearer token | aggregate health/capacity counters; no request dimensions |
| `POST /internal/v1/blind-sign` | Bearer token | bounded `ANBS` blind signature |

The signing request body is `ANBI || admission_version || issuer_key_id ||
blinded_message`. It contains no entitlement or user metadata. The Rust crate
exports `encode_sign_request`, `decode_sign_response`, and
`decode_epoch_snapshot` as the canonical backend-side codec reference.

The JSON status snapshot contains only process-wide counts: active/key count,
signer generation, reload attempts/successes/rejections, last aggregate reload
attempt/success time, in-flight capacity, breaker state (`circuit_open` and
`circuit_half_open`), signing successes, backend failures, timeouts, and
rejections. It never includes blinded bytes, key IDs, wallet/account/device
identifiers, IP addresses, request IDs, timestamps per request, issuance
history, key paths, custody providers, or reload error details. Treat the
endpoint as operator telemetry and keep bearer authentication enabled even on
loopback.

## Safe Rotation

1. Generate a new private key as `aeronyx-issuer`; never copy it to nodes.
2. Add a future `[[keys]]` epoch while retaining the currently active epoch.
3. Atomically install the updated config, then run
   `sudo systemctl reload aeronyx-blind-issuer.service`.
4. Require `reload_attempted` to increase by one, `reload_succeeded` to increase
   by one, `reload_rejected` to remain unchanged, `signer_generation` to
   increase, and `/internal/v1/health` to remain `204`. A rejected reload
   increments only the attempt/rejection audit counters and leaves the previous
   generation serving unchanged.
5. Fetch `/internal/v1/issuer-epochs` through the authenticated backend path.
6. Publish the new **public** DER epoch to storage-node configuration before
   its activation time; nodes then expose signed public issuer directories.
7. Activate client selection only after audited nodes advertise the same key ID.
8. Keep the old public epoch on nodes until it expires. Remove its private key
   from online custody only after issuance has stopped and the rollback window
   has closed.

Rotation overlap is for public-key propagation and clock tolerance. It must not
be used to add account tiers, application IDs, or redemption callbacks to the
signer. SIGHUP reloads only `[[keys]]`; listener, token path, capacity, timeout,
and circuit-policy changes are rejected and require a controlled restart. Every
reload candidate must contain an active key and preserve every unexpired epoch
byte-for-byte, so already issued clients cannot lose their signing epoch.
