# [PHALA-NODE-IMAGE 2026-10-06 by Codex]
# Build from the Rust workspace root:
# docker build -f deploy/node/Dockerfile -t aeronyx-node:local .
# [PHALA-TDX-IMAGE-ARCH 2026-10-06 by Codex] The authorized Phala target is
# Intel TDX; force both compilation and runtime layers to its Linux x86_64 ABI.
FROM --platform=linux/amd64 rust:1.97.1-bookworm AS builder

WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
# [PHALA-UNWIND-BUILD-CONTRACT 2026-10-07 by Codex] release uses abort
# and cannot run the reverse owners' panic/drain contracts. Use one explicit
# profile for both compilation and the copied executable; no override argument.
RUN cargo build --profile phala --locked -p aeronyx-server

FROM --platform=linux/amd64 debian:bookworm-slim AS runtime-common
# [PHALA-AGENT-MOUNTPOINT 2026-10-06 by Codex] Keep the fixed bind target
# present in the immutable image before Compose attaches one selected socket.
# [PHALA-PEER-LEAST-PRIVILEGE 2026-10-06 by Codex] This peer image does not
# configure a VPN interface, so it carries no runtime network tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/aeronyx /var/lib/aeronyx /var/run/aeronyx

COPY --from=builder /workspace/target/phala/aeronyx-server /usr/local/bin/aeronyx-server
# [PHALA-CLOUD-IMAGE-CONFIG 2026-10-06 by Codex] Cloud deployments use the
# baked non-secret template; the isolated local profile still bind-mounts its
# own config at the same path.
# [PHALA-STANDALONE-IDENTITY 2026-10-06 by Codex] Both Phala services disable
# CMS management and persist their independently generated server identity in
# their own state volume on first start.
COPY deploy/node/server.phala.peer.example.toml /etc/aeronyx/server.toml
ENTRYPOINT ["/usr/local/bin/aeronyx-server"]
CMD ["start", "--config", "/etc/aeronyx/server.toml"]

# [PHALA-ISOLATED-VPN-IMAGE 2026-10-08 by Codex] Only the explicitly
# selected isolated trial stage carries the Linux TUN lifecycle's ip command.
# The Compose profile still owns its bounded capability/device permissions.
FROM runtime-common AS isolated-vpn
RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/node/server.phala.example.toml /etc/aeronyx/server.toml

# [PHALA-ISOLATED-VPN-IMAGE 2026-10-08 by Codex] The default final stage
# remains the tool-free peer image; isolated-vpn must be selected explicitly.
FROM runtime-common AS peer
