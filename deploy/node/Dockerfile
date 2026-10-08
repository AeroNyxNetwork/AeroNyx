# ============================================
# File: deploy/node/Dockerfile
# ============================================
# Creation Reason:
# - [TEE-NODE-IMAGE 2026-10-09 by Claude] One verifiable linux/amd64 image of
#   the node for dstack CVMs (Phala Cloud or self-hosted TDX). CI builds it
#   from a tagged commit and publishes the digest; the CVM attestation then
#   measures that digest, so anyone can tie a running node back to source.
#
# Main Functionality:
# - Builds `aeronyx-server` with the pinned toolchain and the locked graph.
# - Runtime holds only the binary, CA roots, the TEE config template, and the
#   entrypoint. No VPN tooling: this role does not create a TUN device.
#
# Important Notes for Next Developer:
# - Base images are pinned by digest. Bump them as an audited change.
# - Build context is the repository root (see /.dockerignore).
# ============================================

FROM rust:1.97.1-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97 AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --release --locked -p aeronyx-server --bin aeronyx-server

FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/aeronyx /var/lib/aeronyx
COPY --from=build /src/target/release/aeronyx-server /usr/local/bin/aeronyx-server
COPY deploy/node/server.tee.toml /etc/aeronyx/server.tee.toml
COPY deploy/node/tee-entrypoint.sh /usr/local/bin/aeronyx-tee-entrypoint
VOLUME ["/var/lib/aeronyx"]
EXPOSE 8422
ENTRYPOINT ["/usr/local/bin/aeronyx-tee-entrypoint"]
