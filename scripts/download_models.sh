#!/usr/bin/env bash
# [MEMCHAIN-PHALA-ONLY 2026-10-06 by Codex]
# Retained only so legacy deployment commands fail with an explicit message.
# MemChain model inference is delegated to the configured Phala ACI route;
# ordinary Rust nodes must not download or run local inference models.

set -euo pipefail
printf '%s\n' "Local MemChain model downloads are disabled; configure the attested Phala ACI route instead." >&2
exit 1
