# MemChain Blind Cognition — Design (relocating cognition off the node)

> Status: **IMPLEMENTED (node side) — all bricks landed on `main`.** Builds on the node-blind path (`docs/memchain-node-blind-refactor.md`, Brick 1). Bricks: 3a `0a04f59`, 3b `460dee2`, 3c `7534f11`+`a781f72`, 3d `382a47d`. (Client-side artifact production — tokenize/hash/extract/summarize — is separate and not yet built.)
> Goal: give **node-blind** records the same retrieval quality as sighted ones — **full-text (BM25), graph, and summary** — while the node **still never reads plaintext**. Cognition (tokenize / extract / embed / summarize) moves to the **client** (which holds the plaintext) or a **user-chosen LLM**; the node stores and indexes only client-supplied, privacy-preserving artifacts.
> Constraint: **additive, default-off, never touch the VPN core.**

---

## 0. Where we are

Brick 1 made blind records **storable** and **vector-searchable** (the client ships a precomputed embedding; the node searches opaque vectors and returns ciphertext in the recall `sealed` list). What blind records still lack — because those run on node-side **plaintext** today — is **BM25 full-text**, **graph traversal**, and **summaries**. This document is the plan to close that gap without giving the node plaintext.

## 1. Principle

> The client computes searchable artifacts from its plaintext and sends them alongside the sealed record. The node indexes **opaque** artifacts. The node's own cognition (embed / NER / FTS / Miner / ONNX models) is **never** run on blind records.

Per-owner artifact keys (`k_fts`, `k_graph`, …) are derived from the P2P identity on the client and **never sent to the node**.

## 2. Technique 1 — Blind full-text (BM25) via keyed token-hashes

- **Client:** tokenize plaintext; for each token `t`, compute `h = HMAC-SHA256(k_fts, normalize(t))[:8]` as hex. Send the **ordered multiset** of token-hashes (repeats preserve term frequency).
- **Node:** index the space-joined token-hashes as an FTS document. **BM25 needs only term/document frequencies**, which the hashes preserve — so BM25 works over hashes. Query time: the client sends `HMAC(k_fts, query_token)` hashes; the node runs `MATCH` + BM25 over hashes and surfaces the matching blind record ids into the recall `sealed` list. **The node never sees plaintext tokens or `k_fts`.**

- **⚠️ GOTCHA — do NOT reuse the existing `fts_index`.** It is `fts5(..., tokenize='porter unicode61')` (`storage_fts.rs`). The **porter** stemmer mangles hex tokens: a pure-hex token ending in `ed` (both `e` and `d` are hex) would be wrongly stemmed, corrupting matches. Blind terms **must** be indexed with a **non-stemming** tokenizer.
  → Add a dedicated `CREATE VIRTUAL TABLE blind_fts USING fts5(source_id, owner_hex, terms, tokenize='unicode61')` (no porter) with its own `bm25_search_blind(query_terms, owner)`. Keep it **separate** from the porter `fts_index`.

- **Leakage (honest):** the node sees per-record token-hash **frequency and co-occurrence** patterns — weaker than plaintext FTS, but **not** zero-leak searchable encryption. Mitigations (later): per-epoch `k_fts` rotation, dummy-term padding. Acceptable for v1 (the content itself stays encrypted; this matches the "encrypt content, accept metadata patterns" posture already chosen).

## 3. Technique 2 — Blind graph

- **Client:** extract entities/relations from plaintext; send **entity-hashes** (`HMAC(k_graph, entity)`) and edges `(src_hash, dst_hash, weight)`.
- **Node:** store edges in `memory_edges` (reuse) keyed by hashes; `graph.rs` BFS traverses hashes. The node never sees entity names. Recall graph-expands over hashes → blind record ids → `sealed` list.

## 4. Technique 3 — Blind summary / derived artifacts

- Summaries/reflections are produced by the **client** or a **user-chosen LLM** (consent-gated), **not** the node Miner. A summary is committed as **another sealed record** (blind, encrypted, provenance `derived_from` the source record hashes). The node stores/serves it like any blind record; recall returns it in the `sealed` list. **The node Miner never runs on blind records.**

## 5. Node vs client split

| Capability | Client computes (has plaintext) | Node stores / does (blind) |
|---|---|---|
| Vector | embedding | opaque vector search — **done (Brick 1)** |
| Full-text | keyed token-hashes | `blind_fts` index + BM25 over hashes |
| Graph | entity-hashes + edges | `memory_edges` + BFS over hashes |
| Summary | summary text (client / user-LLM) | store as a sealed **derived** record |

## 6. Brick sequence — ✅ all landed on `main`

- **3a — blind FTS (write + index): ✅ `0a04f59`.** `remember_sealed` accepts `fts_terms` (hex token-hashes); the node indexes them into the non-stemming `blind_fts` table. Storage test included.
- **3b — recall by blind terms: ✅ `460dee2`.** `recall` accepts `query_terms` (hashed); `bm25_search_blind` surfaces blind records into the `sealed` list, deduped with the vector matches.
- **3c — blind graph: ✅ `7534f11` (write) + `a781f72` (recall).** Implemented as **client-declared record↔record edges** — simpler than entity-hashes and reuses `memory_edges`: the client sends `related_records`; `recall` graph-expands via `get_neighbors` into the `sealed` list (gated by `include_graph`). *(Entity-hash graph per §3 remains a possible future refinement.)*
- **3d — blind derived/summary records: ✅ `382a47d`.** `remember_sealed` accepts `derived_from` (source record_ids), stored in an idempotent `blind_provenance` table; record detail returns it. A summary is stored as an ordinary blind record.

## 7. Reuse (don't rebuild)

FTS5 (a **new** non-stemming table beside the porter `fts_index`), `memory_edges` + `graph.rs` BFS, the Brick-1 sealed-record + recall `sealed`-list plumbing, per-owner key derivation from the P2P identity.

## 8. Honest constraints

- **Client-coupled:** the node bricks are inert until the client tokenizes / extracts / hashes / summarizes (Dart work). They are testable node-side with synthetic hashes, but end-to-end value needs the client.
- **Frequency/co-occurrence leakage** on hashed terms/entities (documented; not full SSE/ORAM — matches the v1 posture).
- Blind records **never** touch the node Miner or the node ONNX models (embed/NER/reranker). Reranking of blind results, if wanted, is a client-side or user-LLM step.

Does not touch the VPN core.
