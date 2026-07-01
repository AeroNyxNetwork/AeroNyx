# MemChain Node-Blind Refactor — Audit & Phased Plan

> Status: **Brick 1 IMPLEMENTED on `main`; blind cognition (full-text / graph / provenance) also landed.** Commits: storage foundation `6f49362`, sealed write `c2df34b`, vector recall `a126e37`; blind cognition `0a04f59`→`382a47d` (see `memchain-blind-cognition-design.md`). Remaining, later bricks: metadata minimization, node-to-node replication, encrypted-BM25 hardening. Investigation date 2026-07-01, verified against `main` (commit `ef564ad`).
> Goal: make the node **blind** to user memory — E2E-encrypted, per-identity, content-addressed — the same "nodes can't read your data" property the rest of AeroNyx (VPN / chat / onion relay) already has. Today the MemChain node is **fully sighted**.
> Method: **additive, default-off, brick-by-brick.** Each brick builds green (`cargo test`) and is pushed independently. **The VPN core flow is never touched** (see §5).

---

## 0. TL;DR

- The **data model is already good** and is a stable, widely-depended-on, tested contract — we do **not** fork it. `MemoryRecord` is already content-addressed, owner-signable *by design*, and carries encrypted content.
- The problem is **wiring + location of cognition**, not the structs: (R1) the node derives the content key from a secret **it holds**, (R2) the node **signs** on the user's behalf, (R3) every cognition stage (embeddings, NER, FTS/BM25, graph, Miner, recall) runs on **decrypted plaintext on the node**.
- Making the node blind is therefore a **deep re-architecture**, not a small patch. We stage it. **Brick 1** establishes node-blind *storage + server-side vector recall* behind a default-off flag, leaving the sighted path untouched.
- A key leverage point already exists: the recall **query embedding is client-supplied** (`recall_handler.rs:268`), and vector similarity needs no plaintext — so a blind record that ships its own precomputed embedding stays **semantically searchable on the node** while its content remains opaque.

---

## 1. Verified current state — the node is fully sighted

### 1.1 Data model (sound — keep, do not fork)
- `crates/aeronyx-core/src/ledger/record.rs` — `MemoryRecord` (MRS-1): content-addressed `record_id = SHA-256(owner, timestamp, layer, topic_tags, source_ai, encrypted_content)` (`compute_record_id` `record.rs:275-308`; `verify_id` `:313-323`). Fields already present: `owner:[u8;32]`, `encrypted_content:Vec<u8>`, `signature:[u8;64]` ("Ed25519 over record_id, proving owner authenticity"). `new()` leaves `signature` zeroed for the **owner** to fill. Append-only; runtime metadata (embedding/status/feedback/…) excluded from the hash. Documented as "the most widely depended-upon struct in the entire codebase" with a full test suite.
- `crates/aeronyx-core/src/ledger/block.rs` — `BlockHeader{height,timestamp,prev_block_hash,merkle_root,block_type}`, 81-byte canonical hash (stable contract). `RecordBlock` = header + `Vec<MemoryRecord>`. **No owner / no signature on blocks** → per-node chain.

**The intent was always owner-encryption + owner-signing.** The bug is that the *node* fills those roles.

### 1.2 Write path — node holds the key, encrypts, and signs
- **Owner** comes from the **auth context**, never the request body: `Local` bearer → owner = the **node's own** pubkey; `Remote` Ed25519 → owner = a **client-supplied pubkey** from `X-MemChain-PublicKey`, and the request is Ed25519-signed by it (verified: freshness ±300s + sig over `SHA256(ts‖method‖path‖SHA256(body))`); `SaaS` JWT → owner = `sub`. Consumed at `mpi_handlers.rs:112`, `log_handler.rs:658`.
- **Key (R1):** `derive_record_key` / `derive_rawlog_key` = HKDF-SHA256 over an Ed25519 **private** key (`storage_crypto.rs:61,77`). Single-tenant passes the **node's own** identity: `derive_record_key(&self.identity.to_bytes())` (`server.rs:1203`; rawlog `:640,:1008`, `log_handler.rs:716`). SaaS derives per-user from the **public** key — explicitly *"TENANT ISOLATION, not E2E … the server holds this key and can decrypt"* (`storage_pool.rs:198-201`, `:341-349`). Either way the node can decrypt.
- **Signing (R2):** `record.signature = state.identity.sign(&record.record_id)` (`mpi_handlers.rs:194`, `log_handler.rs:813`). No client signature is ever stored; the Remote-auth request signature authenticates the HTTP request and is discarded.
- **Encryption:** client sends **plaintext**; the handler puts it in `record.encrypted_content` unencrypted (`mpi_handlers.rs:184`), and storage encrypts with the node key inside `insert()` (`storage.rs:1392-1409`, encrypt at `:1399`; deterministic HMAC nonce for dedup, `storage_crypto.rs:100-125`). `record_id` is computed over the **plaintext**.
- **Persistence:** SQLite `records` (`storage.rs:348-371`, insert `:1417-1444`) stores `owner`, `topic_tags`, `source_ai`, `timestamp`, `embedding` as **plaintext** (owner is indexed, `:372`); only `encrypted_content` is ciphertext. FTS5 indexes **plaintext** content. `raw_logs` (`storage.rs:377-389`, insert `storage_ops.rs:137-153`) stores rawlog-encrypted content. (Note: the `/remember` + `/log` write path targets SQLite, **not** the AOF; the AOF `.memchain` is the separate mined Fact/Block ledger — `aof.rs`, writers `miner/reflection.rs`, `server.rs:4049/4108`, `api/local.rs`.)

### 1.3 Recall path — node decrypts and returns plaintext
- `mpi_recall` (`recall_handler.rs:122`) → hybrid retrieval: **vector** (`:268`, query embedding is **client-supplied** — leverage point), **BM25/FTS** (`:282`, reads the plaintext `fts_index`), **graph BFS** (`:398-420`) → RRF+MVF fusion → **cross-encoder rerank on plaintext** (`:662-673`). Content is decrypted upstream in row→record (`storage.rs:1674-1700`, decrypt `:1691`) and returned as **plaintext** JSON (`RecalledMemory.content`). `mode:"index"` truncates to 80 chars but is still plaintext.

### 1.4 Cognition (v2.5.0 +SuperNode) — all on plaintext, on the node
- Three **bundled local ONNX models** via `ort 2.0.0-rc.11`: embedding (MiniLM-L6 / EmbeddingGemma, `embed.rs`), NER (GLiNER, `ner.rs`), cross-encoder reranker (`reranker.rs`). Vector index (`vector.rs`), quantization (`quantize.rs`), MVF scoring (`mvf.rs`), co-occurrence graph (`graph.rs`), and the background **Miner** (`miner/reflection.rs`, `storage_miner.rs`) all consume decrypted content. Plaintext dependency sites include ingest NER/rule-engine (`log_handler.rs`), row decrypt (`storage.rs:1691`, `storage_ops.rs:180,706`), and **FTS build which inserts plaintext into `fts_index`** (`storage_fts.rs:570-589`).
- **SuperNode LLM** (`llm_openai.rs`, `llm_anthropic.rs`, `llm_router.rs`): outbound only when enabled; **default-off**; uses the **node's own** API key (not per-user); `PrivacyLevel` default `Structured` = metadata only, `Full` (raw turns) requires an explicit whitelist. Separable and already gated.

### 1.5 Wire & sync (context for later bricks)
- `MemChainMessage` (`crates/aeronyx-core/src/protocol/memchain.rs:130-296`): 18 bincode variants — **discriminant order is a hard contract, append-only, never reorder**. Includes `BroadcastRecord(MemoryRecord)`, `SyncRecordRequest{owner,after}`, `SyncRecordResponse{records}` — the receive path already **verifies the record signature + `verify_id()`** before storing (`server.rs:4052-4144`). `BlockAnnounce` is a **log-only stub** (`server.rs:4145-4151`).
- **Important:** today's "sync" is **node ⇄ its own connected clients** (`broadcast_to_all_sessions` → `sessions.all_sessions()`), **not** node-to-node. Genuine inter-node memory replication is **not** wired (only chat has a real peer path via `chat_peer.rs`/`peer_store`). All MemChain feature flags default **false**, so upgrading nodes see no behavior change.

---

## 2. Root causes

| # | Root cause | Anchor |
|---|---|---|
| R1 | Node derives the content key from a secret it holds → can decrypt everything | `storage_crypto.rs:61`, `server.rs:1203`, `storage_pool.rs:201` |
| R2 | Node signs records with its own identity → not owner-authored | `mpi_handlers.rs:194`, `log_handler.rs:813` |
| R3 | All cognition + recall run on node-decrypted plaintext → blindness ⇒ relocate cognition | `storage.rs:1691`, `storage_fts.rs:585`, `miner/reflection.rs`, `recall_handler.rs:662` |

Secondary: deterministic record nonce leaks content-equality (`storage_crypto.rs:100`); `owner`/`topic_tags`/`source_ai`/`timestamp`/`embedding` stored plaintext + indexed (metadata leak beyond content); `record_id` hashes plaintext.

---

## 3. Target

Per-identity, E2E, node-blind memory: the **client** holds the key (derived from its P2P identity), **encrypts** and **signs**; the **node** stores opaque ciphertext + **verifies** the client signature and never holds the key or reads plaintext. Cognition (extraction / embedding / summary / recall-synthesis) relocates to the **client** or a **user-chosen LLM** (BYOK / self-hosted AI node), consent-gated. Durability comes from **replication** of content-addressed ciphertext across the owner's devices + redundant blind nodes — not from consensus.

---

## 4. The honest scope

Node-blindness is **load-bearing all the way down** (it lives at the key-derivation layer). Making the node blind touches the entire write path, storage row-conversion, **FTS5/BM25 (which fundamentally needs plaintext to tokenize)**, the cognitive graph, the whole Miner, and recall's rerank + response. This is months of staged work, not a patch.

**Hard blockers** (need client-side pre-processing or different crypto): FTS/BM25 over ciphertext, NER-driven graph, Miner LLM summaries.
**Leverage already present:** client-supplied query embedding (`recall_handler.rs:268`), content-agnostic quantization, metadata-only MVF scoring, and a wire that already carries + verifies client-authored signed encrypted records.

---

## 5. VPN core — **DO NOT TOUCH**

Never modify (pure VPN tunnel: TUN / UDP / per-packet crypto / session / routing / handshake / IP alloc):

- Entire `crates/aeronyx-transport/` crate.
- `crates/aeronyx-core/src/crypto/{transport,handshake,kdf,keys,mod}.rs`; `crates/aeronyx-core/src/protocol/{codec,messages,auth}.rs`.
- `crates/aeronyx-server/src/services/{session,routing,handshake,ip_pool,traffic_tracker,node_policy,deny_list}.rs`.
- `crates/aeronyx-server/src/api/vpn_health.rs` (read-only status).

**Shared files — edit only the MemChain regions:**
- `handlers/packet.rs`: touch **only** the `0xAE` MemChain arm (`422-470`). Never the VPN/TUN arms (`369-420`, `526-597`), the dispatch order, or the magic constants (`MEMCHAIN_MAGIC=0xAE`, `VOICE_MAGIC=0xAF`).
- `server.rs`: MemChain is `handle_memchain_message` (`4014-4441`) + the `MemChain` match arm (`3984-3994`). Never the VPN→TUN write (`3909-3912`) or TUN→UDP egress (`4447+`).
- `protocol/memchain.rs` (both copies): append-only variants, never reorder.

Safe MemChain territory: `services/memchain/**`, `miner/**`, `ledger/**`, `api/{mpi*,local,chat*,recall*}`, `config_memchain.rs`.

---

## 6. Phased plan (brick-by-brick — additive, default-off, each green + pushed)

1. **Brick 1 — node-blind storage + server-side vector recall (this cycle).** New default-off flag. A "blind" record class where the **client** supplies: the E2E-encrypted content blob (node never decrypts), the **precomputed embedding**, the owner P2P-identity pubkey, and the **record signature**. The node **verifies the client signature**, stores the blob + embedding opaquely (**no node encryption, no re-signing**), skips NER/FTS/graph/Miner for blind records, and on recall returns the **ciphertext blob** (`sealed:true`) for client-side decryption. Vector search still runs (client-supplied embeddings). Sighted path 100% untouched → backward compatible.
2. **Brick 2 — metadata minimization.** Reduce the plaintext `owner`/`topic_tags`/`source_ai` leak for blind records (move discretionary metadata into the sealed blob; keep only what indexing needs).
3. **Brick 3 — relocate cognition.** Client-side (or user-LLM) embedding + extraction + summary at ingest, consent-gated; the node stops needing plaintext for blind records end-to-end.
4. **Brick 4 — real replication (durability).** Turn client↔node record sync into per-owner, content-addressed **have/want** replication across the owner's devices + redundant blind nodes (`BroadcastRecord`/`SyncRecordRequest` are the seed; `BlockAnnounce` stub → real fetch).
5. **Brick 5 — encrypted keyword search.** The hard one: client-side FTS or an encrypted/searchable index to replace server-side BM25 for blind records.

Each brick maps to the broader per-identity E2E design; none requires touching the sighted path until it is fully superseded and explicitly removed (with consent).

---

## 7. Brick 1 — concrete change set (all in safe territory)

- `config_memchain.rs`: add `blind_storage_enabled: bool` (`#[serde(default)]` → **false**), mirroring the SuperNode flag pattern.
- Write DTO (`mpi_handlers.rs` `RememberRequest`, and/or a dedicated endpoint): optional `sealed_content_b64`, `embedding: Vec<f32>`, `owner_pubkey_hex`, `record_signature_b64`.
- Write branch: when a blind record is supplied → build `MemoryRecord` with `encrypted_content = client ciphertext`, `owner = authenticated client pubkey`, `signature = client signature`; **verify** the Ed25519 signature over `record_id`; **skip** node encryption and re-signing; **skip** NER / FTS / rule-engine.
- Storage: a "store-as-sealed" path guarding `storage.rs:1392` so already-sealed content is **not** re-encrypted; still store the client embedding for vector search.
- Recall branch: for blind records, return the ciphertext (base64) + `sealed:true` instead of decrypting.
- `record_id` contract: blind records are content-addressed over the **ciphertext** (client computes; node verifies) — distinguished from sighted records by the blind flag/field so the existing plaintext-hash contract is unchanged.
- Tests: seal→store→verify-signature→vector-recall→return-ciphertext roundtrip; assert the node cannot decrypt; assert sighted path unchanged.

---

## 8. Risks / verify during implementation

- **Double-encryption:** confirm whether the P2P `BroadcastRecord` / `SyncRecordResponse` `insert()` re-encrypts already-sealed `encrypted_content` when a node `record_key` is present (`storage.rs:1392`). Brick 1's "store-as-sealed" guard must cover this path too.
- **`record_id` divergence:** sighted = hash over plaintext; blind = hash over ciphertext. Keep them cleanly distinguished so dedup/verify contracts don't collide.
- **Metadata leak persists** until Brick 2 — documented, not hidden.
- **SaaS vs Local vs Remote:** Brick 1 targets the Remote/blind path; Local (owner = node) and SaaS (server-derived key) semantics are unchanged and remain sighted by design until later bricks.
