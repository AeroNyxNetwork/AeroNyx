// [ARCH-SPLIT 2026-10-02]
// Create and migrate the SQLite schema. MemoryStorage::open calls this before any read.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl MemoryStorage {
    pub(super) fn create_schema(conn: &Connection) -> Result<(), String> {
        // ── v1-v4: Original tables (records, raw_logs, memory_edges, etc.) ──
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS records (
                record_id           BLOB PRIMARY KEY,
                owner               BLOB NOT NULL,
                timestamp           INTEGER NOT NULL,
                layer               INTEGER NOT NULL,
                topic_tags          TEXT NOT NULL DEFAULT '[]',
                source_ai           TEXT NOT NULL DEFAULT '',
                status              INTEGER NOT NULL DEFAULT 0,
                supersedes          BLOB,
                encrypted_content   BLOB NOT NULL DEFAULT x'',
                embedding           BLOB,
                embedding_model     TEXT NOT NULL DEFAULT '',
                embedding_dim       INTEGER NOT NULL DEFAULT 0,
                signature           BLOB NOT NULL,
                access_count        INTEGER NOT NULL DEFAULT 0,
                created_at          INTEGER NOT NULL,
                archived_at         INTEGER,
                positive_feedback   INTEGER NOT NULL DEFAULT 0,
                negative_feedback   INTEGER NOT NULL DEFAULT 0,
                conflict_with       BLOB,
                project_id          TEXT,
                session_id          TEXT,
                episode_id          TEXT,
                blind               INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_owner ON records(owner);
            CREATE INDEX IF NOT EXISTS idx_owner_layer_status ON records(owner, layer, status);
            CREATE INDEX IF NOT EXISTS idx_status_layer ON records(status, layer);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp);

            -- v8: node-blind commitment chain. Blocks intentionally contain
            -- only opaque record ids; memory owners and ciphertext payloads
            -- remain in the separately authorised records table.
            CREATE TABLE IF NOT EXISTS record_commitment_blocks (
                height              INTEGER PRIMARY KEY CHECK(height > 0),
                block_hash          BLOB NOT NULL UNIQUE CHECK(length(block_hash) = 32),
                chain_id            BLOB NOT NULL CHECK(length(chain_id) = 32),
                protocol_version    INTEGER NOT NULL,
                timestamp           INTEGER NOT NULL,
                prev_block_hash     BLOB NOT NULL CHECK(length(prev_block_hash) = 32),
                merkle_root         BLOB NOT NULL CHECK(length(merkle_root) = 32),
                record_count        INTEGER NOT NULL,
                proposer            BLOB NOT NULL CHECK(length(proposer) = 32),
                proposer_signature  BLOB NOT NULL CHECK(length(proposer_signature) = 64),
                payload             BLOB NOT NULL,
                received_from       BLOB,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_record_blocks_hash
                ON record_commitment_blocks(block_hash);

            CREATE TABLE IF NOT EXISTS record_block_commitments (
                record_id       BLOB PRIMARY KEY CHECK(length(record_id) = 32),
                block_height    INTEGER NOT NULL,
                FOREIGN KEY(block_height) REFERENCES record_commitment_blocks(height)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_block_commitments_height
                ON record_block_commitments(block_height);

            -- v9: bounded local proof evidence. The exact signed peer frame is
            -- retained for operator-side verification, but never leaves the
            -- node through status, heartbeat, logs, or public protocol APIs.
            CREATE TABLE IF NOT EXISTS record_checkpoint_evidence (
                evidence_digest    BLOB PRIMARY KEY CHECK(length(evidence_digest) = 32),
                observed_at        INTEGER NOT NULL CHECK(observed_at >= 0),
                relation           TEXT NOT NULL CHECK(relation IN
                    ('converged','remote_ahead','remote_behind','diverged')),
                local_tip_height   INTEGER NOT NULL CHECK(local_tip_height >= 0),
                remote_tip_height  INTEGER NOT NULL CHECK(remote_tip_height >= 0),
                checkpoint_height  INTEGER NOT NULL CHECK(checkpoint_height >= 0),
                signed_response    BLOB NOT NULL CHECK(length(signed_response) > 0
                    AND length(signed_response) <= 4096),
                created_at         INTEGER NOT NULL CHECK(created_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_record_checkpoint_evidence_observed
                ON record_checkpoint_evidence(observed_at);

            -- v10: compact, append-only proof that one explicitly trusted
            -- witness signed incompatible hashes for the same chain position.
            -- The referenced signed frames remain in the bounded v9 vault and
            -- cannot be pruned while an incident depends on them.
            CREATE TABLE IF NOT EXISTS record_checkpoint_equivocations (
                responder               BLOB NOT NULL CHECK(length(responder) = 32),
                conflict_scope          TEXT NOT NULL CHECK(conflict_scope IN
                    ('checkpoint','tip')),
                conflict_height         INTEGER NOT NULL CHECK(conflict_height >= 0),
                first_evidence_digest   BLOB NOT NULL CHECK(length(first_evidence_digest) = 32),
                second_evidence_digest  BLOB NOT NULL CHECK(length(second_evidence_digest) = 32),
                detected_at             INTEGER NOT NULL CHECK(detected_at >= 0),
                PRIMARY KEY (responder, conflict_scope, conflict_height),
                CHECK(first_evidence_digest != second_evidence_digest),
                FOREIGN KEY(first_evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                    ON DELETE RESTRICT,
                FOREIGN KEY(second_evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_checkpoint_equivocations_detected
                ON record_checkpoint_equivocations(detected_at);

            -- v11: first durable proof that one operator-pinned witness signed
            -- a shared checkpoint inconsistent with the locally audited chain.
            -- Later convergence cannot overwrite this incident.
            CREATE TABLE IF NOT EXISTS record_checkpoint_trusted_divergences (
                responder          BLOB PRIMARY KEY CHECK(length(responder) = 32),
                evidence_digest    BLOB NOT NULL UNIQUE CHECK(length(evidence_digest) = 32),
                checkpoint_height  INTEGER NOT NULL CHECK(checkpoint_height >= 0),
                detected_at        INTEGER NOT NULL CHECK(detected_at >= 0),
                FOREIGN KEY(evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_checkpoint_trusted_divergences_detected
                ON record_checkpoint_trusted_divergences(detected_at);

            -- v12: one immutable threshold certificate per retained local
            -- checkpoint. Members reference exact v9 signed response frames;
            -- no synthetic aggregate signature or peer-count consensus exists.
            CREATE TABLE IF NOT EXISTS record_checkpoint_certificates (
                checkpoint_height  INTEGER PRIMARY KEY CHECK(checkpoint_height > 0),
                chain_id           BLOB NOT NULL CHECK(length(chain_id) = 32),
                checkpoint_hash    BLOB NOT NULL CHECK(length(checkpoint_hash) = 32),
                certificate_digest BLOB NOT NULL UNIQUE CHECK(length(certificate_digest) = 32),
                required_signers   INTEGER NOT NULL CHECK(required_signers BETWEEN 2 AND 3),
                signer_count       INTEGER NOT NULL CHECK(signer_count BETWEEN required_signers AND 3),
                certified_at       INTEGER NOT NULL CHECK(certified_at >= 0),
                FOREIGN KEY(checkpoint_height) REFERENCES record_commitment_blocks(height)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_checkpoint_certificates_time
                ON record_checkpoint_certificates(certified_at);

            CREATE TABLE IF NOT EXISTS record_checkpoint_certificate_members (
                checkpoint_height INTEGER NOT NULL,
                responder         BLOB NOT NULL CHECK(length(responder) = 32),
                evidence_digest   BLOB NOT NULL UNIQUE CHECK(length(evidence_digest) = 32),
                PRIMARY KEY(checkpoint_height, responder),
                FOREIGN KEY(checkpoint_height) REFERENCES record_checkpoint_certificates(checkpoint_height)
                    ON DELETE CASCADE,
                FOREIGN KEY(evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_checkpoint_certificate_members_height
                ON record_checkpoint_certificate_members(checkpoint_height);

            -- v13: one durable exclusive process lease per configured
            -- coordinator identity on this follower witness. No endpoint,
            -- host identity, process id, or memory/user data is retained.
            CREATE TABLE IF NOT EXISTS record_coordinator_leases (
                coordinator      BLOB PRIMARY KEY CHECK(length(coordinator) = 32),
                chain_id         BLOB NOT NULL CHECK(length(chain_id) = 32),
                instance_id      BLOB NOT NULL CHECK(length(instance_id) = 32),
                lease_epoch      INTEGER NOT NULL CHECK(lease_epoch > 0),
                lease_expires_at INTEGER NOT NULL CHECK(lease_expires_at >= 0),
                updated_at       INTEGER NOT NULL CHECK(updated_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_record_coordinator_leases_expiry
                ON record_coordinator_leases(lease_expires_at);

            -- v14: one aggregate-only high-water decision per admitted node.
            -- The digest commits a requester-signed local anchor without
            -- exposing its delivery count, timestamp, routes, or payload data.
            CREATE TABLE IF NOT EXISTS verified_delivery_anchor_witnesses (
                requester      BLOB PRIMARY KEY CHECK(length(requester) = 32),
                generation     INTEGER NOT NULL CHECK(generation > 0),
                anchor_digest  BLOB NOT NULL CHECK(length(anchor_digest) = 32),
                observed_at    INTEGER NOT NULL CHECK(observed_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_verified_delivery_anchor_witnesses_observed
                ON verified_delivery_anchor_witnesses(observed_at);

            -- v15: replayable coordinator authority history. Each transition
            -- is dual-signed, anchored to one existing commitment prefix, and
            -- retained permanently so cold followers can verify the proposer
            -- authorised at every historical block height. This table stores
            -- no memory owner, payload, route, endpoint, or user metadata.
            CREATE TABLE IF NOT EXISTS record_coordinator_handovers (
                authority_epoch      INTEGER PRIMARY KEY CHECK(authority_epoch > 0),
                activation_height    INTEGER NOT NULL UNIQUE CHECK(activation_height > 1),
                previous_height      INTEGER NOT NULL UNIQUE CHECK(previous_height > 0),
                chain_id             BLOB NOT NULL CHECK(length(chain_id) = 32),
                protocol_version     INTEGER NOT NULL,
                previous_tip_hash    BLOB NOT NULL CHECK(length(previous_tip_hash) = 32),
                previous_coordinator BLOB NOT NULL CHECK(length(previous_coordinator) = 32),
                next_coordinator     BLOB NOT NULL CHECK(length(next_coordinator) = 32),
                authorization_id     BLOB NOT NULL UNIQUE CHECK(length(authorization_id) = 16),
                issued_at            INTEGER NOT NULL CHECK(issued_at > 0),
                previous_signature   BLOB NOT NULL CHECK(length(previous_signature) = 64),
                next_signature       BLOB NOT NULL CHECK(length(next_signature) = 64),
                payload              BLOB NOT NULL CHECK(length(payload) > 0 AND length(payload) <= 1024),
                accepted_at          INTEGER NOT NULL CHECK(accepted_at > 0),
                CHECK(activation_height = previous_height + 1),
                FOREIGN KEY(previous_height) REFERENCES record_commitment_blocks(height)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_coordinator_handovers_activation
                ON record_coordinator_handovers(activation_height);

            -- v16: one opaque custody-checkpoint high-water decision per
            -- producer. This namespace is intentionally independent from the
            -- delivery-cache witness table because their generations have no
            -- semantic relationship.
            CREATE TABLE IF NOT EXISTS custody_audit_anchor_witnesses (
                producer       BLOB PRIMARY KEY CHECK(length(producer) = 32),
                generation     INTEGER NOT NULL CHECK(generation > 0),
                frame_sha256   BLOB NOT NULL CHECK(length(frame_sha256) = 32),
                observed_at    INTEGER NOT NULL CHECK(observed_at > 0)
            );
            CREATE INDEX IF NOT EXISTS idx_custody_audit_anchor_witnesses_observed
                ON custody_audit_anchor_witnesses(observed_at);

            -- v17: immutable producer-side portable witness receipts. The
            -- exact bounded frame is retained for restart-time signature and
            -- column revalidation; no custody counts or payload data enter it.
            CREATE TABLE IF NOT EXISTS custody_audit_witness_receipt_evidence (
                receipt_digest        BLOB PRIMARY KEY CHECK(length(receipt_digest) = 32),
                producer              BLOB NOT NULL CHECK(length(producer) = 32),
                witness               BLOB NOT NULL CHECK(length(witness) = 32),
                requested_generation  INTEGER NOT NULL CHECK(requested_generation > 0),
                requested_frame_sha256 BLOB NOT NULL CHECK(length(requested_frame_sha256) = 32),
                retained_generation   INTEGER NOT NULL CHECK(retained_generation > 0),
                retained_frame_sha256 BLOB NOT NULL CHECK(length(retained_frame_sha256) = 32),
                outcome               INTEGER NOT NULL CHECK(outcome BETWEEN 0 AND 4),
                observed_at           INTEGER NOT NULL CHECK(observed_at > 0),
                receipt_frame         BLOB NOT NULL CHECK(length(receipt_frame) > 0
                    AND length(receipt_frame) <= 320),
                persisted_at          INTEGER NOT NULL CHECK(persisted_at > 0),
                admission_kind        INTEGER NOT NULL DEFAULT 0
                    CHECK(admission_kind BETWEEN 0 AND 1),
                admission_max_delay_secs INTEGER NOT NULL DEFAULT 60
                    CHECK(admission_max_delay_secs BETWEEN 60 AND 604800),
                CHECK(producer != witness)
            );
            CREATE INDEX IF NOT EXISTS idx_custody_witness_receipts_anchor
                ON custody_audit_witness_receipt_evidence(
                    producer, requested_generation, requested_frame_sha256, observed_at
                );
            CREATE INDEX IF NOT EXISTS idx_custody_witness_receipts_observed
                ON custody_audit_witness_receipt_evidence(observed_at);

            CREATE TABLE IF NOT EXISTS raw_logs (
                log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                turn_index      INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         BLOB NOT NULL,
                source_ai       TEXT NOT NULL DEFAULT '',
                recall_context  TEXT DEFAULT NULL,
                extractable     INTEGER DEFAULT 1,
                feedback_signal INTEGER DEFAULT NULL,
                encrypted       INTEGER DEFAULT 0,
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rawlogs_session ON raw_logs(session_id, turn_index);
            CREATE INDEX IF NOT EXISTS idx_rawlogs_feedback ON raw_logs(feedback_signal);

            CREATE TABLE IF NOT EXISTS memory_edges (
                source_id BLOB NOT NULL, target_id BLOB NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'co_occurred',
                weight REAL NOT NULL DEFAULT 1.0, created_at INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);

            CREATE TABLE IF NOT EXISTS user_weights (
                owner BLOB PRIMARY KEY, weights BLOB NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner BLOB NOT NULL, memory_id BLOB NOT NULL,
                session_id TEXT NOT NULL, turn_index INTEGER NOT NULL,
                signal INTEGER NOT NULL, features BLOB, prediction REAL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_owner ON memory_feedback(owner);
            CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);

            CREATE TABLE IF NOT EXISTS chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);",
        )
        .map_err(|e| format!("Schema creation failed (base tables): {}", e))?;

        // ── v5 (v2.4.0): Three-layer cognitive graph tables ──
        conn.execute_batch(
            "-- Episode layer: complete original conversations (non-lossy)
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                episode_type        TEXT NOT NULL,
                source              TEXT NOT NULL,
                session_id          TEXT,
                encrypted_content   BLOB NOT NULL,
                content_hash        TEXT NOT NULL,
                embedding           BLOB,
                token_count         INTEGER,
                created_at          INTEGER NOT NULL,
                ingested_at         INTEGER NOT NULL,
                metadata_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_owner ON episodes(owner, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(owner, episode_type);

            -- Semantic Entity layer: GLiNER-extracted entity nodes
            CREATE TABLE IF NOT EXISTS entities (
                entity_id           TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                name_normalized     TEXT NOT NULL,
                entity_type         TEXT NOT NULL,
                description         TEXT,
                embedding           BLOB,
                community_id        TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL,
                mention_count       INTEGER DEFAULT 1,
                metadata_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entities_owner ON entities(owner);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(owner, entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(owner, name_normalized);
            CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(owner, community_id);

            -- Semantic Entity layer: temporal knowledge edges
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                edge_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                owner               BLOB NOT NULL,
                source_id           TEXT NOT NULL,
                target_id           TEXT NOT NULL,
                relation_type       TEXT NOT NULL,
                fact_text           TEXT,
                weight              REAL DEFAULT 1.0,
                confidence          REAL DEFAULT 1.0,
                embedding           BLOB,
                valid_from          INTEGER NOT NULL,
                valid_until         INTEGER,
                episode_id          TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kedges_source ON knowledge_edges(owner, source_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_kedges_target ON knowledge_edges(owner, target_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_kedges_valid ON knowledge_edges(owner, valid_until);
            CREATE INDEX IF NOT EXISTS idx_kedges_episode ON knowledge_edges(episode_id);

            -- Bridge layer: Episode ↔ Entity bidirectional links
            CREATE TABLE IF NOT EXISTS episode_edges (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                owner               BLOB NOT NULL,
                episode_id          TEXT NOT NULL,
                entity_id           TEXT NOT NULL,
                role                TEXT NOT NULL,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ep_edges_episode ON episode_edges(episode_id);
            CREATE INDEX IF NOT EXISTS idx_ep_edges_entity ON episode_edges(entity_id);

            -- Community layer: auto-clustering via label propagation
            CREATE TABLE IF NOT EXISTS communities (
                community_id        TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                summary             TEXT,
                description         TEXT,
                entity_count        INTEGER DEFAULT 0,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_communities_owner ON communities(owner);

            -- Projects: Community specialization for code projects
            CREATE TABLE IF NOT EXISTS projects (
                project_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                status              TEXT DEFAULT 'active',
                community_id        TEXT NOT NULL,
                summary             TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL,
                last_active_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner, status);
            CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(owner, last_active_at DESC);

            -- Sessions: conversation session metadata (v2.5.0: added title column)
            CREATE TABLE IF NOT EXISTS sessions (
                session_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                project_id          TEXT,
                session_type        TEXT DEFAULT 'chat',
                started_at          INTEGER NOT NULL,
                ended_at            INTEGER,
                turn_count          INTEGER DEFAULT 0,
                title               TEXT,
                summary             TEXT,
                key_decisions       TEXT,
                files_touched       TEXT,
                entities_extracted  INTEGER DEFAULT 0,
                summary_generated   INTEGER DEFAULT 0,
                artifacts_extracted INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_pending ON sessions(entities_extracted, summary_generated);

            -- Artifacts: code/document artifacts with version chains
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id         TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                session_id          TEXT NOT NULL,
                project_id          TEXT,
                artifact_type       TEXT NOT NULL,
                filename            TEXT,
                language            TEXT,
                version             INTEGER DEFAULT 1,
                parent_id           TEXT,
                encrypted_content   BLOB NOT NULL,
                content_hash        TEXT NOT NULL,
                embedding           BLOB,
                line_count          INTEGER,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id, filename, version DESC);
            CREATE INDEX IF NOT EXISTS idx_artifacts_file ON artifacts(owner, filename, version DESC);"
        ).map_err(|e| format!("Schema creation failed (v5 cognitive graph tables): {}", e))?;

        // ── v2.4.0+BM25: Full-text search index ──
        // FTS5 content table indexes text from records, entities, and sessions
        // for BM25 keyword matching in hybrid recall.
        //
        // Design: external content table (contentless) — FTS5 stores only the
        // inverted index, not the original text. This saves ~40% space vs
        // content-based FTS5. Queries join back to source tables for content.
        //
        // Columns:
        //   source_type: 'record' | 'entity' | 'session' — identifies source table
        //   source_id: record_id hex / entity_id / session_id — for join-back
        //   owner_hex: owner public key hex — for access control filtering
        //   content: searchable text content
        //   tags: topic_tags or entity_type for boosting
        match conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                source_type,
                source_id,
                owner_hex,
                content,
                tags,
                tokenize='porter unicode61'
            );",
        ) {
            Ok(_) => info!("[STORAGE] ✅ FTS5 index ready"),
            Err(e) => warn!("[STORAGE] ⚠️ FTS5 creation failed (BM25 disabled): {}", e),
        }

        // ── node-blind full-text: a SEPARATE FTS5 index over client-supplied
        //    keyed token-hashes. NON-stemming (unicode61, NOT porter) — porter
        //    would mangle hex token-hashes (e.g. a hash ending in "ed"). Created
        //    idempotently on every open, so no schema-version bump is needed.
        //   source_id: record_id hex; owner_hex: access control;
        //   terms: space-joined HMAC(k_fts, token) hex hashes supplied by the client.
        match conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS blind_fts USING fts5(
                source_id,
                owner_hex,
                terms,
                tokenize='unicode61'
            );",
        ) {
            Ok(_) => info!("[STORAGE] ✅ blind FTS5 index ready"),
            Err(e) => warn!("[STORAGE] ⚠️ blind FTS5 creation failed: {}", e),
        }

        // ── node-blind derived-record provenance (Brick 3d) ──
        // Links a derived record (e.g. a client/LLM summary) to the source
        // record_ids it was built from. Opaque hexes only — the node learns the
        // provenance DAG shape, never the content. Idempotent, no version bump.
        let _ = conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS blind_provenance (
                record_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                owner_hex TEXT NOT NULL,
                PRIMARY KEY (record_id, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_blind_prov_owner ON blind_provenance(owner_hex);",
        );

        // ── v6 (v2.5.0-SuperNode): Cognitive task queue + LLM usage log ──
        // cognitive_tasks: async LLM task queue processed by TaskWorker
        // llm_usage_log: per-call token usage + latency for cost tracking
        //
        // Design notes:
        //   - Tasks are claimed atomically (UPDATE ... WHERE status='pending' LIMIT 1)
        //   - result / prompt_messages stored as JSON TEXT for flexibility
        //   - token_usage stored as JSON: {"input": N, "output": N, "cached": N}
        //   - cost_usd NOT stored — computed dynamically at query time from token counts
        //     so rate changes don't affect historical accuracy
        conn.execute_batch(
            "-- Cognitive task queue (SuperNode async LLM tasks)
            CREATE TABLE IF NOT EXISTS cognitive_tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type       TEXT NOT NULL,
                priority        INTEGER DEFAULT 5,
                status          TEXT DEFAULT 'pending',
                payload         TEXT NOT NULL,
                result          TEXT,
                prompt_messages TEXT,
                target_table    TEXT,
                target_id       TEXT,
                privacy_level   TEXT DEFAULT 'structured',
                provider_used   TEXT,
                model_used      TEXT,
                token_usage     TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                completed_at    INTEGER,
                retry_count     INTEGER DEFAULT 0,
                max_retries     INTEGER DEFAULT 3,
                error_message   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ct_status ON cognitive_tasks(status, priority DESC, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_ct_target ON cognitive_tasks(target_table, target_id, task_type);
            CREATE INDEX IF NOT EXISTS idx_ct_type ON cognitive_tasks(task_type, status);

            -- LLM usage log (token counts + latency per call)
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER,
                provider        TEXT NOT NULL,
                model           TEXT NOT NULL,
                input_tokens    INTEGER NOT NULL,
                output_tokens   INTEGER NOT NULL,
                cached_tokens   INTEGER DEFAULT 0,
                latency_ms      INTEGER NOT NULL,
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_time ON llm_usage_log(created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_provider ON llm_usage_log(provider, created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_task ON llm_usage_log(task_id);"
        ).map_err(|e| format!("Schema creation failed (v6 SuperNode tables): {}", e))?;

        // [MEMORY-SEALED-V2 2026-10-02 by Codex] Additive opaque storage for
        // V2. create_schema runs on every open, so existing V1 databases gain
        // this table without rewriting or restricting legacy rows. No
        // plaintext metadata, vectors, FTS terms, or relationship columns are
        // admitted here.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_sealed_v2 (
                record_id   BLOB PRIMARY KEY CHECK(length(record_id) = 32),
                owner       BLOB NOT NULL CHECK(length(owner) = 32),
                created_at  INTEGER NOT NULL CHECK(created_at >= 0),
                envelope    BLOB NOT NULL CHECK(length(envelope) BETWEEN 37 AND 16421),
                signature   BLOB NOT NULL CHECK(length(signature) = 64),
                status      INTEGER NOT NULL DEFAULT 0 CHECK(status IN (0, 2)),
                inserted_at INTEGER NOT NULL CHECK(inserted_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_sealed_v2_owner_cursor
                ON memory_sealed_v2(owner, record_id)
                WHERE status = 0;",
        )
        .map_err(|e| format!("Schema creation failed (V2 sealed memory): {e}"))?;

        // Insert schema version if not present
        let existing: Option<u32> = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(|e| format!("Read schema version: {}", e))?;

        if existing.is_none() {
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?1)",
                params![SCHEMA_VERSION],
            )
            .map_err(|e| format!("Insert schema version: {}", e))?;
        }

        Ok(())
    }

    pub(super) fn maybe_migrate(conn: &Connection) -> Result<(), String> {
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(1);

        // v1 → v2: embedding column
        if current < 2 {
            info!("[STORAGE] Migrating schema v{} → v2", current);
            let has_embedding = conn
                .prepare("SELECT embedding FROM records LIMIT 0")
                .is_ok();
            if !has_embedding {
                let _ = conn.execute_batch("ALTER TABLE records ADD COLUMN embedding BLOB;");
                info!("[STORAGE] Added `embedding` column");
            }
            // ⚠️ hardcoded 2, not SCHEMA_VERSION — prevents skipping v4/v5/v6
            conn.execute("UPDATE schema_version SET version = 2", [])
                .map_err(|e| format!("Update schema version to v2: {}", e))?;
            info!("[STORAGE] ✅ Migration to v2 complete");
        }

        // v2 → v4: MVF feedback + conflict
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(2);

        if current < 4 {
            info!("[STORAGE] Migrating schema v{} → v4", current);

            if conn
                .prepare("SELECT positive_feedback FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN positive_feedback INTEGER NOT NULL DEFAULT 0;",
                )
                .map_err(|e| format!("Add positive_feedback: {}", e))?;
            }
            if conn
                .prepare("SELECT negative_feedback FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN negative_feedback INTEGER NOT NULL DEFAULT 0;",
                )
                .map_err(|e| format!("Add negative_feedback: {}", e))?;
            }
            if conn
                .prepare("SELECT conflict_with FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN conflict_with BLOB;")
                    .map_err(|e| format!("Add conflict_with: {}", e))?;
            }

            // ⚠️ hardcoded 4, not SCHEMA_VERSION
            conn.execute("UPDATE schema_version SET version = 4", [])
                .map_err(|e| format!("Update schema version to v4: {}", e))?;
            info!("[STORAGE] ✅ Migration to v4 complete");
        }

        // v4 → v5 (v2.4.0-GraphCognition): Three-layer cognitive graph
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(4);

        if current < 5 {
            info!(
                "[STORAGE] Migrating schema v{} → v5 (cognitive graph)",
                current
            );

            // 5a: ALTER records — add project_id, session_id, episode_id
            if conn
                .prepare("SELECT project_id FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN project_id TEXT;")
                    .map_err(|e| format!("Add project_id to records: {}", e))?;
                info!("[STORAGE] Added records.project_id");
            }
            if conn
                .prepare("SELECT session_id FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN session_id TEXT;")
                    .map_err(|e| format!("Add session_id to records: {}", e))?;
                info!("[STORAGE] Added records.session_id");
            }
            if conn
                .prepare("SELECT episode_id FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN episode_id TEXT;")
                    .map_err(|e| format!("Add episode_id to records: {}", e))?;
                info!("[STORAGE] Added records.episode_id");
            }

            // 5b: Create indexes for new records columns
            let _ = conn.execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_records_project ON records(project_id, timestamp DESC);
                 CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
                 CREATE INDEX IF NOT EXISTS idx_records_episode ON records(episode_id);"
            );

            // 5c: Create v5 tables (IF NOT EXISTS — safe if create_schema already ran)
            let v5_tables = [
                "episodes",
                "entities",
                "knowledge_edges",
                "episode_edges",
                "communities",
                "projects",
                "sessions",
                "artifacts",
            ];
            for table in &v5_tables {
                let exists: bool = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        params![table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;

                if !exists {
                    warn!(
                        "[STORAGE] v5 table '{}' missing after create_schema — \
                         this is expected when upgrading from v2.3.0. \
                         Table will be created by create_schema on next restart.",
                        table
                    );
                }
            }

            // 5d: Migrate memory_edges → knowledge_edges
            //
            // ⚠️ BUG FIX (v2.5.2+Provenance): the original migration used
            // `source_id` (a BLOB record_id) as the `owner` column in the INSERT.
            // knowledge_edges.owner must be a 32-byte Ed25519 public key, NOT a
            // record_id. Records in memory_edges do not carry an owner column.
            //
            // Fix: join records on source_id to look up the actual owner.
            // Edges whose source_id has no matching record are skipped.
            {
                let migrated: bool = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = 'memory_edges_migrated_v5'",
                        [],
                        |row| {
                            let v: Vec<u8> = row.get(0)?;
                            Ok(v == b"1")
                        },
                    )
                    .unwrap_or(false);

                if !migrated {
                    let edge_count: i64 = conn
                        .query_row("SELECT COUNT(*) FROM memory_edges", [], |row| row.get(0))
                        .unwrap_or(0);

                    if edge_count > 0 {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64;

                        let ke_exists: bool = conn.query_row(
                            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='knowledge_edges'",
                            [],
                            |row| row.get::<_, i64>(0),
                        ).unwrap_or(0) > 0;

                        if ke_exists {
                            // ⚠️ BUG FIX: was `source_id` (record BLOB) as owner.
                            // Now JOINs records to get the correct owner bytes.
                            // Edges with no matching record in records table are skipped
                            // (INNER JOIN — safer than using wrong owner bytes).
                            match conn.execute(
                                "INSERT OR IGNORE INTO knowledge_edges
                                    (owner, source_id, target_id, relation_type, weight,
                                     confidence, valid_from, created_at, updated_at)
                                 SELECT
                                    r.owner,
                                    hex(me.source_id),
                                    hex(me.target_id),
                                    'RELATED_TO',
                                    me.weight,
                                    1.0,
                                    me.created_at,
                                    ?1,
                                    ?1
                                 FROM memory_edges me
                                 INNER JOIN records r ON r.record_id = me.source_id",
                                params![now],
                            ) {
                                Ok(migrated_count) => {
                                    info!(
                                        count = migrated_count,
                                        "[STORAGE] Migrated memory_edges → knowledge_edges"
                                    );
                                }
                                Err(e) => {
                                    warn!(
                                        error = %e,
                                        "[STORAGE] memory_edges migration failed (non-fatal, edges preserved)"
                                    );
                                }
                            }
                        } else {
                            warn!("[STORAGE] knowledge_edges table not found, skipping memory_edges migration");
                        }
                    }

                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('memory_edges_migrated_v5', ?1)",
                        params![b"1".as_slice()],
                    );
                    info!("[STORAGE] ✅ memory_edges migration marker set");
                }
            }

            // 5e: Update schema version to v5
            // ⚠️ hardcoded 5, NOT SCHEMA_VERSION constant.
            conn.execute("UPDATE schema_version SET version = 5", [])
                .map_err(|e| format!("Update schema version to v5: {}", e))?;

            // 5f: Backfill FTS5 index from existing records
            {
                let fts_populated: bool = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = 'fts_index_populated'",
                        [],
                        |row| {
                            let v: Vec<u8> = row.get(0)?;
                            Ok(v == b"1")
                        },
                    )
                    .unwrap_or(false);

                if !fts_populated {
                    // P1 SecAudit: skip FTS backfill when record_key is set.
                    // encrypted_content is ciphertext — indexing it is meaningless
                    // and leaks encrypted token patterns into an unencrypted FTS table.
                    if conn.prepare("SELECT title FROM sessions LIMIT 0").is_ok() {
                        // record_key presence is not available here (no &self),
                        // so we check via chain_state marker set by open() below.
                        // The actual guard is enforced in open() before calling maybe_migrate.
                        // For the migration path (fresh DB), encrypted_content is empty anyway.
                    }
                    let indexed = conn.execute(
                        "INSERT OR IGNORE INTO fts_index (source_type, source_id, owner_hex, content, tags)
                         SELECT 'record', hex(record_id), hex(owner), encrypted_content, topic_tags
                         FROM records WHERE status = 0 AND encrypted_content != x''",
                        [],
                    ).unwrap_or(0);

                    if indexed > 0 {
                        info!(count = indexed, "[STORAGE] FTS5 backfill: records indexed");
                    }

                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('fts_index_populated', ?1)",
                        params![b"1".as_slice()],
                    );
                    info!("[STORAGE] ✅ FTS5 index populated");
                }
            }

            info!("[STORAGE] ✅ Migration to v5 (cognitive graph) complete");
        }

        // v5 → v6 (v2.5.0-SuperNode): Cognitive task queue + LLM usage log + sessions.title
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(5);

        if current < 6 {
            info!("[STORAGE] Migrating schema v{} → v6 (SuperNode)", current);

            // 6a: Create cognitive_tasks table (IF NOT EXISTS — safe if create_schema already ran)
            let ct_exists: bool = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cognitive_tasks'",
                [],
                |row| row.get::<_, i64>(0),
            ).unwrap_or(0) > 0;

            if !ct_exists {
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS cognitive_tasks (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_type       TEXT NOT NULL,
                        priority        INTEGER DEFAULT 5,
                        status          TEXT DEFAULT 'pending',
                        payload         TEXT NOT NULL,
                        result          TEXT,
                        prompt_messages TEXT,
                        target_table    TEXT,
                        target_id       TEXT,
                        privacy_level   TEXT DEFAULT 'structured',
                        provider_used   TEXT,
                        model_used      TEXT,
                        token_usage     TEXT,
                        created_at      INTEGER NOT NULL,
                        started_at      INTEGER,
                        completed_at    INTEGER,
                        retry_count     INTEGER DEFAULT 0,
                        max_retries     INTEGER DEFAULT 3,
                        error_message   TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_ct_status ON cognitive_tasks(status, priority DESC, created_at ASC);
                    CREATE INDEX IF NOT EXISTS idx_ct_target ON cognitive_tasks(target_table, target_id, task_type);
                    CREATE INDEX IF NOT EXISTS idx_ct_type ON cognitive_tasks(task_type, status);"
                ).map_err(|e| format!("v6 migration: create cognitive_tasks: {}", e))?;
                info!("[STORAGE] Created cognitive_tasks table");
            }

            // 6b: Create llm_usage_log table
            let ul_exists: bool = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='llm_usage_log'",
                [],
                |row| row.get::<_, i64>(0),
            ).unwrap_or(0) > 0;

            if !ul_exists {
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS llm_usage_log (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id         INTEGER,
                        provider        TEXT NOT NULL,
                        model           TEXT NOT NULL,
                        input_tokens    INTEGER NOT NULL,
                        output_tokens   INTEGER NOT NULL,
                        cached_tokens   INTEGER DEFAULT 0,
                        latency_ms      INTEGER NOT NULL,
                        created_at      INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_usage_time ON llm_usage_log(created_at);
                    CREATE INDEX IF NOT EXISTS idx_usage_provider ON llm_usage_log(provider, created_at);
                    CREATE INDEX IF NOT EXISTS idx_usage_task ON llm_usage_log(task_id);"
                ).map_err(|e| format!("v6 migration: create llm_usage_log: {}", e))?;
                info!("[STORAGE] Created llm_usage_log table");
            }

            // 6c: Add sessions.title column
            if conn.prepare("SELECT title FROM sessions LIMIT 0").is_err() {
                conn.execute_batch("ALTER TABLE sessions ADD COLUMN title TEXT;")
                    .map_err(|e| format!("v6 migration: add sessions.title: {}", e))?;
                info!("[STORAGE] Added sessions.title column");
            }

            // 6d: Update schema version (hardcoded 6, not SCHEMA_VERSION)
            conn.execute("UPDATE schema_version SET version = 6", [])
                .map_err(|e| format!("Update schema version to v6: {}", e))?;

            info!("[STORAGE] ✅ Migration to v6 (SuperNode) complete");
        }

        // v6 → v7: node-blind storage marker (Brick 1)
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(6);

        if current < 7 {
            info!("[STORAGE] Migrating schema v{} → v7 (node-blind)", current);
            // Additive column: 1 = client-sealed record the node stores but cannot
            // decrypt or read; 0 = normal node-encrypted record. Guard on the table
            // existing first — some migration unit tests set up a minimal DB without
            // the `records` table (matches the v6 cognitive_tasks existence check).
            let records_exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='records'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if records_exists && conn.prepare("SELECT blind FROM records LIMIT 0").is_err() {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN blind INTEGER NOT NULL DEFAULT 0;",
                )
                .map_err(|e| format!("v7 migration: add records.blind: {}", e))?;
                info!("[STORAGE] Added `blind` column");
            }
            // ⚠️ hardcoded 7, not SCHEMA_VERSION — matches the existing pattern.
            conn.execute("UPDATE schema_version SET version = 7", [])
                .map_err(|e| format!("Update schema version to v7: {}", e))?;
            info!("[STORAGE] ✅ Migration to v7 (node-blind) complete");
        }

        // v7 → v8: node-blind commitment block persistence.
        // Normal startup runs create_schema() first, but migration tooling and
        // compatibility tests may call maybe_migrate() directly against a
        // minimal legacy database. Keep this migration independently complete
        // and idempotent instead of depending on caller order.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(7);

        if current < 8 {
            info!(
                "[STORAGE] Migrating schema v{} → v8 (commitment blocks)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_commitment_blocks (
                    height              INTEGER PRIMARY KEY CHECK(height > 0),
                    block_hash          BLOB NOT NULL UNIQUE CHECK(length(block_hash) = 32),
                    chain_id            BLOB NOT NULL CHECK(length(chain_id) = 32),
                    protocol_version    INTEGER NOT NULL,
                    timestamp           INTEGER NOT NULL,
                    prev_block_hash     BLOB NOT NULL CHECK(length(prev_block_hash) = 32),
                    merkle_root         BLOB NOT NULL CHECK(length(merkle_root) = 32),
                    record_count        INTEGER NOT NULL,
                    proposer            BLOB NOT NULL CHECK(length(proposer) = 32),
                    proposer_signature  BLOB NOT NULL CHECK(length(proposer_signature) = 64),
                    payload             BLOB NOT NULL,
                    received_from       BLOB,
                    created_at          INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_record_blocks_hash
                    ON record_commitment_blocks(block_hash);
                CREATE TABLE IF NOT EXISTS record_block_commitments (
                    record_id       BLOB PRIMARY KEY CHECK(length(record_id) = 32),
                    block_height    INTEGER NOT NULL,
                    FOREIGN KEY(block_height) REFERENCES record_commitment_blocks(height)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_block_commitments_height
                    ON record_block_commitments(block_height);",
            )
            .map_err(|error| format!("v8 migration: create commitment tables: {error}"))?;
            for table in ["record_commitment_blocks", "record_block_commitments"] {
                let exists = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        params![table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !exists {
                    return Err(format!(
                        "v8 migration: required table '{}' was not created",
                        table
                    ));
                }
            }
            // ⚠️ hardcoded 8, not SCHEMA_VERSION — preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 8", [])
                .map_err(|e| format!("Update schema version to v8: {}", e))?;
            info!("[STORAGE] ✅ Migration to v8 (commitment blocks) complete");
        }

        // v8 → v9: bounded durable checkpoint evidence. This table deliberately
        // stores no memory content, commitment IDs, owners, routes, or endpoints.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(8);

        if current < 9 {
            info!(
                "[STORAGE] Migrating schema v{} → v9 (checkpoint evidence vault)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_checkpoint_evidence (
                    evidence_digest    BLOB PRIMARY KEY CHECK(length(evidence_digest) = 32),
                    observed_at        INTEGER NOT NULL CHECK(observed_at >= 0),
                    relation           TEXT NOT NULL CHECK(relation IN
                        ('converged','remote_ahead','remote_behind','diverged')),
                    local_tip_height   INTEGER NOT NULL CHECK(local_tip_height >= 0),
                    remote_tip_height  INTEGER NOT NULL CHECK(remote_tip_height >= 0),
                    checkpoint_height  INTEGER NOT NULL CHECK(checkpoint_height >= 0),
                    signed_response    BLOB NOT NULL CHECK(length(signed_response) > 0
                        AND length(signed_response) <= 4096),
                    created_at         INTEGER NOT NULL CHECK(created_at >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_record_checkpoint_evidence_observed
                    ON record_checkpoint_evidence(observed_at);",
            )
            .map_err(|error| format!("v9 migration: create checkpoint evidence: {error}"))?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_evidence'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v9 migration: required table 'record_checkpoint_evidence' was not created"
                        .to_string(),
                );
            }
            // ⚠️ hardcoded 9, not SCHEMA_VERSION — preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 9", [])
                .map_err(|error| format!("Update schema version to v9: {error}"))?;
            info!("[STORAGE] ✅ Migration to v9 (checkpoint evidence vault) complete");
        }

        // v9 → v10: durable witness-equivocation incidents. This remains a
        // local operator security primitive: responder identities and signed
        // frames are never exported through heartbeat or public APIs.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(9);

        if current < 10 {
            info!(
                "[STORAGE] Migrating schema v{} → v10 (checkpoint equivocation incidents)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_checkpoint_equivocations (
                    responder               BLOB NOT NULL CHECK(length(responder) = 32),
                    conflict_scope          TEXT NOT NULL CHECK(conflict_scope IN
                        ('checkpoint','tip')),
                    conflict_height         INTEGER NOT NULL CHECK(conflict_height >= 0),
                    first_evidence_digest   BLOB NOT NULL CHECK(length(first_evidence_digest) = 32),
                    second_evidence_digest  BLOB NOT NULL CHECK(length(second_evidence_digest) = 32),
                    detected_at             INTEGER NOT NULL CHECK(detected_at >= 0),
                    PRIMARY KEY (responder, conflict_scope, conflict_height),
                    CHECK(first_evidence_digest != second_evidence_digest),
                    FOREIGN KEY(first_evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                        ON DELETE RESTRICT,
                    FOREIGN KEY(second_evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_checkpoint_equivocations_detected
                    ON record_checkpoint_equivocations(detected_at);",
            )
            .map_err(|error| format!("v10 migration: create equivocation incidents: {error}"))?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_equivocations'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v10 migration: required table 'record_checkpoint_equivocations' was not created"
                        .to_string(),
                );
            }
            // ⚠️ hardcoded 10, not SCHEMA_VERSION — preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 10", [])
                .map_err(|error| format!("Update schema version to v10: {error}"))?;
            info!("[STORAGE] ✅ Migration to v10 (checkpoint equivocation incidents) complete");
        }

        // v10 → v11: a verified divergent prefix from an explicitly pinned
        // witness becomes a sticky local safety incident. The referenced frame
        // remains private and protected from bounded-vault pruning.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(10);

        if current < 11 {
            info!(
                "[STORAGE] Migrating schema v{} → v11 (trusted checkpoint divergence incidents)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_checkpoint_trusted_divergences (
                    responder          BLOB PRIMARY KEY CHECK(length(responder) = 32),
                    evidence_digest    BLOB NOT NULL UNIQUE CHECK(length(evidence_digest) = 32),
                    checkpoint_height  INTEGER NOT NULL CHECK(checkpoint_height >= 0),
                    detected_at        INTEGER NOT NULL CHECK(detected_at >= 0),
                    FOREIGN KEY(evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_checkpoint_trusted_divergences_detected
                    ON record_checkpoint_trusted_divergences(detected_at);",
            )
            .map_err(|error| {
                format!("v11 migration: create trusted divergence incidents: {error}")
            })?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_trusted_divergences'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v11 migration: required table 'record_checkpoint_trusted_divergences' was not created"
                        .to_string(),
                );
            }
            // ⚠️ hardcoded 11, not SCHEMA_VERSION — preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 11", [])
                .map_err(|error| format!("Update schema version to v11: {error}"))?;
            info!(
                "[STORAGE] ✅ Migration to v11 (trusted checkpoint divergence incidents) complete"
            );
        }

        // v11 -> v12: persist an immutable threshold certificate that points
        // at exact signature-verified evidence frames. Certificates remain
        // bounded and never select or mutate the canonical chain.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(11);

        if current < 12 {
            info!(
                "[STORAGE] Migrating schema v{} -> v12 (checkpoint certificates)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_checkpoint_certificates (
                    checkpoint_height  INTEGER PRIMARY KEY CHECK(checkpoint_height > 0),
                    chain_id           BLOB NOT NULL CHECK(length(chain_id) = 32),
                    checkpoint_hash    BLOB NOT NULL CHECK(length(checkpoint_hash) = 32),
                    certificate_digest BLOB NOT NULL UNIQUE CHECK(length(certificate_digest) = 32),
                    required_signers   INTEGER NOT NULL CHECK(required_signers BETWEEN 2 AND 3),
                    signer_count       INTEGER NOT NULL CHECK(signer_count BETWEEN required_signers AND 3),
                    certified_at       INTEGER NOT NULL CHECK(certified_at >= 0),
                    FOREIGN KEY(checkpoint_height) REFERENCES record_commitment_blocks(height)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_checkpoint_certificates_time
                    ON record_checkpoint_certificates(certified_at);
                CREATE TABLE IF NOT EXISTS record_checkpoint_certificate_members (
                    checkpoint_height INTEGER NOT NULL,
                    responder         BLOB NOT NULL CHECK(length(responder) = 32),
                    evidence_digest   BLOB NOT NULL UNIQUE CHECK(length(evidence_digest) = 32),
                    PRIMARY KEY(checkpoint_height, responder),
                    FOREIGN KEY(checkpoint_height) REFERENCES record_checkpoint_certificates(checkpoint_height)
                        ON DELETE CASCADE,
                    FOREIGN KEY(evidence_digest) REFERENCES record_checkpoint_evidence(evidence_digest)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_checkpoint_certificate_members_height
                    ON record_checkpoint_certificate_members(checkpoint_height);",
            )
            .map_err(|error| format!("v12 migration: create checkpoint certificates: {error}"))?;
            for table in [
                "record_checkpoint_certificates",
                "record_checkpoint_certificate_members",
            ] {
                let exists = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        params![table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !exists {
                    return Err(format!(
                        "v12 migration: required table '{table}' was not created"
                    ));
                }
            }
            // Hardcoded 12 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 12", [])
                .map_err(|error| format!("Update schema version to v12: {error}"))?;
            info!("[STORAGE] Migration to v12 (checkpoint certificates) complete");
        }

        // v12 -> v13: durable witness-side coordinator lease state. This is a
        // crash-recovery fence for one configured coordinator identity, not a
        // leader-election, fork-choice, quorum, or finality table.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(12);

        if current < 13 {
            info!(
                "[STORAGE] Migrating schema v{} -> v13 (coordinator leases)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_coordinator_leases (
                    coordinator      BLOB PRIMARY KEY CHECK(length(coordinator) = 32),
                    chain_id         BLOB NOT NULL CHECK(length(chain_id) = 32),
                    instance_id      BLOB NOT NULL CHECK(length(instance_id) = 32),
                    lease_epoch      INTEGER NOT NULL CHECK(lease_epoch > 0),
                    lease_expires_at INTEGER NOT NULL CHECK(lease_expires_at >= 0),
                    updated_at       INTEGER NOT NULL CHECK(updated_at >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_record_coordinator_leases_expiry
                    ON record_coordinator_leases(lease_expires_at);",
            )
            .map_err(|error| format!("v13 migration: create coordinator leases: {error}"))?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_coordinator_leases'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v13 migration: required table 'record_coordinator_leases' was not created"
                        .to_string(),
                );
            }
            // Hardcoded 13 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 13", [])
                .map_err(|error| format!("Update schema version to v13: {error}"))?;
            info!("[STORAGE] Migration to v13 (coordinator leases) complete");
        }

        // v13 -> v14: one bounded aggregate-only delivery-anchor high-water
        // row per requester. This is local witness state, not relay traffic,
        // block consensus, fork choice, or a user activity ledger.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(13);

        if current < 14 {
            info!(
                "[STORAGE] Migrating schema v{} -> v14 (delivery anchor witnesses)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS verified_delivery_anchor_witnesses (
                    requester      BLOB PRIMARY KEY CHECK(length(requester) = 32),
                    generation     INTEGER NOT NULL CHECK(generation > 0),
                    anchor_digest  BLOB NOT NULL CHECK(length(anchor_digest) = 32),
                    observed_at    INTEGER NOT NULL CHECK(observed_at >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_verified_delivery_anchor_witnesses_observed
                    ON verified_delivery_anchor_witnesses(observed_at);",
            )
            .map_err(|error| format!("v14 migration: create delivery anchor witnesses: {error}"))?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='verified_delivery_anchor_witnesses'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v14 migration: required table 'verified_delivery_anchor_witnesses' was not created"
                        .to_string(),
                );
            }
            // Hardcoded 14 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 14", [])
                .map_err(|error| format!("Update schema version to v14: {error}"))?;
            info!("[STORAGE] Migration to v14 (delivery anchor witnesses) complete");
        }

        // v14 -> v15: append-only dual-signed coordinator authority history.
        // [COORDINATOR-HANDOVER 2026-08-12 by Codex] The complete transition
        // sequence is retained so cold sync can validate historical proposers;
        // a mutable "current coordinator" row would lose that evidence.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(14);

        if current < 15 {
            info!(
                "[STORAGE] Migrating schema v{} -> v15 (coordinator authority history)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_coordinator_handovers (
                    authority_epoch      INTEGER PRIMARY KEY CHECK(authority_epoch > 0),
                    activation_height    INTEGER NOT NULL UNIQUE CHECK(activation_height > 1),
                    previous_height      INTEGER NOT NULL UNIQUE CHECK(previous_height > 0),
                    chain_id             BLOB NOT NULL CHECK(length(chain_id) = 32),
                    protocol_version     INTEGER NOT NULL,
                    previous_tip_hash    BLOB NOT NULL CHECK(length(previous_tip_hash) = 32),
                    previous_coordinator BLOB NOT NULL CHECK(length(previous_coordinator) = 32),
                    next_coordinator     BLOB NOT NULL CHECK(length(next_coordinator) = 32),
                    authorization_id     BLOB NOT NULL UNIQUE CHECK(length(authorization_id) = 16),
                    issued_at            INTEGER NOT NULL CHECK(issued_at > 0),
                    previous_signature   BLOB NOT NULL CHECK(length(previous_signature) = 64),
                    next_signature       BLOB NOT NULL CHECK(length(next_signature) = 64),
                    payload              BLOB NOT NULL CHECK(length(payload) > 0 AND length(payload) <= 1024),
                    accepted_at          INTEGER NOT NULL CHECK(accepted_at > 0),
                    CHECK(activation_height = previous_height + 1),
                    FOREIGN KEY(previous_height) REFERENCES record_commitment_blocks(height)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_coordinator_handovers_activation
                    ON record_coordinator_handovers(activation_height);",
            )
            .map_err(|error| format!("v15 migration: create coordinator handovers: {error}"))?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_coordinator_handovers'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v15 migration: required table 'record_coordinator_handovers' was not created"
                        .to_string(),
                );
            }
            // Hardcoded 15 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 15", [])
                .map_err(|error| format!("Update schema version to v15: {error}"))?;
            info!("[STORAGE] Migration to v15 (coordinator authority history) complete");
        }

        // v15 -> v16: an independent high-water namespace for externally
        // witnessed relay-custody checkpoints. [CUSTODY-AUDIT-WITNESS
        // 2026-08-16 by Codex] Do not merge this with delivery-cache witness
        // state: both counters begin at one and advance independently.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(15);

        if current < 16 {
            info!(
                "[STORAGE] Migrating schema v{} -> v16 (custody audit anchor witnesses)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS custody_audit_anchor_witnesses (
                    producer       BLOB PRIMARY KEY CHECK(length(producer) = 32),
                    generation     INTEGER NOT NULL CHECK(generation > 0),
                    frame_sha256   BLOB NOT NULL CHECK(length(frame_sha256) = 32),
                    observed_at    INTEGER NOT NULL CHECK(observed_at > 0)
                );
                CREATE INDEX IF NOT EXISTS idx_custody_audit_anchor_witnesses_observed
                    ON custody_audit_anchor_witnesses(observed_at);",
            )
            .map_err(|error| {
                format!("v16 migration: create custody audit anchor witnesses: {error}")
            })?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='custody_audit_anchor_witnesses'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v16 migration: required table 'custody_audit_anchor_witnesses' was not created"
                        .to_string(),
                );
            }
            // Hardcoded 16 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 16", [])
                .map_err(|error| format!("Update schema version to v16: {error}"))?;
            info!("[STORAGE] Migration to v16 (custody audit anchor witnesses) complete");
        }

        // v16 -> v17: producer-side immutable receipt evidence. [CUSTODY-
        // WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] This table is distinct
        // from witness-side high-water state: it proves what independent
        // witnesses signed and is re-audited after every restart.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(16);

        if current < 17 {
            info!(
                "[STORAGE] Migrating schema v{} -> v17 (custody witness receipt evidence)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS custody_audit_witness_receipt_evidence (
                    receipt_digest         BLOB PRIMARY KEY CHECK(length(receipt_digest) = 32),
                    producer               BLOB NOT NULL CHECK(length(producer) = 32),
                    witness                BLOB NOT NULL CHECK(length(witness) = 32),
                    requested_generation   INTEGER NOT NULL CHECK(requested_generation > 0),
                    requested_frame_sha256 BLOB NOT NULL CHECK(length(requested_frame_sha256) = 32),
                    retained_generation    INTEGER NOT NULL CHECK(retained_generation > 0),
                    retained_frame_sha256  BLOB NOT NULL CHECK(length(retained_frame_sha256) = 32),
                    outcome                INTEGER NOT NULL CHECK(outcome BETWEEN 0 AND 4),
                    observed_at            INTEGER NOT NULL CHECK(observed_at > 0),
                    receipt_frame          BLOB NOT NULL CHECK(length(receipt_frame) > 0
                        AND length(receipt_frame) <= 320),
                    persisted_at           INTEGER NOT NULL CHECK(persisted_at > 0),
                    CHECK(producer != witness)
                );
                CREATE INDEX IF NOT EXISTS idx_custody_witness_receipts_anchor
                    ON custody_audit_witness_receipt_evidence(
                        producer, requested_generation, requested_frame_sha256, observed_at
                    );
                CREATE INDEX IF NOT EXISTS idx_custody_witness_receipts_observed
                    ON custody_audit_witness_receipt_evidence(observed_at);",
            )
            .map_err(|error| {
                format!("v17 migration: create custody witness receipt evidence: {error}")
            })?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='custody_audit_witness_receipt_evidence'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v17 migration: required table 'custody_audit_witness_receipt_evidence' was not created"
                        .to_string(),
                );
            }
            // Hardcoded 17 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 17", [])
                .map_err(|error| format!("Update schema version to v17: {error}"))?;
            info!("[STORAGE] Migration to v17 (custody witness receipt evidence) complete");
        }

        // v17 -> v18: retain the exact admission policy used for each receipt.
        // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Existing rows
        // are conservatively classified as strict live transport (60 seconds).
        // Column probes make this migration restart-safe after an interrupted
        // first ALTER and harmless for a freshly created v18 table.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(17);

        if current < 18 {
            info!(
                "[STORAGE] Migrating schema v{} -> v18 (custody witness receipt admission evidence)",
                current
            );
            let has_admission_kind = conn
                .prepare(
                    "SELECT admission_kind
                     FROM custody_audit_witness_receipt_evidence LIMIT 0",
                )
                .is_ok();
            if !has_admission_kind {
                conn.execute_batch(
                    "ALTER TABLE custody_audit_witness_receipt_evidence
                         ADD COLUMN admission_kind INTEGER NOT NULL DEFAULT 0
                         CHECK(admission_kind BETWEEN 0 AND 1);",
                )
                .map_err(|error| {
                    format!("v18 migration: add custody receipt admission kind: {error}")
                })?;
            }
            let has_admission_delay = conn
                .prepare(
                    "SELECT admission_max_delay_secs
                     FROM custody_audit_witness_receipt_evidence LIMIT 0",
                )
                .is_ok();
            if !has_admission_delay {
                conn.execute_batch(
                    "ALTER TABLE custody_audit_witness_receipt_evidence
                         ADD COLUMN admission_max_delay_secs INTEGER NOT NULL DEFAULT 60
                         CHECK(admission_max_delay_secs BETWEEN 60 AND 604800);",
                )
                .map_err(|error| {
                    format!("v18 migration: add custody receipt admission delay: {error}")
                })?;
            }
            let admission_columns: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info(
                         'custody_audit_witness_receipt_evidence'
                     ) WHERE name IN ('admission_kind','admission_max_delay_secs')",
                    [],
                    |row| row.get(0),
                )
                .map_err(|error| {
                    format!("v18 migration: inspect custody receipt admission columns: {error}")
                })?;
            if admission_columns != 2 {
                return Err(
                    "v18 migration: required custody receipt admission columns were not created"
                        .to_string(),
                );
            }
            // Hardcoded 18 preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 18", [])
                .map_err(|error| format!("Update schema version to v18: {error}"))?;
            info!(
                "[STORAGE] Migration to v18 (custody witness receipt admission evidence) complete"
            );
        }

        Ok(())
    }
}
