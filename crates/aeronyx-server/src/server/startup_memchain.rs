// [ARCH-SPLIT 2026-10-02]
// MemChain storage startup. Disabled mode still returns from Server::run before this opens storage.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    pub(super) async fn init_memchain(
        &self,
    ) -> Result<(
        Arc<MemoryStorage>,
        Arc<VectorIndex>,
        Arc<MemPool>,
        Arc<TokioMutex<AofWriter>>,
    )> {
        let db_path = &self.config.memchain.db_path;
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("DB dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        let record_key = derive_record_key(&self.identity.to_bytes());
        info!("[MEMCHAIN] Record content encryption enabled");

        let storage = Arc::new(
            MemoryStorage::open(db_path, Some(record_key))
                .map_err(|e| ServerError::startup_failed(format!("SQLite: {}", e)))?,
        );
        // [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Install the
        // operator-validated root before any chain audit. The root stays
        // process-local; logs expose only whether height-scoped authority
        // enforcement is active, never the identity itself.
        let commitment_authority_root = self
            .config
            .memchain
            .effective_commitment_authority_root_node_id(&self.identity.public_key_bytes());
        storage
            .configure_record_commitment_authority_root(commitment_authority_root)
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain commitment authority root: {error}"))
            })?;
        info!(
            enabled = commitment_authority_root.is_some(),
            explicit = !self
                .config
                .memchain
                .commitment_authority_root_node_id
                .trim()
                .is_empty(),
            "[MEMCHAIN_BLOCK] Commitment proposer authority configured"
        );
        let commitment_durability = storage
            .configure_record_commitment_durability(
                self.config.memchain.commitment_coordinator_enabled,
            )
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain commitment durability: {error}"))
            })?;
        let coordinator_fence_state = storage
            .record_commitment_chain_integrity_status()
            .coordinator_fence_state;
        info!(
            durability_mode = commitment_durability,
            coordinator = self.config.memchain.commitment_coordinator_enabled,
            coordinator_fence_state,
            "[MEMCHAIN_BLOCK] Commitment durability gate passed"
        );
        let commitment_audit = storage
            .audit_record_commitment_chain()
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain commitment integrity audit: {error}"))
            })?;
        let commitment_integrity = storage.record_commitment_chain_integrity_status();
        info!(
            blocks = commitment_audit.block_count,
            commitments = commitment_audit.commitment_count,
            tip_height = commitment_audit.tip_height,
            duration_ms = commitment_integrity.verification_duration_ms.unwrap_or(0),
            "[MEMCHAIN_BLOCK] Persisted commitment chain audit passed"
        );
        if self.config.memchain.commitment_coordinator_enabled {
            let anchor_state = storage
                .configure_record_commitment_tip_anchor(
                    self.config.memchain.effective_commitment_tip_anchor_path(),
                    &self.identity,
                )
                .await
                .map_err(|error| {
                    ServerError::startup_failed(format!(
                        "MemChain commitment tip rollback guard: {error}"
                    ))
                })?;
            info!(
                state = anchor_state,
                tip_height = commitment_audit.tip_height,
                scope = "local_db_file_rollback_only",
                "[MEMCHAIN_BLOCK] Signed commitment tip rollback guard passed"
            );
        }
        let checkpoint_evidence_audit = storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!("MemChain checkpoint evidence audit: {error}"))
            })?;
        info!(
            evidence_records = checkpoint_evidence_audit.evidence_records,
            applicable_records = checkpoint_evidence_audit.applicable_evidence_records,
            deferred_records = checkpoint_evidence_audit.deferred_evidence_records,
            divergence_records = checkpoint_evidence_audit.divergence_evidence_records,
            equivocation_incidents = checkpoint_evidence_audit.equivocation_incidents,
            trusted_divergence_incidents = checkpoint_evidence_audit.trusted_divergence_incidents,
            checkpoint_certificates = checkpoint_evidence_audit.checkpoint_certificates,
            latest_certified_height = checkpoint_evidence_audit.latest_certified_height,
            latest_certificate_signers = checkpoint_evidence_audit.latest_certificate_signers,
            "[MEMCHAIN_BLOCK] Persisted checkpoint evidence audit passed"
        );
        if self.config.memchain.commitment_coordinator_enabled {
            let certificate_anchor_state = storage
                .configure_record_commitment_checkpoint_certificate_anchor(
                    self.config.memchain.effective_commitment_tip_anchor_path(),
                    &self.identity,
                )
                .await
                .map_err(|error| {
                    ServerError::startup_failed(format!(
                        "MemChain checkpoint certificate rollback guard: {error}"
                    ))
                })?;
            info!(
                state = certificate_anchor_state,
                certificate_height = checkpoint_evidence_audit
                    .latest_certified_height
                    .unwrap_or(0),
                scope = "local_certificate_vault_rollback_only",
                "[MEMCHAIN_BLOCK] Signed checkpoint certificate rollback guard passed"
            );
        }

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let vector_index = Arc::new(if quantization_enabled {
            let sat = if self.config.memchain.vector_early_termination {
                0.001_f32
            } else {
                0.0_f32
            };
            info!(
                quantization = "scalar_uint8",
                "[MEMCHAIN] VectorIndex with scalar quantization"
            );
            VectorIndex::with_config(true, sat)
        } else {
            VectorIndex::new()
        });

        // [MEMCHAIN-SEALED-VECTOR-BOUNDARY 2026-10-05 by Codex] Keep historic
        // vectors in SQLite for compatibility/recovery, but never read them
        // into an ordinary node's transient semantic index.
        let rebuild_count = vector_index.total_vectors();

        let record_count = storage.count().await;
        info!(
            db = %db_path,
            records = record_count,
            vectors = rebuild_count,
            rebuild_scope = "disabled_node_blind_policy",
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );

        let aof_path = &self.config.memchain.aof_path;
        if let Some(parent) = std::path::Path::new(aof_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ServerError::startup_failed(format!("AOF dir '{}': {}", parent.display(), e))
                })?;
            }
        }

        let (existing_facts, last_block) = AofWriter::replay(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF replay: {}", e)))?;

        let mempool = Arc::new(MemPool::new());
        let mut loaded = 0u64;
        for fact in existing_facts {
            if mempool.add_fact(fact) {
                loaded += 1;
            }
        }

        let aof_writer = AofWriter::open(aof_path)
            .await
            .map_err(|e| ServerError::startup_failed(format!("AOF open: {}", e)))?;
        aof_writer.set_chain_state(last_block.as_ref());
        let aof_writer = Arc::new(TokioMutex::new(aof_writer));

        info!(
            facts = loaded,
            "[MEMCHAIN] Legacy MemPool + AOF initialized"
        );
        Ok((storage, vector_index, mempool, aof_writer))
    }
}
