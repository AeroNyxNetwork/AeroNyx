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

        let owner = self.identity.public_key_bytes();
        let rebuild_all_owners =
            self.config.memchain.blind_storage_enabled || self.config.memchain.allow_remote_storage;
        let records_with_model = if rebuild_all_owners {
            storage.get_all_records_with_embedding().await
        } else {
            storage.get_records_with_embedding(&owner).await
        };
        let mut rebuilt_owners = std::collections::HashSet::new();
        let mut rebuilt_partitions = std::collections::HashSet::new();
        let mut integrity_rejected = 0usize;
        for (r, model) in records_with_model {
            if r.has_embedding() {
                if let Some(reason) = memchain_index_rejection_reason(&r) {
                    integrity_rejected += 1;
                    warn!(
                        reason,
                        blind = r.blind,
                        "[MEMCHAIN] Persisted record rejected from vector rebuild"
                    );
                    continue;
                }
                rebuilt_owners.insert(r.owner);
                rebuilt_partitions.insert((r.owner, model.clone()));
                vector_index.upsert(
                    r.record_id,
                    r.embedding.clone(),
                    r.layer,
                    r.timestamp,
                    &r.owner,
                    &model,
                );
            }
        }
        let rebuild_count = vector_index.total_vectors();

        info!(
            db = %db_path,
            records = storage.count().await,
            vectors = rebuild_count,
            owners = rebuilt_owners.len(),
            partitions = rebuilt_partitions.len(),
            integrity_rejected,
            rebuild_scope = if rebuild_all_owners { "all_active_owners" } else { "local_owner" },
            "[MEMCHAIN] SQLite + VectorIndex initialized"
        );
        if integrity_rejected > 0 {
            warn!(
                integrity_rejected,
                "[MEMCHAIN] Integrity audit quarantined persisted records from recall"
            );
        }

        if quantization_enabled && rebuild_count > 0 {
            // Each owner/model pair is an independent security partition. A
            // blind storage node may host many such partitions, so restoring
            // only the node identity's quantizer would silently degrade remote
            // recall after restart.
            for (partition_owner, model_name) in rebuilt_partitions {
                let owner_hex = hex::encode(partition_owner);
                let cal_key = format!("{}:{}:{}", QUANTIZER_CAL_KEY_PREFIX, owner_hex, model_name);

                let restored = {
                    let conn = storage.conn_lock().await;
                    let cal_data: Option<Vec<u8>> = conn
                        .query_row(
                            "SELECT value FROM chain_state WHERE key = ?1",
                            rusqlite::params![cal_key],
                            |row| row.get::<_, Vec<u8>>(0),
                        )
                        .optional()
                        .unwrap_or(None);
                    drop(conn);
                    if let Some(data) = cal_data {
                        vector_index.restore_quantizer(&partition_owner, &model_name, &data)
                    } else {
                        false
                    }
                };

                if restored {
                    info!(
                        owner = %owner_hex,
                        model = %model_name,
                        "[VECTOR] Quantizer restored"
                    );
                    continue;
                }

                vector_index.calibrate_partition(&partition_owner, &model_name);
                if let Some(cal_bytes) =
                    vector_index.get_quantizer_bytes(&partition_owner, &model_name)
                {
                    let conn = storage.conn_lock().await;
                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES (?1, ?2)",
                        rusqlite::params![cal_key, cal_bytes.as_slice()],
                    );
                    drop(conn);
                    info!(
                        owner = %owner_hex,
                        model = %model_name,
                        "[VECTOR] Quantizer calibrated and persisted"
                    );
                }
            }
        }

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

    // [NODE-ROLES 2026-10-09 by Claude] Moved verbatim out of `Server::run`:
    // what the memory role prepares before the node API starts.
    pub(super) async fn prepare_memory_runtime(
        &self,
        stores: &MemoryStores,
        llm_router: &Option<Arc<LlmRouter>>,
    ) -> Result<MemoryRuntime> {
        let st = &stores.storage;
        let vi = &stores.vector_index;
        let llm_router = llm_router.clone();

        let is_saas = self.config.memchain.mode == MemChainMode::Saas;

        let user_weights = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()));

        if !is_saas {
            let owner = self.identity.public_key_bytes();
            if let Some(blob) = st.load_user_weights(&owner).await {
                if let Some(w) = crate::services::memchain::mvf::WeightVector::from_bytes(&blob) {
                    let mut map = user_weights.write();
                    map.insert(hex::encode(owner), w);
                    info!("[MEMCHAIN] Loaded MVF user weights from SQLite");
                }
            }
        }

        let mvf_baseline: Option<BaselineSnapshot> = if !is_saas {
            let conn = st.conn_lock().await;
            let raw: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT value FROM chain_state WHERE key = 'mvf_baseline'",
                    [],
                    |row: &rusqlite::Row<'_>| row.get::<_, Vec<u8>>(0),
                )
                .optional()
                .unwrap_or(None);
            drop(conn);
            raw.and_then(|bytes| {
                serde_json::from_str::<BaselineSnapshot>(&String::from_utf8_lossy(&bytes)).ok()
            })
        } else {
            None
        };

        let owner_key = self.identity.public_key_bytes();
        let api_secret = self
            .config
            .memchain
            .effective_api_secret()
            .map(|s| s.to_string());

        let embed_engine = self.init_embed_engine();
        let ner_engine = self.init_ner_engine();
        let reranker_engine = self.init_reranker_engine();

        if llm_router.is_some() {
            // [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex] Startup and the
            // live worker must agree on when a processing claim is expired.
            // The grace interval prevents a second process from stealing a
            // task while its original owner persists the timeout outcome.
            let timeout_secs = self
                .config
                .memchain
                .supernode
                .worker
                .stale_claim_recovery_secs();
            let recovered = st.reset_stale_processing_tasks(timeout_secs).await;
            if recovered > 0 {
                info!(recovered, timeout_secs, "[SUPERNODE] Recovered stale tasks");
            }
        }

        let mpi_state = if is_saas {
            self.init_saas_mpi_state(
                st,
                owner_key,
                api_secret,
                Arc::clone(&user_weights),
                embed_engine.clone(),
                ner_engine.clone(),
                reranker_engine,
                llm_router.clone(),
            )
            .await?
        } else {
            let mpi = MpiState::local(
                Arc::clone(st),
                Arc::clone(vi),
                self.identity.clone(),
                parking_lot::RwLock::new(std::collections::HashMap::new()),
                std::sync::atomic::AtomicBool::new(false),
                Arc::clone(&user_weights),
                self.config.memchain.mvf_alpha,
                self.config.memchain.mvf_enabled,
                parking_lot::RwLock::new(SessionEmbeddingCache::default()),
                parking_lot::RwLock::new(mvf_baseline),
                owner_key,
                api_secret,
                embed_engine.clone(),
                self.config.memchain.allow_remote_storage,
                self.config.memchain.blind_storage_enabled,
                self.config.memchain.max_remote_owners,
                ner_engine.clone(),
                self.config.memchain.graph_enabled,
                self.config.memchain.entropy_filter_enabled,
                reranker_engine,
                Some(derive_rawlog_key(&self.identity.to_bytes())),
                llm_router.clone(),
            );
            Arc::new(mpi)
        };

        if !is_saas {
            let owner_hex = hex::encode(owner_key);
            let identity_records = st
                .get_active_records(
                    &owner_key,
                    Some(aeronyx_core::ledger::MemoryLayer::Identity),
                    100,
                )
                .await;
            if !identity_records.is_empty() {
                let mut cache = mpi_state.identity_cache.write();
                cache.insert(owner_hex, identity_records);
            }
            mpi_state
                .index_ready
                .store(true, std::sync::atomic::Ordering::Relaxed);

            if self.config.memchain.mvf_enabled && mpi_state.mvf_baseline.read().is_none() {
                let feedback = st.get_recent_feedback(200).await;
                if !feedback.is_empty() {
                    let positive = feedback.iter().filter(|(s, _)| *s == 1).count();
                    let rate = positive as f32 / feedback.len() as f32;
                    let now_ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64;

                    let baseline = BaselineSnapshot {
                        positive_rate: rate,
                        sample_size: feedback.len(),
                        frozen_at: now_ts,
                    };

                    if let Ok(json) = serde_json::to_string(&baseline) {
                        let conn = st.conn_lock().await;
                        let _ = conn.execute(
                                "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('mvf_baseline', ?1)",
                                rusqlite::params![json.as_bytes()],
                            );
                    }

                    *mpi_state.mvf_baseline.write() = Some(baseline);
                    info!(rate, samples = feedback.len(), "[MVF] Baseline frozen");
                }
            }
        }

        // The coordinator's local tip channel and the follower's remote
        // announcement channel are deliberately separate. Each is bounded
        // to one pending height because its consumer always re-reads the
        // current audited chain rather than trusting event payload state.
        let (commitment_sync_tip_tx, commitment_sync_tip_rx) = mpsc::channel(1);
        let commitment_sync_tip_notifier = self
            .config
            .memchain
            .commitment_sync_enabled
            .then_some(commitment_sync_tip_tx);
        let (commitment_tip_tx, commitment_tip_rx) = mpsc::channel(1);

        Ok(MemoryRuntime {
            is_saas,
            mpi_state,
            user_weights,
            embed_engine,
            ner_engine,
            commitment_sync_tip_notifier,
            commitment_sync_tip_rx,
            commitment_tip_tx,
            commitment_tip_rx,
        })
    }

    // [NODE-ROLES 2026-10-09 by Claude] Moved verbatim out of `Server::run`:
    // the memory role's commitment, worker, pool and miner tasks, started
    // after the node API.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn spawn_memory_tasks(
        &self,
        stores: &MemoryStores,
        runtime: MemoryRuntime,
        plane: &DataPlane,
        peer_store: &Arc<PeerStore>,
        llm_router: &Option<Arc<LlmRouter>>,
        peer_http_clients: &PeerHttpClients,
        commitment_coordinator_lease_instance: Option<[u8; 32]>,
        tasks: &mut RuntimeTaskRegistry,
        critical_failure_tx: &mpsc::Sender<CriticalRuntimeFailure>,
    ) -> Result<()> {
        let st = &stores.storage;
        let vi = &stores.vector_index;
        let mp = &stores.mempool;
        let aw = &stores.aof_writer;
        let MemoryRuntime {
            is_saas,
            mpi_state,
            user_weights,
            embed_engine,
            ner_engine,
            commitment_sync_tip_rx,
            commitment_tip_tx,
            commitment_tip_rx,
            ..
        } = runtime;
        let sessions = Arc::clone(&plane.sessions);
        let udp = Arc::clone(&plane.udp);
        let peer_store = Arc::clone(peer_store);
        let llm_router = llm_router.clone();

        if let Some(sync_task) = self.spawn_memchain_commitment_sync_task(
            Arc::clone(st),
            Arc::clone(&peer_store),
            commitment_sync_tip_rx,
            Arc::clone(&peer_http_clients.sync),
        )? {
            tasks.push((
                "memchain-block-sync",
                Self::supervise_required_runtime_task(
                    "memchain-block-sync",
                    sync_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
        }
        if let Some(reconciliation_task) = self.spawn_memchain_commitment_reconciliation_task(
            Arc::clone(st),
            Arc::clone(&peer_store),
            commitment_tip_rx,
            Arc::clone(&peer_http_clients.sync),
        ) {
            tasks.push(("memchain-checkpoint-witness", reconciliation_task));
        }
        if let Some(lease_task) = self.spawn_memchain_commitment_coordinator_lease_task(
            Arc::clone(st),
            Arc::clone(&peer_store),
            commitment_coordinator_lease_instance,
            Arc::clone(&peer_http_clients.control),
        ) {
            tasks.push(("memchain-coordinator-lease", lease_task));
        }

        if let Some(ref router) = llm_router {
            let worker = TaskWorker::new(
                Arc::clone(st),
                Arc::clone(router),
                self.config.memchain.supernode.worker.clone(),
            );
            let worker_shutdown = self.shutdown_tx.subscribe();
            // [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] Once the
            // operator enables SuperNode, its worker is required runtime.
            // Unexpected return or panic must revoke process readiness.
            let worker_task = tokio::spawn(async move {
                worker.run(worker_shutdown).await;
            });
            tasks.push((
                "supernode-worker",
                Self::supervise_required_runtime_task(
                    "supernode-worker",
                    worker_task,
                    Arc::clone(&self.shutdown),
                    critical_failure_tx.clone(),
                ),
            ));
            info!(
                poll_interval = self.config.memchain.supernode.worker.poll_interval_secs,
                max_concurrent = self.config.memchain.supernode.worker.max_concurrent,
                "[SUPERNODE] TaskWorker spawned"
            );
        }

        if is_saas {
            if let (Some(ref sp), Some(ref vp)) = (&mpi_state.storage_pool, &mpi_state.vector_pool)
            {
                let sp_clone = Arc::clone(sp);
                let vp_clone = Arc::clone(vp);
                let mut evict_rx = self.shutdown_tx.subscribe();
                tasks.push((
                    "pool-eviction",
                    tokio::spawn(async move {
                        let mut interval =
                            tokio::time::interval(Duration::from_secs(POOL_EVICTION_INTERVAL_SECS));
                        loop {
                            tokio::select! {
                                _ = evict_rx.recv() => break,
                                _ = interval.tick() => {
                                    let evicted_s = sp_clone.evict_idle().await;
                                    let evicted_v = vp_clone.evict_idle();
                                    if evicted_s + evicted_v > 0 {
                                        info!(
                                            storage = evicted_s,
                                            vector  = evicted_v,
                                            "[POOL] Evicted idle connections"
                                        );
                                    }
                                }
                            }
                        }
                    }),
                ));
                info!(
                    interval_secs = POOL_EVICTION_INTERVAL_SECS,
                    "[POOL] Eviction timer started"
                );
            }
        }

        if self.config.memchain.miner_interval_secs > 0 {
            if is_saas {
                if let (Some(ref sp), Some(ref sys_db)) =
                    (&mpi_state.storage_pool, &mpi_state.system_db)
                {
                    let scheduler = crate::miner::MinerScheduler::new(
                        Arc::clone(sp),
                        Arc::clone(sys_db),
                        self.config
                            .memchain
                            .saas
                            .as_ref()
                            .map(|s| s.miner_max_owners_per_tick)
                            .unwrap_or(10),
                        self.config
                            .memchain
                            .saas
                            .as_ref()
                            .map(|s| s.miner_max_rounds_per_hour)
                            .unwrap_or(6) as u32,
                        self.identity.clone(),
                        llm_router.clone(),
                        embed_engine.clone(),
                        ner_engine.clone(),
                    )
                    .await?;
                    let mut sched_rx = self.shutdown_tx.subscribe();
                    tasks.push((
                        "miner-scheduler",
                        tokio::spawn(async move {
                            let mut interval = tokio::time::interval(Duration::from_secs(
                                MINER_SCHEDULER_TICK_SECS,
                            ));
                            loop {
                                tokio::select! {
                                    _ = sched_rx.recv() => break,
                                    _ = interval.tick() => { scheduler.tick().await; }
                                }
                            }
                        }),
                    ));
                    info!(
                        tick_secs = MINER_SCHEDULER_TICK_SECS,
                        "[MINER] SaaS MinerScheduler started"
                    );
                }
            } else {
                let miner = ReflectionMiner::new(
                    self.config.memchain.miner_interval_secs,
                    Arc::clone(st),
                    Arc::clone(vi),
                    self.identity.clone(),
                    Arc::clone(mp),
                    Arc::clone(aw),
                    Arc::clone(&sessions),
                    Arc::clone(&udp),
                )
                .with_compaction_threshold(self.config.memchain.compaction_threshold)
                .with_mvf(self.config.memchain.mvf_enabled, Arc::clone(&user_weights))
                .with_commitment_coordinator(self.config.memchain.commitment_coordinator_enabled)
                .with_commitment_tip_notifier(commitment_tip_tx);

                let miner = if let Some(ref ee) = embed_engine {
                    miner.with_embed_engine(Arc::clone(ee))
                } else {
                    miner
                };
                let miner = if let Some(ref ne) = ner_engine {
                    miner.with_ner_engine(Arc::clone(ne))
                } else {
                    miner
                };
                let miner = if let Some(ref lr) = llm_router {
                    miner.with_llm_router(Arc::clone(lr))
                } else {
                    miner
                };

                // Reconcile a bounded backlog before announcing startup.
                // This packs only verified opaque commitments; it never
                // copies memory payloads to peers. Remaining backlog, if
                // any, is drained by the normal bounded miner tick.
                let bootstrap_blocks = miner.pack_commitment_blocks(64).await;
                if bootstrap_blocks > 0 {
                    info!(
                        blocks = bootstrap_blocks,
                        "[MEMCHAIN_BLOCK] Startup commitment backlog reconciled"
                    );
                }

                let miner_shutdown = self.shutdown_tx.subscribe();
                tasks.push((
                    "miner",
                    tokio::spawn(async move {
                        miner.run(miner_shutdown).await;
                    }),
                ));
            }
        } else {
            info!("[MINER] Disabled (interval=0)");
        }
        Ok(())
    }
}
