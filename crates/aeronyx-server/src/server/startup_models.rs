// [ARCH-SPLIT 2026-10-02]
// Model and provider startup. Server::run calls these before descriptors or gossip.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    // ============================================
    // SaaS MpiState init
    // ============================================

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn init_saas_mpi_state(
        &self,
        _server_storage: &Arc<MemoryStorage>,
        owner_key: [u8; 32],
        api_secret: Option<String>,
        user_weights: Arc<
            parking_lot::RwLock<
                std::collections::HashMap<String, crate::services::memchain::mvf::WeightVector>,
            >,
        >,
        embed_engine: Option<Arc<EmbedEngine>>,
        ner_engine: Option<Arc<NerEngine>>,
        reranker_engine: Option<Arc<RerankerEngine>>,
        llm_router: Option<Arc<LlmRouter>>,
    ) -> Result<Arc<MpiState>> {
        let saas_cfg = self.config.memchain.saas.as_ref().ok_or_else(|| {
            ServerError::startup_failed("mode=saas requires [memchain.saas] config section")
        })?;

        let data_root = &saas_cfg.data_root;

        tokio::fs::create_dir_all(data_root).await.map_err(|e| {
            ServerError::startup_failed(format!("SaaS data_root '{}': {}", data_root.display(), e))
        })?;

        let system_db = SystemDb::open(&data_root.join("system.db"))
            .await
            .map_err(|e| ServerError::startup_failed(format!("SystemDb: {}", e)))?;

        let volumes_config_path = ensure_volumes_config(data_root)
            .map_err(|e| ServerError::startup_failed(format!("volumes.toml: {}", e)))?;

        let volume_router = VolumeRouter::new(&volumes_config_path, Arc::clone(&system_db))
            .await
            .map_err(|e| ServerError::startup_failed(format!("VolumeRouter: {}", e)))?;

        let storage_pool = StoragePool::new(
            Arc::clone(&volume_router),
            Arc::clone(&system_db),
            saas_cfg.pool_max_connections,
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
        );

        let quantization_enabled =
            self.config.memchain.vector_quantization == VectorQuantizationMode::ScalarUint8;
        let saturation_threshold = if self.config.memchain.vector_early_termination {
            0.001_f32
        } else {
            0.0_f32
        };

        let vector_pool = VectorIndexPool::new(
            Arc::clone(&volume_router),
            Duration::from_secs(saas_cfg.pool_idle_timeout_secs),
            quantization_enabled,
            saturation_threshold,
        );

        let jwt_secret = ensure_jwt_secret(
            self.config.memchain.jwt_secret.as_deref(),
            self.config_path.as_deref(),
        )
        .map_err(|e| ServerError::startup_failed(format!("jwt_secret: {}", e)))?;

        info!(
            data_root     = %data_root.display(),
            pool_max      = saas_cfg.pool_max_connections,
            idle_timeout  = saas_cfg.pool_idle_timeout_secs,
            "[SAAS] Infrastructure initialized"
        );

        let mpi_state = Arc::new(MpiState {
            mode: Mode::Saas,
            storage: None,
            vector_index: None,
            identity: self.identity.clone(),
            identity_cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
            index_ready: std::sync::atomic::AtomicBool::new(true),
            user_weights,
            mvf_alpha: self.config.memchain.mvf_alpha,
            mvf_enabled: self.config.memchain.mvf_enabled,
            session_embeddings: parking_lot::RwLock::new(SessionEmbeddingCache::default()),
            mvf_baseline: parking_lot::RwLock::new(None),
            owner_key,
            api_secret,
            embed_engine,
            allow_remote_storage: false,
            blind_storage_enabled: self.config.memchain.blind_storage_enabled,
            max_remote_owners: 0,
            ner_engine,
            graph_enabled: self.config.memchain.graph_enabled,
            entropy_filter_enabled: self.config.memchain.entropy_filter_enabled,
            reranker_engine,
            rawlog_key: Some(derive_rawlog_key(&self.identity.to_bytes())),
            llm_router,
            storage_pool: Some(storage_pool),
            vector_pool: Some(vector_pool),
            volume_router: Some(volume_router),
            system_db: Some(system_db),
            jwt_secret: Some(jwt_secret),
            token_ttl_secs: self.config.memchain.token_ttl_secs,
            pool_max_connections: saas_cfg.pool_max_connections,
            pool_idle_timeout_secs: saas_cfg.pool_idle_timeout_secs,
        });

        Ok(mpi_state)
    }

    // ============================================
    // Engine initialization
    // ============================================

    pub(super) fn init_embed_engine(&self) -> Option<Arc<EmbedEngine>> {
        // [MEMCHAIN-PHALA-ONLY 2026-10-05 by Codex] Never load a local model
        // on ordinary nodes; the legacy config path remains parse-compatible.
        info!("[EMBED] Local model loading is fail-closed; embeddings require attested Phala ACI");
        None
    }

    pub(super) fn init_ner_engine(&self) -> Option<Arc<NerEngine>> {
        // [MEMCHAIN-PHALA-ONLY 2026-10-05 by Codex] Extraction is model
        // inference too; never load the legacy local GLiNER bundle.
        debug!("[NER] Local model loading is fail-closed; extraction requires attested Phala ACI");
        None
    }

    pub(super) fn init_reranker_engine(&self) -> Option<Arc<RerankerEngine>> {
        // [MEMCHAIN-PHALA-ONLY 2026-10-05 by Codex] Cross-encoder inference
        // also stays in the attested Phala route, never in a node process.
        debug!("[RERANKER] Local model loading is fail-closed; ranking requires attested Phala ACI");
        None
    }

    // ============================================
    // LlmRouter initialization
    // ============================================

    /// Builds the complete configured SuperNode provider set transactionally.
    ///
    /// [SUPERNODE-STARTUP-INTEGRITY 2026-08-14 by Codex] `enabled=true` is a
    /// runtime contract. One invalid provider can be referenced by routing or
    /// fallback policy, so partial initialization is rejected rather than
    /// silently changing operator intent. Error output is a fixed reason code.
    pub(super) fn init_llm_router(&self) -> Result<Option<Arc<LlmRouter>>> {
        use crate::config_supernode::ProviderType;
        use crate::services::memchain::{
            AnthropicProvider, LlmProvider, LlmProviderInitError, OpenAiCompatProvider,
        };

        if !self.config.memchain.is_supernode_enabled() {
            debug!("[SUPERNODE] Disabled");
            return Ok(None);
        }
        // [MEMCHAIN-PHALA-E2EE-BOUNDARY 2026-10-06 by Codex] ACI verification
        // protects workload identity, not payloads from this node. Do not start
        // a server-side model worker until requests and responses are E2EE-bound
        // between the source client and the attested Phala workload.
        if self.config.memchain.supernode.enabled {
            return Err(ServerError::startup_failed(
                "SuperNode initialization failed (client_to_tee_e2ee_transport_unavailable)",
            ));
        }
        // [MEMCHAIN-PHALA-NODE-BOUNDARY 2026-10-05 by Codex] Config validation
        // enforces this too; retain the startup guard so future callers cannot
        // construct a plaintext cognition worker in ordinary node mode.
        if !self.config.memchain.is_saas() {
            return Err(ServerError::startup_failed(
                "SuperNode initialization failed (plaintext_cognition_requires_saas_mode)",
            ));
        }
        if !self.config.memchain.is_enabled() {
            return Err(ServerError::startup_failed(
                "SuperNode initialization failed (memchain_runtime_required)",
            ));
        }

        let supernode = &self.config.memchain.supernode;
        if supernode.providers.is_empty() {
            return Err(ServerError::startup_failed(
                "SuperNode initialization failed (provider_set_empty)",
            ));
        }

        let mut providers: Vec<(String, String, String, Arc<dyn LlmProvider>)> = Vec::new();

        // [MEMCHAIN-PHALA-ROUTING 2026-10-05 by Codex] Resolve the only
        // supported confidential provider to Phala's canonical endpoint.
        for provider_cfg in &supernode.providers {
            let api_base = if provider_cfg.api_base.is_empty()
                && provider_cfg.provider_type == ProviderType::Anthropic
            {
                "https://api.anthropic.com".to_string()
            } else if provider_cfg.api_base.is_empty()
                && provider_cfg.provider_type == ProviderType::PhalaAci
            {
                crate::services::memchain::llm_provider::PHALA_ACI_API_BASE_DEFAULT.to_owned()
            } else {
                provider_cfg.api_base.clone()
            };

            let provider_result: std::result::Result<Arc<dyn LlmProvider>, LlmProviderInitError> =
                match provider_cfg.provider_type {
                    ProviderType::OpenaiCompatible => OpenAiCompatProvider::new(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        provider_cfg.api_key.clone().unwrap_or_default(),
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    )
                    .map(|provider| Arc::new(provider) as Arc<dyn LlmProvider>),
                    ProviderType::Anthropic => AnthropicProvider::new_with_api_base(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        provider_cfg.api_key.clone().unwrap_or_default(),
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                    )
                    .map(|provider| Arc::new(provider) as Arc<dyn LlmProvider>),
                    ProviderType::PhalaAci => OpenAiCompatProvider::new_phala_aci(
                        provider_cfg.name.clone(),
                        api_base.clone(),
                        provider_cfg.api_key.clone().unwrap_or_default(),
                        provider_cfg.model.clone(),
                        provider_cfg.max_tokens,
                        provider_cfg.temperature,
                        &supernode.accepted_compose_hashes,
                        // [MEMCHAIN-PHALA-SOURCE-PROVENANCE 2026-10-06 by Codex]
                        &supernode.accepted_source_provenance,
                        &supernode.accepted_kms_root_public_keys,
                    )
                    .map(|provider| Arc::new(provider) as Arc<dyn LlmProvider>),
                };

            let provider = provider_result.map_err(|error| {
                let reason = error.reason_code();
                warn!(
                    reason,
                    "[SUPERNODE] Required provider initialization rejected"
                );
                ServerError::startup_failed(format!("SuperNode initialization failed ({reason})"))
            })?;

            // [MEMCHAIN-PHALA-VERIFIER-GATE 2026-10-05 by Codex] SuperNode
            // is a required runtime contract: do not start its queue worker
            // unless the provider both requests and verifies ACI cryptographically.
            if !provider.supports_aci_verified() || !provider.has_cryptographic_aci_verifier() {
                warn!(
                    reason = "aci_verifier_unavailable",
                    "[SUPERNODE] Confidential provider is not ready"
                );
                return Err(ServerError::startup_failed(
                    "SuperNode initialization failed (aci_verifier_unavailable)",
                ));
            }

            info!(type_ = ?provider_cfg.provider_type, model = %provider_cfg.model, "[SUPERNODE] Provider registered");
            providers.push((
                provider_cfg.name.clone(),
                api_base,
                provider_cfg.model.clone(),
                provider,
            ));
        }

        let mut router = LlmRouter::new(providers, supernode.routing.clone());
        // [MEMCHAIN-PHALA-EMBEDDINGS 2026-10-06 by Codex] Semantic vectors
        // are opt-in and bind to one named Phala ACI provider/model; no generic
        // provider or local engine is considered as a fallback.
        if let (Some(provider), Some(model)) = (
            supernode.embedding_provider.as_deref(),
            supernode.embedding_model.as_deref(),
        ) {
            router = router.with_embedding_route(provider, model).map_err(|_| {
                ServerError::startup_failed(
                    "SuperNode initialization failed (phala_embedding_route_invalid)",
                )
            })?;
        }
        info!(providers = supernode.providers.len(), fallback = ?supernode.routing.fallback, "[SUPERNODE] LlmRouter initialized");
        Ok(Some(Arc::new(router)))
    }
}
