// [ARCH-SPLIT 2026-10-02]
// Directory chain open, then producer-isolated replica open.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    pub(super) async fn init_directory_chain(
        &self,
        peer_store: &PeerStore,
    ) -> Result<Option<Arc<DirectoryChainStore>>> {
        let Some(path) = self.config.discovery.directory_chain_path.clone() else {
            info!("[DIRECTORY_CHAIN] Local persistence disabled");
            return Ok(None);
        };
        let producer = self.identity.public_key_bytes();
        let observed_at = unix_now_secs();
        let open_path = path.clone();
        let (store, startup_audit) = tokio::task::spawn_blocking(move || {
            DirectoryChainStore::open(&open_path, producer, observed_at)
        })
        .await
        .map_err(|error| {
            ServerError::startup_failed(format!(
                "Directory Chain store task failed before startup: {}",
                RuntimeTaskJoinFailureKind::classify(&error).blocking_task_reason()
            ))
        })?
        .map_err(|error| {
            ServerError::startup_failed(format!(
                "Directory Chain startup audit failed for '{path}': {error}"
            ))
        })?;
        let store = Arc::new(store);
        info!(
            blocks = startup_audit.blocks,
            commitments = startup_audit.commitments,
            tip_height = startup_audit.tip_height,
            "[DIRECTORY_CHAIN] Persisted local chain audit passed"
        );

        let append = Self::reconcile_directory_chain_once(
            Arc::clone(&store),
            Arc::new(self.identity.clone()),
            peer_store,
        )
        .await
        .map_err(|error| {
            ServerError::startup_failed(format!(
                "Directory Chain startup reconciliation failed for '{path}': {error}"
            ))
        })?;
        let verified_at = unix_now_secs();
        let audit_store = Arc::clone(&store);
        let current_audit = tokio::task::spawn_blocking(move || audit_store.audit(verified_at))
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "Directory Chain post-append audit task failed: {}",
                    RuntimeTaskJoinFailureKind::classify(&error).blocking_task_reason()
                ))
            })?
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "Directory Chain post-append audit failed for '{path}': {error}"
                ))
            })?;
        info!(
            blocks_appended = append.blocks_appended,
            commitments_appended = append.commitments_appended,
            blocks = current_audit.blocks,
            commitments = current_audit.commitments,
            tip_height = current_audit.tip_height,
            "[DIRECTORY_CHAIN] Startup descriptor reconciliation committed"
        );
        Ok(Some(store))
    }

    /// Opens and fully audits the producer-isolated remote replica namespace.
    ///
    /// Replica tables share the Directory Chain SQLite file for one durable
    /// backup boundary, but never share local producer tables or key space.
    pub(super) async fn init_directory_replica(
        &self,
    ) -> Result<Option<Arc<DirectoryReplicaStore>>> {
        let Some(path) = self.config.discovery.directory_chain_path.clone() else {
            return Ok(None);
        };
        let local_node_id = self.identity.public_key_bytes();
        let observed_at = unix_now_secs();
        let open_path = path.clone();
        let identity = self.identity.clone();
        let witness_node_ids = self
            .config
            .discovery
            .directory_chain_sync_peer_node_id_bytes();
        let minimum_witnesses = self
            .config
            .discovery
            .directory_observation_witness_min_verified;
        // [ROUTE-DOMAIN-POLICY-HISTORY 2026-08-03 by Codex] Reconcile the
        // canonical local routing-diversity policy before any listener starts.
        // This signed history attests only to local operator configuration; it
        // is deliberately not described as an AS or operator identity proof.
        let route_domain_assignments = self.config.discovery.pinned_route_domain_assignments();
        let route_domain_strict = self
            .config
            .discovery
            .require_pinned_route_domains_for_multi_hop;
        // [ROUTE-DOMAIN-ATTESTOR-HISTORY 2026-08-03 by Codex] Persist the
        // verifier's local trust roots before route listeners start. Signatures
        // prove attestor authorship only; no log message exports pin identities.
        let route_domain_attestors = self.config.discovery.route_domain_attestor_node_id_bytes();
        let route_domain_attestor_minimum =
            self.config.discovery.route_domain_attestation_min_verified;
        let route_domain_attestor_strict = self
            .config
            .discovery
            .require_route_domain_attestations_for_multi_hop;
        let (store, audit, reaudited, policy, route_domain_policy, route_domain_attestor_policy) =
            tokio::task::spawn_blocking(move || {
                // [STARTUP-AUDIT-ONCE 2026-10-10 by Claude] open() already ran a
                // complete audit. Reconciliation below writes only when the
                // configured policy changed; with an unchanged policy the store
                // is exactly what that audit verified, so repeating it (a second
                // full pass, ~27 s on JP1) proves nothing new.
                let (store, opened_audit) =
                    DirectoryReplicaStore::open(&open_path, local_node_id, observed_at)?;
                let policy = store.reconcile_observation_witness_policy(
                    &identity,
                    &witness_node_ids,
                    minimum_witnesses,
                    observed_at,
                )?;
                let route_domain_policy = store.reconcile_route_domain_policy(
                    &identity,
                    &route_domain_assignments,
                    route_domain_strict,
                    observed_at,
                )?;
                let route_domain_attestor_policy = store.reconcile_route_domain_attestor_policy(
                    &identity,
                    &route_domain_attestors,
                    route_domain_attestor_minimum,
                    route_domain_attestor_strict,
                    observed_at,
                )?;
                let reaudited = policy.appended
                    || route_domain_policy.appended
                    || route_domain_attestor_policy.appended;
                let audit = if reaudited {
                    store.audit(observed_at)?
                } else {
                    opened_audit
                };
                Ok::<_, crate::services::DirectoryReplicaStoreError>((
                    store,
                    audit,
                    reaudited,
                    policy,
                    route_domain_policy,
                    route_domain_attestor_policy,
                ))
            })
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "Directory replica store task failed before startup: {}",
                    RuntimeTaskJoinFailureKind::classify(&error).blocking_task_reason()
                ))
            })?
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "Directory replica startup audit failed for '{path}': {error}"
                ))
            })?;
        info!(
            producers = audit.producers,
            mirror_producers = audit.mirror_producers,
            quarantined_producers = audit.quarantined_producers,
            blocks = audit.blocks,
            commitments = audit.commitments,
            incidents = audit.incidents,
            resolutions = audit.resolutions,
            observation_checkpoints = audit.observation_checkpoints,
            observation_checkpoint_sequence = audit.observation_checkpoint_sequence,
            observation_checkpoint_witnesses = audit.observation_checkpoint_witnesses,
            observation_checkpoint_witnessed_sequence =
                audit.observation_checkpoint_witnessed_sequence,
            witness_policy_appended = policy.appended,
            witness_policy_epoch = policy.epoch,
            witness_policy_digest = %hex::encode(policy.policy_digest),
            witness_policy_activated_at = policy.activated_at,
            witness_policy_members = policy.witness_members,
            witness_policy_threshold = policy.minimum_witnesses,
            witness_policy_anchor_receipts =
                audit.observation_witness_policy_anchor_receipts,
            remote_witness_policy_heads = audit.observation_witness_remote_policy_anchors,
            route_domain_policy_appended = route_domain_policy.appended,
            route_domain_policy_epoch = route_domain_policy.epoch,
            route_domain_policy_digest = %hex::encode(route_domain_policy.policy_digest),
            route_domain_policy_activated_at = route_domain_policy.activated_at,
            route_domain_policy_assignments = route_domain_policy.assignments,
            route_domain_policy_strict = route_domain_policy.strict_required,
            route_domain_attestor_policy_appended = route_domain_attestor_policy.appended,
            route_domain_attestor_policy_epoch = route_domain_attestor_policy.epoch,
            route_domain_attestor_policy_digest =
                %hex::encode(route_domain_attestor_policy.policy_digest),
            route_domain_attestor_policy_activated_at =
                route_domain_attestor_policy.activated_at,
            route_domain_attestor_policy_members = route_domain_attestor_policy.attestors,
            route_domain_attestor_policy_threshold =
                route_domain_attestor_policy.minimum_attestors,
            route_domain_attestor_policy_strict = route_domain_attestor_policy.strict_required,
            retry_states = audit.retry_states,
            post_reconcile_audit = reaudited,
            "[DIRECTORY_REPLICA] Startup audit and local policy reconciliation passed"
        );
        Ok(Some(Arc::new(store)))
    }
}
