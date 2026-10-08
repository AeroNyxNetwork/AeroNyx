// [ARCH-SPLIT 2026-10-02]
// Blind admission readiness, issuer epochs, and lease provisioning.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn replica_authorization_take<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], BlindVaultReplicaJobAuthorizationError> {
    let end = cursor
        .checked_add(N)
        .ok_or(BlindVaultReplicaJobAuthorizationError::Rejected)?;
    let value = bytes
        .get(*cursor..end)
        .ok_or(BlindVaultReplicaJobAuthorizationError::Rejected)?
        .try_into()
        .map_err(|_| BlindVaultReplicaJobAuthorizationError::Rejected)?;
    *cursor = end;
    Ok(value)
}

pub(super) fn build_blind_issuer_runtime(
    generation: u64,
    mut epochs: Vec<BlindVaultBlindIssuerEpoch>,
    validated_at_ms: u64,
    require_active_epoch: bool,
    node_max_lease_ttl_ms: u64,
    node_identity: &IdentityKeyPair,
) -> Result<BlindAdmissionIssuerRuntime, BlindVaultServiceError> {
    if generation > i64::MAX as u64 {
        return Err(BlindVaultServiceError::IssuerDirectoryGenerationOutOfRange);
    }
    epochs.sort_by_key(|epoch| epoch.issuer_key_id);
    if epochs
        .iter()
        .any(|epoch| epoch.max_lease_ttl_ms > node_max_lease_ttl_ms)
    {
        return Err(BlindVaultServiceError::AdmissionConfigurationInvalid);
    }

    // [BLIND-VAULT-ISSUER-RUNTIME 2026-07-23 by Codex] Reuse the protocol's
    // canonical directory validator instead of maintaining a weaker parallel
    // interpretation of epoch bounds, ordering, and key fingerprints.
    let mut validated_directory = BlindVaultBlindIssuerDirectory::new(
        validated_at_ms,
        node_identity.public_key_bytes(),
        epochs.clone(),
    );
    validated_directory
        .sign(node_identity)
        .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?;
    if require_active_epoch
        && !epochs.iter().any(|epoch| {
            epoch.not_before_ms <= validated_at_ms && validated_at_ms < epoch.expires_at_ms
        })
    {
        return Err(BlindVaultServiceError::IssuerDirectoryNoActiveEpoch);
    }

    let mut issuers = HashMap::with_capacity(epochs.len());
    for epoch in &epochs {
        let public_key = PublicKeySha384PSSRandomized::from_der(&epoch.public_key_der)
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?;
        let canonical_der = public_key
            .to_der()
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?;
        if canonical_der != epoch.public_key_der {
            return Err(BlindVaultServiceError::AdmissionConfigurationInvalid);
        }
        if issuers
            .insert(
                epoch.issuer_key_id,
                BlindAdmissionIssuer {
                    public_key: Arc::new(public_key),
                    not_before_ms: epoch.not_before_ms,
                    expires_at_ms: epoch.expires_at_ms,
                    max_lease_ttl_ms: epoch.max_lease_ttl_ms,
                },
            )
            .is_some()
        {
            return Err(BlindVaultServiceError::AdmissionConfigurationInvalid);
        }
    }
    let digest = blind_issuer_set_digest(&epochs)?;
    Ok(BlindAdmissionIssuerRuntime {
        generation,
        digest,
        updated_at_ms: validated_at_ms,
        epochs,
        issuers,
    })
}

pub(super) fn blind_issuer_set_digest(
    epochs: &[BlindVaultBlindIssuerEpoch],
) -> Result<[u8; 32], BlindVaultServiceError> {
    let epoch_count = u16::try_from(epochs.len())
        .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?;
    let mut hasher = Sha256::new();
    hasher.update(BLIND_ISSUER_SET_DIGEST_DOMAIN);
    hasher.update(epoch_count.to_be_bytes());
    for epoch in epochs {
        let key_length = u16::try_from(epoch.public_key_der.len())
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?;
        hasher.update(epoch.admission_version.to_be_bytes());
        hasher.update(epoch.issuer_key_id);
        hasher.update(key_length.to_be_bytes());
        hasher.update(&epoch.public_key_der);
        hasher.update(epoch.not_before_ms.to_be_bytes());
        hasher.update(epoch.expires_at_ms.to_be_bytes());
        hasher.update(epoch.max_lease_ttl_ms.to_be_bytes());
    }
    Ok(hasher.finalize().into())
}

pub(super) fn load_persisted_blind_issuer_runtime(
    connection: &Connection,
    bootstrap: BlindAdmissionIssuerRuntime,
    node_max_lease_ttl_ms: u64,
    node_identity: &IdentityKeyPair,
) -> Result<BlindAdmissionIssuerRuntime, BlindVaultServiceError> {
    let persisted: Option<(i64, Vec<u8>, i64)> = connection
        .query_row(
            "SELECT generation, digest, updated_at_ms
             FROM blind_vault_blind_issuer_state WHERE state_id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((generation, digest, updated_at_ms)) = persisted else {
        return Ok(bootstrap);
    };
    let generation = non_negative_u64(generation)?;
    let updated_at_ms = non_negative_u64(updated_at_ms)?;
    let persisted_digest: [u8; 32] = fixed_array(&digest)?;

    let mut statement = connection.prepare(
        "SELECT issuer_key_id, admission_version, public_key_der,
                not_before_ms, expires_at_ms, max_lease_ttl_ms
         FROM blind_vault_blind_issuer_epochs
         ORDER BY issuer_key_id ASC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, Vec<u8>>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, Vec<u8>>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, i64>(5)?,
        ))
    })?;
    let mut epochs = Vec::new();
    for row in rows {
        let (
            issuer_key_id,
            admission_version,
            public_key_der,
            not_before_ms,
            expires_at_ms,
            max_lease_ttl_ms,
        ) = row?;
        let admission_version =
            u16::try_from(admission_version).map_err(|_| BlindVaultServiceError::CorruptState)?;
        epochs.push(BlindVaultBlindIssuerEpoch {
            admission_version,
            issuer_key_id: fixed_array(&issuer_key_id)?,
            public_key_der,
            not_before_ms: non_negative_u64(not_before_ms)?,
            expires_at_ms: non_negative_u64(expires_at_ms)?,
            max_lease_ttl_ms: non_negative_u64(max_lease_ttl_ms)?,
        });
    }
    if epochs.is_empty() {
        return Err(BlindVaultServiceError::CorruptState);
    }
    let runtime = build_blind_issuer_runtime(
        generation,
        epochs,
        updated_at_ms,
        false,
        node_max_lease_ttl_ms,
        node_identity,
    )
    .map_err(|_| BlindVaultServiceError::CorruptState)?;
    if runtime.digest != persisted_digest {
        return Err(BlindVaultServiceError::CorruptState);
    }
    Ok(runtime)
}

pub(super) fn persist_blind_issuer_runtime(
    connection: &Mutex<Connection>,
    runtime: &BlindAdmissionIssuerRuntime,
    updated_at_ms: u64,
) -> Result<(), BlindVaultServiceError> {
    let generation = i64::try_from(runtime.generation)
        .map_err(|_| BlindVaultServiceError::IssuerDirectoryGenerationOutOfRange)?;
    let updated_at_ms = sqlite_i64(updated_at_ms)?;
    let mut connection = connection.lock();
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute("DELETE FROM blind_vault_blind_issuer_epochs", [])?;
    for epoch in &runtime.epochs {
        transaction.execute(
            "INSERT INTO blind_vault_blind_issuer_epochs
             (issuer_key_id, admission_version, public_key_der, not_before_ms,
              expires_at_ms, max_lease_ttl_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &epoch.issuer_key_id[..],
                i64::from(epoch.admission_version),
                &epoch.public_key_der,
                sqlite_i64(epoch.not_before_ms)?,
                sqlite_i64(epoch.expires_at_ms)?,
                sqlite_i64(epoch.max_lease_ttl_ms)?,
            ],
        )?;
    }
    transaction.execute(
        "INSERT INTO blind_vault_blind_issuer_state
         (state_id, generation, digest, updated_at_ms)
         VALUES (1, ?1, ?2, ?3)
         ON CONFLICT(state_id) DO UPDATE SET
           generation = excluded.generation,
           digest = excluded.digest,
           updated_at_ms = excluded.updated_at_ms",
        params![generation, &runtime.digest[..], updated_at_ms],
    )?;
    transaction.commit()?;
    Ok(())
}

pub(super) fn derive_node_key(
    seed: &[u8],
    domain: &[u8],
) -> Result<Zeroizing<[u8; 32]>, BlindVaultServiceError> {
    let mut key_deriver =
        HmacSha256::new_from_slice(seed).map_err(|_| BlindVaultServiceError::CorruptState)?;
    key_deriver.update(domain);
    Ok(Zeroizing::new(key_deriver.finalize().into_bytes().into()))
}

impl BlindVaultService {
    /// Whether general direct Blind Vault HTTP operations are enabled.
    // [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex]
    pub(crate) const fn public_api_enabled(&self) -> bool {
        self.config.public_api_enabled
    }

    /// Checks that the durable store can answer a private terminal Pull.
    /// This deliberately does not require public mutation issuers or free
    /// admission capacity: Pull is read-only and independently source-sealed.
    // [PRIVATE-ONION-PULL-READINESS 2026-10-05 by Codex]
    pub fn private_terminal_readiness(&self) -> Result<(), BlindVaultServiceError> {
        self.connection.lock().query_row("SELECT 1", [], |_| Ok(()))?;
        Ok(())
    }

    /// Returns a single privacy-safe admission decision for discovery,
    /// heartbeat, and local operations.
    ///
    /// # Errors
    /// Returns a storage error when the durable logical-capacity singleton
    /// cannot be read. Callers must treat errors as not ready.
    pub fn admission_readiness(
        &self,
        now_ms: u64,
    ) -> Result<BlindVaultAdmissionReadiness, BlindVaultServiceError> {
        // [BLIND-VAULT-ADMISSION-READINESS 2026-08-28 by Codex] Authority-only
        // bootstrap is intentionally not ready until an authenticated active
        // V2 generation arrives. A configured V1 issuer remains compatible.
        if !self.config.public_api_enabled {
            return Ok(BlindVaultAdmissionReadiness::PolicyUnavailable);
        }
        let has_v1_issuer = !self.admission_issuers.is_empty();
        let has_active_v2_issuer = self
            .blind_admission_issuers
            .read()
            .epochs
            .iter()
            .any(|epoch| epoch.not_before_ms <= now_ms && now_ms < epoch.expires_at_ms);
        if !has_v1_issuer && !has_active_v2_issuer {
            return Ok(BlindVaultAdmissionReadiness::PolicyUnavailable);
        }

        let (committed_leases, committed_bytes): (i64, i64) = self.connection.lock().query_row(
            "SELECT committed_lease_count, committed_ciphertext_bytes
                 FROM blind_vault_capacity_state WHERE state_id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if non_negative_u64(committed_leases)? >= self.config.max_live_leases
            || non_negative_u64(committed_bytes)? >= self.config.max_total_ciphertext_bytes
        {
            return Ok(BlindVaultAdmissionReadiness::LogicalCapacityExhausted);
        }

        match self.ensure_filesystem_capacity(0) {
            Ok(()) => Ok(BlindVaultAdmissionReadiness::Ready),
            Err(BlindVaultServiceError::NodeCapacityExceeded) => {
                Ok(BlindVaultAdmissionReadiness::PhysicalCapacityExhausted)
            }
            Err(BlindVaultServiceError::FilesystemCapacityUnavailable) => {
                Ok(BlindVaultAdmissionReadiness::PhysicalCapacityUnknown)
            }
            Err(error) => Err(error),
        }
    }

    /// Returns deterministic, public, non-expired V2 issuer epochs.
    ///
    /// Future keys are intentionally included so clients can prefetch rotation
    /// material. Expired keys and all issuer-private state are omitted.
    ///
    /// # Errors
    /// Returns `AdmissionConfigurationInvalid` if canonical key material no
    /// longer matches the validated configuration snapshot.
    pub fn blind_admission_issuer_epochs(
        &self,
        now_ms: u64,
    ) -> Result<Vec<BlindVaultBlindIssuerEpoch>, BlindVaultServiceError> {
        Ok(self
            .blind_admission_issuers
            .read()
            .epochs
            .iter()
            .filter(|epoch| epoch.expires_at_ms > now_ms)
            .cloned()
            .collect())
    }

    /// Returns aggregate-only runtime issuer state.
    #[must_use]
    pub fn blind_admission_issuer_runtime_status(
        &self,
        now_ms: u64,
    ) -> BlindVaultIssuerRuntimeStatus {
        let runtime = self.blind_admission_issuers.read();
        BlindVaultIssuerRuntimeStatus {
            generation: runtime.generation,
            updated_at_ms: runtime.updated_at_ms,
            epoch_count: runtime.epochs.len(),
            active_epoch_count: runtime
                .epochs
                .iter()
                .filter(|epoch| epoch.not_before_ms <= now_ms && now_ms < epoch.expires_at_ms)
                .count(),
        }
    }

    /// Verifies and durably installs one authority-signed issuer generation.
    ///
    /// The signed object is independent of the management transport. Nodes
    /// therefore enforce the same authority, freshness, monotonicity, and
    /// continuity policy for an authenticated backend channel, offline
    /// operator tool, or future node-to-node control plane.
    ///
    /// # Errors
    /// Returns a coarse fail-closed error for an unpinned authority, malformed
    /// signature, stale update, rollback, conflict, or continuity failure.
    pub fn install_signed_blind_admission_issuer_update(
        &self,
        update: &BlindVaultBlindIssuerUpdate,
        now_ms: u64,
    ) -> Result<BlindVaultIssuerInstallOutcome, BlindVaultServiceError> {
        // [BLIND-VAULT-ISSUER-AUTHORITY 2026-07-23 by Codex] Resolve by the
        // signed authority ID before verification and collapse all protocol
        // failures so management callers cannot use the node as a signature or
        // freshness oracle.
        let authority = self
            .blind_issuer_update_authorities
            .get(&update.authority_id)
            .ok_or(BlindVaultServiceError::IssuerDirectoryAuthorityRejected)?;
        update
            .validate_and_verify(
                now_ms,
                self.config.blind_issuer_update_max_age_ms(),
                self.config.mutation_clock_skew_ms(),
                authority,
            )
            .map_err(|_| BlindVaultServiceError::IssuerDirectoryUpdateRejected)?;
        self.install_blind_admission_issuer_epochs(update.generation, update.epochs.clone(), now_ms)
    }

    /// Durably installs one cryptographically authenticated issuer generation.
    ///
    /// This private state-machine boundary independently validates canonical
    /// RSA public keys, bounded policy, monotonic generation, active-key
    /// availability, and continuity of every still-valid published epoch.
    ///
    /// # Errors
    /// Returns a coarse fail-closed error for malformed, stale, conflicting, or
    /// continuity-breaking candidates and leaves the active runtime unchanged.
    pub(super) fn install_blind_admission_issuer_epochs(
        &self,
        generation: u64,
        epochs: Vec<BlindVaultBlindIssuerEpoch>,
        now_ms: u64,
    ) -> Result<BlindVaultIssuerInstallOutcome, BlindVaultServiceError> {
        let candidate = build_blind_issuer_runtime(
            generation,
            epochs,
            now_ms,
            true,
            self.config.max_lease_ttl_ms(),
            &self.node_identity,
        )?;
        let mut current = self.blind_admission_issuers.write();
        if now_ms < current.updated_at_ms {
            return Err(BlindVaultServiceError::IssuerDirectoryRollback);
        }
        if candidate.generation < current.generation {
            return Err(BlindVaultServiceError::IssuerDirectoryRollback);
        }
        if candidate.generation == current.generation {
            return if candidate.digest == current.digest {
                Ok(BlindVaultIssuerInstallOutcome::Unchanged { generation })
            } else {
                Err(BlindVaultServiceError::IssuerDirectoryGenerationConflict)
            };
        }
        for current_epoch in current
            .epochs
            .iter()
            .filter(|epoch| epoch.expires_at_ms > now_ms)
        {
            let unchanged = candidate.epochs.iter().any(|candidate_epoch| {
                candidate_epoch.issuer_key_id == current_epoch.issuer_key_id
                    && candidate_epoch == current_epoch
            });
            if !unchanged {
                return Err(BlindVaultServiceError::IssuerDirectoryContinuity);
            }
        }

        persist_blind_issuer_runtime(&self.connection, &candidate, now_ms)?;
        let installed_generation = candidate.generation;
        *current = candidate;
        Ok(BlindVaultIssuerInstallOutcome::Installed {
            generation: installed_generation,
        })
    }

    /// Atomically consumes one operator-approved anonymous bearer ticket and
    /// provisions its self-authenticating lease. Exact retries remain
    /// idempotent; the same ticket cannot create a second lease.
    pub fn provision_lease_with_admission(
        &self,
        request: &BlindVaultLeaseAdmissionRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        let issuer_key = self
            .admission_issuers
            .get(&request.admission.issuer_id)
            .ok_or(BlindVaultServiceError::AdmissionIssuerRejected)?;
        request.validate_and_verify(
            now_ms,
            self.config.max_lease_ttl_ms(),
            self.config.max_admission_ticket_ttl_ms(),
            issuer_key,
        )?;

        self.provision_validated_admission(
            &request.lease,
            &request.admission.token_id,
            request.admission.expires_at_ms,
            now_ms,
        )
    }

    /// Verifies and atomically spends an unlinkable RFC 9474 credential.
    pub fn provision_lease_with_blind_admission(
        &self,
        request: &BlindVaultBlindLeaseAdmissionRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        let issuer = self.verify_blind_admission_token(&request.admission, now_ms)?;
        request.lease.validate_and_verify(
            now_ms,
            self.config.max_lease_ttl_ms().min(issuer.max_lease_ttl_ms),
        )?;

        self.provision_validated_admission(
            &request.lease,
            &request.admission.spend_id(),
            issuer.expires_at_ms,
            now_ms,
        )
    }

    /// Verifies one RFC 9474 credential against an active pinned issuer epoch.
    ///
    /// [BLIND-VAULT-BLIND-ADMISSION-VERIFY 2026-08-28 by Codex] Provisioning
    /// and renewal share this cryptographic boundary so token shape, epoch
    /// activity, randomized-message verification, and failure semantics cannot
    /// drift between two capacity-consuming state transitions.
    pub(super) fn verify_blind_admission_token(
        &self,
        token: &BlindVaultBlindAdmissionToken,
        now_ms: u64,
    ) -> Result<BlindAdmissionIssuer, BlindVaultServiceError> {
        token.validate_shape()?;
        let issuer = self
            .blind_admission_issuers
            .read()
            .issuers
            .get(&token.issuer_key_id)
            .cloned()
            .ok_or(BlindVaultServiceError::AdmissionIssuerRejected)?;
        if now_ms < issuer.not_before_ms || now_ms >= issuer.expires_at_ms {
            return Err(BlindVaultServiceError::AdmissionIssuerRejected);
        }
        let signature = Signature::new(token.signature.clone());
        let randomizer = MessageRandomizer::new(token.message_randomizer);
        issuer
            .public_key
            .verify(&signature, Some(randomizer), token.message_bytes())
            .map_err(|_| BlindVaultServiceError::AdmissionProofRejected)?;
        Ok(issuer)
    }

    pub(super) fn provision_validated_admission(
        &self,
        lease: &BlindVaultLeaseCreateRequest,
        spend_id: &[u8; 32],
        spend_expires_at_ms: u64,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseProvisionOutcome, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let lease_expires_at = sqlite_i64(lease.expires_at_ms)?;
        let spend_expires_at = sqlite_i64(spend_expires_at_ms)?;
        let read_tag = self.read_capability_tag(&lease.lease_id, &lease.read_capability_hash)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(outcome) = existing_lease_outcome(&transaction, lease, &read_tag)? {
            return Ok(outcome);
        }
        // [BLIND-VAULT-LEASE-RETIRE-REPLAY 2026-08-28 by Codex] A retired
        // random lease identifier cannot be resurrected while its bounded
        // retirement proof remains valid. Expired markers are removed before
        // allowing a genuinely new admission transaction.
        let retired: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_lease_tombstones
                WHERE lease_id = ?1 AND expires_at_ms > ?2
             )",
            params![&lease.lease_id[..], now],
            |row| row.get(0),
        )?;
        if retired {
            return Err(BlindVaultServiceError::LeaseConflict);
        }
        transaction.execute(
            "DELETE FROM blind_vault_lease_tombstones
             WHERE lease_id = ?1 AND expires_at_ms <= ?2",
            params![&lease.lease_id[..], now],
        )?;
        let spent: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_admission_spends WHERE token_id = ?1
             )",
            params![&spend_id[..]],
            |row| row.get(0),
        )?;
        if spent {
            return Err(BlindVaultServiceError::AdmissionSpent);
        }
        ensure_lease_request_available(&transaction, &lease.request_id)?;
        // [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] Capacity is checked
        // after exact-retry/conflict handling but before consuming the blind
        // credential. The immediate transaction serializes anonymous lease
        // admissions, so concurrent valid credentials cannot oversubscribe the
        // operator's retained-lease commitment.
        ensure_node_admission_capacity(
            &transaction,
            self.config.max_live_leases,
            self.config.max_total_ciphertext_bytes,
        )?;
        // [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] A lease row consumes
        // little space but admission still stops once the safety watermark is
        // crossed. Keep this after exact-retry handling and before token spend.
        self.ensure_filesystem_capacity(0)?;

        // [BLIND-VAULT-ADMISSION 2026-07-23 by Codex] V1 raw token IDs and
        // V2 domain-separated spend IDs share one transaction and replay table.
        // A crash commits both spend and lease or neither.
        transaction.execute(
            "INSERT INTO blind_vault_admission_spends
             (token_id, consumed_at_ms, expires_at_ms) VALUES (?1, ?2, ?3)",
            params![&spend_id[..], now, spend_expires_at],
        )?;
        insert_lease_row(&transaction, lease, &read_tag, now, lease_expires_at)?;
        transaction.commit()?;
        Ok(BlindVaultLeaseProvisionOutcome::Created)
    }

    /// Verifies explicit replication authority without exposing the verifier.
    ///
    /// The single query reads no ciphertext and this method performs no
    /// transaction, durable mutation, token spend, counter update, telemetry,
    /// recovery staging, or network work on either success or failure.
    /// This proves only the exact typed claims in `authorization`: a future
    /// coordinator must stage and dispatch those same source, target, and
    /// bundle claims, never verify one target/bundle and substitute another.
    pub(crate) fn verify_replica_job_authorization(
        &self,
        authorization: &BlindVaultReplicaJobAuthorizationV1,
        now_ms: u64,
    ) -> Result<(), BlindVaultReplicaJobAuthorizationError> {
        // [BLIND-VAULT-REPLICA-AUTH 2026-09-01 by Codex] Validate every
        // caller-controlled field before resolving private lease authority.
        authorization.validate_shape(now_ms)?;
        let claims = authorization.claims();
        let authority = {
            let connection = self.connection.lock();
            load_replica_job_authority(
                &connection,
                &claims.source_lease_id,
                &claims.source_object_id,
            )?
            .ok_or(BlindVaultReplicaJobAuthorizationError::Rejected)?
        };

        if authority.object_expires_at_ms > authority.lease_expires_at_ms
            || authority.ciphertext_commitment == [0; 32]
        {
            return Err(BlindVaultReplicaJobAuthorizationError::Unavailable);
        }
        // [BLIND-VAULT-REPLICA-AUTH 2026-09-01 by Codex] A node may already
        // administer an otherwise valid lease, but its identity is never
        // client replication authority. Reject the collision only here so
        // ordinary lease provisioning and opaque object storage stay intact.
        if authority.admin_verifying_key == self.node_identity.public_key_bytes() {
            return Err(BlindVaultReplicaJobAuthorizationError::Rejected);
        }
        if authority.lease_expires_at_ms <= now_ms
            || authority.object_expires_at_ms <= now_ms
            || claims.expires_at_ms > authority.lease_expires_at_ms
            || claims.expires_at_ms > authority.object_expires_at_ms
            || authority.ciphertext_commitment != claims.source_ciphertext_commitment
        {
            return Err(BlindVaultReplicaJobAuthorizationError::Rejected);
        }
        let admin_key = IdentityPublicKey::from_bytes(&authority.admin_verifying_key)
            .map_err(|_| BlindVaultReplicaJobAuthorizationError::Unavailable)?;
        admin_key
            .verify(&authorization.signing_bytes(), &authorization.signature)
            .map_err(|_| BlindVaultReplicaJobAuthorizationError::Rejected)
    }
}
