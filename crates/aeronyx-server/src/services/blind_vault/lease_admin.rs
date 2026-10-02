// [ARCH-SPLIT 2026-10-02]
// Lease renew, status, inventory, retire, and admin authentication.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl BlindVaultService {
    /// Loads one immutable lease-authority snapshot without opening a write
    /// transaction. Callers must revalidate the relevant authority bytes after
    /// acquiring their mutation transaction because an expired lease can be
    /// removed and its random identifier reprovisioned between these phases.
    pub(super) fn lease_runtime_snapshot(
        &self,
        lease_id: &[u8; 32],
    ) -> Result<LeaseRuntimeRow, BlindVaultServiceError> {
        let connection = self.connection.lock();
        load_lease_runtime(&connection, lease_id)?.ok_or(BlindVaultServiceError::LeaseNotFound)
    }

    /// Authenticates one private read-only administration observation before
    /// running its potentially larger SQLite snapshot query.
    pub(super) fn authenticate_admin_observation<R>(
        &self,
        request: &R,
        now_ms: u64,
    ) -> Result<[u8; 32], BlindVaultServiceError>
    where
        R: BlindVaultAdminObservationRequest,
    {
        let lease = self.lease_runtime_snapshot(request.lease_id())?;
        if lease.expires_at_ms <= now_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        let admin_key = IdentityPublicKey::from_bytes(&lease.admin_verifying_key)
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        request.validate_and_verify(now_ms, self.config.mutation_clock_skew_ms(), &admin_key)?;
        Ok(lease.admin_verifying_key)
    }

    /// Authenticates a complete lease retirement before acquiring the WAL
    /// writer. Exact retained retries use their tombstone authority and do not
    /// become invalid merely because the original freshness window elapsed.
    pub(super) fn authenticate_lease_retire_request(
        &self,
        request: &BlindVaultLeaseRetireRequest,
        now_ms: u64,
    ) -> Result<[u8; 32], BlindVaultServiceError> {
        let authority = {
            let connection = self.connection.lock();
            if let Some(lease) = load_lease_runtime(&connection, &request.lease_id)? {
                LeaseRetirementAuthoritySnapshot::Live {
                    admin_verifying_key: lease.admin_verifying_key,
                }
            } else if let Some(retired) =
                load_active_lease_retirement(&connection, &request.lease_id, now_ms)?
            {
                LeaseRetirementAuthoritySnapshot::Retired(retired)
            } else {
                return Err(BlindVaultServiceError::LeaseNotFound);
            }
        };

        let admin_verifying_key = match &authority {
            LeaseRetirementAuthoritySnapshot::Live {
                admin_verifying_key,
            } => *admin_verifying_key,
            LeaseRetirementAuthoritySnapshot::Retired(retired) => {
                if retired.request_id != request.request_id
                    || retired.request_commitment != request.commitment()
                {
                    return Err(BlindVaultServiceError::RequestConflict);
                }
                retired.admin_verifying_key
            }
        };
        let admin_key = IdentityPublicKey::from_bytes(&admin_verifying_key)
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        match authority {
            LeaseRetirementAuthoritySnapshot::Live { .. } => request.validate_and_verify(
                now_ms,
                self.config.mutation_clock_skew_ms(),
                &admin_key,
            )?,
            LeaseRetirementAuthoritySnapshot::Retired(_) => {
                request.validate_and_verify_signature(&admin_key)?
            }
        }
        Ok(admin_verifying_key)
    }

    /// Consumes one blind credential and extends one live anonymous lease.
    ///
    /// [BLIND-VAULT-LEASE-RENEWAL-TX 2026-08-28 by Codex] Exact retained
    /// retries are resolved before the spend table is consulted. A first-time
    /// renewal then consumes its credential, compare-and-swaps the expected
    /// lease generation, and records reply evidence in one immediate
    /// transaction; a crash commits all three effects or none.
    pub fn renew_lease_with_blind_admission(
        &self,
        request: &BlindVaultBlindLeaseRenewalRequest,
        now_ms: u64,
    ) -> Result<BlindVaultBlindLeaseRenewedReceipt, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        request.admission.validate_shape()?;
        let spend_id = request.admission.spend_id();
        let request_commitment = request.renewal.commitment();

        // A retained exact retry remains valid after the original mutation
        // freshness window and issuer epoch have elapsed. The marker exists
        // only because the credential and signature were verified at commit.
        {
            let connection = self.connection.lock();
            if let Some(existing) = load_active_lease_renewal(
                &connection,
                &request.renewal.lease_id,
                &request.renewal.request_id,
                now_ms,
            )? {
                let lease = load_lease_runtime(&connection, &request.renewal.lease_id)?
                    .ok_or(BlindVaultServiceError::CorruptState)?;
                ensure_exact_lease_renewal(&existing, request)?;
                let admin_key = IdentityPublicKey::from_bytes(&lease.admin_verifying_key)
                    .map_err(|_| BlindVaultServiceError::CorruptState)?;
                request.renewal.validate_and_verify_signature(&admin_key)?;
                return self.sign_lease_renewal_receipt(request, existing.renewed_at_ms);
            }
        }

        let issuer = self.verify_blind_admission_token(&request.admission, now_ms)?;
        let authenticated_admin_key = {
            let lease = self.lease_runtime_snapshot(&request.renewal.lease_id)?;
            let admin_key = IdentityPublicKey::from_bytes(&lease.admin_verifying_key)
                .map_err(|_| BlindVaultServiceError::CorruptState)?;
            request.renewal.validate_and_verify(
                now_ms,
                self.config.max_lease_ttl_ms().min(issuer.max_lease_ttl_ms),
                self.config.mutation_clock_skew_ms(),
                &admin_key,
            )?;
            lease.admin_verifying_key
        };
        let now = sqlite_i64(now_ms)?;
        let requested_expiry = sqlite_i64(request.renewal.requested_expires_at_ms)?;
        let spend_expiry = sqlite_i64(issuer.expires_at_ms)?;
        let marker_expiry_ms = now_ms
            .checked_add(self.config.tombstone_ttl_secs.saturating_mul(1_000))
            .ok_or(BlindVaultServiceError::TimestampOutOfRange)?;
        let marker_expiry = sqlite_i64(marker_expiry_ms)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let renewal = if let Some(existing) = load_active_lease_renewal(
            &transaction,
            &request.renewal.lease_id,
            &request.renewal.request_id,
            now_ms,
        )? {
            ensure_exact_lease_renewal(&existing, request)?;
            existing
        } else {
            let lease = load_lease_runtime(&transaction, &request.renewal.lease_id)?
                .ok_or(BlindVaultServiceError::LeaseNotFound)?;
            if lease.admin_verifying_key != authenticated_admin_key {
                return Err(BlindVaultServiceError::LeaseConflict);
            }
            if lease.expires_at_ms <= now_ms {
                return Err(BlindVaultServiceError::LeaseExpired);
            }
            if lease.expires_at_ms != request.renewal.expected_expires_at_ms {
                return Err(BlindVaultServiceError::LeaseConflict);
            }
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
            transaction.execute(
                "INSERT INTO blind_vault_admission_spends
                 (token_id, consumed_at_ms, expires_at_ms) VALUES (?1, ?2, ?3)",
                params![&spend_id[..], now, spend_expiry],
            )?;
            let updated = transaction.execute(
                "UPDATE blind_vault_leases SET expires_at_ms = ?2
                 WHERE lease_id = ?1 AND expires_at_ms = ?3",
                params![
                    &request.renewal.lease_id[..],
                    requested_expiry,
                    sqlite_i64(request.renewal.expected_expires_at_ms)?,
                ],
            )?;
            if updated != 1 {
                return Err(BlindVaultServiceError::LeaseConflict);
            }
            transaction.execute(
                "INSERT INTO blind_vault_lease_renewals
                 (lease_id, request_id, admission_spend_id, request_commitment,
                  previous_expires_at_ms, renewed_expires_at_ms, renewed_at_ms,
                  expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    &request.renewal.lease_id[..],
                    &request.renewal.request_id[..],
                    &spend_id[..],
                    &request_commitment[..],
                    sqlite_i64(request.renewal.expected_expires_at_ms)?,
                    requested_expiry,
                    now,
                    marker_expiry,
                ],
            )?;
            LeaseRenewalRow {
                admission_spend_id: spend_id,
                request_commitment,
                previous_expires_at_ms: request.renewal.expected_expires_at_ms,
                renewed_expires_at_ms: request.renewal.requested_expires_at_ms,
                renewed_at_ms: now_ms,
            }
        };
        transaction.commit()?;
        self.sign_lease_renewal_receipt(request, renewal.renewed_at_ms)
    }

    /// Returns one administration-authorized, terminal-signed live-lease view.
    ///
    /// [BLIND-VAULT-LEASE-STATUS-SNAPSHOT 2026-08-28 by Codex] Signature
    /// verification happens before querying aggregate usage. The second
    /// authority check and all returned counters come from one SQLite
    /// statement, so retirement, reprovisioning, cleanup, or renewal cannot
    /// produce a receipt assembled from different lease generations.
    pub fn lease_status(
        &self,
        request: &BlindVaultLeaseStatusRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseStatusReceipt, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        let authenticated_admin_key = self.authenticate_admin_observation(request, now_ms)?;

        let observation = {
            let connection = self.connection.lock();
            load_live_lease_status(&connection, &request.lease_id, now_ms)?
                .ok_or(BlindVaultServiceError::LeaseNotFound)?
        };
        if observation.admin_verifying_key != authenticated_admin_key {
            return Err(BlindVaultServiceError::LeaseConflict);
        }
        if observation.expires_at_ms <= now_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }

        let mut receipt = BlindVaultLeaseStatusReceipt::new(
            request,
            observation.expires_at_ms,
            observation.live_object_count,
            observation.live_ciphertext_bytes,
            now_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    /// Returns one administration-authorized commitment to the live object set.
    ///
    /// [BLIND-VAULT-LEASE-INVENTORY-SNAPSHOT 2026-08-28 by Codex] A deferred
    /// read transaction pins the lease generation and ordered object scan to
    /// one SQLite snapshot. The streaming builder retains only the previous
    /// object ID and aggregate hash state, keeping memory bounded independently
    /// from the lease object limit.
    pub fn lease_inventory(
        &self,
        request: &BlindVaultLeaseInventoryRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseInventoryReceipt, BlindVaultServiceError> {
        if !self.config.public_api_enabled {
            return Err(BlindVaultServiceError::AdmissionUnavailable);
        }
        let authenticated_admin_key = self.authenticate_admin_observation(request, now_ms)?;

        let (lease_expires_at_ms, summary) = {
            let mut connection = self.connection.lock();
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
            let lease = load_lease_runtime(&transaction, &request.lease_id)?
                .ok_or(BlindVaultServiceError::LeaseNotFound)?;
            if lease.admin_verifying_key != authenticated_admin_key {
                return Err(BlindVaultServiceError::LeaseConflict);
            }
            if lease.expires_at_ms <= now_ms {
                return Err(BlindVaultServiceError::LeaseExpired);
            }
            let summary = load_live_lease_inventory(
                &transaction,
                &request.lease_id,
                now_ms,
                lease.expires_at_ms,
            )?;
            transaction.commit()?;
            (lease.expires_at_ms, summary)
        };

        let mut receipt = BlindVaultLeaseInventoryReceipt::new(
            request,
            lease_expires_at_ms,
            summary,
            now_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    /// Retires one complete anonymous lease and returns a node-signed receipt.
    ///
    /// The transaction inserts a bounded idempotency tombstone before deleting
    /// the lease row; foreign-key cascades then remove every ciphertext object
    /// and object tombstone atomically. Exact retries return the original
    /// aggregate result and retirement time.
    pub fn retire_lease(
        &self,
        request: &BlindVaultLeaseRetireRequest,
        now_ms: u64,
    ) -> Result<BlindVaultLeaseRetiredReceipt, BlindVaultServiceError> {
        let authenticated_admin_key = self.authenticate_lease_retire_request(request, now_ms)?;
        let request_commitment = request.commitment();
        let now = sqlite_i64(now_ms)?;
        let tombstone_expiry_ms = now_ms
            .checked_add(self.config.tombstone_ttl_secs.saturating_mul(1_000))
            .ok_or(BlindVaultServiceError::TimestampOutOfRange)?;
        let tombstone_expiry = sqlite_i64(tombstone_expiry_ms)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let retirement = if let Some(lease) = load_lease_runtime(&transaction, &request.lease_id)? {
            if lease.admin_verifying_key != authenticated_admin_key {
                return Err(BlindVaultServiceError::LeaseConflict);
            }
            if load_active_lease_retirement(&transaction, &request.lease_id, now_ms)?.is_some() {
                return Err(BlindVaultServiceError::CorruptState);
            }
            // [BLIND-VAULT-LEASE-RETIRE-USAGE 2026-08-28 by Codex] A signed
            // aggregate receipt must describe rows removed by this exact
            // transaction. Refuse to sign if cached quota counters drifted.
            let deleted_usage = load_lease_object_usage(&transaction, &request.lease_id)?;
            if lease.object_count != deleted_usage.object_count
                || lease.byte_count != deleted_usage.ciphertext_bytes
            {
                return Err(BlindVaultServiceError::CorruptState);
            }
            transaction.execute(
                "DELETE FROM blind_vault_lease_tombstones
                 WHERE lease_id = ?1 AND expires_at_ms <= ?2",
                params![&request.lease_id[..], now],
            )?;
            transaction.execute(
                "INSERT INTO blind_vault_lease_tombstones
                 (lease_id, request_id, admin_verifying_key, retired_at_ms,
                  request_commitment, deleted_object_count,
                  deleted_ciphertext_bytes, expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    &request.lease_id[..],
                    &request.request_id[..],
                    &authenticated_admin_key[..],
                    now,
                    &request_commitment[..],
                    sqlite_i64(deleted_usage.object_count)?,
                    sqlite_i64(deleted_usage.ciphertext_bytes)?,
                    tombstone_expiry,
                ],
            )?;
            let deleted = transaction.execute(
                "DELETE FROM blind_vault_leases WHERE lease_id = ?1",
                params![&request.lease_id[..]],
            )?;
            if deleted != 1 {
                return Err(BlindVaultServiceError::LeaseConflict);
            }
            LeaseRetirementRow {
                request_id: request.request_id,
                admin_verifying_key: authenticated_admin_key,
                request_commitment,
                retired_at_ms: now_ms,
                deleted_object_count: deleted_usage.object_count,
                deleted_ciphertext_bytes: deleted_usage.ciphertext_bytes,
            }
        } else {
            let retired = load_active_lease_retirement(&transaction, &request.lease_id, now_ms)?
                .ok_or(BlindVaultServiceError::LeaseNotFound)?;
            if retired.admin_verifying_key != authenticated_admin_key
                || retired.request_id != request.request_id
                || retired.request_commitment != request_commitment
            {
                return Err(BlindVaultServiceError::RequestConflict);
            }
            retired
        };
        transaction.commit()?;

        let mut receipt = BlindVaultLeaseRetiredReceipt::new(
            request,
            retirement.retired_at_ms,
            retirement.deleted_object_count,
            retirement.deleted_ciphertext_bytes,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    pub(super) fn sign_lease_renewal_receipt(
        &self,
        request: &BlindVaultBlindLeaseRenewalRequest,
        renewed_at_ms: u64,
    ) -> Result<BlindVaultBlindLeaseRenewedReceipt, BlindVaultServiceError> {
        let mut receipt = BlindVaultBlindLeaseRenewedReceipt::new(
            request,
            renewed_at_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }
}
