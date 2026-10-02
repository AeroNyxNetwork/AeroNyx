// [ARCH-SPLIT 2026-10-02]
// Ciphertext put, pull page, delete, and pull-cursor authentication.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl BlindVaultService {
    /// Authenticates an object write before competing for `SQLite`'s single `WAL`
    /// writer. This prevents invalid-signature traffic from holding an
    /// immediate transaction during public-key verification.
    pub(super) fn authenticate_put_request(
        &self,
        request: &BlindVaultPutRequest,
        now_ms: u64,
    ) -> Result<[u8; 32], BlindVaultServiceError> {
        let lease = self.lease_runtime_snapshot(&request.lease_id)?;
        if lease.expires_at_ms <= now_ms || request.expires_at_ms > lease.expires_at_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        let write_key = IdentityPublicKey::from_bytes(&lease.write_verifying_key)
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        request.validate_and_verify(now_ms, self.config.max_object_ttl_ms(), &write_key)?;
        Ok(lease.write_verifying_key)
    }

    /// Authenticates a deletion before competing for `SQLite`'s single `WAL`
    /// writer. Lease expiry semantics remain backward compatible: an
    /// administration key may delete retained ciphertext until cleanup removes
    /// the lease.
    pub(super) fn authenticate_delete_request(
        &self,
        request: &BlindVaultDeleteRequest,
        now_ms: u64,
    ) -> Result<[u8; 32], BlindVaultServiceError> {
        let lease = self.lease_runtime_snapshot(&request.lease_id)?;
        let admin_key = IdentityPublicKey::from_bytes(&lease.admin_verifying_key)
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        request.validate_and_verify(now_ms, self.config.mutation_clock_skew_ms(), &admin_key)?;
        Ok(lease.admin_verifying_key)
    }

    /// Stores one immutable ciphertext and returns a node-signed acceptance
    /// receipt. Exact retries return a new signature over the original
    /// acceptance time without increasing usage.
    pub fn put(
        &self,
        request: &BlindVaultPutRequest,
        now_ms: u64,
    ) -> Result<BlindVaultStoredReceipt, BlindVaultServiceError> {
        // [BLIND-VAULT-AUTH-PIPELINE 2026-07-23 by Codex] Expensive Ed25519
        // verification happens before BEGIN IMMEDIATE. The transaction still
        // compares the authority snapshot to close the delete/reprovision race.
        let authenticated_write_key = self.authenticate_put_request(request, now_ms)?;
        let now = sqlite_i64(now_ms)?;
        let expires_at = sqlite_i64(request.expires_at_ms)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let lease = load_lease_runtime(&transaction, &request.lease_id)?
            .ok_or(BlindVaultServiceError::LeaseNotFound)?;
        if lease.write_verifying_key != authenticated_write_key {
            return Err(BlindVaultServiceError::LeaseConflict);
        }
        if lease.expires_at_ms <= now_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        if request.expires_at_ms > lease.expires_at_ms {
            return Err(BlindVaultServiceError::LeaseExpired);
        }

        let tombstoned: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_tombstones
                WHERE lease_id = ?1 AND object_id = ?2
             )",
            params![&request.lease_id[..], &request.object_id[..]],
            |row| row.get(0),
        )?;
        if tombstoned {
            return Err(BlindVaultServiceError::ObjectDeleted);
        }

        if let Some(existing) = load_existing_object(&transaction, request)? {
            if existing.request_id != request.request_id
                || existing.ciphertext_commitment != request.ciphertext_commitment
                || existing.expires_at_ms != request.expires_at_ms
            {
                return Err(BlindVaultServiceError::ObjectConflict);
            }
            transaction.commit()?;
            return self.sign_store_receipt(request, existing.created_at_ms);
        }

        let request_conflict: bool = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM blind_vault_objects
                WHERE lease_id = ?1 AND request_id = ?2
             )",
            params![&request.lease_id[..], &request.request_id[..]],
            |row| row.get(0),
        )?;
        if request_conflict {
            return Err(BlindVaultServiceError::RequestConflict);
        }

        let object_bytes = u64::try_from(request.ciphertext.len())
            .map_err(|_| BlindVaultServiceError::QuotaExceeded)?;
        let next_count = lease
            .object_count
            .checked_add(1)
            .ok_or(BlindVaultServiceError::QuotaExceeded)?;
        let next_bytes = lease
            .byte_count
            .checked_add(object_bytes)
            .ok_or(BlindVaultServiceError::QuotaExceeded)?;
        if next_count > self.config.max_objects_per_lease
            || next_bytes > self.config.max_bytes_per_lease
        {
            return Err(BlindVaultServiceError::QuotaExceeded);
        }
        // [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] Exact retries have
        // already returned above. This global check and the write share one
        // immediate transaction, preventing concurrent writes from passing on
        // the same remaining capacity. Cached lease bytes may conservatively
        // include objects awaiting bounded expiry cleanup; fail closed until
        // maintenance releases that commitment.
        ensure_node_ciphertext_capacity(
            &transaction,
            object_bytes,
            self.config.max_total_ciphertext_bytes,
        )?;
        // [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] Preserve the physical
        // reserve after this ciphertext write. Exact retries returned above,
        // so a full node can still acknowledge already durable objects.
        self.ensure_filesystem_capacity(object_bytes)?;

        transaction.execute(
            "INSERT INTO blind_vault_objects
             (lease_id, object_id, request_id, ciphertext, ciphertext_commitment,
              created_at_ms, expires_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &request.lease_id[..],
                &request.object_id[..],
                &request.request_id[..],
                &request.ciphertext,
                &request.ciphertext_commitment[..],
                now,
                expires_at,
            ],
        )?;
        transaction.execute(
            "UPDATE blind_vault_leases
             SET object_count = ?2, byte_count = ?3
             WHERE lease_id = ?1",
            params![
                &request.lease_id[..],
                sqlite_i64(next_count)?,
                sqlite_i64(next_bytes)?
            ],
        )?;
        transaction.commit()?;
        self.sign_store_receipt(request, now_ms)
    }

    /// Pulls one bounded page after authenticating a random read capability.
    pub fn pull_page(
        &self,
        lease_id: &[u8; 32],
        read_capability: &[u8; 32],
        continuation_cursor: Option<&[u8]>,
        requested_limit: usize,
        now_ms: u64,
    ) -> Result<BlindVaultPullPage, BlindVaultServiceError> {
        if requested_limit == 0 {
            return Err(BlindVaultServiceError::ReadUnauthorized);
        }
        let now = sqlite_i64(now_ms)?;
        let limit = requested_limit.min(self.config.max_pull_objects);
        let query_limit = i64::try_from(limit.saturating_add(1))
            .map_err(|_| BlindVaultServiceError::CorruptState)?;

        let connection = self.connection.lock();
        let (read_tag, lease_expiry): (Vec<u8>, i64) = connection
            .query_row(
                "SELECT read_capability_tag, expires_at_ms
                 FROM blind_vault_leases WHERE lease_id = ?1",
                params![&lease_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .ok_or(BlindVaultServiceError::LeaseNotFound)?;
        if lease_expiry <= now {
            return Err(BlindVaultServiceError::LeaseExpired);
        }
        self.verify_read_capability(lease_id, read_capability, &read_tag)?;

        // [BLIND-VAULT-CURSOR 2026-07-23 by Codex] The first page freezes a
        // sequence ceiling. Subsequent pages can neither reveal raw SQLite
        // positions nor drift into objects written after recovery began.
        let (after_sequence, ceiling_sequence) = if let Some(cursor) = continuation_cursor {
            self.decode_pull_cursor(lease_id, cursor)?
        } else {
            let ceiling: i64 = connection.query_row(
                "SELECT COALESCE(MAX(sequence), 0)
                 FROM blind_vault_objects
                 WHERE lease_id = ?1 AND expires_at_ms > ?2",
                params![&lease_id[..], now],
                |row| row.get(0),
            )?;
            (
                0,
                u64::try_from(ceiling).map_err(|_| BlindVaultServiceError::CorruptState)?,
            )
        };
        let after = sqlite_i64(after_sequence)?;
        let ceiling = sqlite_i64(ceiling_sequence)?;

        let mut statement = connection.prepare(
            "SELECT sequence, object_id, ciphertext, ciphertext_commitment, expires_at_ms
             FROM blind_vault_objects
             WHERE lease_id = ?1 AND sequence > ?2 AND sequence <= ?3
               AND expires_at_ms > ?4
             ORDER BY sequence ASC LIMIT ?5",
        )?;
        let rows = statement.query_map(
            params![&lease_id[..], after, ceiling, now, query_limit],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )?;

        let mut decoded = Vec::new();
        for row in rows {
            let (sequence, object_id, ciphertext, commitment, expires_at_ms) = row?;
            decoded.push((
                u64::try_from(sequence).map_err(|_| BlindVaultServiceError::CorruptState)?,
                BlindVaultStoredObject {
                    object_id: fixed_array(&object_id)?,
                    ciphertext,
                    ciphertext_commitment: fixed_array(&commitment)?,
                    expires_at_ms: u64::try_from(expires_at_ms)
                        .map_err(|_| BlindVaultServiceError::CorruptState)?,
                },
            ));
        }
        let has_more = decoded.len() > limit;
        if has_more {
            decoded.truncate(limit);
        }
        let continuation_cursor = if has_more {
            let position = decoded
                .last()
                .map(|(sequence, _)| *sequence)
                .ok_or(BlindVaultServiceError::CorruptState)?;
            Some(self.encode_pull_cursor(lease_id, position, ceiling_sequence)?)
        } else {
            None
        };
        Ok(BlindVaultPullPage {
            objects: decoded.into_iter().map(|(_, object)| object).collect(),
            continuation_cursor,
        })
    }

    /// Deletes one object under the separate administration key and returns a
    /// signed receipt. A retained tombstone makes retries idempotent.
    pub fn delete(
        &self,
        request: &BlindVaultDeleteRequest,
        now_ms: u64,
    ) -> Result<BlindVaultDeletedReceipt, BlindVaultServiceError> {
        // [BLIND-VAULT-AUTH-PIPELINE 2026-07-23 by Codex] Reject forged or
        // stale administration requests without acquiring the WAL write lock.
        let authenticated_admin_key = self.authenticate_delete_request(request, now_ms)?;
        let now = sqlite_i64(now_ms)?;
        let tombstone_expiry = now_ms
            .checked_add(self.config.tombstone_ttl_secs.saturating_mul(1_000))
            .ok_or(BlindVaultServiceError::TimestampOutOfRange)?;
        let tombstone_expiry = sqlite_i64(tombstone_expiry)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let lease = load_lease_runtime(&transaction, &request.lease_id)?
            .ok_or(BlindVaultServiceError::LeaseNotFound)?;
        if lease.admin_verifying_key != authenticated_admin_key {
            return Err(BlindVaultServiceError::LeaseConflict);
        }

        let existing: Option<(Vec<u8>, i64)> = transaction
            .query_row(
                "SELECT ciphertext_commitment, length(ciphertext)
                 FROM blind_vault_objects
                 WHERE lease_id = ?1 AND object_id = ?2",
                params![&request.lease_id[..], &request.object_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        let (commitment, deleted_at_ms) = if let Some((commitment, object_bytes)) = existing {
            transaction.execute(
                "DELETE FROM blind_vault_objects
                 WHERE lease_id = ?1 AND object_id = ?2",
                params![&request.lease_id[..], &request.object_id[..]],
            )?;
            transaction.execute(
                "UPDATE blind_vault_leases
                 SET object_count = MAX(object_count - 1, 0),
                     byte_count = MAX(byte_count - ?2, 0)
                 WHERE lease_id = ?1",
                params![&request.lease_id[..], object_bytes],
            )?;
            transaction.execute(
                "INSERT INTO blind_vault_tombstones
                 (lease_id, object_id, ciphertext_commitment, deleted_at_ms, expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(lease_id, object_id) DO UPDATE SET
                   ciphertext_commitment = excluded.ciphertext_commitment,
                   deleted_at_ms = excluded.deleted_at_ms,
                   expires_at_ms = excluded.expires_at_ms",
                params![
                    &request.lease_id[..],
                    &request.object_id[..],
                    &commitment,
                    now,
                    tombstone_expiry,
                ],
            )?;
            (fixed_array(&commitment)?, now_ms)
        } else {
            let tombstone: Option<(Vec<u8>, i64)> = transaction
                .query_row(
                    "SELECT ciphertext_commitment, deleted_at_ms
                     FROM blind_vault_tombstones
                     WHERE lease_id = ?1 AND object_id = ?2 AND expires_at_ms > ?3",
                    params![&request.lease_id[..], &request.object_id[..], now],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let (commitment, deleted_at) =
                tombstone.ok_or(BlindVaultServiceError::ObjectNotFound)?;
            (
                fixed_array(&commitment)?,
                u64::try_from(deleted_at).map_err(|_| BlindVaultServiceError::CorruptState)?,
            )
        };
        transaction.commit()?;

        let mut receipt = BlindVaultDeletedReceipt::new(
            request,
            commitment,
            deleted_at_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    pub(super) fn sign_store_receipt(
        &self,
        request: &BlindVaultPutRequest,
        accepted_at_ms: u64,
    ) -> Result<BlindVaultStoredReceipt, BlindVaultServiceError> {
        let mut receipt = BlindVaultStoredReceipt::from_put(
            request,
            accepted_at_ms,
            request.expires_at_ms,
            self.node_identity.public_key_bytes(),
        );
        receipt.sign(&self.node_identity)?;
        Ok(receipt)
    }

    pub(super) fn read_capability_tag(
        &self,
        lease_id: &[u8; 32],
        capability_hash: &[u8; 32],
    ) -> Result<[u8; 32], BlindVaultServiceError> {
        // [BLIND-VAULT-PANIC-FREE-READ-TAG 2026-08-30 by Codex] A fixed-size
        // HMAC key should always be accepted, but crypto-provider failure is a
        // typed fail-closed service error rather than a node process panic.
        let mut mac = HmacSha256::new_from_slice(self.read_auth_key.as_ref())
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        mac.update(READ_AUTH_TAG_DOMAIN);
        mac.update(lease_id);
        mac.update(capability_hash);
        Ok(mac.finalize().into_bytes().into())
    }

    pub(super) fn verify_read_capability(
        &self,
        lease_id: &[u8; 32],
        capability: &[u8; 32],
        stored_tag: &[u8],
    ) -> Result<(), BlindVaultServiceError> {
        let capability_hash: [u8; 32] = Sha256::digest(capability).into();
        let mut mac = HmacSha256::new_from_slice(self.read_auth_key.as_ref())
            .map_err(|_| BlindVaultServiceError::CorruptState)?;
        mac.update(READ_AUTH_TAG_DOMAIN);
        mac.update(lease_id);
        mac.update(&capability_hash);
        mac.verify_slice(stored_tag)
            .map_err(|_| BlindVaultServiceError::ReadUnauthorized)
    }

    pub(super) fn pull_cursor_aad(lease_id: &[u8; 32]) -> Vec<u8> {
        let mut aad = Vec::with_capacity(PULL_CURSOR_AAD_DOMAIN.len() + lease_id.len());
        aad.extend_from_slice(PULL_CURSOR_AAD_DOMAIN);
        aad.extend_from_slice(lease_id);
        aad
    }

    pub(super) fn encode_pull_cursor(
        &self,
        lease_id: &[u8; 32],
        position: u64,
        ceiling: u64,
    ) -> Result<Vec<u8>, BlindVaultServiceError> {
        if position > ceiling || ceiling > i64::MAX as u64 {
            return Err(BlindVaultServiceError::PullCursorEncryptionFailed);
        }
        let mut plaintext = [0_u8; PULL_CURSOR_PLAINTEXT_BYTES];
        plaintext[..8].copy_from_slice(&position.to_le_bytes());
        plaintext[8..].copy_from_slice(&ceiling.to_le_bytes());

        let mut nonce = [0_u8; PULL_CURSOR_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(self.pull_cursor_key.as_ref()));
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &plaintext,
                    aad: &Self::pull_cursor_aad(lease_id),
                },
            )
            .map_err(|_| BlindVaultServiceError::PullCursorEncryptionFailed)?;
        if ciphertext.len() != PULL_CURSOR_PLAINTEXT_BYTES + PULL_CURSOR_TAG_BYTES {
            return Err(BlindVaultServiceError::PullCursorEncryptionFailed);
        }

        let mut encoded = Vec::with_capacity(PULL_CURSOR_BYTES);
        encoded.push(PULL_CURSOR_VERSION);
        encoded.extend_from_slice(&nonce);
        encoded.extend_from_slice(&ciphertext);
        Ok(encoded)
    }

    pub(super) fn decode_pull_cursor(
        &self,
        lease_id: &[u8; 32],
        encoded: &[u8],
    ) -> Result<(u64, u64), BlindVaultServiceError> {
        if encoded.len() != PULL_CURSOR_BYTES
            || encoded.first().copied() != Some(PULL_CURSOR_VERSION)
        {
            return Err(BlindVaultServiceError::InvalidPullCursor);
        }
        let nonce_start = 1;
        let ciphertext_start = nonce_start + PULL_CURSOR_NONCE_BYTES;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(self.pull_cursor_key.as_ref()));
        let plaintext = cipher
            .decrypt(
                XNonce::from_slice(&encoded[nonce_start..ciphertext_start]),
                Payload {
                    msg: &encoded[ciphertext_start..],
                    aad: &Self::pull_cursor_aad(lease_id),
                },
            )
            .map_err(|_| BlindVaultServiceError::InvalidPullCursor)?;
        if plaintext.len() != PULL_CURSOR_PLAINTEXT_BYTES {
            return Err(BlindVaultServiceError::InvalidPullCursor);
        }

        let mut position = [0_u8; 8];
        position.copy_from_slice(&plaintext[..8]);
        let mut ceiling = [0_u8; 8];
        ceiling.copy_from_slice(&plaintext[8..]);
        let position = u64::from_le_bytes(position);
        let ceiling = u64::from_le_bytes(ceiling);
        if position > ceiling || ceiling > i64::MAX as u64 {
            return Err(BlindVaultServiceError::InvalidPullCursor);
        }
        Ok((position, ceiling))
    }
}
