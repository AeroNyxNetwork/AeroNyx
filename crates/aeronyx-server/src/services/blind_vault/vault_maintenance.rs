// [ARCH-SPLIT 2026-10-02]
// Cleanup and node-blind service status.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl BlindVaultService {
    /// Removes a bounded number of expired rows and repairs lease usage inside
    /// one immediate transaction.
    pub fn run_cleanup(
        &self,
        now_ms: u64,
    ) -> Result<BlindVaultCleanupReport, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let tombstone_expiry = sqlite_i64(
            now_ms
                .checked_add(self.config.tombstone_ttl_secs.saturating_mul(1_000))
                .ok_or(BlindVaultServiceError::TimestampOutOfRange)?,
        )?;
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let expired_objects = select_expired_objects(&transaction, now)?;
        for object in &expired_objects {
            transaction.execute(
                "DELETE FROM blind_vault_objects WHERE sequence = ?1",
                params![object.sequence],
            )?;
            transaction.execute(
                "UPDATE blind_vault_leases
                 SET object_count = MAX(object_count - 1, 0),
                     byte_count = MAX(byte_count - ?2, 0)
                 WHERE lease_id = ?1",
                params![&object.lease_id, object.ciphertext_bytes],
            )?;
            transaction.execute(
                "INSERT OR IGNORE INTO blind_vault_tombstones
                 (lease_id, object_id, ciphertext_commitment, deleted_at_ms, expires_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    &object.lease_id,
                    &object.object_id,
                    &object.ciphertext_commitment,
                    now,
                    tombstone_expiry,
                ],
            )?;
        }

        let expired_leases = select_expired_leases(&transaction, now)?;
        for lease_id in &expired_leases {
            transaction.execute(
                "DELETE FROM blind_vault_leases WHERE lease_id = ?1",
                params![lease_id],
            )?;
        }

        let expired_tombstones = transaction.execute(
            "DELETE FROM blind_vault_tombstones
             WHERE (lease_id, object_id) IN (
                SELECT lease_id, object_id FROM blind_vault_tombstones
                WHERE expires_at_ms <= ?1 LIMIT ?2
             )",
            params![now, CLEANUP_TOMBSTONE_BATCH as i64],
        )?;
        let expired_lease_tombstones = transaction.execute(
            "DELETE FROM blind_vault_lease_tombstones
             WHERE lease_id IN (
                SELECT lease_id FROM blind_vault_lease_tombstones
                WHERE expires_at_ms <= ?1 LIMIT ?2
             )",
            params![now, CLEANUP_TOMBSTONE_BATCH as i64],
        )?;
        let expired_lease_renewals = transaction.execute(
            "DELETE FROM blind_vault_lease_renewals
             WHERE (lease_id, request_id) IN (
                SELECT lease_id, request_id FROM blind_vault_lease_renewals
                WHERE expires_at_ms <= ?1 LIMIT ?2
             )",
            params![now, CLEANUP_TOMBSTONE_BATCH as i64],
        )?;
        let expired_admission_spends = transaction.execute(
            "DELETE FROM blind_vault_admission_spends
             WHERE token_id IN (
                SELECT token_id FROM blind_vault_admission_spends
                WHERE expires_at_ms <= ?1 LIMIT ?2
             )",
            params![now, CLEANUP_ADMISSION_SPEND_BATCH as i64],
        )?;
        transaction.commit()?;

        Ok(BlindVaultCleanupReport {
            objects_removed: expired_objects.len() as u64,
            leases_removed: expired_leases.len() as u64,
            tombstones_removed: expired_tombstones as u64,
            lease_tombstones_removed: expired_lease_tombstones as u64,
            lease_renewals_removed: expired_lease_renewals as u64,
            admission_spends_removed: expired_admission_spends as u64,
        })
    }

    /// Returns only node-wide aggregate health information.
    pub fn status(&self, now_ms: u64) -> Result<BlindVaultStatus, BlindVaultServiceError> {
        let now = sqlite_i64(now_ms)?;
        let (available_disk_bytes, physical_capacity_ready) = self.filesystem_capacity_status();
        let connection = self.connection.lock();
        let live_leases: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_leases WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        let (committed_leases, committed_bytes): (i64, i64) = connection.query_row(
            "SELECT committed_lease_count, committed_ciphertext_bytes
             FROM blind_vault_capacity_state WHERE state_id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let (live_objects, live_bytes): (i64, i64) = connection.query_row(
            "SELECT COUNT(*), COALESCE(SUM(length(ciphertext)), 0)
             FROM blind_vault_objects WHERE expires_at_ms > ?1",
            params![now],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let tombstones: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_tombstones WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        let lease_tombstones: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_lease_tombstones WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        let lease_renewals: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_lease_renewals WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        let retained_admission_spends: i64 = connection.query_row(
            "SELECT COUNT(*) FROM blind_vault_admission_spends WHERE expires_at_ms > ?1",
            params![now],
            |row| row.get(0),
        )?;
        let live_leases = non_negative_u64(live_leases)?;
        let live_objects = non_negative_u64(live_objects)?;
        let live_bytes = non_negative_u64(live_bytes)?;
        let committed_leases = non_negative_u64(committed_leases)?;
        let committed_bytes = non_negative_u64(committed_bytes)?;
        Ok(BlindVaultStatus {
            enabled: true,
            live_leases,
            committed_leases,
            live_objects,
            live_ciphertext_bytes: live_bytes,
            committed_ciphertext_bytes: committed_bytes,
            max_live_leases: self.config.max_live_leases,
            remaining_live_leases: self.config.max_live_leases.saturating_sub(committed_leases),
            max_total_ciphertext_bytes: self.config.max_total_ciphertext_bytes,
            remaining_ciphertext_bytes: self
                .config
                .max_total_ciphertext_bytes
                .saturating_sub(committed_bytes),
            min_free_disk_bytes: self.config.min_free_disk_bytes,
            available_disk_bytes,
            physical_capacity_ready,
            tombstones: non_negative_u64(tombstones)?,
            lease_tombstones: non_negative_u64(lease_tombstones)?,
            lease_renewals: non_negative_u64(lease_renewals)?,
            retained_admission_spends: non_negative_u64(retained_admission_spends)?,
        })
    }
}
