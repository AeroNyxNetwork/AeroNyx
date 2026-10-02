// [ARCH-SPLIT 2026-10-02]
// Sync runtime, SQLite durability, coordinator fence, and chain integrity status.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

// [CUSTODY-WITNESS-FULL-DURABILITY 2026-09-24 by Codex] The commitment
// coordinator and receipt vault share one SQLite connection and must interpret
// its effective synchronous mode identically. An unknown mode is never proof
// that an acknowledged transaction survives host power loss.
pub(in crate::services::memchain) fn read_sqlite_durability(
    connection: &rusqlite::Connection,
) -> Result<(i64, &'static str), String> {
    let level: i64 = connection
        .query_row("PRAGMA synchronous", [], |row| row.get(0))
        .map_err(|error| format!("read SQLite durability: {error}"))?;
    match level {
        0 => Ok((level, "off")),
        1 => Ok((level, "normal")),
        2 => Ok((level, "full")),
        3 => Ok((level, "extra")),
        _ => Err(format!("unsupported SQLite synchronous level {level}")),
    }
}

pub(in crate::services::memchain) fn ensure_full_sqlite_durability(
    connection: &rusqlite::Connection,
) -> Result<(i64, &'static str), String> {
    if read_sqlite_durability(connection)?.0 < 2 {
        connection
            .pragma_update(None, "synchronous", "FULL")
            .map_err(|error| format!("set SQLite FULL durability: {error}"))?;
    }
    let (level, mode) = read_sqlite_durability(connection)?;
    if level < 2 {
        return Err(format!(
            "SQLite FULL durability is required; effective mode is {mode}"
        ));
    }
    Ok((level, mode))
}

impl MemoryStorage {
    // [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] The private receipt
    // module reports the verified effective mode without exposing the atomic
    // or widening access to the raw SQLite connection.
    pub(in crate::services::memchain) fn note_effective_sqlite_durability(&self, level: i64) {
        self.commitment_durability
            .store(level as u64, Ordering::Release);
    }

    /// Configures the process-local Block Sync role once startup validation has
    /// completed. This does not alter SQLite or the canonical block chain.
    pub fn configure_record_commitment_sync(&self, coordinator: bool, follower: bool) {
        let mut runtime = self.commitment_sync.write();
        *runtime = RecordCommitmentSyncRuntime::default();
        match (coordinator, follower) {
            (true, false) => {
                runtime.role = "coordinator";
                runtime.state = "producing";
            }
            (false, true) => {
                runtime.role = "follower";
                runtime.state = "starting";
                runtime.enabled = true;
                runtime.certificate_policy_state = "disabled";
            }
            (false, false) => {}
            (true, true) => {
                runtime.state = "configuration_error";
                runtime.last_error_code = Some("role_conflict".to_string());
            }
        }
    }

    /// Configures how long one signed equal-tip follower observation may be
    /// treated as current.
    ///
    /// [FOLLOWER-READINESS-FRESHNESS 2026-07-30 by Codex] Three missed
    /// operator-configured poll windows are tolerated. This process-local
    /// deadline cannot affect chain choice, certificate policy, or storage.
    pub fn configure_record_commitment_sync_readiness_freshness(&self, poll_interval_secs: u64) {
        const MISSED_POLL_WINDOWS_BEFORE_STALE: u64 = 3;

        let mut runtime = self.commitment_sync.write();
        if runtime.enabled && runtime.role == "follower" {
            runtime.follower_readiness_max_age_secs =
                Some(poll_interval_secs.saturating_mul(MISSED_POLL_WINDOWS_BEFORE_STALE));
        }
    }

    /// Configures and verifies `SQLite` commit durability before chain audit.
    ///
    /// WAL + `NORMAL` preserves database consistency but may lose a recently
    /// acknowledged transaction after host power failure. The single-writer
    /// coordinator therefore upgrades the shared connection to `FULL` and
    /// refuses startup unless `SQLite` reports FULL-or-stronger. Followers keep
    /// the existing mode until a receipt-vault write requires FULL durability.
    /// This setting contains no chain or user data.
    ///
    /// # Errors
    ///
    /// Returns an error when the local production fence cannot be acquired,
    /// the durability pragma cannot be applied or read, or a coordinator does
    /// not receive `FULL`-or-stronger durability.
    pub async fn configure_record_commitment_durability(
        &self,
        coordinator: bool,
    ) -> Result<&'static str, String> {
        let coordinator_fence_state =
            self.configure_record_commitment_coordinator_fence(coordinator)?;
        let conn = self.conn.lock().await;
        // [CUSTODY-WITNESS-FULL-DURABILITY 2026-09-24 by Codex] The coordinator
        // and receipt vault use one effective-mode parser and FULL readback.
        let (level, mode) = if coordinator {
            ensure_full_sqlite_durability(&conn)?
        } else {
            read_sqlite_durability(&conn)?
        };
        drop(conn);
        self.commitment_durability
            .store(level as u64, Ordering::Release);
        info!(
            role = if coordinator {
                "coordinator"
            } else {
                "non_coordinator"
            },
            durability_mode = mode,
            coordinator_fence_state,
            "[MEMCHAIN_BLOCK] SQLite commitment durability configured"
        );
        Ok(mode)
    }

    /// Acquires the process-lifetime local coordinator production fence.
    ///
    /// The lock is non-blocking and precedes `SQLite` durability configuration,
    /// chain audit, sidecar verification, listener startup, and mining. A
    /// second process targeting the same on-disk database therefore fails
    /// closed instead of waiting or racing the canonical writer. The kernel
    /// releases the advisory lock when the owning process exits.
    ///
    /// This is a local duplicate-process guard only. It cannot fence another
    /// host with a copied database or identity and is not a distributed lease,
    /// leader election, consensus, quorum, fork choice, or finality.
    pub(super) fn configure_record_commitment_coordinator_fence(
        &self,
        coordinator: bool,
    ) -> Result<&'static str, String> {
        let mut runtime = self.commitment_coordinator_fence.write();
        if !coordinator {
            if runtime.handle.is_none() {
                runtime.state = "not_required";
                runtime.acquired_at = None;
            }
            let state = runtime.state;
            drop(runtime);
            return Ok(state);
        }
        if runtime.handle.is_some() {
            drop(runtime);
            return Ok("held");
        }
        let Some(database_path) = self.database_path.as_deref() else {
            runtime.state = "isolated_in_memory";
            runtime.acquired_at = None;
            let state = runtime.state;
            drop(runtime);
            return Ok(state);
        };

        match acquire_commitment_coordinator_fence(database_path) {
            Ok(handle) => {
                let acquired_at = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                runtime.handle = Some(handle);
                runtime.state = "held";
                runtime.acquired_at = Some(acquired_at);
                let state = runtime.state;
                drop(runtime);
                info!(
                    state,
                    scope = "same_host_same_database_only",
                    "[MEMCHAIN_BLOCK] Coordinator production fence acquired"
                );
                Ok(state)
            }
            Err(error) => {
                runtime.handle = None;
                runtime.state = error.state();
                runtime.acquired_at = None;
                runtime.acquisition_failures_total =
                    runtime.acquisition_failures_total.saturating_add(1);
                drop(runtime);
                Err(error.message().to_string())
            }
        }
    }

    /// Returns the privacy-safe chain-integrity baseline for this process.
    ///
    /// A persisted flag is intentionally not used: every process lifetime must
    /// re-establish the baseline from the complete `SQLite` chain before it may
    /// report `verified`.
    pub fn record_commitment_chain_integrity_status(&self) -> RecordCommitmentChainIntegrityStatus {
        const POLICY: &str = "full snapshot-consistent startup audit plus transactionally verified appends; coordinator requires an exclusive local production fence, SQLite FULL-or-stronger commit durability, a signed local tip anchor, and any explicitly enabled all-witness coordinator lease; no lock path, process id, instance id, block hashes, proposer identities, commitment ids, owners, payloads, peers, endpoints, routes, or client metadata";
        const COORDINATOR_FENCE_SCOPE: &str = "prevents duplicate coordinator processes on one host from producing against the same database while the OS lock holder remains alive; does not fence copied databases or identities on other hosts and is not a distributed lease, leader election, consensus, quorum, fork choice, or finality";
        const COORDINATOR_LEASE_SCOPE: &str = "when explicitly enabled, requires short-lived grants from every operator-pinned audited follower before local production; prevents concurrent copied coordinator instances while at least one honest available witness retains the conflicting lease, but is not permissionless consensus, Byzantine finality, fork choice, or proof of global uniqueness";
        const ROLLBACK_GUARD_SCOPE: &str = "detects commitment SQLite rollback or replacement while the host-side signed anchor remains; does not detect whole-host or whole-disk snapshot rollback and is not consensus, quorum, or finality";
        let durability_mode = match self.commitment_durability.load(Ordering::Acquire) {
            0 => "off",
            1 => "normal",
            2 => "full",
            3 => "extra",
            _ => "unknown",
        };
        let (coordinator_fence_state, coordinator_fence_acquired_at, coordinator_fence_failures) = {
            let runtime = self.commitment_coordinator_fence.read();
            (
                runtime.state,
                runtime.acquired_at,
                runtime.acquisition_failures_total,
            )
        };
        let (
            coordinator_lease_state,
            coordinator_lease_granted_witnesses,
            coordinator_lease_required_witnesses,
            coordinator_lease_expires_at,
            coordinator_lease_seconds_remaining,
            coordinator_lease_production_permitted,
            coordinator_lease_last_attempted_at,
            coordinator_lease_last_renewed_at,
            coordinator_lease_last_failure_at,
            coordinator_lease_renewal_failures_total,
            coordinator_lease_consecutive_failures,
            coordinator_lease_recoveries_total,
        ) = {
            let runtime = self.commitment_coordinator_lease.read();
            let now = Instant::now();
            let seconds_remaining = runtime.valid_until.map(|deadline| {
                deadline.checked_duration_since(now).map_or(0, |remaining| {
                    remaining
                        .as_secs()
                        .saturating_add(u64::from(remaining.subsec_nanos() > 0))
                })
            });
            let valid = seconds_remaining.is_some_and(|remaining| remaining > 0);
            let state = if runtime.required
                && !valid
                && matches!(runtime.state, "held" | "renewal_degraded")
            {
                "expired"
            } else {
                runtime.state
            };
            (
                state,
                runtime.granted_witnesses,
                runtime.required_witnesses,
                runtime.expires_at,
                seconds_remaining,
                !runtime.required || valid,
                runtime.last_attempted_at,
                runtime.last_renewed_at,
                runtime.last_failure_at,
                runtime.renewal_failures_total,
                runtime.consecutive_failures,
                runtime.recoveries_total,
            )
        };
        let (
            rollback_guard_state,
            rollback_guard_height,
            rollback_guard_last_verified_at,
            rollback_guard_last_persisted_at,
            rollback_guard_write_failures_total,
        ) = {
            let runtime = self.commitment_tip_anchor.read();
            (
                runtime.state,
                runtime.anchored_height,
                runtime.last_verified_at,
                runtime.last_persisted_at,
                runtime.write_failures_total,
            )
        };
        let integrity = *self.commitment_integrity.read();
        match integrity {
            Some(runtime) => RecordCommitmentChainIntegrityStatus {
                contract_version: "record_commitment_integrity.v1",
                state: "verified",
                baseline_verified_at: Some(runtime.baseline_verified_at),
                last_verified_at: Some(runtime.last_verified_at),
                verification_duration_ms: Some(runtime.verification_duration_ms),
                verified_block_count: runtime.verified_block_count,
                verified_commitment_count: runtime.verified_commitment_count,
                verified_tip_height: runtime.verified_tip_height,
                durability_mode,
                coordinator_fence_state,
                coordinator_fence_acquired_at,
                coordinator_fence_acquisition_failures_total: coordinator_fence_failures,
                coordinator_fence_scope: COORDINATOR_FENCE_SCOPE,
                coordinator_lease_state,
                coordinator_lease_granted_witnesses,
                coordinator_lease_required_witnesses,
                coordinator_lease_expires_at,
                coordinator_lease_seconds_remaining,
                coordinator_lease_production_permitted,
                coordinator_lease_last_attempted_at,
                coordinator_lease_last_renewed_at,
                coordinator_lease_last_failure_at,
                coordinator_lease_renewal_failures_total,
                coordinator_lease_consecutive_failures,
                coordinator_lease_recoveries_total,
                coordinator_lease_scope: COORDINATOR_LEASE_SCOPE,
                rollback_guard_state,
                rollback_guard_height,
                rollback_guard_last_verified_at,
                rollback_guard_last_persisted_at,
                rollback_guard_write_failures_total,
                rollback_guard_scope: ROLLBACK_GUARD_SCOPE,
                verification_policy: POLICY,
            },
            None => RecordCommitmentChainIntegrityStatus {
                contract_version: "record_commitment_integrity.v1",
                state: "not_verified",
                baseline_verified_at: None,
                last_verified_at: None,
                verification_duration_ms: None,
                verified_block_count: 0,
                verified_commitment_count: 0,
                verified_tip_height: 0,
                durability_mode,
                coordinator_fence_state,
                coordinator_fence_acquired_at,
                coordinator_fence_acquisition_failures_total: coordinator_fence_failures,
                coordinator_fence_scope: COORDINATOR_FENCE_SCOPE,
                coordinator_lease_state,
                coordinator_lease_granted_witnesses,
                coordinator_lease_required_witnesses,
                coordinator_lease_expires_at,
                coordinator_lease_seconds_remaining,
                coordinator_lease_production_permitted,
                coordinator_lease_last_attempted_at,
                coordinator_lease_last_renewed_at,
                coordinator_lease_last_failure_at,
                coordinator_lease_renewal_failures_total,
                coordinator_lease_consecutive_failures,
                coordinator_lease_recoveries_total,
                coordinator_lease_scope: COORDINATOR_LEASE_SCOPE,
                rollback_guard_state,
                rollback_guard_height,
                rollback_guard_last_verified_at,
                rollback_guard_last_persisted_at,
                rollback_guard_write_failures_total,
                rollback_guard_scope: ROLLBACK_GUARD_SCOPE,
                verification_policy: POLICY,
            },
        }
    }

    /// Returns aggregate chain health without exposing record commitments,
    /// proposer identities, or peer metadata.
    pub async fn record_commitment_chain_status(&self) -> RecordCommitmentChainStatus {
        let conn = self.conn.lock().await;
        let (block_count, commitment_count): (u64, u64) = conn
            .query_row(
                "SELECT COUNT(*),COALESCE(SUM(record_count),0)
                 FROM record_commitment_blocks",
                [],
                |row| Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64)),
            )
            .unwrap_or((0, 0));
        let tip: Option<(u64, Vec<u8>)> = conn
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, i64>(0)? as u64, row.get(1)?)),
            )
            .optional()
            .unwrap_or(None);
        let (tip_height, tip_hash) = match tip {
            Some((height, hash)) if hash.len() == 32 => (height, Some(hex::encode(hash))),
            _ => (0, None),
        };
        let integrity = self.record_commitment_chain_integrity_status();
        let checkpoint = self.record_commitment_checkpoint_status();
        drop(conn);
        RecordCommitmentChainStatus {
            contract_version: "record_commitment_chain.v1",
            chain_id: hex::encode(AERONYX_MEMCHAIN_MAINNET_CHAIN_ID),
            block_count,
            commitment_count,
            tip_height,
            tip_hash,
            payload_policy: "opaque_record_commitments_only_no_memory_payload_or_owner_metadata",
            integrity,
            checkpoint,
        }
    }
}
