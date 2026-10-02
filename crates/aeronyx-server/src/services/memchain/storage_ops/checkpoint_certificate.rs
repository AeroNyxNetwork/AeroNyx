// [ARCH-SPLIT 2026-10-02]
// Checkpoint certificate bundle, policy, and certificate-anchor persistence.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn checkpoint_certificate_anchor_signing_bytes(
    chain_id: &[u8; 32],
    state: CheckpointCertificateAnchorState,
    signer: &[u8; 32],
    updated_at: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(CHECKPOINT_CERTIFICATE_ANCHOR_DOMAIN.len() + 160);
    bytes.extend_from_slice(CHECKPOINT_CERTIFICATE_ANCHOR_DOMAIN);
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&state.certificate_height.to_le_bytes());
    bytes.extend_from_slice(&state.checkpoint_hash);
    bytes.extend_from_slice(&state.certificate_digest);
    bytes.extend_from_slice(&state.required_signers.to_le_bytes());
    bytes.extend_from_slice(&state.signer_count.to_le_bytes());
    bytes.extend_from_slice(signer);
    bytes.extend_from_slice(&updated_at.to_le_bytes());
    bytes
}

pub(super) fn decode_checkpoint_certificate_anchor_hex<const N: usize>(
    value: &str,
    label: &str,
) -> Result<[u8; N], String> {
    let decoded = hex::decode(value)
        .map_err(|_| format!("checkpoint certificate anchor {label} is not hexadecimal"))?;
    decoded.try_into().map_err(|decoded: Vec<u8>| {
        format!(
            "checkpoint certificate anchor {label} has invalid length {}",
            decoded.len()
        )
    })
}

pub(super) fn checkpoint_certificate_anchor_path(
    tip_anchor_path: &Path,
) -> Result<PathBuf, String> {
    let mut file_name = tip_anchor_path
        .file_name()
        .ok_or_else(|| "commitment tip anchor path has no file name".to_string())?
        .to_os_string();
    file_name.push(".checkpoint-certificate-v1.json");
    Ok(tip_anchor_path.with_file_name(file_name))
}

pub(super) async fn read_checkpoint_certificate_anchor(
    path: &Path,
) -> Result<Option<RecordCheckpointCertificateAnchorV1>, String> {
    let Some(bytes) = read_signed_local_anchor_bytes(
        path,
        "checkpoint certificate anchor",
        MAX_CHECKPOINT_CERTIFICATE_ANCHOR_BYTES,
    )
    .await?
    else {
        return Ok(None);
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| format!("decode checkpoint certificate anchor: {error}"))
}

pub(super) fn write_checkpoint_certificate_anchor_atomic(
    path: &Path,
    bytes: &[u8],
) -> Result<(), String> {
    write_signed_local_anchor_atomic(path, bytes, "checkpoint certificate anchor")
}

pub(super) async fn persist_checkpoint_certificate_anchor(
    path: PathBuf,
    state: CheckpointCertificateAnchorState,
    identity: &IdentityKeyPair,
) -> Result<u64, String> {
    let updated_at = unix_now_secs();
    let anchor = RecordCheckpointCertificateAnchorV1::new_signed(state, identity, updated_at);
    let bytes = serde_json::to_vec(&anchor)
        .map_err(|error| format!("encode checkpoint certificate anchor: {error}"))?;
    run_blocking_local_anchor_write("checkpoint certificate anchor", move || {
        write_checkpoint_certificate_anchor_atomic(&path, &bytes)
    })
    .await?;
    Ok(updated_at)
}

pub(super) fn read_latest_checkpoint_certificate_anchor_state(
    connection: &rusqlite::Connection,
) -> Result<Option<CheckpointCertificateAnchorState>, String> {
    let row: Option<LatestCheckpointCertificateAnchorRow> = connection
        .query_row(
            "SELECT checkpoint_height,chain_id,checkpoint_hash,certificate_digest,
                    required_signers,signer_count
             FROM record_checkpoint_certificates
             ORDER BY checkpoint_height DESC LIMIT 1",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            },
        )
        .optional()
        .map_err(|error| format!("read latest checkpoint certificate anchor state: {error}"))?;
    let Some((height, chain_id, checkpoint_hash, certificate_digest, required, signers)) = row
    else {
        return Ok(None);
    };
    let certificate_height = u64::try_from(height)
        .map_err(|_| "latest checkpoint certificate height is invalid".to_string())?;
    let chain_id: [u8; 32] = chain_id
        .as_slice()
        .try_into()
        .map_err(|_| "latest checkpoint certificate chain id has invalid length".to_string())?;
    let checkpoint_hash: [u8; 32] = checkpoint_hash
        .as_slice()
        .try_into()
        .map_err(|_| "latest checkpoint certificate hash has invalid length".to_string())?;
    let certificate_digest: [u8; 32] = certificate_digest
        .as_slice()
        .try_into()
        .map_err(|_| "latest checkpoint certificate digest has invalid length".to_string())?;
    let required_signers = u64::try_from(required)
        .map_err(|_| "latest checkpoint certificate threshold is invalid".to_string())?;
    let signer_count = u64::try_from(signers)
        .map_err(|_| "latest checkpoint certificate signer count is invalid".to_string())?;
    if certificate_height == 0
        || chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || !(2..=MAX_CHECKPOINT_CERTIFICATE_SIGNERS as u64).contains(&required_signers)
        || signer_count < required_signers
        || signer_count > MAX_CHECKPOINT_CERTIFICATE_SIGNERS as u64
    {
        return Err("latest checkpoint certificate metadata is invalid".to_string());
    }
    Ok(Some(CheckpointCertificateAnchorState {
        certificate_height,
        checkpoint_hash,
        certificate_digest,
        required_signers,
        signer_count,
    }))
}

impl MemoryStorage {
    /// Returns the latest certificate only when it matches the caller's exact
    /// audited tip. The complete evidence vault is re-audited in the same
    /// SQLite snapshot before any witness frame leaves storage.
    pub(crate) async fn record_commitment_checkpoint_certificate_bundle(
        &self,
        expected_height: u64,
        expected_hash: &[u8; 32],
    ) -> Result<Option<RecordCommitmentCheckpointCertificateBundle>, String> {
        let expected_height_i64 = i64::try_from(expected_height)
            .map_err(|_| "checkpoint certificate height exceeds SQLite range".to_string())?;
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin checkpoint certificate export snapshot: {error}"))?;
        let report = audit_checkpoint_evidence_snapshot(&transaction)?;
        let latest: Option<(i64, Vec<u8>, Vec<u8>, i64, i64)> = transaction
            .query_row(
                "SELECT checkpoint_height,checkpoint_hash,certificate_digest,
                        required_signers,signer_count
                 FROM record_checkpoint_certificates
                 ORDER BY checkpoint_height DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| format!("read latest checkpoint certificate: {error}"))?;

        let bundle = if let Some((
            checkpoint_height_i64,
            checkpoint_hash,
            certificate_digest,
            required_signers_i64,
            signer_count_i64,
        )) = latest
        {
            let checkpoint_hash: [u8; 32] = checkpoint_hash
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate hash has invalid length".to_string())?;
            let certificate_digest: [u8; 32] = certificate_digest
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate digest has invalid length".to_string())?;
            let required_signers = usize::try_from(required_signers_i64)
                .map_err(|_| "checkpoint certificate threshold is invalid".to_string())?;
            let signer_count = usize::try_from(signer_count_i64)
                .map_err(|_| "checkpoint certificate signer count is invalid".to_string())?;
            if checkpoint_height_i64 != expected_height_i64 || checkpoint_hash != *expected_hash {
                None
            } else {
                let mut member_frames = Vec::with_capacity(signer_count);
                {
                    let mut statement = transaction
                        .prepare(
                            "SELECT evidence.signed_response
                             FROM record_checkpoint_certificate_members AS member
                             JOIN record_checkpoint_evidence AS evidence
                               ON evidence.evidence_digest=member.evidence_digest
                             WHERE member.checkpoint_height=?1
                             ORDER BY member.responder ASC",
                        )
                        .map_err(|error| {
                            format!("prepare checkpoint certificate export members: {error}")
                        })?;
                    let mut rows =
                        statement
                            .query(params![expected_height_i64])
                            .map_err(|error| {
                                format!("query checkpoint certificate export members: {error}")
                            })?;
                    while let Some(row) = rows.next().map_err(|error| {
                        format!("read checkpoint certificate export member: {error}")
                    })? {
                        member_frames.push(row.get::<_, Vec<u8>>(0).map_err(|error| {
                            format!("read checkpoint certificate export frame: {error}")
                        })?);
                    }
                }
                if member_frames.len() != signer_count {
                    return Err("checkpoint certificate export signer count mismatch".to_string());
                }
                Some(RecordCommitmentCheckpointCertificateBundle {
                    checkpoint_height: expected_height,
                    checkpoint_hash,
                    certificate_digest,
                    required_signers,
                    member_frames,
                })
            }
        } else {
            None
        };
        transaction
            .commit()
            .map_err(|error| format!("finish checkpoint certificate export snapshot: {error}"))?;
        self.apply_record_commitment_checkpoint_audit(&report);
        Ok(bundle)
    }

    /// Checks whether the exact current-tip certificate satisfies local policy.
    ///
    /// [FOLLOWER-CERTIFICATE-SYNC 2026-07-29 by Codex] Height alone is not
    /// enough: an operator may rotate witness pins while retaining the same
    /// audited chain tip. This path re-audits the complete bounded evidence
    /// vault, then requires every immutable member to remain in the current
    /// allowlist and the stored threshold to meet the current local minimum.
    pub(crate) async fn record_commitment_checkpoint_certificate_satisfies_policy(
        &self,
        expected_height: u64,
        allowed_witnesses: &[[u8; 32]],
        minimum_required_signers: usize,
    ) -> Result<bool, String> {
        if !(2..=MAX_CHECKPOINT_CERTIFICATE_SIGNERS).contains(&minimum_required_signers)
            || allowed_witnesses.len() < minimum_required_signers
            || expected_height == 0
        {
            return Ok(false);
        }
        let (checkpoint_height, checkpoint_hash, tip_height, tip_hash) = self
            .record_commitment_chain_checkpoint(expected_height)
            .await?;
        if checkpoint_height != expected_height
            || tip_height != expected_height
            || checkpoint_hash != tip_hash
        {
            return Ok(false);
        }
        let Some(bundle) = self
            .record_commitment_checkpoint_certificate_bundle(expected_height, &tip_hash)
            .await?
        else {
            return Ok(false);
        };
        if bundle.required_signers < minimum_required_signers
            || bundle.member_frames.len() < minimum_required_signers
            || bundle.member_frames.len() > MAX_CHECKPOINT_CERTIFICATE_SIGNERS
        {
            return Ok(false);
        }

        let mut responders = Vec::with_capacity(bundle.member_frames.len());
        for frame in &bundle.member_frames {
            let responder = decode_checkpoint_evidence_claims(frame)?.responder;
            if !allowed_witnesses.contains(&responder) || responders.contains(&responder) {
                return Ok(false);
            }
            responders.push(responder);
        }
        Ok(responders.len() == bundle.member_frames.len())
    }

    /// Freezes one immutable certificate from a single pinned-witness round.
    ///
    /// Only exact evidence digests from the current round are considered. Every
    /// member must be a distinct explicitly allowed witness whose signed frame
    /// confirms the current local tip (`converged` or `remote_ahead`). A
    /// behind, divergent, permissionless, stale, or duplicate witness cannot
    /// contribute. This proves a threshold of observations, not global finality.
    pub(crate) async fn persist_record_commitment_checkpoint_certificate(
        &self,
        certified_at: u64,
        required_signers: usize,
        allowed_witnesses: &[[u8; 32]],
        evidence_digests: &[[u8; 32]],
    ) -> Result<bool, String> {
        if !(2..=MAX_CHECKPOINT_CERTIFICATE_SIGNERS).contains(&required_signers) {
            return Err("checkpoint certificate threshold is outside supported bounds".to_string());
        }
        if allowed_witnesses.len() < required_signers
            || evidence_digests.len() < required_signers
            || evidence_digests.len() > MAX_CHECKPOINT_CERTIFICATE_SIGNERS
        {
            return Ok(false);
        }
        let certified_at = i64::try_from(certified_at)
            .map_err(|_| "checkpoint certificate time exceeds SQLite range".to_string())?;

        let anchor_enabled = self
            .commitment_checkpoint_certificate_anchor
            .read()
            .config
            .is_some();
        let _anchor_write_guard = if anchor_enabled {
            Some(
                self.commitment_checkpoint_certificate_anchor_write
                    .lock()
                    .await,
            )
        } else {
            None
        };
        if anchor_enabled
            && !matches!(
                self.commitment_checkpoint_certificate_anchor.read().state,
                "initialized" | "verified" | "repaired"
            )
        {
            return Err(
                "checkpoint certificate anchor is not ready; restart and complete startup audit"
                    .to_string(),
            );
        }

        let (report, anchor_state) = {
            let mut conn = self.conn.lock().await;
            let transaction = conn
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .map_err(|error| format!("begin checkpoint certificate transaction: {error}"))?;
            let (tip_height_i64, tip_hash): (i64, Vec<u8>) = transaction
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| format!("read checkpoint certificate local tip: {error}"))?
            .ok_or_else(|| "checkpoint certificate local tip is unavailable".to_string())?;
            let tip_height = u64::try_from(tip_height_i64)
                .map_err(|_| "checkpoint certificate local tip is invalid".to_string())?;
            let tip_hash: [u8; 32] = tip_hash
                .as_slice()
                .try_into()
                .map_err(|_| "checkpoint certificate local hash has invalid length".to_string())?;
            if tip_height == 0 {
                return Ok(false);
            }

            let mut members = Vec::with_capacity(evidence_digests.len());
            for evidence_digest in evidence_digests {
                let (relation, local_tip_height_i64, frame): (String, i64, Vec<u8>) = transaction
                    .query_row(
                        "SELECT relation,local_tip_height,signed_response
                     FROM record_checkpoint_evidence WHERE evidence_digest=?1",
                        params![evidence_digest.as_slice()],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .map_err(|_| "checkpoint certificate evidence is unavailable".to_string())?;
                let local_tip_height = u64::try_from(local_tip_height_i64)
                    .map_err(|_| "checkpoint certificate evidence height is invalid".to_string())?;
                let claim = decode_checkpoint_evidence_claims(&frame)?;
                if !matches!(relation.as_str(), "converged" | "remote_ahead")
                    || local_tip_height != tip_height
                    || claim.checkpoint_height != tip_height
                    || claim.checkpoint_hash != tip_hash
                    || !allowed_witnesses.contains(&claim.responder)
                    || members
                        .iter()
                        .any(|(responder, _)| *responder == claim.responder)
                {
                    continue;
                }
                members.push((claim.responder, *evidence_digest));
            }
            members.sort_unstable_by_key(|member| member.0);
            if members.len() < required_signers {
                return Ok(false);
            }
            let certificate_digest = record_checkpoint_certificate_digest_v1(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                tip_height,
                &tip_hash,
                required_signers,
                &members,
            );

            let existing: Option<(Vec<u8>, i64, i64, Vec<u8>)> = transaction
                .query_row(
                    "SELECT checkpoint_hash,required_signers,signer_count,certificate_digest
                 FROM record_checkpoint_certificates WHERE checkpoint_height=?1",
                    params![tip_height_i64],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()
                .map_err(|error| format!("read existing checkpoint certificate: {error}"))?;
            if let Some((stored_hash, stored_required, stored_count, stored_digest)) = existing {
                let stored_required = usize::try_from(stored_required).map_err(|_| {
                    "existing checkpoint certificate threshold is invalid".to_string()
                })?;
                let stored_count = usize::try_from(stored_count).map_err(|_| {
                    "existing checkpoint certificate signer count is invalid".to_string()
                })?;
                if stored_hash.as_slice() != tip_hash
                    || !(2..=MAX_CHECKPOINT_CERTIFICATE_SIGNERS).contains(&stored_required)
                    || stored_count < stored_required
                    || stored_count > MAX_CHECKPOINT_CERTIFICATE_SIGNERS
                    || stored_digest.len() != 32
                {
                    return Err(
                        "existing checkpoint certificate conflicts with local tip".to_string()
                    );
                }
                let report = audit_checkpoint_evidence_snapshot(&transaction)?;
                let mut current_policy_members = 0usize;
                {
                    let mut statement = transaction
                        .prepare(
                            "SELECT responder FROM record_checkpoint_certificate_members
                         WHERE checkpoint_height=?1 ORDER BY responder ASC",
                        )
                        .map_err(|error| {
                            format!("prepare existing checkpoint certificate members: {error}")
                        })?;
                    let mut rows = statement.query(params![tip_height_i64]).map_err(|error| {
                        format!("query existing checkpoint certificate members: {error}")
                    })?;
                    while let Some(row) = rows.next().map_err(|error| {
                        format!("read existing checkpoint certificate member: {error}")
                    })? {
                        let responder: Vec<u8> = row.get(0).map_err(|error| {
                            format!("read existing checkpoint certificate responder: {error}")
                        })?;
                        let responder: [u8; 32] =
                            responder.as_slice().try_into().map_err(|_| {
                                "existing checkpoint certificate responder has invalid length"
                                    .to_string()
                            })?;
                        if allowed_witnesses.contains(&responder) {
                            current_policy_members = current_policy_members.saturating_add(1);
                        }
                    }
                }
                let satisfies_current_policy = stored_required >= required_signers
                    && stored_count >= required_signers
                    && current_policy_members == stored_count;
                transaction
                    .commit()
                    .map_err(|error| format!("finish existing checkpoint certificate: {error}"))?;
                self.apply_record_commitment_checkpoint_audit(&report);
                return Ok(satisfies_current_policy);
            }

            let certificate_count = transaction
                .query_row(
                    "SELECT COUNT(*) FROM record_checkpoint_certificates",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(|error| format!("count checkpoint certificates: {error}"))?;
            let excess = certificate_count
                .saturating_add(1)
                .saturating_sub(CHECKPOINT_CERTIFICATE_CAPACITY as i64);
            if excess > 0 {
                transaction
                    .execute(
                        "DELETE FROM record_checkpoint_certificates WHERE checkpoint_height IN (
                        SELECT checkpoint_height FROM record_checkpoint_certificates
                        ORDER BY checkpoint_height ASC LIMIT ?1
                    )",
                        params![excess],
                    )
                    .map_err(|error| format!("prune checkpoint certificates: {error}"))?;
            }
            transaction
                .execute(
                    "INSERT INTO record_checkpoint_certificates
                 (checkpoint_height,chain_id,checkpoint_hash,certificate_digest,
                  required_signers,signer_count,certified_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        tip_height_i64,
                        AERONYX_MEMCHAIN_MAINNET_CHAIN_ID.as_slice(),
                        tip_hash.as_slice(),
                        certificate_digest.as_slice(),
                        i64::try_from(required_signers)
                            .map_err(|_| "checkpoint certificate threshold exceeds SQLite range")?,
                        i64::try_from(members.len()).map_err(|_| {
                            "checkpoint certificate signer count exceeds SQLite range"
                        })?,
                        certified_at,
                    ],
                )
                .map_err(|error| format!("insert checkpoint certificate: {error}"))?;
            for (responder, evidence_digest) in &members {
                transaction
                    .execute(
                        "INSERT INTO record_checkpoint_certificate_members
                     (checkpoint_height,responder,evidence_digest) VALUES (?1,?2,?3)",
                        params![
                            tip_height_i64,
                            responder.as_slice(),
                            evidence_digest.as_slice(),
                        ],
                    )
                    .map_err(|error| format!("insert checkpoint certificate member: {error}"))?;
            }
            let report = audit_checkpoint_evidence_snapshot(&transaction)?;
            let anchor_state = CheckpointCertificateAnchorState {
                certificate_height: tip_height,
                checkpoint_hash: tip_hash,
                certificate_digest,
                required_signers: required_signers as u64,
                signer_count: members.len() as u64,
            };
            transaction
                .commit()
                .map_err(|error| format!("commit checkpoint certificate: {error}"))?;
            (report, anchor_state)
        };
        self.apply_record_commitment_checkpoint_audit(&report);
        self.persist_checkpoint_certificate_anchor_after_commit(anchor_state)
            .await?;
        Ok(true)
    }

    /// Configures identity-blind follower certificate-policy readiness.
    ///
    /// [FOLLOWER-CERTIFICATE-READINESS 2026-07-29 by Codex] Only aggregate
    /// count and threshold enter runtime status. The caller has already
    /// validated every identity and must never pass pins into this surface.
    pub fn configure_record_commitment_certificate_policy(
        &self,
        witnesses_configured: usize,
        minimum_signers: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            runtime.certificate_policy_state = "not_applicable";
            runtime.certificate_policy_ready = false;
            runtime.certificate_policy_last_evaluated_at = None;
            runtime.certificate_policy_evaluated_tip_height = None;
            runtime.certificate_witnesses_configured = 0;
            runtime.certificate_minimum_signers = 0;
            return;
        }

        runtime.certificate_witnesses_configured = witnesses_configured;
        runtime.certificate_minimum_signers = minimum_signers;
        runtime.certificate_policy_ready = false;
        runtime.certificate_policy_last_evaluated_at = None;
        runtime.certificate_policy_evaluated_tip_height = None;
        runtime.certificate_policy_state = if minimum_signers < 2 {
            "disabled"
        } else if witnesses_configured < minimum_signers
            || minimum_signers > MAX_CHECKPOINT_CERTIFICATE_SIGNERS
        {
            "configuration_error"
        } else {
            "waiting_for_convergence"
        };
    }

    /// Verifies or initializes the signed certificate-vault high-water mark.
    ///
    /// This must run after the complete commitment-chain and checkpoint-vault
    /// audits. The method re-audits the bounded vault in one `SQLite` snapshot,
    /// then compares its latest immutable certificate with a separately signed
    /// sidecar. A lower `SQLite` height or a same-height metadata mismatch fails
    /// startup closed. A higher fully audited certificate repairs the sidecar,
    /// covering the expected crash window after a DB commit.
    ///
    /// Scope: this detects certificate-vault rollback while the host-side
    /// sidecar remains. A whole-host snapshot can roll back both artifacts and
    /// still requires external pinned-witness evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when the prerequisite audit is absent, the signed
    /// sidecar is malformed or conflicts with the audited vault, rollback is
    /// detected, or durable sidecar replacement fails.
    pub async fn configure_record_commitment_checkpoint_certificate_anchor(
        &self,
        commitment_tip_anchor_path: impl AsRef<Path>,
        identity: &IdentityKeyPair,
    ) -> Result<&'static str, String> {
        if self.commitment_checkpoint.read().evidence_state != "verified" {
            return Err(
                "checkpoint certificate anchor requires a successful evidence-vault audit"
                    .to_string(),
            );
        }
        let path = checkpoint_certificate_anchor_path(commitment_tip_anchor_path.as_ref())?;
        let _write_guard = self
            .commitment_checkpoint_certificate_anchor_write
            .lock()
            .await;
        {
            let mut runtime = self.commitment_checkpoint_certificate_anchor.write();
            *runtime = RecordCommitmentCheckpointCertificateAnchorRuntime::default();
            runtime.config = Some(RecordCommitmentCheckpointCertificateAnchorConfig {
                path: path.clone(),
                identity: identity.clone(),
            });
            runtime.state = "checking";
        }

        let (report, current_state) = {
            let mut conn = self.conn.lock().await;
            let transaction = conn
                .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
                .map_err(|error| {
                    self.fail_record_commitment_checkpoint_certificate_anchor("invalid", 0, false);
                    format!("begin checkpoint certificate anchor audit: {error}")
                })?;
            let report = audit_checkpoint_evidence_snapshot(&transaction).map_err(|error| {
                self.fail_record_commitment_checkpoint_certificate_anchor("invalid", 0, false);
                error
            })?;
            let current_state = read_latest_checkpoint_certificate_anchor_state(&transaction)
                .map_err(|error| {
                    self.fail_record_commitment_checkpoint_certificate_anchor("invalid", 0, false);
                    error
                })?
                .unwrap_or(CheckpointCertificateAnchorState::EMPTY);
            transaction.commit().map_err(|error| {
                self.fail_record_commitment_checkpoint_certificate_anchor("invalid", 0, false);
                format!("finish checkpoint certificate anchor audit: {error}")
            })?;
            (report, current_state)
        };
        self.apply_record_commitment_checkpoint_audit(&report);

        let stored = match read_checkpoint_certificate_anchor(&path).await {
            Ok(stored) => stored,
            Err(error) => {
                self.fail_record_commitment_checkpoint_certificate_anchor("invalid", 0, false);
                return Err(error);
            }
        };
        let now = unix_now_secs();
        match stored {
            None => {
                let persisted_at =
                    persist_checkpoint_certificate_anchor(path, current_state, identity)
                        .await
                        .map_err(|error| {
                            self.fail_record_commitment_checkpoint_certificate_anchor(
                                "write_failed",
                                current_state.certificate_height,
                                true,
                            );
                            error
                        })?;
                let mut runtime = self.commitment_checkpoint_certificate_anchor.write();
                runtime.state = "initialized";
                runtime.anchored_height = current_state.certificate_height;
                runtime.last_verified_at = Some(now);
                runtime.last_persisted_at = Some(persisted_at);
                info!(
                    certificate_height = current_state.certificate_height,
                    "[MEMCHAIN_BLOCK] Signed checkpoint certificate anchor initialized"
                );
                Ok("initialized")
            }
            Some(anchor) => {
                let verified = anchor
                    .verify(&identity.public_key_bytes())
                    .map_err(|error| {
                        self.fail_record_commitment_checkpoint_certificate_anchor(
                            "invalid", 0, false,
                        );
                        error
                    })?;
                if verified.state.certificate_height > current_state.certificate_height {
                    self.fail_record_commitment_checkpoint_certificate_anchor(
                        "rollback_detected",
                        verified.state.certificate_height,
                        false,
                    );
                    return Err(format!(
                        "checkpoint certificate SQLite height {} is behind signed local anchor height {}",
                        current_state.certificate_height, verified.state.certificate_height
                    ));
                }
                if verified.state.certificate_height == current_state.certificate_height {
                    if verified.state != current_state {
                        self.fail_record_commitment_checkpoint_certificate_anchor(
                            "rollback_detected",
                            verified.state.certificate_height,
                            false,
                        );
                        return Err(format!(
                            "signed checkpoint certificate anchor conflicts at height {}",
                            verified.state.certificate_height
                        ));
                    }
                    let mut runtime = self.commitment_checkpoint_certificate_anchor.write();
                    runtime.state = "verified";
                    runtime.anchored_height = current_state.certificate_height;
                    runtime.last_verified_at = Some(now);
                    runtime.last_persisted_at = Some(verified.updated_at);
                    info!(
                        certificate_height = current_state.certificate_height,
                        "[MEMCHAIN_BLOCK] Signed checkpoint certificate anchor verified"
                    );
                    return Ok("verified");
                }

                let persisted_at =
                    persist_checkpoint_certificate_anchor(path, current_state, identity)
                        .await
                        .map_err(|error| {
                            self.fail_record_commitment_checkpoint_certificate_anchor(
                                "write_failed",
                                verified.state.certificate_height,
                                true,
                            );
                            error
                        })?;
                let mut runtime = self.commitment_checkpoint_certificate_anchor.write();
                runtime.state = "repaired";
                runtime.anchored_height = current_state.certificate_height;
                runtime.last_verified_at = Some(now);
                runtime.last_persisted_at = Some(persisted_at);
                info!(
                    previous_height = verified.state.certificate_height,
                    certificate_height = current_state.certificate_height,
                    "[MEMCHAIN_BLOCK] Signed checkpoint certificate anchor advanced after audited DB-ahead recovery"
                );
                Ok("repaired")
            }
        }
    }

    pub(super) fn fail_record_commitment_checkpoint_certificate_anchor(
        &self,
        state: &'static str,
        anchored_height: u64,
        write_failure: bool,
    ) {
        let mut runtime = self.commitment_checkpoint_certificate_anchor.write();
        runtime.state = state;
        runtime.anchored_height = anchored_height;
        runtime.last_verified_at = None;
        if write_failure {
            runtime.write_failures_total = runtime.write_failures_total.saturating_add(1);
        }
        drop(runtime);
        *self.commitment_integrity.write() = None;
    }

    /// Advances the sidecar after a certificate DB transaction has committed.
    /// The caller must hold `commitment_checkpoint_certificate_anchor_write`
    /// from before opening that transaction until this method returns.
    pub(super) async fn persist_checkpoint_certificate_anchor_after_commit(
        &self,
        state: CheckpointCertificateAnchorState,
    ) -> Result<(), String> {
        let anchor_config = self
            .commitment_checkpoint_certificate_anchor
            .read()
            .config
            .as_ref()
            .map(|config| (config.path.clone(), config.identity.clone()));
        let Some((path, identity)) = anchor_config else {
            return Ok(());
        };
        let anchored_height = self
            .commitment_checkpoint_certificate_anchor
            .read()
            .anchored_height;
        if state.certificate_height < anchored_height {
            self.fail_record_commitment_checkpoint_certificate_anchor(
                "rollback_detected",
                anchored_height,
                false,
            );
            return Err(
                "checkpoint certificate committed below the signed local high-water mark"
                    .to_string(),
            );
        }
        let persisted_at = persist_checkpoint_certificate_anchor(path, state, &identity)
            .await
            .map_err(|error| {
                self.fail_record_commitment_checkpoint_certificate_anchor(
                    "write_failed",
                    anchored_height,
                    true,
                );
                format!(
                    "checkpoint certificate was committed but signed anchor persistence failed; restart and re-audit: {error}"
                )
            })?;
        let mut runtime = self.commitment_checkpoint_certificate_anchor.write();
        runtime.state = "verified";
        runtime.anchored_height = state.certificate_height;
        runtime.last_verified_at = Some(persisted_at);
        runtime.last_persisted_at = Some(persisted_at);
        Ok(())
    }
}
