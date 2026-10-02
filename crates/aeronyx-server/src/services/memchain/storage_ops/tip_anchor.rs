// [ARCH-SPLIT 2026-10-02]
// Signed tip-anchor read, atomic write, and MemoryStorage configuration.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(in crate::services::memchain) fn read_record_commitment_tip_transaction(
    transaction: &rusqlite::Transaction<'_>,
    context: &str,
) -> Result<(u64, [u8; 32]), String> {
    let tip: Option<(i64, Vec<u8>)> = transaction
        .query_row(
            "SELECT height,block_hash FROM record_commitment_blocks
             ORDER BY height DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(|error| format!("read {context} tip: {error}"))?;
    match tip {
        Some((height, hash)) => {
            let height =
                u64::try_from(height).map_err(|_| format!("{context} tip height is invalid"))?;
            let hash: [u8; 32] = hash
                .try_into()
                .map_err(|hash: Vec<u8>| format!("{context} tip hash length {}", hash.len()))?;
            Ok((height, hash))
        }
        None => Ok((0, GENESIS_PREV_HASH)),
    }
}

pub(super) fn record_commitment_tip_anchor_signing_bytes(
    chain_id: &[u8; 32],
    tip_height: u64,
    tip_hash: &[u8; 32],
    signer: &[u8; 32],
    updated_at: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(COMMITMENT_TIP_ANCHOR_DOMAIN.len() + 112);
    bytes.extend_from_slice(COMMITMENT_TIP_ANCHOR_DOMAIN);
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&tip_height.to_le_bytes());
    bytes.extend_from_slice(tip_hash);
    bytes.extend_from_slice(signer);
    bytes.extend_from_slice(&updated_at.to_le_bytes());
    bytes
}

pub(super) fn decode_fixed_hex<const N: usize>(
    value: &str,
    label: &str,
) -> Result<[u8; N], String> {
    let decoded = hex::decode(value)
        .map_err(|_| format!("commitment tip anchor {label} is not hexadecimal"))?;
    decoded.try_into().map_err(|decoded: Vec<u8>| {
        format!(
            "commitment tip anchor {label} has invalid length {}",
            decoded.len()
        )
    })
}

pub(in crate::services::memchain) fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(super) async fn read_record_commitment_tip_anchor(
    path: &Path,
) -> Result<Option<RecordCommitmentTipAnchorV1>, String> {
    let Some(bytes) = read_signed_local_anchor_bytes(
        path,
        "commitment tip anchor",
        MAX_COMMITMENT_TIP_ANCHOR_BYTES,
    )
    .await?
    else {
        return Ok(None);
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| format!("decode commitment tip anchor: {error}"))
}

pub(super) async fn read_signed_local_anchor_bytes(
    path: &Path,
    label: &'static str,
    max_bytes: u64,
) -> Result<Option<Vec<u8>>, String> {
    let metadata = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("inspect {label}: {error}")),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular file"));
    }
    if metadata.len() > max_bytes {
        return Err(format!("{label} exceeds the defensive size bound"));
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| format!("read {label}: {error}"))?;
    if bytes.len() as u64 > max_bytes {
        return Err(format!("{label} exceeds the defensive size bound"));
    }
    Ok(Some(bytes))
}

pub(super) fn write_record_commitment_tip_anchor_atomic(
    path: &Path,
    bytes: &[u8],
) -> Result<(), String> {
    write_signed_local_anchor_atomic(path, bytes, "commitment tip anchor")
}

pub(super) fn write_signed_local_anchor_atomic(
    path: &Path,
    bytes: &[u8],
    label: &'static str,
) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create {label} directory: {error}"))?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(format!("{label} target must be a regular file"));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect {label} target: {error}")),
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{label} path has no valid file name"))?;
    let nonce = SIGNED_LOCAL_ANCHOR_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let time_nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = parent.join(format!(
        ".{file_name}.{}.{}.{}.tmp",
        std::process::id(),
        nonce,
        time_nonce
    ));

    let result = (|| -> Result<(), String> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temp_path)
            .map_err(|error| format!("create {label} temp file: {error}"))?;
        file.write_all(bytes)
            .map_err(|error| format!("write {label}: {error}"))?;
        file.flush()
            .map_err(|error| format!("flush {label}: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync {label}: {error}"))?;
        drop(file);
        std::fs::rename(&temp_path, path).map_err(|error| format!("replace {label}: {error}"))?;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("sync {label} directory: {error}"))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// Runs one signed local-anchor write without exposing Tokio panic payloads.
///
/// [ANCHOR-WORKER-PRIVACY 2026-07-30 by Codex] Anchor persistence failures
/// may reach startup, readiness, or operator logs. The caller-provided label is
/// static and the JoinError is collapsed before it crosses that boundary.
pub(super) async fn run_blocking_local_anchor_write<F>(
    label: &'static str,
    write: F,
) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    tokio::task::spawn_blocking(write).await.map_err(|error| {
        let failure = RuntimeTaskJoinFailureKind::classify(&error);
        format!("{label} write worker {failure}")
    })?
}

pub(in crate::services::memchain) async fn persist_record_commitment_tip_anchor(
    path: PathBuf,
    tip_height: u64,
    tip_hash: [u8; 32],
    identity: &IdentityKeyPair,
) -> Result<u64, String> {
    let updated_at = unix_now_secs();
    let anchor =
        RecordCommitmentTipAnchorV1::new_signed(tip_height, tip_hash, identity, updated_at);
    let bytes = serde_json::to_vec(&anchor)
        .map_err(|error| format!("encode commitment tip anchor: {error}"))?;
    run_blocking_local_anchor_write("commitment tip anchor", move || {
        write_record_commitment_tip_anchor_atomic(&path, &bytes)
    })
    .await?;
    Ok(updated_at)
}

impl MemoryStorage {
    /// Verifies or initializes the coordinator's signed tip high-water mark.
    ///
    /// This must run after the complete SQLite chain audit. A stored anchor may
    /// be behind the audited tip only when its exact height/hash remains an
    /// ancestor of that chain; in that case it is atomically advanced. An
    /// anchor ahead of SQLite, a same-height mismatch, an invalid signature,
    /// or an ancestry mismatch clears runtime integrity and fails startup.
    ///
    /// Scope: this detects SQLite rollback/replacement while the host-side
    /// anchor remains. Replaying a whole disk/VM snapshot can roll back both
    /// files and requires an external peer witness in a later protocol layer.
    pub async fn configure_record_commitment_tip_anchor(
        &self,
        path: impl AsRef<Path>,
        identity: &IdentityKeyPair,
    ) -> Result<&'static str, String> {
        let path = path.as_ref().to_path_buf();
        let (current_height, current_hash) = self
            .commitment_integrity
            .read()
            .as_ref()
            .map(|runtime| (runtime.verified_tip_height, runtime.verified_tip_hash))
            .ok_or_else(|| {
                "commitment tip anchor requires a successful full chain audit".to_string()
            })?;
        {
            let mut runtime = self.commitment_tip_anchor.write();
            *runtime = Default::default();
            runtime.config = Some(RecordCommitmentTipAnchorConfig {
                path: path.clone(),
                identity: identity.clone(),
            });
            runtime.state = "checking";
        }

        let stored = match read_record_commitment_tip_anchor(&path).await {
            Ok(stored) => stored,
            Err(error) => {
                self.fail_record_commitment_tip_anchor("invalid", 0, false);
                return Err(error);
            }
        };
        let now = unix_now_secs();

        match stored {
            None => {
                let persisted_at = persist_record_commitment_tip_anchor(
                    path,
                    current_height,
                    current_hash,
                    identity,
                )
                .await
                .map_err(|error| {
                    self.fail_record_commitment_tip_anchor("write_failed", current_height, true);
                    error
                })?;
                let mut runtime = self.commitment_tip_anchor.write();
                runtime.state = "initialized";
                runtime.anchored_height = current_height;
                runtime.last_verified_at = Some(now);
                runtime.last_persisted_at = Some(persisted_at);
                info!(
                    tip_height = current_height,
                    "[MEMCHAIN_BLOCK] Signed commitment tip anchor initialized"
                );
                Ok("initialized")
            }
            Some(anchor) => {
                let verified = match anchor.verify(&identity.public_key_bytes()) {
                    Ok(verified) => verified,
                    Err(error) => {
                        self.fail_record_commitment_tip_anchor("invalid", 0, false);
                        return Err(error);
                    }
                };
                if verified.tip_height > current_height {
                    self.fail_record_commitment_tip_anchor(
                        "rollback_detected",
                        verified.tip_height,
                        false,
                    );
                    return Err(format!(
                        "commitment SQLite tip height {current_height} is behind signed local anchor height {}",
                        verified.tip_height
                    ));
                }

                let ancestor_hash = if verified.tip_height == 0 {
                    GENESIS_PREV_HASH
                } else {
                    let height = i64::try_from(verified.tip_height).map_err(|_| {
                        self.fail_record_commitment_tip_anchor(
                            "rollback_detected",
                            verified.tip_height,
                            false,
                        );
                        "commitment tip anchor height exceeds SQLite range".to_string()
                    })?;
                    let conn = self.conn.lock().await;
                    let hash: Option<Vec<u8>> = conn
                        .query_row(
                            "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                            params![height],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(|error| {
                            self.fail_record_commitment_tip_anchor(
                                "invalid",
                                verified.tip_height,
                                false,
                            );
                            format!("read anchored commitment ancestor: {error}")
                        })?;
                    hash.ok_or_else(|| {
                        self.fail_record_commitment_tip_anchor(
                            "rollback_detected",
                            verified.tip_height,
                            false,
                        );
                        "signed commitment tip anchor is not present in the audited chain"
                            .to_string()
                    })?
                    .try_into()
                    .map_err(|hash: Vec<u8>| {
                        self.fail_record_commitment_tip_anchor(
                            "rollback_detected",
                            verified.tip_height,
                            false,
                        );
                        format!(
                            "anchored commitment ancestor hash has invalid length {}",
                            hash.len()
                        )
                    })?
                };
                if ancestor_hash != verified.tip_hash {
                    self.fail_record_commitment_tip_anchor(
                        "rollback_detected",
                        verified.tip_height,
                        false,
                    );
                    return Err(format!(
                        "signed commitment tip anchor ancestry mismatch at height {}",
                        verified.tip_height
                    ));
                }

                if verified.tip_height == current_height {
                    let mut runtime = self.commitment_tip_anchor.write();
                    runtime.state = "verified";
                    runtime.anchored_height = current_height;
                    runtime.last_verified_at = Some(now);
                    runtime.last_persisted_at = Some(verified.updated_at);
                    info!(
                        tip_height = current_height,
                        "[MEMCHAIN_BLOCK] Signed commitment tip anchor verified"
                    );
                    return Ok("verified");
                }

                let persisted_at = persist_record_commitment_tip_anchor(
                    path,
                    current_height,
                    current_hash,
                    identity,
                )
                .await
                .map_err(|error| {
                    self.fail_record_commitment_tip_anchor(
                        "write_failed",
                        verified.tip_height,
                        true,
                    );
                    error
                })?;
                let mut runtime = self.commitment_tip_anchor.write();
                runtime.state = "repaired";
                runtime.anchored_height = current_height;
                runtime.last_verified_at = Some(now);
                runtime.last_persisted_at = Some(persisted_at);
                info!(
                    previous_height = verified.tip_height,
                    tip_height = current_height,
                    "[MEMCHAIN_BLOCK] Signed commitment tip anchor advanced after audited DB-ahead recovery"
                );
                Ok("repaired")
            }
        }
    }

    pub(in crate::services::memchain) fn fail_record_commitment_tip_anchor(
        &self,
        state: &'static str,
        anchored_height: u64,
        write_failure: bool,
    ) {
        let mut runtime = self.commitment_tip_anchor.write();
        runtime.state = state;
        runtime.anchored_height = anchored_height;
        runtime.last_verified_at = None;
        if write_failure {
            runtime.write_failures_total = runtime.write_failures_total.saturating_add(1);
        }
        drop(runtime);
        *self.commitment_integrity.write() = None;
    }
}
