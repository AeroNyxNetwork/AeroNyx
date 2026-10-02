// [ARCH-SPLIT 2026-10-02]
// Bounded file reads and process-local readiness observations used by startup.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl Server {
    /// Starts bounded authority pulls and optional non-authoritative mirroring.
    ///
    /// Empty pins disable checkpoints/witnesses but may still run opt-in mirror
    /// transport. Once either mode is configured, initialization and task
    /// liveness are required process state.
    /// Read a recovery snapshot through a fixed-length view of the file.
    /// Metadata provides an inexpensive rejection, while `take(max + 1)` also
    /// catches a file that grows after the metadata check.
    pub(super) async fn read_bounded_file(
        path: &Path,
        max_bytes: usize,
    ) -> std::io::Result<Vec<u8>> {
        let file = tokio::fs::File::open(path).await?;
        let declared_length = file.metadata().await?.len();
        if declared_length > max_bytes as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file exceeds configured recovery limit",
            ));
        }

        let initial_capacity = declared_length.min(max_bytes as u64) as usize;
        let mut reader = file.take(max_bytes.saturating_add(1) as u64);
        let mut body = Vec::with_capacity(initial_capacity);
        reader.read_to_end(&mut body).await?;
        if body.len() > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file grew beyond configured recovery limit",
            ));
        }
        Ok(body)
    }

    pub(super) fn bounded_file_error_reason(error: &std::io::Error) -> &'static str {
        if error.kind() == std::io::ErrorKind::InvalidData {
            "file_too_large"
        } else {
            "read_failed"
        }
    }

    pub(super) fn classify_reqwest_error(phase: &str, error: &reqwest::Error) -> String {
        if error.is_timeout() {
            return format!("{phase}_timeout");
        }
        if error.is_connect() {
            return format!("{phase}_connect");
        }
        if error.is_status() {
            if let Some(status) = error.status() {
                return format!("{phase}_http_{}", status.as_u16());
            }
            return format!("{phase}_http_status");
        }
        if error.is_decode() {
            return format!("{phase}_decode");
        }
        if error.is_body() {
            return format!("{phase}_body");
        }
        if error.is_request() {
            return format!("{phase}_request");
        }
        format!("{phase}_unknown")
    }

    /// Observes aggregate Blind Vault admission readiness away from an async
    /// executor worker. Any missing service, storage error, probe error, or
    /// worker failure is not ready.
    pub(super) async fn observe_blind_vault_admission_readiness(
        blind_vault: Option<Arc<BlindVaultService>>,
        now_secs: u64,
    ) -> bool {
        let Some(blind_vault) = blind_vault else {
            return false;
        };
        // [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] SQLite and
        // statvfs are synchronous capabilities. Never block Tokio's discovery
        // or heartbeat worker while deriving a signed advertisement decision.
        tokio::task::spawn_blocking(move || {
            blind_vault
                .admission_readiness(now_secs.saturating_mul(1_000))
                .map(|readiness| readiness.is_ready())
                .unwrap_or(false)
        })
        .await
        .unwrap_or(false)
    }
}
