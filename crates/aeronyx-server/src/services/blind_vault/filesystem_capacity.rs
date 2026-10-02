// [ARCH-SPLIT 2026-10-02]
// Physical capacity probe used before admission and writes.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

impl BlindVaultService {
    /// Enforces the physical reserve for a capacity-consuming transition.
    /// Probe errors are unavailable rather than capacity: callers may retry a
    /// different healthy replica, while this node remains fail-closed.
    pub(super) fn ensure_filesystem_capacity(
        &self,
        additional_bytes: u64,
    ) -> Result<(), BlindVaultServiceError> {
        if self.config.min_free_disk_bytes == 0 {
            return Ok(());
        }
        let transient_payload_bytes = additional_bytes
            .checked_mul(2)
            .ok_or(BlindVaultServiceError::NodeCapacityExceeded)?;
        let required_available = self
            .config
            .min_free_disk_bytes
            .checked_add(transient_payload_bytes)
            .and_then(|bytes| bytes.checked_add(FILESYSTEM_WRITE_OVERHEAD_BYTES))
            .ok_or(BlindVaultServiceError::NodeCapacityExceeded)?;
        let available = self
            .filesystem_capacity_probe
            .available_bytes(&self.filesystem_capacity_path)
            .map_err(|_| BlindVaultServiceError::FilesystemCapacityUnavailable)?;
        if available < required_available {
            return Err(BlindVaultServiceError::NodeCapacityExceeded);
        }
        Ok(())
    }

    /// Returns aggregate-only physical capacity state without turning a local
    /// probe outage into a status-endpoint failure.
    pub(super) fn filesystem_capacity_status(&self) -> (Option<u64>, bool) {
        if self.config.min_free_disk_bytes == 0 {
            return (None, true);
        }
        match self
            .filesystem_capacity_probe
            .available_bytes(&self.filesystem_capacity_path)
        {
            Ok(available) => (
                Some(available),
                available >= self.config.min_free_disk_bytes,
            ),
            Err(_) => (None, false),
        }
    }
}
