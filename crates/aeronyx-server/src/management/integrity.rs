//! # Binary Integrity Check

use sha2::{Digest, Sha256};
use tracing::{debug, warn};

pub fn compute_binary_hash() -> String {
    match std::env::current_exe() {
        Ok(path) => {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let mut hasher = Sha256::new();
                    hasher.update(&bytes);
                    let hash = hex::encode(hasher.finalize());
                    debug!("Binary hash: {}...", &hash[..16]);
                    hash
                }
                Err(e) => {
                    warn!("Failed to read binary: {}", e);
                    "unable_to_read".to_string()
                }
            }
        }
        Err(_) => "unknown".to_string()
    }
}

pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
