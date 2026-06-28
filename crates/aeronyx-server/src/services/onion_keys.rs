// ============================================
// File: crates/aeronyx-server/src/services/onion_keys.rs
// ============================================
//! # Onion Routing — Rotating Onion Keys (Forward Secrecy)
//!
//! ## Creation Reason
//! Onion routing v1 originally derived each hop's KEM key from the node's
//! long-term Ed25519 identity. That has **no forward secrecy**: if the identity
//! secret leaks, an adversary who recorded past onions addressed to this node
//! can re-derive every layer key and recover the routing metadata (its
//! next-hops) retroactively.
//!
//! Forward secrecy fundamentally requires that old key material be **deleted and
//! unrecoverable**. This module therefore manages a dedicated **onion KEM
//! keypair, in memory only, that rotates on a schedule** and keeps the previous
//! key for a short overlap window. This matches Tor's onion-key model (rotation
//! with overlap) and the architecture decision in
//! `docs/onion-routing-architecture-decision.md` (D-B).
//!
//! ## Design
//! - The keypair is **never persisted**. A process restart yields a fresh key
//!   and the old secret is gone — strictly stronger forward secrecy than
//!   on-disk onion keys. The node re-publishes and re-gossips its descriptor on
//!   startup, so peers pick up the new public key promptly.
//! - The node's long-term **Ed25519 identity is unchanged** and still signs
//!   descriptors. Identity ≠ onion key, exactly like Tor (identity vs onion).
//! - **Reads never mutate.** `current_public_key()` (used when building the
//!   self descriptor) and `peel_secrets()` (used when peeling a received layer)
//!   take a read lock only. Rotation happens exclusively via `tick_rotation()`,
//!   which the discovery background task calls on its cadence. This keeps the
//!   process-global deterministic for unit tests, which never rotate.
//!
//! ## Process-global rationale
//! The onion key is per-process node state (one keypair per node), read by both
//! the descriptor builder and the relay peel path. A managed process-global
//! avoids threading an `Arc<RwLock<…>>` through a dozen unrelated signatures;
//! it is initialized once at startup via [`init_shared`].
//!
//! ## ⚠️ Important Notes for Next Developer
//! - Never log or persist the secret bytes. They are zeroized on drop.
//! - `peel_secrets` returns `current` plus `previous` only while the latter is
//!   within the rotation + grace window; do not widen this.
//! - When `kem_alg = 2` (X-Wing) lands, generate the hybrid keypair here and
//!   keep the same rotate/grace lifecycle.
//!
//! ## Last Modified
//! v1.1.0-OnionFS — Initial rotating onion key manager for forward secrecy

use std::sync::{Arc, OnceLock, RwLock};

use rand::rngs::OsRng;
use rand::RngCore;
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::Zeroize;

/// How often the onion keypair rotates. Decoupled from (and longer than) the
/// descriptor TTL so a fresh descriptor always carries a live onion key.
pub const ONION_KEY_ROTATION_SECS: u64 = 24 * 60 * 60; // 24 hours

/// Floor for how long the previous onion key is retained after a rotation, so
/// onions built against the just-superseded descriptor still peel. The EFFECTIVE
/// grace is set at startup to `descriptor_ttl_secs + GRACE_SKEW_SECS` (see
/// [`init_shared`]) because a client may build an onion against the current
/// descriptor right up to its expiry; the grace must cover the full descriptor
/// TTL or peels fail near each rotation. This const is only the lower bound /
/// lazy-default used before `init_shared` runs (e.g. in unit tests).
pub const ONION_KEY_GRACE_SECS: u64 = 60 * 60; // 1 hour floor

/// Extra grace beyond the descriptor TTL to absorb clock skew and in-flight time.
pub const GRACE_SKEW_SECS: u64 = 10 * 60; // 10 minutes

/// One onion key generation. Stores the raw secret bytes (reconstructed into a
/// `StaticSecret` on demand) and the derived public key.
struct OnionKeyEpoch {
    secret: [u8; 32],
    public: [u8; 32],
    created_at: u64,
}

impl OnionKeyEpoch {
    fn generate(now: u64) -> Self {
        let mut secret = [0u8; 32];
        OsRng.fill_bytes(&mut secret);
        // `StaticSecret::from` clamps a copy internally; deriving the public key
        // from it keeps the published key consistent with `static_secret()`.
        let public = PublicKey::from(&StaticSecret::from(secret)).to_bytes();
        Self {
            secret,
            public,
            created_at: now,
        }
    }

    fn static_secret(&self) -> StaticSecret {
        StaticSecret::from(self.secret)
    }
}

impl Drop for OnionKeyEpoch {
    fn drop(&mut self) {
        self.secret.zeroize();
    }
}

/// In-memory rotating onion key store: the current key plus an optional
/// previous key kept for the rotation grace window.
pub struct OnionKeyManager {
    current: OnionKeyEpoch,
    previous: Option<OnionKeyEpoch>,
    rotation_secs: u64,
    grace_secs: u64,
}

impl OnionKeyManager {
    fn new(now: u64) -> Self {
        Self {
            current: OnionKeyEpoch::generate(now),
            previous: None,
            rotation_secs: ONION_KEY_ROTATION_SECS,
            grace_secs: ONION_KEY_GRACE_SECS,
        }
    }

    fn current_public(&self) -> [u8; 32] {
        self.current.public
    }

    /// Candidate secrets to attempt when peeling: always the current key, plus
    /// the previous key while it is still inside the rotation + grace window.
    fn peel_secrets(&self, now: u64) -> Vec<StaticSecret> {
        let mut secrets = vec![self.current.static_secret()];
        if let Some(previous) = &self.previous {
            if now.saturating_sub(previous.created_at) <= self.rotation_secs + self.grace_secs {
                secrets.push(previous.static_secret());
            }
        }
        secrets
    }

    /// Rotates the current key once it reaches `rotation_secs`, retaining the old
    /// key as `previous`. Drops `previous` once it falls outside the grace
    /// window. Idempotent and cheap; safe to call frequently.
    fn tick_rotation(&mut self, now: u64) {
        if now.saturating_sub(self.current.created_at) >= self.rotation_secs {
            let superseded = std::mem::replace(&mut self.current, OnionKeyEpoch::generate(now));
            self.previous = Some(superseded);
        }
        if let Some(previous) = &self.previous {
            if now.saturating_sub(previous.created_at) > self.rotation_secs + self.grace_secs {
                self.previous = None;
            }
        }
    }
}

static SHARED: OnceLock<Arc<RwLock<OnionKeyManager>>> = OnceLock::new();

fn shared() -> &'static Arc<RwLock<OnionKeyManager>> {
    // Lazy default (created_at = 0) covers any read that races ahead of
    // `init_shared` (e.g. a unit test). `init_shared` re-stamps it at startup.
    SHARED.get_or_init(|| Arc::new(RwLock::new(OnionKeyManager::new(0))))
}

/// Initializes the process-global onion key at server startup. Sets the grace
/// window from the descriptor TTL (so a previous key outlives any descriptor
/// still in circulation) and stamps the key with a real timestamp so the first
/// scheduled rotation respects the full rotation period.
///
/// `descriptor_ttl_secs` is `discovery.descriptor_ttl_secs`. The effective grace
/// becomes `descriptor_ttl_secs + GRACE_SKEW_SECS` (floored at
/// `ONION_KEY_GRACE_SECS`). This MUST stay well below `ONION_KEY_ROTATION_SECS`
/// so the single retained previous key fully covers the grace window; the
/// descriptor TTL is expected to be a few hours at most.
pub fn init_shared(now: u64, descriptor_ttl_secs: u64) {
    if let Ok(mut manager) = shared().write() {
        manager.grace_secs = effective_grace_secs(descriptor_ttl_secs);
        if manager.current.created_at == 0 {
            manager.current = OnionKeyEpoch::generate(now);
            manager.previous = None;
        }
    }
}

/// Effective previous-key grace window: the descriptor TTL plus clock-skew
/// allowance, never below the floor. A previous key must outlive any descriptor
/// still in circulation, so this is tied to the descriptor TTL.
fn effective_grace_secs(descriptor_ttl_secs: u64) -> u64 {
    descriptor_ttl_secs
        .saturating_add(GRACE_SKEW_SECS)
        .max(ONION_KEY_GRACE_SECS)
}

/// The current onion public key to publish in the node's signed descriptor.
/// Read-only; never rotates.
#[must_use]
pub fn current_public_key() -> [u8; 32] {
    shared()
        .read()
        .map(|manager| manager.current_public())
        .unwrap_or([0u8; 32])
}

/// Candidate secrets for peeling a received onion layer (current + in-grace
/// previous). Read-only; never rotates.
#[must_use]
pub fn peel_secrets(now: u64) -> Vec<StaticSecret> {
    shared()
        .read()
        .map(|manager| manager.peel_secrets(now))
        .unwrap_or_default()
}

/// Advances key rotation. Called only by the discovery background task on its
/// cadence, never by request paths — this keeps reads deterministic for tests.
pub fn tick_rotation(now: u64) {
    if let Ok(mut manager) = shared().write() {
        manager.tick_rotation(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_moves_current_to_previous_and_keeps_both_peelable() {
        let mut manager = OnionKeyManager::new(1_000);
        let first = manager.current_public();
        assert!(manager.peel_secrets(1_000).len() == 1);

        // Before the rotation period: no rotation.
        manager.tick_rotation(1_000 + ONION_KEY_ROTATION_SECS - 1);
        assert_eq!(manager.current_public(), first);

        // At the rotation period: rotate, previous retained.
        manager.tick_rotation(1_000 + ONION_KEY_ROTATION_SECS);
        let second = manager.current_public();
        assert_ne!(second, first);
        assert_eq!(
            manager.peel_secrets(1_000 + ONION_KEY_ROTATION_SECS).len(),
            2
        );
    }

    #[test]
    fn previous_key_dropped_after_grace() {
        let mut manager = OnionKeyManager::new(0);
        manager.tick_rotation(ONION_KEY_ROTATION_SECS); // rotate once
        assert_eq!(manager.peel_secrets(ONION_KEY_ROTATION_SECS).len(), 2);

        // Well past rotation + grace from the previous key's creation (t=0).
        let far = ONION_KEY_ROTATION_SECS + ONION_KEY_GRACE_SECS + 1;
        manager.tick_rotation(far);
        assert_eq!(manager.peel_secrets(far).len(), 1);
    }

    #[test]
    fn grace_window_covers_descriptor_ttl() {
        // Grace must be >= descriptor TTL (else onions built against a still-valid
        // descriptor fail to peel after rotation). Check the example TTL (7200)
        // and the default (3600), plus the floor for tiny TTLs.
        assert!(effective_grace_secs(7200) >= 7200);
        assert!(effective_grace_secs(3600) >= 3600);
        assert_eq!(effective_grace_secs(7200), 7200 + GRACE_SKEW_SECS);
        assert_eq!(effective_grace_secs(60), ONION_KEY_GRACE_SECS); // floor
                                                                    // Must stay safely below the rotation period for the single-previous model.
        assert!(effective_grace_secs(7200) < ONION_KEY_ROTATION_SECS);
    }

    #[test]
    fn generated_public_matches_static_secret() {
        let epoch = OnionKeyEpoch::generate(42);
        let derived = PublicKey::from(&epoch.static_secret()).to_bytes();
        assert_eq!(derived, epoch.public);
    }
}
