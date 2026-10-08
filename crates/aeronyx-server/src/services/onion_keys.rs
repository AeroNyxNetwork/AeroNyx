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
//!   within its fixed retirement grace window; do not widen this.
//! [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Grace starts when a key is
//! actually superseded, not when it was created. Delayed rotation must not
//! discard a key still referenced by a recently issued signed descriptor.
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

// [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] A single previous generation
// must retire before the next scheduled rotation, without silently clipping
// the configured descriptor lifetime or extending the rotation interval.
pub const MAX_ONION_DESCRIPTOR_TTL_SECS: u64 = ONION_KEY_ROTATION_SECS - GRACE_SKEW_SECS - 1;

pub(crate) const fn descriptor_ttl_is_supported(ttl: u64) -> bool {
    ttl >= 60 && ttl <= MAX_ONION_DESCRIPTOR_TTL_SECS
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum OnionKeyError {
    #[error("onion key local time rejected")]
    Time,
    #[error("onion key descriptor lifetime rejected")]
    DescriptorLifetime,
    #[error("onion key manager unavailable")]
    Unavailable,
}

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

// [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Retention is immutable for
// this retired epoch, including if another embedded owner initializes later.
struct RetiredOnionKeyEpoch {
    key: OnionKeyEpoch,
    retired_at: u64,
    retained_until: u64,
}

impl RetiredOnionKeyEpoch {
    fn is_live_at(&self, now: u64) -> bool {
        now >= self.retired_at && now <= self.retained_until
    }
}

/// In-memory rotating onion key store: the current key plus an optional
/// previous key kept for the rotation grace window.
pub struct OnionKeyManager {
    current: OnionKeyEpoch,
    previous: Option<RetiredOnionKeyEpoch>,
    rotation_secs: u64,
    grace_secs: u64,
    observed_at: u64,
}

impl OnionKeyManager {
    fn new(now: u64) -> Self {
        Self {
            current: OnionKeyEpoch::generate(now),
            previous: None,
            rotation_secs: ONION_KEY_ROTATION_SECS,
            grace_secs: ONION_KEY_GRACE_SECS,
            observed_at: now,
        }
    }

    fn current_public(&self) -> [u8; 32] {
        self.current.public
    }

    /// Candidate secrets to attempt when peeling: always the current key, plus
    /// the previous key while it is still inside its retirement grace window.
    fn peel_secrets(&self, now: u64) -> Vec<StaticSecret> {
        if now == 0 || now < self.observed_at {
            return Vec::new();
        }
        let mut secrets = vec![self.current.static_secret()];
        if let Some(previous) = &self.previous {
            if previous.is_live_at(now) {
                secrets.push(previous.key.static_secret());
            }
        }
        secrets
    }

    /// Rotates the current key once it reaches `rotation_secs`, retaining the old
    /// key as `previous`. Drops `previous` once it falls outside the grace
    /// window. Idempotent and cheap; safe to call frequently.
    fn tick_rotation(&mut self, now: u64) -> Result<(), OnionKeyError> {
        // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] A rollback cannot
        // revive an old overlap or alter a key generation. Reject before RNG
        // or mutation, and keep the last successful rotation observation.
        if now == 0 || now < self.observed_at {
            return Err(OnionKeyError::Time);
        }
        if now.saturating_sub(self.current.created_at) >= self.rotation_secs {
            let retained_until = now.checked_add(self.grace_secs).ok_or(OnionKeyError::Time)?;
            let superseded = std::mem::replace(&mut self.current, OnionKeyEpoch::generate(now));
            self.previous = Some(RetiredOnionKeyEpoch {
                key: superseded,
                retired_at: now,
                retained_until,
            });
        }
        if let Some(previous) = &self.previous {
            if now > previous.retained_until {
                self.previous = None;
            }
        }
        self.observed_at = now;
        Ok(())
    }

    fn public_keys_at(&self, now: u64) -> ([u8; 32], Option<[u8; 32]>) {
        if now == 0 || now < self.observed_at {
            return ([0; 32], None);
        }
        (
            self.current_public(),
            self.previous.as_ref()
                .filter(|epoch| epoch.is_live_at(now))
                .map(|epoch| epoch.key.public),
        )
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
    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Retain the legacy public
    // signature; server composition uses the checked entry below.
    let _ = try_init_shared(now, descriptor_ttl_secs);
}

pub(crate) fn try_init_shared(now: u64, descriptor_ttl_secs: u64) -> Result<(), OnionKeyError> {
    let grace = effective_grace_secs(descriptor_ttl_secs)?;
    if now == 0 || now.checked_add(grace).is_none() {
        return Err(OnionKeyError::Time);
    }
    let mut manager = shared().write().map_err(|_| OnionKeyError::Unavailable)?;
    if now < manager.observed_at {
        return Err(OnionKeyError::Time);
    }
    // Never shorten an already published owner's future overlap. A retired
    // epoch's fixed retained_until is neither extended nor shortened here.
    manager.grace_secs = manager.grace_secs.max(grace);
    if manager.current.created_at == 0 {
        manager.current = OnionKeyEpoch::generate(now);
        manager.previous = None;
    }
    manager.observed_at = now;
    Ok(())
}

/// Effective previous-key grace window: the descriptor TTL plus clock-skew
/// allowance, never below the floor. A previous key must outlive any descriptor
/// still in circulation, so this is tied to the descriptor TTL.
fn effective_grace_secs(descriptor_ttl_secs: u64) -> Result<u64, OnionKeyError> {
    if !descriptor_ttl_is_supported(descriptor_ttl_secs) {
        return Err(OnionKeyError::DescriptorLifetime);
    }
    Ok((descriptor_ttl_secs + GRACE_SKEW_SECS).max(ONION_KEY_GRACE_SECS))
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

// [REVERSE-ONION-KEM-PUBLIC-OVERLAP 2026-10-04 by Codex] Expose only the
// public KEM generations that a descriptor may still advertise.  The private
// epochs remain inaccessible so restart/replay validation cannot obtain key
// material or alter the rotation lifecycle.
#[must_use]
pub fn advertised_public_keys(now: u64) -> ([u8; 32], Option<[u8; 32]>) {
    shared()
        .read()
        .map(|manager| manager.public_keys_at(now))
        .unwrap_or(([0u8; 32], None))
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
    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Legacy callers cannot
    // mutate a failed epoch; production discovery observes the checked error.
    let _ = try_tick_rotation(now);
}

pub(crate) fn try_tick_rotation(now: u64) -> Result<(), OnionKeyError> {
    shared().write().map_err(|_| OnionKeyError::Unavailable)?.tick_rotation(now)
}

#[cfg(test)]
mod tests {
    use super::*;
    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] These fixtures use
    // local managers only; no process-global mutation or test execution.
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::onion::{build_onion_envelope, try_open_onion_layer, OnionHop};

    #[test]
    fn rotation_moves_current_to_previous_and_keeps_both_peelable() {
        let mut manager = OnionKeyManager::new(1_000);
        let first = manager.current_public();
        assert!(manager.peel_secrets(1_000).len() == 1);

        // Before the rotation period: no rotation.
        manager.tick_rotation(1_000 + ONION_KEY_ROTATION_SECS - 1).unwrap();
        assert_eq!(manager.current_public(), first);

        // At the rotation period: rotate, previous retained.
        manager.tick_rotation(1_000 + ONION_KEY_ROTATION_SECS).unwrap();
        let second = manager.current_public();
        assert_ne!(second, first);
        assert_eq!(
            manager.peel_secrets(1_000 + ONION_KEY_ROTATION_SECS).len(),
            2
        );
    }

    #[test]
    fn advertised_public_keys_expose_only_current_and_in_grace_previous() {
        let mut manager = OnionKeyManager::new(1_000);
        let first = manager.current_public();
        manager.tick_rotation(1_000 + ONION_KEY_ROTATION_SECS).unwrap();
        let second = manager.current_public();
        let previous = manager.previous.as_ref().expect("rotation keeps previous");
        assert_eq!(previous.key.public, first);
        assert_eq!(second, manager.current_public());
        let within_grace = 1_000 + ONION_KEY_ROTATION_SECS + ONION_KEY_GRACE_SECS;
        let outside_grace = within_grace + 1;
        // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Exercise actual
        // candidate selection on both sides of the inclusive grace boundary.
        assert_eq!(manager.public_keys_at(within_grace), (second, Some(first)));
        assert_eq!(manager.peel_secrets(within_grace).len(), 2);
        assert_eq!(manager.public_keys_at(outside_grace), (second, None));
        assert_eq!(manager.peel_secrets(outside_grace).len(), 1);
        assert_eq!(manager.previous.as_ref().unwrap().retained_until, within_grace);
    }

    #[test]
    fn previous_key_dropped_after_grace() {
        let mut manager = OnionKeyManager::new(1);
        let rotated_at = 1 + ONION_KEY_ROTATION_SECS;
        manager.tick_rotation(rotated_at).unwrap();
        assert_eq!(manager.peel_secrets(rotated_at).len(), 2);

        // One second past the fixed deadline from actual retirement.
        let far = rotated_at + ONION_KEY_GRACE_SECS + 1;
        manager.tick_rotation(far).unwrap();
        assert_eq!(manager.peel_secrets(far).len(), 1);
        assert!(manager.previous.is_none());
    }

    #[test]
    fn grace_window_covers_descriptor_ttl() {
        // Grace must be >= descriptor TTL (else onions built against a still-valid
        // descriptor fail to peel after rotation). Check the example TTL (7200)
        // and the default (3600), plus the floor for tiny TTLs.
        assert!(effective_grace_secs(7200).unwrap() >= 7200);
        assert!(effective_grace_secs(3600).unwrap() >= 3600);
        assert_eq!(effective_grace_secs(7200).unwrap(), 7200 + GRACE_SKEW_SECS);
        assert_eq!(effective_grace_secs(60).unwrap(), ONION_KEY_GRACE_SECS);
        assert_eq!(effective_grace_secs(MAX_ONION_DESCRIPTOR_TTL_SECS).unwrap(),
            ONION_KEY_ROTATION_SECS - 1);
        for ttl in [0, 59, MAX_ONION_DESCRIPTOR_TTL_SECS + 1, u64::MAX] {
            assert_eq!(effective_grace_secs(ttl), Err(OnionKeyError::DescriptorLifetime));
        }
    }

    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Authored only:
    // these rejected initializations return before accessing global state.
    #[test]
    fn checked_initialization_rejects_bad_policy_or_time_before_shared_access() {
        for ttl in [0, 59, MAX_ONION_DESCRIPTOR_TTL_SECS + 1, u64::MAX] {
            assert_eq!(try_init_shared(1_000, ttl), Err(OnionKeyError::DescriptorLifetime));
        }
        for now in [0, u64::MAX - ONION_KEY_GRACE_SECS + 1, u64::MAX] {
            assert_eq!(try_init_shared(now, 60), Err(OnionKeyError::Time));
        }
    }

    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Authored only:
    // delayed discovery used to discard this just-retired key immediately.
    #[test]
    fn delayed_rotation_keeps_exact_old_ciphertext_for_full_retirement_grace() {
        let mut manager = OnionKeyManager::new(1_000);
        manager.grace_secs = effective_grace_secs(7200).unwrap();
        let old = manager.current_public();
        let retired_at = 1_000 + ONION_KEY_ROTATION_SECS + manager.grace_secs + 100;
        let source = IdentityKeyPair::from_bytes(&[57; 32]).unwrap();
        let recipient = IdentityKeyPair::from_bytes(&[58; 32]).unwrap();
        let hop = OnionHop { node_id: recipient.public_key_bytes(), kem_pub: old };
        let payload = b"synthetic delayed onion";
        let envelope = build_onion_envelope(&[hop], payload, [59; 16], 2, retired_at - 1, &source).unwrap();
        manager.tick_rotation(retired_at).unwrap();
        let current = manager.current_public();
        let deadline = retired_at + manager.grace_secs;
        for now in [retired_at, deadline - 1, deadline] {
            assert_eq!(manager.public_keys_at(now), (current, Some(old)));
            let peeled = try_open_onion_layer(&envelope.encrypted_blob, &manager.peel_secrets(now)).unwrap();
            assert_eq!(peeled.inner.as_slice(), payload);
            assert_eq!(peeled.next_hop, None);
        }
        assert!(try_open_onion_layer(&envelope.encrypted_blob, &manager.peel_secrets(deadline + 1)).is_err());
        let current_envelope = build_onion_envelope(
            &[OnionHop { node_id: recipient.public_key_bytes(), kem_pub: current }],
            payload, [60; 16], 2, deadline + 1, &source,
        ).unwrap();
        assert!(try_open_onion_layer(&current_envelope.encrypted_blob,
            &manager.peel_secrets(deadline + 1)).is_ok());
        // Reads beyond expiry neither delete nor re-stamp the retired epoch.
        assert_eq!(manager.previous.as_ref().unwrap().retained_until, deadline);
        manager.tick_rotation(deadline + 1).unwrap();
        assert!(manager.previous.is_none());
    }

    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Authored only:
    // reject before mutation/RNG, including checked deadline overflow.
    #[test]
    fn invalid_rotation_observations_do_not_replace_or_revive_keys() {
        let mut manager = OnionKeyManager::new(1_000);
        let retired_at = 1_000 + ONION_KEY_ROTATION_SECS;
        manager.tick_rotation(retired_at).unwrap();
        let public = manager.current_public();
        let deadline = manager.previous.as_ref().unwrap().retained_until;
        let previous_public = manager.previous.as_ref().unwrap().key.public;
        for now in [0, retired_at - 1, u64::MAX] {
            assert_eq!(manager.tick_rotation(now), Err(OnionKeyError::Time));
            assert_eq!(manager.observed_at, retired_at);
            assert_eq!(manager.current_public(), public);
            assert_eq!(manager.previous.as_ref().unwrap().retained_until, deadline);
            assert_eq!(manager.previous.as_ref().unwrap().key.public, previous_public);
        }
        for now in [0, retired_at - 1] {
            assert!(manager.peel_secrets(now).is_empty());
            assert_eq!(manager.public_keys_at(now), ([0; 32], None));
        }
        manager.tick_rotation(deadline + 1).unwrap();
        assert_eq!(manager.tick_rotation(deadline), Err(OnionKeyError::Time));
        assert!(manager.peel_secrets(deadline).is_empty());
        assert!(manager.previous.is_none());
    }

    // [PHALA-KEM-RETIREMENT 2026-10-08 by Codex] Authored only:
    // two generations suffice at the maximum accepted TTL; restart does not
    // reconstruct any old ephemeral secret from a signing identity.
    #[test]
    fn repeated_rotation_stays_bounded_and_restart_loses_old_keys() {
        let mut manager = OnionKeyManager::new(1_000);
        manager.grace_secs = effective_grace_secs(MAX_ONION_DESCRIPTOR_TTL_SECS).unwrap();
        let source = IdentityKeyPair::from_bytes(&[61; 32]).unwrap();
        for round in 1..=3 {
            let old = manager.current_public();
            let now = 1_000 + round * ONION_KEY_ROTATION_SECS;
            manager.tick_rotation(now).unwrap();
            assert_eq!(manager.public_keys_at(now), (manager.current_public(), Some(old)));
            assert_eq!(manager.peel_secrets(now).len(), 2);
            assert_eq!(manager.peel_secrets(now + manager.grace_secs + 1).len(), 1);
        }
        let now = manager.observed_at;
        let envelope = build_onion_envelope(
            &[OnionHop { node_id: source.public_key_bytes(), kem_pub: manager.current_public() }],
            b"synthetic restart", [62; 16], 2, now, &source,
        ).unwrap();
        assert!(try_open_onion_layer(&envelope.encrypted_blob, &manager.peel_secrets(now)).is_ok());
        let restarted = OnionKeyManager::new(now);
        assert!(restarted.previous.is_none());
        assert!(try_open_onion_layer(&envelope.encrypted_blob, &restarted.peel_secrets(now)).is_err());
    }

    #[test]
    fn generated_public_matches_static_secret() {
        let epoch = OnionKeyEpoch::generate(42);
        let derived = PublicKey::from(&epoch.static_secret()).to_bytes();
        assert_eq!(derived, epoch.public);
    }
}
