// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn peer_guard_rejects_replay_and_enforces_rate_limit() {
    let peer = [0x11; 32];
    let wall_now = 1_700_000_000;
    let monotonic_now = Instant::now();
    let mut guard = PeerRequestGuard::default();
    assert!(guard.admit_at(peer, [0x01; 16], wall_now, monotonic_now));
    assert!(!guard.admit_at(peer, [0x01; 16], wall_now, monotonic_now));
    for value in 2..MAX_REQUESTS_PER_PEER_PER_MINUTE {
        let mut request_id = [0u8; 16];
        request_id[..4].copy_from_slice(&value.to_le_bytes());
        assert!(guard.admit_at(peer, request_id, wall_now, monotonic_now));
    }
    assert!(!guard.admit_at(peer, [0xFF; 16], wall_now, monotonic_now));
    assert!(guard.admit_at(
        peer,
        [0xFF; 16],
        wall_now + 61,
        monotonic_now + PEER_RATE_LIMIT_WINDOW,
    ));
}

#[test]
fn peer_guard_wall_clock_corrections_cannot_reset_rate_budget() {
    let peer = [0x12; 32];
    let wall_now = 1_700_000_000;
    let monotonic_now = Instant::now();
    let mut guard = PeerRequestGuard::default();

    for value in 0..MAX_REQUESTS_PER_PEER_PER_MINUTE {
        let mut request_id = [0u8; 16];
        request_id[..4].copy_from_slice(&value.to_le_bytes());
        assert!(guard.admit_at(peer, request_id, wall_now, monotonic_now));
    }

    assert!(!guard.admit_at(
        peer,
        [0xFE; 16],
        wall_now.saturating_sub(3_600),
        monotonic_now,
    ));
    assert!(!guard.admit_at(peer, [0xFD; 16], wall_now + 3_600, monotonic_now,));
    assert!(guard.admit_at(
        peer,
        [0xFC; 16],
        wall_now + 3_600,
        monotonic_now + PEER_RATE_LIMIT_WINDOW,
    ));
}

#[test]
fn peer_guard_allows_idempotent_hint_retries_within_shared_rate_limit() {
    let peer = [0x22; 32];
    let wall_now = 1_700_000_000;
    let monotonic_now = Instant::now();
    let mut guard = PeerRequestGuard::default();

    for _ in 0..MAX_REQUESTS_PER_PEER_PER_MINUTE {
        assert!(guard.admit_idempotent_hint_at(peer, wall_now, monotonic_now));
    }
    assert!(!guard.admit_idempotent_hint_at(peer, wall_now, monotonic_now));
    assert!(guard.admit_idempotent_hint_at(peer, wall_now, monotonic_now + PEER_RATE_LIMIT_WINDOW,));
}
