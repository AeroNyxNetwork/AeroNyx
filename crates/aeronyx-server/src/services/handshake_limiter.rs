// ============================================
// File: crates/aeronyx-server/src/services/handshake_limiter.rs
// ============================================
//! # Handshake rate limiter
//!
//! [2026-09-12 by Claude] A ClientHello costs an Ed25519 verify, an X25519
//! exchange, an Ed25519 sign, a session and a virtual IP — and anyone can
//! sign one with a key they just made up. Before this, ~1000 hellos filled a
//! node (`max_connections`) in well under a second and the sessions stayed
//! for the idle timeout. Two token buckets: one per source IP, one global.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Instant;

use parking_lot::Mutex;

struct Bucket {
    tokens: f64,
    last: Instant,
}

/// Per-IP + global token buckets for ClientHello admission.
pub struct HandshakeLimiter {
    per_ip_rate: f64,
    per_ip_burst: f64,
    global_rate: f64,
    global_burst: f64,
    per_ip: Mutex<HashMap<IpAddr, Bucket>>,
    global: Mutex<Bucket>,
}

impl HandshakeLimiter {
    /// Production defaults: 2 hellos/s with a burst of 6 per address; 300/s
    /// with a burst of 600 node-wide (a busy node sees tens per minute).
    #[must_use]
    pub fn production() -> Self {
        Self::new(2.0, 6.0, 300.0, 600.0)
    }

    #[must_use]
    pub fn new(per_ip_rate: f64, per_ip_burst: f64, global_rate: f64, global_burst: f64) -> Self {
        Self {
            per_ip_rate,
            per_ip_burst,
            global_rate,
            global_burst,
            per_ip: Mutex::new(HashMap::new()),
            global: Mutex::new(Bucket {
                tokens: global_burst,
                last: Instant::now(),
            }),
        }
    }

    /// Admit one hello from `ip` now?
    pub fn allow(&self, ip: IpAddr) -> bool {
        self.allow_at(ip, Instant::now())
    }

    fn take(bucket: &mut Bucket, rate: f64, burst: f64, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(bucket.last).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate).min(burst);
        bucket.last = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    pub fn allow_at(&self, ip: IpAddr, now: Instant) -> bool {
        {
            let mut global = self.global.lock();
            if !Self::take(&mut global, self.global_rate, self.global_burst, now) {
                return false;
            }
        }
        let mut per_ip = self.per_ip.lock();
        if per_ip.len() > 50_000 {
            // Forget addresses that have had time to refill fully.
            let full_after = self.per_ip_burst / self.per_ip_rate;
            per_ip.retain(|_, b| now.saturating_duration_since(b.last).as_secs_f64() < full_after);
        }
        let bucket = per_ip.entry(ip).or_insert_with(|| Bucket {
            tokens: self.per_ip_burst,
            last: now,
        });
        Self::take(bucket, self.per_ip_rate, self.per_ip_burst, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_burst_then_the_rate() {
        let limiter = HandshakeLimiter::new(2.0, 6.0, 1000.0, 1000.0);
        let ip: IpAddr = "203.0.113.7".parse().unwrap();
        let t0 = Instant::now();
        for _ in 0..6 {
            assert!(limiter.allow_at(ip, t0));
        }
        assert!(!limiter.allow_at(ip, t0), "seventh hello in the same instant is refused");
        assert!(limiter.allow_at(ip, t0 + Duration::from_millis(600)), "refills at 2/s");
        let other: IpAddr = "203.0.113.8".parse().unwrap();
        assert!(limiter.allow_at(other, t0), "another address has its own bucket");
    }

    #[test]
    fn the_global_bucket_caps_many_addresses() {
        let limiter = HandshakeLimiter::new(100.0, 100.0, 10.0, 10.0);
        let t0 = Instant::now();
        let mut admitted = 0;
        for i in 0..50u8 {
            let ip: IpAddr = format!("198.51.100.{}", i).parse().unwrap();
            if limiter.allow_at(ip, t0) {
                admitted += 1;
            }
        }
        assert_eq!(admitted, 10);
    }
}
