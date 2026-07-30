// ============================================
// File: crates/aeronyx-server/src/services/dns_proxy.rs
// ============================================
//! Privacy-safe UDP DNS forwarding for VPN clients.
//!
//! ## Creation Reason
//! Commercial VPN clients need DNS resolution through the VPN gateway address.
//! The health endpoint already verifies `gateway_ip:53`, but the Rust node did
//! not provide a DNS listener. This module starts a small UDP forwarder bound to
//! the VPN gateway and forwards each DNS datagram to an upstream resolver.
//!
//! ## Privacy Boundary
//! The proxy does not parse, log, persist, or report queried domains. It only
//! forwards opaque DNS UDP payload bytes and reports aggregate startup/errors in
//! process logs. Nodeboard continues to receive only health-check status from:
//!   - Rust: /root/open/AeroNyx/crates/aeronyx-server/src/api/vpn_health.rs
//!   - Backend: /root/aeronyx/privacy_network/api/vpn_observability.py
//!   - Frontend: /root/open/nodeboard/app/dashboard/services/page.tsx
//!
//! ## Last Modified
//! v1.1.0 - [DNS-RUNTIME-OWNERSHIP 2026-07-30 by Codex] Pre-bind the
//! required listener, bound forwarding concurrency, and own child tasks.
//! v1.0.0 - Add VPN gateway DNS forwarding stub
// ============================================

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio::sync::broadcast;
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

const DNS_PORT: u16 = 53;
const DNS_BUFFER_BYTES: usize = 4096;
const DNS_MAX_IN_FLIGHT_QUERIES: usize = 256;
const DNS_QUERY_TIMEOUT: Duration = Duration::from_secs(3);
const UPSTREAM_DNS: [&str; 2] = ["1.1.1.1:53", "8.8.8.8:53"];

/// Bind and start the required VPN gateway DNS forwarder transactionally.
///
/// Unlike [`spawn_dns_proxy`], this function does not return until
/// `gateway_ip:53` is bound. Production startup must use this entry point so
/// systemd readiness cannot precede listener availability.
///
/// # Errors
///
/// Returns the operating-system bind error without spawning a background task.
pub async fn start_dns_proxy(
    gateway_ip: Ipv4Addr,
    shutdown_rx: broadcast::Receiver<()>,
) -> std::io::Result<JoinHandle<()>> {
    start_dns_proxy_on(SocketAddr::from((gateway_ip, DNS_PORT)), shutdown_rx).await
}

/// Spawn the VPN gateway DNS forwarder.
///
/// The listener binds to `gateway_ip:53`, receives opaque DNS UDP payloads from
/// VPN clients, forwards them to an upstream resolver, and returns the upstream
/// response to the original client. Query names are never decoded or logged.
///
/// This compatibility entry point retains the original non-blocking signature.
/// New production startup code must use [`start_dns_proxy`] so bind failure is
/// part of the readiness transaction.
pub fn spawn_dns_proxy(
    gateway_ip: Ipv4Addr,
    shutdown_rx: broadcast::Receiver<()>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let listen_addr = SocketAddr::from((gateway_ip, DNS_PORT));
        let socket = match bind_dns_proxy_socket(listen_addr).await {
            Ok(socket) => socket,
            Err(err) => {
                error!(
                    error = %err,
                    listen_addr = %listen_addr,
                    "[DNS] Failed to bind VPN DNS forwarder"
                );
                return;
            }
        };
        run_dns_proxy(socket, listen_addr, shutdown_rx).await;
    })
}

async fn start_dns_proxy_on(
    listen_addr: SocketAddr,
    shutdown_rx: broadcast::Receiver<()>,
) -> std::io::Result<JoinHandle<()>> {
    // [DNS-STARTUP-READINESS 2026-07-30 by Codex] Bind in the caller's
    // startup transaction. No task exists when the required address is
    // unavailable.
    let socket = bind_dns_proxy_socket(listen_addr).await?;
    Ok(tokio::spawn(run_dns_proxy(
        socket,
        listen_addr,
        shutdown_rx,
    )))
}

async fn bind_dns_proxy_socket(listen_addr: SocketAddr) -> std::io::Result<Arc<UdpSocket>> {
    UdpSocket::bind(listen_addr).await.map(Arc::new)
}

async fn run_dns_proxy(
    socket: Arc<UdpSocket>,
    requested_listen_addr: SocketAddr,
    mut shutdown_rx: broadcast::Receiver<()>,
) {
    let listen_addr = socket.local_addr().unwrap_or(requested_listen_addr);
    info!(
        listen_addr = %listen_addr,
        upstreams = ?UPSTREAM_DNS,
        max_in_flight = DNS_MAX_IN_FLIGHT_QUERIES,
        "[DNS] VPN DNS forwarder started"
    );

    let mut dropped_overload = 0u64;
    let mut forward_tasks = JoinSet::new();
    let mut buf = vec![0u8; DNS_BUFFER_BYTES];
    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.recv() => {
                info!("[DNS] VPN DNS forwarder shutting down");
                break;
            }
            completed = forward_tasks.join_next(), if !forward_tasks.is_empty() => {
                if let Some(Err(join_error)) = completed {
                    let reason = if join_error.is_panic() {
                        "panic"
                    } else if join_error.is_cancelled() {
                        "cancelled"
                    } else {
                        "join_failed"
                    };
                    warn!(
                        reason,
                        "[DNS] Forwarding worker ended unexpectedly"
                    );
                }
            }
            received = socket.recv_from(&mut buf) => {
                let (len, client_addr) = match received {
                    Ok(value) => value,
                    Err(err) => {
                        // [DNS-RUNTIME-OWNERSHIP 2026-07-30 by Codex] A
                        // bound UDP listener returning an OS receive error is
                        // no longer allowed to hot-loop. Returning lets the
                        // required-task supervisor recover the process.
                        error!(error = %err, "[DNS] Required listener receive failed");
                        break;
                    }
                };

                if len == 0 {
                    continue;
                }

                if forward_tasks.len() >= DNS_MAX_IN_FLIGHT_QUERIES {
                    dropped_overload = dropped_overload.saturating_add(1);
                    if dropped_overload == 1 || dropped_overload.is_power_of_two() {
                        warn!(
                            dropped_overload,
                            max_in_flight = DNS_MAX_IN_FLIGHT_QUERIES,
                            "[DNS] Query dropped by forwarding concurrency limit"
                        );
                    }
                    continue;
                }

                let query = buf[..len].to_vec();
                let reply_socket = Arc::clone(&socket);
                forward_tasks.spawn(async move {
                    if let Err(err) =
                        forward_dns_query(reply_socket, client_addr, query).await
                    {
                        debug!(error = %err, "[DNS] DNS forward failed");
                    }
                });
            }
        }
    }

    // [DNS-RUNTIME-OWNERSHIP 2026-07-30 by Codex] JoinSet owns every query
    // worker. Abort and reap them before the parent reports completion so no
    // resolver request survives node shutdown or process recovery.
    forward_tasks.abort_all();
    while forward_tasks.join_next().await.is_some() {}
}

async fn forward_dns_query(
    reply_socket: Arc<UdpSocket>,
    client_addr: SocketAddr,
    query: Vec<u8>,
) -> std::io::Result<()> {
    let query_id = query.get(0..2).map(|value| [value[0], value[1]]);

    for upstream in UPSTREAM_DNS {
        let upstream_socket = UdpSocket::bind("0.0.0.0:0").await?;
        upstream_socket.connect(upstream).await?;
        upstream_socket.send(&query).await?;

        let mut response = vec![0u8; DNS_BUFFER_BYTES];
        let received = timeout(DNS_QUERY_TIMEOUT, upstream_socket.recv(&mut response)).await;
        let len = match received {
            Ok(Ok(len)) => len,
            Ok(Err(err)) => {
                debug!(error = %err, upstream, "[DNS] Upstream receive failed");
                continue;
            }
            Err(_) => {
                debug!(upstream, "[DNS] Upstream receive timed out");
                continue;
            }
        };

        if !matches_dns_query_id(query_id, &response[..len]) {
            debug!(upstream, "[DNS] Ignoring response with mismatched DNS id");
            continue;
        }

        reply_socket.send_to(&response[..len], client_addr).await?;
        return Ok(());
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "all upstream DNS resolvers failed",
    ))
}

fn matches_dns_query_id(query_id: Option<[u8; 2]>, response: &[u8]) -> bool {
    match (query_id, response.get(0..2)) {
        (Some(expected), Some(actual)) => actual == expected,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{matches_dns_query_id, start_dns_proxy_on};
    use std::time::Duration;
    use tokio::net::UdpSocket;

    #[tokio::test]
    async fn transactional_start_rejects_an_occupied_listener() {
        // [DNS-STARTUP-READINESS 2026-07-30 by Codex] Bind failure must reach
        // the startup caller synchronously instead of disappearing inside a
        // detached task after readiness.
        let occupied = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let listen_addr = occupied.local_addr().unwrap();
        let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);

        let result = start_dns_proxy_on(listen_addr, shutdown_rx).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::AddrInUse);
    }

    #[tokio::test]
    async fn transactional_start_owns_graceful_shutdown() {
        // [DNS-RUNTIME-OWNERSHIP 2026-07-30 by Codex] The pre-bound task must
        // observe the retained broadcast and join without an orphan worker.
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);
        let task = start_dns_proxy_on("127.0.0.1:0".parse().unwrap(), shutdown_rx)
            .await
            .unwrap();

        shutdown_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
    }

    #[test]
    fn response_must_match_the_opaque_query_id() {
        assert!(matches_dns_query_id(Some([0x12, 0x34]), &[0x12, 0x34]));
        assert!(!matches_dns_query_id(Some([0x12, 0x34]), &[0x12, 0x35]));
        assert!(!matches_dns_query_id(Some([0x12, 0x34]), &[0x12]));
        assert!(!matches_dns_query_id(None, &[0x12, 0x34]));
    }
}
