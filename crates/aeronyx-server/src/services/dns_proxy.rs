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
//! v1.0.0 - Add VPN gateway DNS forwarding stub
// ============================================

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

const DNS_PORT: u16 = 53;
const DNS_BUFFER_BYTES: usize = 4096;
const DNS_QUERY_TIMEOUT: Duration = Duration::from_secs(3);
const UPSTREAM_DNS: [&str; 2] = ["1.1.1.1:53", "8.8.8.8:53"];

/// Spawn the VPN gateway DNS forwarder.
///
/// The listener binds to `gateway_ip:53`, receives opaque DNS UDP payloads from
/// VPN clients, forwards them to an upstream resolver, and returns the upstream
/// response to the original client. Query names are never decoded or logged.
pub fn spawn_dns_proxy(
    gateway_ip: Ipv4Addr,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let listen_addr = SocketAddr::from((gateway_ip, DNS_PORT));
        let socket = match UdpSocket::bind(listen_addr).await {
            Ok(socket) => Arc::new(socket),
            Err(err) => {
                error!(
                    error = %err,
                    listen_addr = %listen_addr,
                    "[DNS] Failed to bind VPN DNS forwarder"
                );
                return;
            }
        };

        info!(
            listen_addr = %listen_addr,
            upstreams = ?UPSTREAM_DNS,
            "[DNS] VPN DNS forwarder started"
        );

        let mut buf = vec![0u8; DNS_BUFFER_BYTES];
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("[DNS] VPN DNS forwarder shutting down");
                    break;
                }
                received = socket.recv_from(&mut buf) => {
                    let (len, client_addr) = match received {
                        Ok(value) => value,
                        Err(err) => {
                            warn!(error = %err, "[DNS] Failed to receive DNS datagram");
                            continue;
                        }
                    };

                    if len == 0 {
                        continue;
                    }

                    let query = buf[..len].to_vec();
                    let reply_socket = Arc::clone(&socket);
                    tokio::spawn(async move {
                        if let Err(err) = forward_dns_query(reply_socket, client_addr, query).await {
                            debug!(error = %err, "[DNS] DNS forward failed");
                        }
                    });
                }
            }
        }
    })
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
