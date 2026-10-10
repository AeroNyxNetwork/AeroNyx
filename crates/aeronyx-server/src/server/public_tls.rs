// ============================================
// File: crates/aeronyx-server/src/server/public_tls.rs
// ============================================
//! # Identity-bound TLS on the public API port
//!
//! [NODE-TLS-BINDING 2026-10-10 by Claude] The public listener serves plain
//! HTTP and TLS 1.3 on the same port. A TLS record always starts with the
//! handshake content type `0x16`; an HTTP/1 request starts with an ASCII method
//! name. The first byte therefore decides, and operators open nothing new.
//!
//! The certificate is self-signed and carries a binding signed by this node's
//! Ed25519 identity (`aeronyx-node-tls`). A client or peer that knows the node
//! id from a signed descriptor verifies that binding instead of a CA chain, so
//! a node reachable only by IP address still gets authenticated, encrypted
//! transport. The identity key signs one binding per certificate and never a
//! TLS handshake.
//!
//! Connection handling mirrors `axum::serve` (HTTP/1 with upgrades, graceful
//! shutdown that drains open connections) so plain HTTP behaves as before.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::IdentityKeyPair;
use arc_swap::ArcSwap;
use axum::body::Body;
use axum::Router;
use hyper::body::Incoming;
use hyper::Request;
use hyper_util::rt::TokioIo;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;
use rustls::ServerConfig;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio_rustls::TlsAcceptor;
use tower::Service;
use tracing::{debug, info, warn};

/// Lifetime of one certificate binding. Verifiers reject anything claiming
/// more than `aeronyx_node_tls::MAX_BINDING_LIFETIME_SECS`.
pub(super) const CERT_LIFETIME_SECS: u64 = 7 * 24 * 60 * 60;
/// A fresh key and certificate replace the current one after this long, so a
/// served certificate always has days left even for a client whose clock runs
/// ahead.
pub(super) const CERT_ROTATE_AFTER_SECS: u64 = 24 * 60 * 60;
const ROTATION_CHECK_INTERVAL: Duration = Duration::from_secs(10 * 60);
/// A connection that sends nothing, or does not finish its TLS handshake, is
/// dropped after this long.
const FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(10);
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const TLS_HANDSHAKE_CONTENT_TYPE: u8 = 0x16;

/// TLS acceptor whose certificate is bound to this node and rotated in place.
pub(super) struct IdentityTls {
    identity: Arc<IdentityKeyPair>,
    provider: Arc<CryptoProvider>,
    resolver: Arc<IdentityCertResolver>,
    acceptor: TlsAcceptor,
}

#[derive(Debug)]
struct IdentityCertResolver {
    current: ArcSwap<CurrentCertificate>,
}

#[derive(Debug)]
struct CurrentCertificate {
    key: Arc<CertifiedKey>,
    issued_at: u64,
}

impl ResolvesServerCert for IdentityCertResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(Arc::clone(&self.current.load().key))
    }
}

impl IdentityTls {
    /// Generates the first certificate and builds a TLS 1.3-only acceptor.
    pub(super) fn new(identity: Arc<IdentityKeyPair>, now: u64) -> Result<Self, String> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let key = certified_key(&provider, &identity, now)?;
        let resolver = Arc::new(IdentityCertResolver {
            current: ArcSwap::from_pointee(CurrentCertificate {
                key,
                issued_at: now,
            }),
        });
        let mut config = ServerConfig::builder_with_provider(Arc::clone(&provider))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|error| format!("TLS configuration: {error}"))?
            .with_no_client_auth()
            .with_cert_resolver(Arc::clone(&resolver) as Arc<dyn ResolvesServerCert>);
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(Self {
            identity,
            provider,
            resolver,
            acceptor: TlsAcceptor::from(Arc::new(config)),
        })
    }

    /// Replaces the certificate once it is older than [`CERT_ROTATE_AFTER_SECS`]
    /// (or the clock went backwards). Returns whether it rotated. On failure
    /// the current certificate stays in service.
    pub(super) fn rotate_if_due(&self, now: u64) -> Result<bool, String> {
        let issued_at = self.resolver.current.load().issued_at;
        if now >= issued_at && now - issued_at < CERT_ROTATE_AFTER_SECS {
            return Ok(false);
        }
        let key = certified_key(&self.provider, &self.identity, now)?;
        self.resolver.current.store(Arc::new(CurrentCertificate {
            key,
            issued_at: now,
        }));
        Ok(true)
    }

    /// DER of the certificate currently served.
    #[cfg(test)]
    pub(super) fn current_certificate_der(&self) -> Vec<u8> {
        self.resolver.current.load().key.cert[0].to_vec()
    }
}

fn certified_key(
    provider: &CryptoProvider,
    identity: &IdentityKeyPair,
    now: u64,
) -> Result<Arc<CertifiedKey>, String> {
    let generated = aeronyx_node_tls::generate_certificate(
        identity.public_key_bytes(),
        now,
        CERT_LIFETIME_SECS,
        |message| identity.sign(message),
    )
    .map_err(|error| error.to_string())?;
    let signing_key = provider
        .key_provider
        .load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
            generated.key_pkcs8_der,
        )))
        .map_err(|error| format!("TLS key: {error}"))?;
    Ok(Arc::new(CertifiedKey::new(
        vec![CertificateDer::from(generated.cert_der)],
        signing_key,
    )))
}

/// Serves `app` on `listener`, plain HTTP and identity-bound TLS alike, until
/// `shutdown` resolves; then stops accepting and drains open connections.
pub(super) async fn serve_http_and_identity_tls(
    listener: TcpListener,
    app: Router,
    tls: Arc<IdentityTls>,
    now: impl Fn() -> u64,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    // Dropping `signal_rx` tells every connection task to shut down; each
    // task holds a `close_rx` clone, so `close_tx.closed()` is the drain.
    let (signal_tx, signal_rx) = watch::channel(());
    let signal_tx = Arc::new(signal_tx);
    tokio::spawn(async move {
        shutdown.await;
        drop(signal_rx);
    });
    let (close_tx, close_rx) = watch::channel(());
    // The certificate was issued at construction; the first check is due one
    // interval from now, not immediately.
    let mut rotation = tokio::time::interval_at(
        tokio::time::Instant::now() + ROTATION_CHECK_INTERVAL,
        ROTATION_CHECK_INTERVAL,
    );
    rotation.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        let (stream, remote) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(accepted) => accepted,
                Err(error) => {
                    if !is_connection_error(&error) {
                        // Out of descriptors or similar: back off like axum.
                        warn!(%error, "[API] Public listener accept failed");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                    continue;
                }
            },
            _ = rotation.tick() => {
                match tls.rotate_if_due(now()) {
                    Ok(true) => info!("[API] Rotated identity-bound TLS certificate"),
                    Ok(false) => {}
                    Err(error) => warn!(%error, "[API] TLS certificate rotation failed; keeping current"),
                }
                continue;
            }
            _ = signal_tx.closed() => break,
        };
        let app = app.clone();
        let acceptor = tls.acceptor.clone();
        let signal_tx = Arc::clone(&signal_tx);
        let close_rx = close_rx.clone();
        tokio::spawn(async move {
            let accepted = tokio::select! {
                accepted = classify(stream, acceptor) => accepted,
                _ = signal_tx.closed() => None,
            };
            match accepted {
                Some(Accepted::Plain(stream)) => drive(stream, app, &signal_tx).await,
                Some(Accepted::Tls(stream)) => drive(*stream, app, &signal_tx).await,
                None => debug!(%remote, "[API] Public connection closed before a request"),
            }
            drop(close_rx);
        });
    }

    drop(close_rx);
    drop(listener);
    close_tx.closed().await;
    Ok(())
}

enum Accepted {
    Plain(TcpStream),
    Tls(Box<tokio_rustls::server::TlsStream<TcpStream>>),
}

async fn classify(stream: TcpStream, acceptor: TlsAcceptor) -> Option<Accepted> {
    let mut first = [0u8; 1];
    match tokio::time::timeout(FIRST_BYTE_TIMEOUT, stream.peek(&mut first)).await {
        Ok(Ok(read)) if read > 0 => {}
        _ => return None,
    }
    if first[0] != TLS_HANDSHAKE_CONTENT_TYPE {
        return Some(Accepted::Plain(stream));
    }
    match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
        Ok(Ok(stream)) => Some(Accepted::Tls(Box::new(stream))),
        _ => None,
    }
}

async fn drive<I>(io: I, app: Router, signal_tx: &watch::Sender<()>)
where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = hyper::service::service_fn(move |request: Request<Incoming>| {
        let mut app = app.clone();
        async move { app.call(request.map(Body::new)).await }
    });
    let connection = hyper::server::conn::http1::Builder::new()
        .serve_connection(TokioIo::new(io), service)
        .with_upgrades();
    tokio::pin!(connection);
    let signal_closed = signal_tx.closed();
    tokio::pin!(signal_closed);
    let mut draining = false;
    loop {
        tokio::select! {
            result = connection.as_mut() => {
                if let Err(error) = result {
                    debug!(%error, "[API] Public connection ended with an error");
                }
                break;
            }
            _ = &mut signal_closed, if !draining => {
                draining = true;
                connection.as_mut().graceful_shutdown();
            }
        }
    }
}

fn is_connection_error(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
    )
}

/// Address string for logs: `http(s)://addr`.
pub(super) fn describe(listen_addr: SocketAddr, tls: bool) -> String {
    if tls {
        format!("http://{listen_addr} + identity-bound https://{listen_addr}")
    } else {
        format!("http://{listen_addr}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{ServerName, UnixTime};
    use rustls::{ClientConfig, DigitallySignedStruct, SignatureScheme};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const NOW: u64 = 1_790_000_000;

    /// What a client does: accept the certificate only for the node it meant
    /// to reach, and check the TLS 1.3 handshake signature with its key.
    #[derive(Debug)]
    struct NodeVerifier {
        node_id: [u8; 32],
        now: u64,
        provider: Arc<CryptoProvider>,
    }

    impl ServerCertVerifier for NodeVerifier {
        fn verify_server_cert(
            &self,
            end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            aeronyx_node_tls::verify_certificate(end_entity, &self.node_id, self.now)
                .map(|()| ServerCertVerified::assertion())
                .map_err(|error| rustls::Error::General(error.to_string()))
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Err(rustls::Error::General("TLS 1.2 is not offered".into()))
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            rustls::crypto::verify_tls13_signature(
                message,
                cert,
                dss,
                &self.provider.signature_verification_algorithms,
            )
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            self.provider
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    fn identity(seed: u8) -> Arc<IdentityKeyPair> {
        Arc::new(IdentityKeyPair::from_bytes(&[seed; 32]).unwrap())
    }

    struct Running {
        addr: SocketAddr,
        tls: Arc<IdentityTls>,
        stop: tokio::sync::oneshot::Sender<()>,
        served: tokio::task::JoinHandle<std::io::Result<()>>,
    }

    async fn start(identity: Arc<IdentityKeyPair>) -> Running {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tls = Arc::new(IdentityTls::new(identity, NOW).unwrap());
        let app = Router::new().route("/ping", axum::routing::get(|| async { "pong" }));
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let served = tokio::spawn(serve_http_and_identity_tls(
            listener,
            app,
            Arc::clone(&tls),
            || NOW,
            async move {
                let _ = stopped.await;
            },
        ));
        // One request proves the accept loop is running.
        let ready = get_ping(TcpStream::connect(addr).await.unwrap()).await;
        assert!(ready.ends_with("pong"), "{ready}");
        Running {
            addr,
            tls,
            stop,
            served,
        }
    }

    async fn get_ping<S: AsyncRead + AsyncWrite + Unpin>(mut stream: S) -> String {
        stream
            .write_all(b"GET /ping HTTP/1.1\r\nHost: node\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    }

    async fn tls_connect(
        addr: SocketAddr,
        node_id: [u8; 32],
        now: u64,
    ) -> std::io::Result<tokio_rustls::client::TlsStream<TcpStream>> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let config = ClientConfig::builder_with_provider(Arc::clone(&provider))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NodeVerifier {
                node_id,
                now,
                provider,
            }))
            .with_no_client_auth();
        let stream = TcpStream::connect(addr).await?;
        tokio_rustls::TlsConnector::from(Arc::new(config))
            .connect(ServerName::IpAddress(addr.ip().into()), stream)
            .await
    }

    #[tokio::test]
    async fn plain_http_and_identity_tls_share_one_port() {
        let node = identity(0x21);
        let running = start(Arc::clone(&node)).await;

        let plain = get_ping(TcpStream::connect(running.addr).await.unwrap()).await;
        assert!(plain.starts_with("HTTP/1.1 200"), "{plain}");
        assert!(plain.ends_with("pong"), "{plain}");

        let tls = tls_connect(running.addr, node.public_key_bytes(), NOW)
            .await
            .unwrap();
        let secure = get_ping(tls).await;
        assert!(secure.starts_with("HTTP/1.1 200"), "{secure}");
        assert!(secure.ends_with("pong"), "{secure}");

        running.stop.send(()).unwrap();
        running.served.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn a_client_expecting_another_node_refuses_the_handshake() {
        let running = start(identity(0x22)).await;
        let other = identity(0x23).public_key_bytes();
        assert!(tls_connect(running.addr, other, NOW).await.is_err());
        running.stop.send(()).unwrap();
        running.served.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn rotation_replaces_the_key_and_the_new_certificate_still_verifies() {
        let node = identity(0x24);
        let running = start(Arc::clone(&node)).await;
        let first = running.tls.current_certificate_der();

        assert!(!running
            .tls
            .rotate_if_due(NOW + CERT_ROTATE_AFTER_SECS - 1)
            .unwrap());
        assert_eq!(running.tls.current_certificate_der(), first);

        let later = NOW + CERT_ROTATE_AFTER_SECS;
        assert!(running.tls.rotate_if_due(later).unwrap());
        let second = running.tls.current_certificate_der();
        assert_ne!(second, first);
        aeronyx_node_tls::verify_certificate(&second, &node.public_key_bytes(), later).unwrap();
        let secure = get_ping(
            tls_connect(running.addr, node.public_key_bytes(), later)
                .await
                .unwrap(),
        )
        .await;
        assert!(secure.ends_with("pong"), "{secure}");

        // A clock that jumps backwards also gets a fresh certificate rather
        // than one issued "in the future".
        assert!(running.tls.rotate_if_due(NOW).unwrap());

        running.stop.send(()).unwrap();
        running.served.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn shutdown_drains_an_open_keep_alive_connection() {
        let running = start(identity(0x25)).await;
        let mut idle = TcpStream::connect(running.addr).await.unwrap();
        idle.write_all(b"GET /ping HTTP/1.1\r\nHost: node\r\n\r\n")
            .await
            .unwrap();
        let mut buffer = [0u8; 256];
        let read = idle.read(&mut buffer).await.unwrap();
        assert!(String::from_utf8_lossy(&buffer[..read]).starts_with("HTTP/1.1 200"));

        running.stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), running.served)
            .await
            .expect("shutdown drained the keep-alive connection")
            .unwrap()
            .unwrap();
        assert_eq!(idle.read(&mut buffer).await.unwrap(), 0);
    }

    /// Resolves reserved peer names to their (loopback) IP for this test only;
    /// production uses `PublicOnlyResolver`, which refuses loopback.
    struct LoopbackPeerResolver;

    impl reqwest::dns::Resolve for LoopbackPeerResolver {
        fn resolve(&self, name: hyper_v014::client::connect::dns::Name) -> reqwest::dns::Resolving {
            let decoded = crate::api::peer_tls::decode_peer_tls_host(name.as_str());
            Box::pin(async move {
                let (_, ip) = decoded.ok_or("not a reserved peer name")?;
                Ok(Box::new(std::iter::once(SocketAddr::new(ip, 0))) as reqwest::dns::Addrs)
            })
        }
    }

    fn unix_now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    // [NODE-TLS-BINDING 2026-10-10 by Claude] The outbound peer stack end to
    // end: rewritten URL -> resolver -> rustls 0.21 verifier -> this listener.
    #[tokio::test]
    async fn the_peer_client_reaches_only_the_pinned_identity() {
        use crate::api::peer_tls::{peer_tls_client_config, peer_tls_host};

        let node = identity(0x31);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tls = Arc::new(IdentityTls::new(Arc::clone(&node), unix_now()).unwrap());
        let app = Router::new().route("/ping", axum::routing::get(|| async { "pong" }));
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let served = tokio::spawn(serve_http_and_identity_tls(
            listener,
            app,
            tls,
            unix_now,
            async move {
                let _ = stopped.await;
            },
        ));

        let client = reqwest::Client::builder()
            .no_proxy()
            .dns_resolver(Arc::new(LoopbackPeerResolver))
            .use_preconfigured_tls(peer_tls_client_config())
            .build()
            .unwrap();
        let url_for = |node_id: &[u8; 32]| {
            format!(
                "https://{}:{}/ping",
                peer_tls_host(node_id, addr.ip()),
                addr.port()
            )
        };

        let response = client
            .get(url_for(&node.public_key_bytes()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.text().await.unwrap(), "pong");

        let other = identity(0x32).public_key_bytes();
        assert!(client.get(url_for(&other)).send().await.is_err());

        // The production resolver refuses a reserved name that carries a
        // non-public address, before any connection is made.
        let production = crate::api::privacy_safe_peer_http_client_builder()
            .build()
            .unwrap();
        assert!(production
            .get(url_for(&node.public_key_bytes()))
            .send()
            .await
            .is_err());

        stop.send(()).unwrap();
        served.await.unwrap().unwrap();
    }
}
