// ============================================
// File: crates/aeronyx-server/src/api/peer_tls.rs
// ============================================
//! # Identity-pinned TLS for node-to-node requests
//!
//! [NODE-TLS-BINDING 2026-10-10 by Claude] A peer whose signed descriptor
//! advertises `NodeProtocolFeature::IdentityBoundTlsV1` serves TLS on its
//! public port with a certificate bound to its identity. Outbound peer
//! requests to such a peer must use that TLS, pinned to exactly that identity,
//! and must never fall back to plain HTTP.
//!
//! Peer request code is spread over many transports that only carry an
//! endpoint string, so the pin is resolved at the one place every outbound
//! peer URL is built ([`super::peer_transport_url`]):
//!
//! 1. [`PeerTlsDirectory`] maps `ip:port` to the node id of the one locally
//!    established peer that claims it and advertises identity-bound TLS. It is
//!    rebuilt from the peer store on every change to the peer map.
//! 2. A matching URL is rewritten to `https://` with a reserved host name that
//!    carries the node id and the IP (see [`peer_tls_host`]). The name never
//!    resolves through DNS: [`super::PublicOnlyResolver`] decodes it back to
//!    the IP, and [`PeerTlsVerifier`] decodes the node id and accepts only a
//!    certificate bound to it. The pin is therefore fixed when the URL is
//!    built and travels with the request.
//! 3. An endpoint claimed by more than one identity, where any claimant
//!    advertises TLS, yields no URL at all (fail closed). A malicious peer that
//!    copies a victim's endpoint can make it unreachable, never impersonate it.
//!
//! Endpoints of peers without the feature (older nodes) are left unchanged
//! and keep plain HTTP; DNS-named HTTPS endpoints keep CA verification.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::discovery::{NodeProtocolFeature, SignedNodeDescriptor};
use arc_swap::ArcSwap;
use rustls_v021 as r21;

/// Reserved suffix of the synthetic peer host names. `.invalid` can never be
/// delegated in public DNS (RFC 6761), so a leaked name resolves nowhere.
const PEER_TLS_HOST_SUFFIX: &str = ".peer.aeronyx.invalid";

/// Who may be reached at one `ip:port` over identity-bound TLS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerTlsClaim {
    /// Exactly one established identity claims the endpoint and serves
    /// identity-bound TLS: connect only over TLS pinned to it.
    Pinned([u8; 32]),
    /// Several identities claim the endpoint and at least one advertises TLS:
    /// refuse to build a URL.
    Ambiguous,
}

/// Endpoint-to-identity map for identity-bound peer TLS.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PeerTlsDirectory {
    claims: HashMap<SocketAddr, PeerTlsClaim>,
}

impl PeerTlsDirectory {
    /// Builds the directory from every locally established descriptor.
    ///
    /// Expired-but-retained descriptors count too: a node's signed statement
    /// that it serves TLS does not lapse into a plain-HTTP fallback.
    pub(crate) fn from_descriptors<'a>(
        descriptors: impl IntoIterator<Item = &'a SignedNodeDescriptor>,
    ) -> Self {
        let mut claimants: HashMap<SocketAddr, (Vec<[u8; 32]>, bool)> = HashMap::new();
        for descriptor in descriptors {
            let Some(addr) = descriptor
                .descriptor
                .public_endpoint
                .as_deref()
                .and_then(ip_literal_socket_addr)
            else {
                continue;
            };
            let entry = claimants.entry(addr).or_default();
            let node_id = descriptor.node_id();
            if !entry.0.contains(&node_id) {
                entry.0.push(node_id);
            }
            entry.1 |= descriptor
                .descriptor
                .advertises_protocol_feature(NodeProtocolFeature::IdentityBoundTlsV1);
        }
        let claims = claimants
            .into_iter()
            .filter(|(_, (_, any_tls))| *any_tls)
            .map(|(addr, (ids, _))| {
                let claim = match ids.as_slice() {
                    [only] => PeerTlsClaim::Pinned(*only),
                    _ => PeerTlsClaim::Ambiguous,
                };
                (addr, claim)
            })
            .collect();
        Self { claims }
    }

    /// The claim for one endpoint, if identity-bound TLS applies to it.
    pub(crate) fn claim(&self, addr: &SocketAddr) -> Option<PeerTlsClaim> {
        self.claims.get(addr).copied()
    }
}

static DIRECTORY: LazyLock<ArcSwap<PeerTlsDirectory>> =
    LazyLock::new(|| ArcSwap::from_pointee(PeerTlsDirectory::default()));

/// Replaces the process-wide directory. Only the production peer store calls
/// this, so tests that build their own stores never affect each other.
pub(crate) fn install_directory(directory: PeerTlsDirectory) {
    DIRECTORY.store(Arc::new(directory));
}

pub(crate) fn current_directory() -> Arc<PeerTlsDirectory> {
    DIRECTORY.load_full()
}

fn ip_literal_socket_addr(endpoint: &str) -> Option<SocketAddr> {
    let url = super::canonical_peer_http_url(endpoint, "/").ok()?;
    let port = url.port_or_known_default()?;
    let ip = match url.host()? {
        url::Host::Ipv4(ip) => IpAddr::V4(ip),
        url::Host::Ipv6(ip) => IpAddr::V6(ip),
        url::Host::Domain(_) => return None,
    };
    Some(SocketAddr::new(ip, port))
}

/// Applies `directory` to a canonical peer URL.
///
/// Returns the URL unchanged when identity-bound TLS does not apply, the
/// rewritten `https://<peer_tls_host>:<port><path>` when it does, and `None`
/// when the endpoint is ambiguous.
pub(crate) fn apply_directory(
    url: reqwest::Url,
    directory: &PeerTlsDirectory,
) -> Option<reqwest::Url> {
    let ip = match url.host() {
        Some(url::Host::Ipv4(ip)) => IpAddr::V4(ip),
        Some(url::Host::Ipv6(ip)) => IpAddr::V6(ip),
        _ => return Some(url),
    };
    let port = url.port_or_known_default()?;
    match directory.claim(&SocketAddr::new(ip, port)) {
        None => Some(url),
        Some(PeerTlsClaim::Ambiguous) => None,
        Some(PeerTlsClaim::Pinned(node_id)) => {
            let rewritten = format!(
                "https://{}:{port}{}",
                peer_tls_host(&node_id, ip),
                url.path()
            );
            reqwest::Url::parse(&rewritten).ok()
        }
    }
}

/// The reserved host name that carries a pinned node id and its IP:
/// `<node id hex, first half>.<second half>.<v4-a-b-c-d | v6-<32 hex>>.peer.aeronyx.invalid`.
/// (A 64-character hex id does not fit one 63-character DNS label.)
pub(crate) fn peer_tls_host(node_id: &[u8; 32], ip: IpAddr) -> String {
    let id = hex::encode(node_id);
    let ip_label = match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, d] = ip.octets();
            format!("v4-{a}-{b}-{c}-{d}")
        }
        IpAddr::V6(ip) => format!("v6-{}", hex::encode(ip.octets())),
    };
    format!(
        "{}.{}.{ip_label}{PEER_TLS_HOST_SUFFIX}",
        &id[..32],
        &id[32..]
    )
}

/// Inverse of [`peer_tls_host`]; `None` for any other name.
pub(crate) fn decode_peer_tls_host(name: &str) -> Option<([u8; 32], IpAddr)> {
    let rest = name.strip_suffix(PEER_TLS_HOST_SUFFIX)?;
    let mut labels = rest.split('.');
    let (Some(first), Some(second), Some(ip_label), None) =
        (labels.next(), labels.next(), labels.next(), labels.next())
    else {
        return None;
    };
    let lower_hex = |value: &str| {
        value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    if first.len() != 32 || second.len() != 32 || !lower_hex(first) || !lower_hex(second) {
        return None;
    }
    let mut node_id = [0u8; 32];
    hex::decode_to_slice(format!("{first}{second}"), &mut node_id).ok()?;
    let ip = if let Some(v4) = ip_label.strip_prefix("v4-") {
        let octets = v4
            .split('-')
            .map(|octet| {
                (!octet.is_empty()
                    && octet.bytes().all(|b| b.is_ascii_digit())
                    && (octet == "0" || !octet.starts_with('0')))
                .then(|| octet.parse::<u8>().ok())
                .flatten()
            })
            .collect::<Option<Vec<_>>>()?;
        let [a, b, c, d] = octets.as_slice() else {
            return None;
        };
        IpAddr::V4(Ipv4Addr::new(*a, *b, *c, *d))
    } else if let Some(v6) = ip_label.strip_prefix("v6-") {
        if v6.len() != 32 || !lower_hex(v6) {
            return None;
        }
        let mut octets = [0u8; 16];
        hex::decode_to_slice(v6, &mut octets).ok()?;
        IpAddr::V6(Ipv6Addr::from(octets))
    } else {
        return None;
    };
    Some((node_id, ip))
}

/// TLS verification for all outbound peer clients: identity binding for
/// reserved peer names, standard CA verification for every other name.
pub(crate) struct PeerTlsVerifier {
    webpki: r21::client::WebPkiVerifier,
}

impl r21::client::ServerCertVerifier for PeerTlsVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &r21::Certificate,
        intermediates: &[r21::Certificate],
        server_name: &r21::ServerName,
        scts: &mut dyn Iterator<Item = &[u8]>,
        ocsp_response: &[u8],
        now: SystemTime,
    ) -> Result<r21::client::ServerCertVerified, r21::Error> {
        if let r21::ServerName::DnsName(name) = server_name {
            if let Some((node_id, _)) = decode_peer_tls_host(name.as_ref()) {
                let now = now
                    .duration_since(UNIX_EPOCH)
                    .map_err(|_| r21::Error::General("clock before epoch".into()))?
                    .as_secs();
                return aeronyx_node_tls::verify_certificate(&end_entity.0, &node_id, now)
                    .map(|()| r21::client::ServerCertVerified::assertion())
                    .map_err(|error| r21::Error::General(error.to_string()));
            }
        }
        self.webpki.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            scts,
            ocsp_response,
            now,
        )
    }
}

/// The rustls configuration every outbound peer client uses. It keeps the
/// Mozilla roots and ALPN that reqwest would otherwise configure itself.
pub(crate) fn peer_tls_client_config() -> r21::ClientConfig {
    let mut roots = r21::RootCertStore::empty();
    roots.add_trust_anchors(webpki_roots::TLS_SERVER_ROOTS.iter().map(|anchor| {
        r21::OwnedTrustAnchor::from_subject_spki_name_constraints(
            anchor.subject,
            anchor.spki,
            anchor.name_constraints,
        )
    }));
    let verifier = PeerTlsVerifier {
        webpki: r21::client::WebPkiVerifier::new(roots, None),
    };
    let mut config = r21::ClientConfig::builder()
        .with_safe_defaults()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::NodeDescriptor;

    fn descriptor(seed: u8, endpoint: &str, tls: bool) -> SignedNodeDescriptor {
        let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            1_700_000_000,
            1_700_003_600,
            "0.1.0",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        if tls {
            descriptor =
                descriptor.with_protocol_features([NodeProtocolFeature::IdentityBoundTlsV1]);
        }
        SignedNodeDescriptor::sign(descriptor, &identity).unwrap()
    }

    fn url(value: &str) -> reqwest::Url {
        reqwest::Url::parse(value).unwrap()
    }

    #[test]
    fn host_names_round_trip_and_reject_lookalikes() {
        let id = [0xab; 32];
        for ip in ["34.51.97.227", "0.0.0.0", "2001:db8::1"] {
            let ip: IpAddr = ip.parse().unwrap();
            let host = peer_tls_host(&id, ip);
            assert!(host.split('.').all(|label| label.len() <= 63), "{host}");
            assert_eq!(decode_peer_tls_host(&host), Some((id, ip)));
        }
        let good = peer_tls_host(&id, "1.2.3.4".parse().unwrap());
        for bad in [
            good.replace(".peer.aeronyx.invalid", ".peer.aeronyx.network"),
            good.to_uppercase(),
            good.replace("v4-1-2-3-4", "v4-1-2-3"),
            good.replace("v4-1-2-3-4", "v4-01-2-3-4"),
            good.replace("v4-1-2-3-4", "v4-1-2-3-256"),
            good.replace("v4-1-2-3-4", "v5-1-2-3-4"),
            format!("x.{good}"),
            "example.com".to_string(),
        ] {
            assert_eq!(decode_peer_tls_host(&bad), None, "{bad}");
        }
    }

    #[test]
    fn only_a_unique_tls_claimant_is_pinned() {
        let a = descriptor(1, "http://34.51.97.227:8422", true);
        let b = descriptor(2, "http://34.32.67.213:8422", false);
        let directory = PeerTlsDirectory::from_descriptors([&a, &b]);

        let pinned = apply_directory(
            url("http://34.51.97.227:8422/api/discovery/gossip"),
            &directory,
        )
        .unwrap();
        assert_eq!(pinned.scheme(), "https");
        assert_eq!(pinned.port(), Some(8422));
        assert_eq!(pinned.path(), "/api/discovery/gossip");
        assert_eq!(
            decode_peer_tls_host(pinned.host_str().unwrap()),
            Some((a.node_id(), "34.51.97.227".parse().unwrap()))
        );

        // A peer without the feature keeps its URL (plain HTTP).
        let plain = url("http://34.32.67.213:8422/api/discovery/gossip");
        assert_eq!(apply_directory(plain.clone(), &directory), Some(plain));
        // DNS names are never rewritten.
        let named = url("https://node.example/api/discovery/gossip");
        assert_eq!(apply_directory(named.clone(), &directory), Some(named));
        // Another port on the same IP is a different endpoint.
        let other_port = url("http://34.51.97.227:9000/x");
        assert_eq!(
            apply_directory(other_port.clone(), &directory),
            Some(other_port)
        );
    }

    #[test]
    fn a_copied_endpoint_fails_closed_instead_of_pinning_the_copier() {
        let victim = descriptor(3, "http://34.51.97.227:8422", false);
        let copier = descriptor(4, "http://34.51.97.227:8422", true);
        let directory = PeerTlsDirectory::from_descriptors([&victim, &copier]);
        assert_eq!(
            directory.claim(&"34.51.97.227:8422".parse().unwrap()),
            Some(PeerTlsClaim::Ambiguous)
        );
        assert_eq!(
            apply_directory(url("http://34.51.97.227:8422/x"), &directory),
            None
        );
    }

    #[test]
    fn the_same_identity_re_announcing_stays_pinned() {
        let first = descriptor(5, "http://34.51.97.227:8422", true);
        let again = descriptor(5, "34.51.97.227:8422", true);
        let directory = PeerTlsDirectory::from_descriptors([&first, &again]);
        assert_eq!(
            directory.claim(&"34.51.97.227:8422".parse().unwrap()),
            Some(PeerTlsClaim::Pinned(first.node_id()))
        );
    }
}
