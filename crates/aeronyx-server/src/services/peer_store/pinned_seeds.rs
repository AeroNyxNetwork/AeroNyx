// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store/pinned_seeds.rs
// ============================================================================
//! Operator-pinned discovery seeds: `<node id>@<public ip>:<port>`.
//!
//! [PINNED-SEEDS 2026-10-11 by Claude] In untrusted-candidate mode a node
//! trusts only identities it already holds. A node with no operator snapshot
//! (the Phala TEE node) therefore trusts only itself and never becomes ready.
//! A pinned seed is the operator's statement "the node at this address is this
//! identity". Its rules, all enforced at the PeerStore write boundary:
//!
//! * The set of pinned ids `P` is fixed when the store is built and never
//!   changes. A pinned identity enters the peer map for the first time only
//!   with a [`PinnedTlsProof`], which only [`fetch_pinned_snapshot`] can mint,
//!   after an exchange over identity-bound TLS pinned to exactly that id.
//! * Every descriptor of a pinned identity, first or refresh, from any source,
//!   must keep the pinned address and advertise identity-bound TLS.
//! * Trust does not travel: other descriptors in a seed's answer stay
//!   untrusted candidates.
//! * Pinned identities are never written to or read from the peer cache. The
//!   quarantine set `Q` (every id ever pinned) is persisted before any import,
//!   so an identity whose pin is later removed cannot come back from an old
//!   cache or cache backup. `Q` only blocks the cache; an operator snapshot is
//!   a separate, explicit authorization and is not overridden by it.
//!
//! Scope of the `Q` guarantee: a new binary that keeps the `Q` file does not
//! restore quarantined ids from a cache. A lost `Q` file, a restored old
//! backup, or a rollback to a binary that does not read `Q` is outside it.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use aeronyx_core::protocol::discovery::{
    NodeBootstrapSnapshot, NodeDiscoveryMessage, NodeProtocolFeature, SignedNodeDescriptor,
};
use serde::{Deserialize, Serialize};

use super::{PeerStore, PeerStoreError, PeerStoreImportReport};
use crate::api::peer_tls::{self, PeerTlsClaim};

/// Maximum pinned seeds one config may list.
pub const MAX_PINNED_SEEDS: usize = 16;
/// Maximum ids the persisted quarantine may hold before startup refuses.
pub const MAX_PIN_QUARANTINE_IDS: usize = 1024;
/// Upper bound on the quarantine file read before parsing.
pub const PIN_QUARANTINE_MAX_BYTES: u64 = 128 * 1024;
const PIN_QUARANTINE_VERSION: u32 = 1;

/// One operator pin: the identity expected at one public address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedSeed {
    /// The identity expected at `addr`.
    pub node_id: [u8; 32],
    /// The public IP literal and port of that identity.
    pub addr: SocketAddr,
}

/// Parses one `<64 lowercase hex>@<ipv4>:<port>` or `...@[<ipv6>]:<port>`
/// entry. Only public unicast addresses are accepted; tests that need a
/// loopback seed build [`PinnedSeed`] directly, never through this parser.
pub fn parse_pinned_seed(entry: &str) -> Result<PinnedSeed, &'static str> {
    let (id_hex, addr) = entry
        .split_once('@')
        .ok_or("expected <node id hex>@<ip>:<port>")?;
    if id_hex.len() != 64
        || !id_hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("node id must be 64 lowercase hex characters");
    }
    let mut node_id = [0u8; 32];
    hex::decode_to_slice(id_hex, &mut node_id).map_err(|_| "node id is not hex")?;
    if addr.contains('%') {
        return Err("IPv6 zone ids are not allowed");
    }
    let addr = SocketAddr::from_str(addr)
        .map_err(|_| "address must be an IP literal with a port (IPv6 in brackets)")?;
    if addr.port() == 0 {
        return Err("port must be 1-65535");
    }
    if let SocketAddr::V6(v6) = addr {
        if v6.scope_id() != 0 || v6.flowinfo() != 0 {
            return Err("IPv6 scope and flow info are not allowed");
        }
    }
    if !crate::api::peer_endpoint_is_permitted(&format!("http://{addr}")) {
        return Err("address must be a public unicast IP");
    }
    Ok(PinnedSeed { node_id, addr })
}

/// Parses and cross-checks every configured pin.
pub fn parse_pinned_seeds(
    entries: &[String],
    self_node_id: Option<&[u8; 32]>,
) -> Result<Vec<PinnedSeed>, String> {
    if entries.len() > MAX_PINNED_SEEDS {
        return Err(format!("at most {MAX_PINNED_SEEDS} pinned seeds"));
    }
    let mut pins: Vec<PinnedSeed> = Vec::with_capacity(entries.len());
    for entry in entries {
        let pin = parse_pinned_seed(entry.trim()).map_err(|reason| format!("{entry}: {reason}"))?;
        if self_node_id == Some(&pin.node_id) {
            return Err(format!("{entry}: must not pin this node's own identity"));
        }
        if pins.iter().any(|other| other.node_id == pin.node_id) {
            return Err(format!("{entry}: node id pinned more than once"));
        }
        if pins.iter().any(|other| other.addr == pin.addr) {
            return Err(format!("{entry}: address pinned to more than one node id"));
        }
        pins.push(pin);
    }
    Ok(pins)
}

/// The fixed pin policy of one PeerStore: pins `P` and quarantine `Q ⊇ P`.
#[derive(Debug, Default)]
pub struct PinPolicy {
    pins: Vec<PinnedSeed>,
    quarantine: HashSet<[u8; 32]>,
}

impl PinPolicy {
    /// No pins and no quarantine: exactly the behavior before pinned seeds.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Builds a policy; the quarantine always contains every pin.
    #[must_use]
    pub fn new(pins: Vec<PinnedSeed>, quarantine: HashSet<[u8; 32]>) -> Self {
        let mut quarantine = quarantine;
        quarantine.extend(pins.iter().map(|pin| pin.node_id));
        Self { pins, quarantine }
    }

    /// The configured pins `P`.
    #[must_use]
    pub fn pins(&self) -> &[PinnedSeed] {
        &self.pins
    }

    /// The pin of one identity, when it is pinned.
    #[must_use]
    pub fn pin_for(&self, node_id: &[u8; 32]) -> Option<&PinnedSeed> {
        self.pins.iter().find(|pin| &pin.node_id == node_id)
    }

    /// Whether the identity is in `Q` (every id ever pinned, including `P`).
    #[must_use]
    pub fn is_quarantined(&self, node_id: &[u8; 32]) -> bool {
        self.quarantine.contains(node_id)
    }
}

/// Where a descriptor came from, as far as pin rules care. Diagnostic source
/// labels stay strings; only this type takes part in admission decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerOrigin {
    /// Any path not listed below (self, bootstrap, gossip, refresh, ...).
    Other,
    CacheValid,
    CacheExpired,
    Promotion,
    /// Only together with a [`PinnedTlsProof`].
    PinnedSeed,
}

/// Why a descriptor was refused by a pin rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PinRejection {
    RequiresTls,
    EndpointMismatch,
    MissingTls,
    FromCache,
    Quarantined,
    Promotion,
}

impl PinRejection {
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::RequiresTls => "pin_requires_tls",
            Self::EndpointMismatch => "pin_endpoint_mismatch",
            Self::MissingTls => "pin_missing_tls",
            Self::FromCache => "pin_from_cache",
            Self::Quarantined => "pin_quarantined",
            Self::Promotion => "pin_promotion",
        }
    }
}

/// Evidence that one answer came over identity-bound TLS pinned to one id.
///
/// The fields are private and the only constructor is private to this module,
/// so nothing outside [`fetch_pinned_snapshot`] can create one (the
/// `cfg(test)` helper aside). It is consumed by the one import of that answer
/// and then dropped; it is never stored.
#[derive(Debug)]
pub struct PinnedTlsProof {
    node_id: [u8; 32],
    addr: SocketAddr,
}

impl PinnedTlsProof {
    fn minted_after_pinned_exchange(pin: &PinnedSeed) -> Self {
        Self {
            node_id: pin.node_id,
            addr: pin.addr,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(pin: &PinnedSeed) -> Self {
        Self::minted_after_pinned_exchange(pin)
    }
}

fn descriptor_endpoint_addr(descriptor: &SignedNodeDescriptor) -> Option<SocketAddr> {
    descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .and_then(peer_tls::ip_literal_socket_addr)
}

impl PeerStore {
    /// The pin rules, checked inside the `peers` write lock right before an
    /// insert (W1 `upsert_verified_with_origin`, W2 the expired-cache insert).
    pub(super) fn pin_admission(
        &self,
        peers: &std::collections::HashMap<[u8; 32], SignedNodeDescriptor>,
        descriptor: &SignedNodeDescriptor,
        origin: PeerOrigin,
        proof: Option<&PinnedTlsProof>,
    ) -> Result<(), PinRejection> {
        let node_id = descriptor.node_id();
        let policy = &self.pin_policy;
        let from_cache = matches!(origin, PeerOrigin::CacheValid | PeerOrigin::CacheExpired);
        let Some(pin) = policy.pin_for(&node_id) else {
            if from_cache && policy.is_quarantined(&node_id) {
                return Err(PinRejection::Quarantined);
            }
            return Ok(());
        };
        if from_cache {
            return Err(PinRejection::FromCache);
        }
        if origin == PeerOrigin::Promotion {
            return Err(PinRejection::Promotion);
        }
        if !peers.contains_key(&node_id) {
            let proven = origin == PeerOrigin::PinnedSeed
                && proof.is_some_and(|proof| proof.node_id == node_id && proof.addr == pin.addr);
            if !proven {
                return Err(PinRejection::RequiresTls);
            }
        }
        if descriptor_endpoint_addr(descriptor) != Some(pin.addr) {
            return Err(PinRejection::EndpointMismatch);
        }
        if !descriptor
            .descriptor
            .advertises_protocol_feature(NodeProtocolFeature::IdentityBoundTlsV1)
        {
            return Err(PinRejection::MissingTls);
        }
        Ok(())
    }

    /// Imports one authenticated pinned-seed answer.
    ///
    /// Only descriptors of the proven identity may use the proof; when the
    /// answer carries more than one distinct descriptor for it, none is used.
    /// Every other descriptor goes through the ordinary untrusted path.
    pub fn apply_pinned_seed_response(
        &self,
        proof: PinnedTlsProof,
        message: &NodeDiscoveryMessage,
        now: u64,
    ) -> PeerStoreImportReport {
        let descriptors: Vec<SignedNodeDescriptor> = match message {
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => vec![descriptor.clone()],
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => snapshot.peers.clone(),
            _ => return PeerStoreImportReport::empty(),
        };
        let (own, others): (Vec<_>, Vec<_>) = descriptors
            .into_iter()
            .partition(|descriptor| descriptor.node_id() == proof.node_id);
        let mut distinct: Vec<SignedNodeDescriptor> = Vec::new();
        for descriptor in own {
            if !distinct.contains(&descriptor) {
                distinct.push(descriptor);
            }
        }
        let mut report = PeerStoreImportReport::empty();
        match distinct.len() {
            0 => {}
            1 => {
                report.total = 1;
                let descriptor = distinct.pop().expect("one descriptor");
                match self.upsert_verified_with_origin(
                    descriptor,
                    now,
                    "pinned_seed",
                    PeerOrigin::PinnedSeed,
                    Some(&proof),
                ) {
                    Ok(true) => report.inserted = 1,
                    Ok(false) => report.unchanged = 1,
                    Err(PeerStoreError::StaleSequence { .. }) => report.stale = 1,
                    Err(_) => report.rejected = 1,
                }
            }
            n => {
                report.total = n;
                report.rejected = n;
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    "pinned_seed",
                    &proof.node_id,
                    None,
                    Some("pin_conflict"),
                );
            }
        }
        drop(proof);
        self.record_import_report(&report, now);
        if !others.is_empty() {
            let rest = NodeDiscoveryMessage::SnapshotResponse {
                snapshot: NodeBootstrapSnapshot::new(now, others),
            };
            self.apply_discovery_message(&rest, now);
        }
        report
    }
}

/// Why one pinned exchange produced no proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinnedFetchError {
    /// The directory does not hold exactly this pin for the address.
    NotPinned,
    /// The rewritten URL does not carry exactly this pin.
    UrlMismatch,
    /// The request did not complete (connect, TLS pin, timeout).
    Transport,
    /// The seed answered with an error status.
    Status,
    /// The answer was not a bounded gossip response.
    Decode,
}

impl PinnedFetchError {
    /// Privacy-safe reason bucket for audit events.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::NotPinned => "pin_ambiguous",
            Self::UrlMismatch => "pin_url_mismatch",
            Self::Transport => "pin_transport",
            Self::Status => "pin_status",
            Self::Decode => "pin_decode",
        }
    }
}

/// The pinned gossip URL, built from ONE directory snapshot: the claim check
/// and the rewrite use the same `directory`.
pub(crate) fn pinned_gossip_url(
    pin: &PinnedSeed,
    directory: &peer_tls::PeerTlsDirectory,
) -> Result<reqwest::Url, PinnedFetchError> {
    if directory.claim(&pin.addr) != Some(PeerTlsClaim::Pinned(pin.node_id)) {
        return Err(PinnedFetchError::NotPinned);
    }
    let base = crate::api::canonical_peer_http_url(
        &format!("http://{}", pin.addr),
        "/api/discovery/gossip",
    )
    .map_err(|_| PinnedFetchError::UrlMismatch)?;
    let url = peer_tls::apply_directory(base, directory).ok_or(PinnedFetchError::NotPinned)?;
    let pinned = url.scheme() == "https"
        && url
            .host_str()
            .and_then(peer_tls::decode_peer_tls_host)
            .is_some_and(|(node_id, ip)| node_id == pin.node_id && ip == pin.addr.ip())
        && url.port_or_known_default() == Some(pin.addr.port());
    if !pinned {
        return Err(PinnedFetchError::UrlMismatch);
    }
    Ok(url)
}

/// The only transport a pinned exchange can run on.
///
/// [PINNED-SEEDS 2026-10-11 by Claude] (review of 6515223) The proof means
/// "this answer came over TLS pinned to this id", so the client may not be the
/// caller's choice. The inner client is private and the one production
/// constructor uses `privacy_safe_peer_http_client_builder`: identity-bound
/// TLS verifier, no redirects, no proxy, public-only resolution.
pub struct PinnedSeedTransport {
    client: reqwest::Client,
}

impl PinnedSeedTransport {
    /// The production transport, with one request timeout.
    pub fn new(timeout: std::time::Duration) -> reqwest::Result<Self> {
        Ok(Self {
            client: crate::api::privacy_safe_peer_http_client_builder()
                .timeout(timeout)
                .build()?,
        })
    }

    /// The same transport, except that synthetic peer names may resolve to
    /// loopback, so tests can run a real identity-bound TLS listener.
    #[cfg(test)]
    pub(crate) fn loopback_for_test(timeout: std::time::Duration) -> reqwest::Result<Self> {
        Ok(Self {
            client: crate::api::privacy_safe_peer_http_client_builder()
                .dns_resolver(std::sync::Arc::new(LoopbackPeerResolver))
                .timeout(timeout)
                .build()?,
        })
    }
}

#[cfg(test)]
struct LoopbackPeerResolver;

#[cfg(test)]
impl reqwest::dns::Resolve for LoopbackPeerResolver {
    fn resolve(&self, name: hyper_v014::client::connect::dns::Name) -> reqwest::dns::Resolving {
        let decoded = peer_tls::decode_peer_tls_host(name.as_str());
        Box::pin(async move {
            match decoded {
                Some((_, ip)) if ip.is_loopback() => {
                    Ok(Box::new(std::iter::once(SocketAddr::new(ip, 0))) as reqwest::dns::Addrs)
                }
                _ => Err("test resolver serves loopback peer names only".into()),
            }
        })
    }
}

async fn pinned_post(
    transport: &PinnedSeedTransport,
    url: reqwest::Url,
    message: &NodeDiscoveryMessage,
) -> Result<reqwest::Response, PinnedFetchError> {
    let response = transport
        .client
        .post(url)
        .json(message)
        .send()
        .await
        .map_err(|_| PinnedFetchError::Transport)?;
    // Only 2xx: `error_for_status` lets a 3xx through, and redirects are
    // never followed.
    if !response.status().is_success() {
        return Err(PinnedFetchError::Status);
    }
    Ok(response)
}

/// Exchanges descriptors with one pinned seed and mints the proof only after
/// a successful answer over the pinned TLS URL.
pub async fn fetch_pinned_snapshot(
    transport: &PinnedSeedTransport,
    pin: &PinnedSeed,
    self_descriptor: SignedNodeDescriptor,
    now: u64,
    limit: u16,
    max_response_bytes: usize,
) -> Result<(PinnedTlsProof, Option<NodeDiscoveryMessage>), PinnedFetchError> {
    let directory = peer_tls::current_directory();
    fetch_pinned_snapshot_with_directory(
        transport,
        pin,
        &directory,
        self_descriptor,
        now,
        limit,
        max_response_bytes,
    )
    .await
}

/// The exchange against one given directory snapshot: the claim check and
/// the URL rewrite both use `directory`.
pub(crate) async fn fetch_pinned_snapshot_with_directory(
    transport: &PinnedSeedTransport,
    pin: &PinnedSeed,
    directory: &peer_tls::PeerTlsDirectory,
    self_descriptor: SignedNodeDescriptor,
    now: u64,
    limit: u16,
    max_response_bytes: usize,
) -> Result<(PinnedTlsProof, Option<NodeDiscoveryMessage>), PinnedFetchError> {
    let url = pinned_gossip_url(pin, directory)?;
    pinned_post(
        transport,
        url.clone(),
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: self_descriptor,
        },
    )
    .await?;
    let response = pinned_post(
        transport,
        url,
        &NodeDiscoveryMessage::SnapshotRequest {
            requested_at: now,
            limit: Some(limit),
        },
    )
    .await?;
    let answer = crate::api::decode_bounded_json_response::<crate::api::discovery::GossipResponse>(
        response,
        max_response_bytes,
    )
    .await
    .map_err(|_| PinnedFetchError::Decode)?;
    Ok((
        PinnedTlsProof::minted_after_pinned_exchange(pin),
        answer.response,
    ))
}

// ── persisted quarantine Q ──────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuarantineFile {
    v: u32,
    ids: Vec<String>,
}

/// `<peer_cache_path>.pin-quarantine.json`.
#[must_use]
pub fn quarantine_path(peer_cache_path: &str) -> PathBuf {
    PathBuf::from(format!("{peer_cache_path}.pin-quarantine.json"))
}

/// The file operations the quarantine needs, so tests can fail each step.
pub trait QuarantineIo {
    /// Opens (creating) and exclusively locks `path` for the whole update.
    fn lock(&self, path: &Path) -> io::Result<File>;
    /// Opens an existing file for a durability flush.
    fn open_existing(&self, path: &Path) -> io::Result<File>;
    /// Reads at most `max + 1` bytes; `NotFound` when the file is absent.
    fn read_bounded(&self, path: &Path, max: u64) -> io::Result<Vec<u8>>;
    /// Creates `path` (which must not exist) holding `bytes`.
    fn write_new(&self, path: &Path, bytes: &[u8]) -> io::Result<File>;
    /// Flushes one file to stable storage.
    fn sync_file(&self, file: &File) -> io::Result<()>;
    /// Atomically replaces `to` with `from`.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    /// Flushes a directory entry change to stable storage.
    fn sync_dir(&self, dir: &Path) -> io::Result<()>;
}

/// The real filesystem.
pub struct FsQuarantineIo;

impl QuarantineIo for FsQuarantineIo {
    fn lock(&self, path: &Path) -> io::Result<File> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        file.lock()?;
        Ok(file)
    }

    fn open_existing(&self, path: &Path) -> io::Result<File> {
        File::open(path)
    }

    fn read_bounded(&self, path: &Path, max: u64) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        File::open(path)?.take(max + 1).read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    fn write_new(&self, path: &Path, bytes: &[u8]) -> io::Result<File> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
        Ok(file)
    }

    fn sync_file(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn sync_dir(&self, dir: &Path) -> io::Result<()> {
        File::open(dir)?.sync_all()
    }
}

fn parse_quarantine(bytes: &[u8]) -> Result<HashSet<[u8; 32]>, String> {
    if bytes.len() as u64 > PIN_QUARANTINE_MAX_BYTES {
        return Err("pin quarantine file is too large".to_string());
    }
    let file: QuarantineFile = serde_json::from_slice(bytes)
        .map_err(|error| format!("pin quarantine file is malformed: {error}"))?;
    if file.v != PIN_QUARANTINE_VERSION {
        return Err(format!(
            "pin quarantine version {} is not supported",
            file.v
        ));
    }
    if file.ids.len() > MAX_PIN_QUARANTINE_IDS {
        return Err("pin quarantine holds too many ids".to_string());
    }
    let mut ids = HashSet::with_capacity(file.ids.len());
    for id in &file.ids {
        let mut node_id = [0u8; 32];
        if id.len() != 64
            || !id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || hex::decode_to_slice(id, &mut node_id).is_err()
        {
            return Err("pin quarantine holds a malformed id".to_string());
        }
        ids.insert(node_id);
    }
    Ok(ids)
}

fn quarantine_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Flushes an existing `Q` file and its directory entry. Run on every start
/// that finds `Q`, changed or not: a previous start may have renamed it into
/// place and then failed its directory sync (review of 6515223).
fn make_quarantine_durable(io: &dyn QuarantineIo, path: &Path) -> Result<(), String> {
    let file = io
        .open_existing(path)
        .map_err(|error| format!("pin quarantine reopen failed: {error}"))?;
    io.sync_file(&file)
        .map_err(|error| format!("pin quarantine sync failed: {error}"))?;
    io.sync_dir(quarantine_dir(path))
        .map_err(|error| format!("pin quarantine directory sync failed: {error}"))
}

/// Loads `Q`, adds the current pins and persists the union before returning.
///
/// Only an absent file reads as empty; every other failure (permission, I/O,
/// size, format, version, capacity, any step of the atomic write or of the
/// durability flush) is an error and startup must refuse. With pins, the
/// exclusive lock spans read-modify-write, so two processes starting together
/// cannot lose each other's ids. Without pins nothing is created: an absent
/// `Q` (even in an absent directory) is simply empty, as before pinned seeds.
pub fn load_and_extend_quarantine(
    io: &dyn QuarantineIo,
    path: &Path,
    pins: &[PinnedSeed],
) -> Result<HashSet<[u8; 32]>, String> {
    if pins.is_empty() {
        // Read-only: `Q` is replaced by rename, so a reader sees one whole
        // version without the lock.
        return match io.read_bounded(path, PIN_QUARANTINE_MAX_BYTES) {
            Ok(bytes) => {
                let ids = parse_quarantine(&bytes)?;
                make_quarantine_durable(io, path)?;
                Ok(ids)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(HashSet::new()),
            Err(error) => Err(format!("pin quarantine read failed: {error}")),
        };
    }
    let lock_path = PathBuf::from(format!("{}.lock", path.display()));
    let _lock = io
        .lock(&lock_path)
        .map_err(|error| format!("pin quarantine lock failed: {error}"))?;
    let (mut ids, existed) = match io.read_bounded(path, PIN_QUARANTINE_MAX_BYTES) {
        Ok(bytes) => (parse_quarantine(&bytes)?, true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (HashSet::new(), false),
        Err(error) => return Err(format!("pin quarantine read failed: {error}")),
    };
    let before = ids.len();
    ids.extend(pins.iter().map(|pin| pin.node_id));
    if ids.len() > MAX_PIN_QUARANTINE_IDS {
        return Err("pin quarantine would exceed its capacity".to_string());
    }
    if ids.len() == before && existed {
        make_quarantine_durable(io, path)?;
        return Ok(ids);
    }
    let mut sorted: Vec<String> = ids.iter().map(hex::encode).collect();
    sorted.sort();
    let bytes = serde_json::to_vec(&QuarantineFile {
        v: PIN_QUARANTINE_VERSION,
        ids: sorted,
    })
    .map_err(|error| format!("pin quarantine encode failed: {error}"))?;
    let tmp = PathBuf::from(format!("{}.tmp.{}", path.display(), std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let written = io
        .write_new(&tmp, &bytes)
        .and_then(|file| io.sync_file(&file))
        .and_then(|()| io.rename(&tmp, path));
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("pin quarantine write failed: {error}"));
    }
    io.sync_dir(quarantine_dir(path))
        .map_err(|error| format!("pin quarantine directory sync failed: {error}"))?;
    Ok(ids)
}

// ── deterministic interleaving seam (tests only) ────────────────────────────

#[cfg(test)]
type PausePoint = (
    [u8; 32],
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
static PAUSE_INSIDE_WRITE_LOCK: parking_lot::Mutex<Option<PausePoint>> =
    parking_lot::Mutex::new(None);

/// Arms a one-shot pause inside W1's `peers` write lock for one identity:
/// the writer arrives at `entered`, then waits for `release`. Tests run in
/// parallel, so the identity must be one no other test writes.
#[cfg(test)]
pub(crate) fn arm_pause_inside_write_lock(
    node_id: [u8; 32],
    entered: std::sync::Arc<std::sync::Barrier>,
    release: std::sync::Arc<std::sync::Barrier>,
) {
    *PAUSE_INSIDE_WRITE_LOCK.lock() = Some((node_id, entered, release));
}

#[cfg(test)]
pub(crate) fn test_pause_point(node_id: &[u8; 32]) {
    let armed = {
        let mut slot = PAUSE_INSIDE_WRITE_LOCK.lock();
        if slot.as_ref().is_some_and(|(id, _, _)| id == node_id) {
            slot.take()
        } else {
            None
        }
    };
    if let Some((_, entered, release)) = armed {
        entered.wait();
        release.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::NodeDescriptor;
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;

    const NOW: u64 = 1_700_000_100;
    const PIN_ENDPOINT: &str = "http://127.0.0.1:18422";

    fn identity(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).unwrap()
    }

    fn descriptor(
        kp: &IdentityKeyPair,
        sequence: u64,
        endpoint: &str,
        tls: bool,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        let mut d = NodeDescriptor::new(
            kp.public_key_bytes(),
            sequence,
            expires_at.saturating_sub(3_600).min(1_700_000_000),
            expires_at,
            "0.1.0",
        );
        d.public_endpoint = Some(endpoint.to_string());
        if tls {
            d = d.with_protocol_features([NodeProtocolFeature::IdentityBoundTlsV1]);
        }
        SignedNodeDescriptor::sign(d, kp).unwrap()
    }

    fn live(
        kp: &IdentityKeyPair,
        sequence: u64,
        endpoint: &str,
        tls: bool,
    ) -> SignedNodeDescriptor {
        descriptor(kp, sequence, endpoint, tls, 1_700_003_600)
    }

    fn pin_of(kp: &IdentityKeyPair) -> PinnedSeed {
        PinnedSeed {
            node_id: kp.public_key_bytes(),
            addr: "127.0.0.1:18422".parse().unwrap(),
        }
    }

    fn pinned_store(pin: PinnedSeed, extra_quarantine: &[[u8; 32]]) -> PeerStore {
        PeerStore::new_with_pin_policy(PinPolicy::new(
            vec![pin],
            extra_quarantine.iter().copied().collect(),
        ))
    }

    fn has(store: &PeerStore, kp: &IdentityKeyPair) -> bool {
        store.peers.read().contains_key(&kp.public_key_bytes())
    }

    fn pinned(
        store: &PeerStore,
        d: SignedNodeDescriptor,
        proof: &PinnedTlsProof,
    ) -> Result<bool, PeerStoreError> {
        store.upsert_verified_with_origin(
            d,
            NOW,
            "pinned_seed",
            PeerOrigin::PinnedSeed,
            Some(proof),
        )
    }

    // ── parsing ────────────────────────────────────────────────────────────

    #[test]
    fn parser_accepts_public_literals_and_normalizes_ipv6() {
        let id = "ab".repeat(32);
        let v4 = parse_pinned_seed(&format!("{id}@8.213.146.244:8422")).unwrap();
        assert_eq!(v4.addr, "8.213.146.244:8422".parse().unwrap());
        assert_eq!(v4.node_id, [0xab; 32]);
        let upper = parse_pinned_seed(&format!("{id}@[2606:4700:0:0::1111]:8422")).unwrap();
        let lower = parse_pinned_seed(&format!("{id}@[2606:4700::1111]:8422")).unwrap();
        assert_eq!(
            upper.addr, lower.addr,
            "IPv6 spellings normalize to one address"
        );
    }

    #[test]
    fn parser_rejects_every_non_public_or_malformed_entry() {
        let id = "ab".repeat(32);
        for bad in [
            format!("{}@8.8.8.8:8422", "AB".repeat(32)),
            format!("{}@8.8.8.8:8422", "ab".repeat(31)),
            format!("{id}@example.com:8422"),
            format!("{id}@8.8.8.8:0"),
            format!("{id}@8.8.8.8"),
            format!("{id}@2606:4700::1111:8422"),
            format!("{id}@[fe80::1%eth0]:8422"),
            format!("{id}@127.0.0.1:8422"),
            format!("{id}@10.0.0.1:8422"),
            format!("{id}@100.64.0.1:8422"),
            format!("{id}@192.0.2.1:8422"),
            format!("{id}@[::1]:8422"),
            "8.8.8.8:8422".to_string(),
        ] {
            assert!(parse_pinned_seed(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn config_cross_checks_duplicates_own_id_and_capacity() {
        let a = format!("{}@8.8.8.8:8422", "aa".repeat(32));
        let b_same_addr = format!("{}@8.8.8.8:8422", "bb".repeat(32));
        let a_other_addr = format!("{}@8.8.4.4:8422", "aa".repeat(32));
        assert!(parse_pinned_seeds(&[a.clone(), b_same_addr], None).is_err());
        assert!(parse_pinned_seeds(&[a.clone(), a_other_addr], None).is_err());
        assert!(parse_pinned_seeds(&[a.clone()], Some(&[0xaa; 32])).is_err());
        let many: Vec<String> = (0..=MAX_PINNED_SEEDS)
            .map(|i| format!("{}@8.8.{}.8:8422", format!("{:02x}", i + 1).repeat(32), i))
            .collect();
        assert!(parse_pinned_seeds(&many, None).is_err());
        assert_eq!(parse_pinned_seeds(&[a], None).unwrap().len(), 1);
    }

    // ── W1 rules ───────────────────────────────────────────────────────────

    #[test]
    fn a_pinned_identity_is_never_first_established_without_tls_proof() {
        let x = identity(1);
        let store = pinned_store(pin_of(&x), &[]);
        let d = live(&x, 1, PIN_ENDPOINT, true);
        // Bootstrap, gossip, refresh, ... (Origin::Other), no proof.
        assert!(store
            .upsert_verified_from_source(d.clone(), NOW, "bootstrap")
            .is_err());
        // The origin label alone is not authority.
        assert!(store
            .upsert_verified_with_origin(
                d.clone(),
                NOW,
                "pinned_seed",
                PeerOrigin::PinnedSeed,
                None
            )
            .is_err());
        // A proof for another pin does not transfer.
        let other = identity(2);
        let wrong = PinnedTlsProof::for_test(&pin_of(&other));
        assert!(pinned(&store, d.clone(), &wrong).is_err());
        assert!(!has(&store, &x));
        let proof = PinnedTlsProof::for_test(&pin_of(&x));
        assert_eq!(pinned(&store, d, &proof).unwrap(), true);
        assert!(has(&store, &x));
    }

    #[test]
    fn a_pinned_identity_must_keep_its_address_and_tls_on_first_and_refresh() {
        let x = identity(1);
        let store = pinned_store(pin_of(&x), &[]);
        let proof = PinnedTlsProof::for_test(&pin_of(&x));
        assert!(pinned(&store, live(&x, 1, "http://127.0.0.1:18423", true), &proof).is_err());
        assert!(pinned(&store, live(&x, 1, PIN_ENDPOINT, false), &proof).is_err());
        let original = live(&x, 1, PIN_ENDPOINT, true);
        assert!(pinned(&store, original.clone(), &proof).unwrap());
        // A higher-sequence plaintext announce may not move or drop TLS.
        for bad in [
            live(&x, 2, "http://127.0.0.1:18423", true),
            live(&x, 2, PIN_ENDPOINT, false),
        ] {
            assert!(store
                .upsert_verified_from_source(bad, NOW, "gossip_announce")
                .is_err());
            assert_eq!(
                store.peers.read().get(&x.public_key_bytes()),
                Some(&original)
            );
        }
        // A conforming refresh from any origin is accepted.
        assert!(store
            .upsert_verified_from_source(
                live(&x, 2, PIN_ENDPOINT, true),
                NOW,
                "local_identity_refresh"
            )
            .unwrap());
    }

    #[test]
    fn identities_that_are_not_pinned_keep_todays_rules() {
        let x = identity(1);
        let y = identity(3);
        let store = pinned_store(pin_of(&x), &[]);
        assert!(store
            .upsert_verified_from_source(
                live(&y, 1, "http://127.0.0.1:19000", false),
                NOW,
                "bootstrap"
            )
            .unwrap());
        assert!(store
            .upsert_verified_from_source(
                live(&y, 2, "http://127.0.0.1:19001", false),
                NOW,
                "gossip_announce"
            )
            .unwrap());
        let empty = PeerStore::new_with_pin_policy(PinPolicy::empty());
        assert!(
            empty
                .upsert_verified_from_source(
                    live(&x, 1, "http://127.0.0.1:1", false),
                    NOW,
                    "bootstrap"
                )
                .unwrap(),
            "P and Q empty: exactly the old behavior"
        );
    }

    #[test]
    fn promotion_cannot_establish_or_refresh_a_pinned_identity() {
        let x = identity(1);
        let store = pinned_store(pin_of(&x), &[]);
        let d = live(&x, 1, PIN_ENDPOINT, true);
        assert!(store
            .upsert_verified_with_origin(
                d,
                NOW,
                "permissionless_promotion",
                PeerOrigin::Promotion,
                None
            )
            .is_err());
        assert!(!has(&store, &x));
        // Once established over TLS, promotion still may not refresh it.
        let proof = PinnedTlsProof::for_test(&pin_of(&x));
        let original = live(&x, 1, PIN_ENDPOINT, true);
        assert!(pinned(&store, original.clone(), &proof).unwrap());
        assert!(store
            .upsert_verified_with_origin(
                live(&x, 2, PIN_ENDPOINT, true),
                NOW,
                "permissionless_promotion",
                PeerOrigin::Promotion,
                None,
            )
            .is_err());
        assert_eq!(
            store.peers.read().get(&x.public_key_bytes()),
            Some(&original)
        );
    }

    // ── caches (W1 valid, W2 expired), export, quarantine ──────────────────

    fn cache_of(descriptors: Vec<SignedNodeDescriptor>) -> NodeBootstrapSnapshot {
        NodeBootstrapSnapshot::new(NOW, descriptors)
    }

    fn expired(kp: &IdentityKeyPair, endpoint: &str) -> SignedNodeDescriptor {
        descriptor(kp, 1, endpoint, true, 1_700_000_050)
    }

    #[test]
    fn pinned_and_quarantined_ids_never_load_from_any_cache_path() {
        let x = identity(1);
        let q = identity(4);
        let store = pinned_store(pin_of(&x), &[q.public_key_bytes()]);
        for source in ["cache", "cache_backup"] {
            let snapshot = cache_of(vec![
                live(&x, 1, PIN_ENDPOINT, true),
                live(&q, 1, "http://127.0.0.1:19004", true),
            ]);
            let report = store.load_peer_cache_snapshot_from_source(&snapshot, NOW, source);
            assert_eq!(report.inserted, 0, "{source}");
            let snapshot = cache_of(vec![
                expired(&x, PIN_ENDPOINT),
                expired(&q, "http://127.0.0.1:19004"),
            ]);
            let report = store.load_peer_cache_snapshot_from_source(&snapshot, NOW, source);
            assert_eq!(report.inserted, 0, "{source} expired (W2)");
        }
        assert!(!has(&store, &x) && !has(&store, &q));
        // Even after x is established over TLS, a conforming cache copy may
        // not refresh it: pins never come from a cache file.
        let original = live(&x, 1, PIN_ENDPOINT, true);
        assert!(pinned(
            &store,
            original.clone(),
            &PinnedTlsProof::for_test(&pin_of(&x))
        )
        .unwrap());
        let newer = cache_of(vec![live(&x, 2, PIN_ENDPOINT, true)]);
        assert_eq!(
            store
                .load_peer_cache_snapshot_from_source(&newer, NOW, "cache")
                .inserted,
            0
        );
        assert_eq!(
            store.peers.read().get(&x.public_key_bytes()),
            Some(&original)
        );
        // Q only blocks caches: an explicit operator snapshot still authorizes q.
        assert!(store
            .upsert_verified_from_source(
                live(&q, 1, "http://127.0.0.1:19004", true),
                NOW,
                "bootstrap"
            )
            .unwrap());
    }

    #[test]
    fn export_never_writes_pins_or_quarantined_ids() {
        let x = identity(1);
        let q = identity(4);
        let y = identity(3);
        let store = pinned_store(pin_of(&x), &[q.public_key_bytes()]);
        let proof = PinnedTlsProof::for_test(&pin_of(&x));
        assert!(pinned(&store, live(&x, 1, PIN_ENDPOINT, true), &proof).unwrap());
        assert!(store
            .upsert_verified_from_source(
                live(&q, 1, "http://127.0.0.1:19004", true),
                NOW,
                "bootstrap"
            )
            .unwrap());
        assert!(store
            .upsert_verified_from_source(
                live(&y, 1, "http://127.0.0.1:19003", true),
                NOW,
                "bootstrap"
            )
            .unwrap());
        let exported: Vec<[u8; 32]> = store
            .export_peer_cache_snapshot(NOW)
            .peers
            .iter()
            .map(SignedNodeDescriptor::node_id)
            .collect();
        assert_eq!(exported, vec![y.public_key_bytes()]);
    }

    #[test]
    fn export_racing_the_first_pinned_insert_never_sees_the_pin() {
        // An identity no other test writes: the pause is keyed by node id and
        // tests run in parallel, so a shared id could take this test's pause.
        let x = identity(201);
        let store = Arc::new(pinned_store(pin_of(&x), &[]));
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        arm_pause_inside_write_lock(x.public_key_bytes(), entered.clone(), release.clone());
        let writer = {
            let store = store.clone();
            let d = live(&x, 1, PIN_ENDPOINT, true);
            let pin = pin_of(&x);
            std::thread::spawn(move || {
                let proof = PinnedTlsProof::for_test(&pin);
                pinned(&store, d, &proof)
            })
        };
        entered.wait(); // The writer now holds the peers write lock, pre-insert.
        let (tx, rx) = mpsc::channel();
        let exporter = {
            let store = store.clone();
            std::thread::spawn(move || {
                let snapshot = store.export_peer_cache_snapshot(NOW);
                tx.send(()).unwrap();
                snapshot
            })
        };
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "export waits for the lock"
        );
        release.wait();
        assert!(writer.join().unwrap().unwrap());
        assert!(
            exporter.join().unwrap().peers.is_empty(),
            "export after the insert excludes P"
        );
        assert!(store.export_peer_cache_snapshot(NOW).peers.is_empty());
        assert!(has(&store, &x));
    }

    #[test]
    fn an_old_cache_cannot_revive_an_id_whose_pin_was_removed_before_any_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let cache_path = dir.path().join("peers-cache.json");
        let qpath = quarantine_path(cache_path.to_str().unwrap());
        let x = identity(1);
        let old_cache = cache_of(vec![live(&x, 1, PIN_ENDPOINT, true)]);
        // Run 1: P = {x}. Q is persisted first; the process then "exits"
        // without ever rewriting the cache.
        let q1 = load_and_extend_quarantine(&FsQuarantineIo, &qpath, &[pin_of(&x)]).unwrap();
        let run1 = PeerStore::new_with_pin_policy(PinPolicy::new(vec![pin_of(&x)], q1));
        assert_eq!(
            run1.load_peer_cache_snapshot_from_source(&old_cache, NOW, "cache")
                .inserted,
            0
        );
        drop(run1);
        // Run 2: the pin was removed. Q still holds x.
        let q2 = load_and_extend_quarantine(&FsQuarantineIo, &qpath, &[]).unwrap();
        let run2 = PeerStore::new_with_pin_policy(PinPolicy::new(Vec::new(), q2));
        for source in ["cache", "cache_backup"] {
            assert_eq!(
                run2.load_peer_cache_snapshot_from_source(&old_cache, NOW, source)
                    .inserted,
                0
            );
        }
        assert!(!has(&run2, &x));
        // Control: with neither P nor Q the same cache restores x (old behavior).
        let control = PeerStore::new_with_pin_policy(PinPolicy::empty());
        assert_eq!(
            control
                .load_peer_cache_snapshot_from_source(&old_cache, NOW, "cache")
                .inserted,
            1
        );
    }

    // ── pinned answers ─────────────────────────────────────────────────────

    #[test]
    fn a_pinned_answer_establishes_only_the_pin_and_trust_does_not_travel() {
        let x = identity(1);
        let z = identity(5);
        let store = pinned_store(pin_of(&x), &[]);
        store.enable_untrusted_discovery_candidate_mode();
        let answer = NodeDiscoveryMessage::SnapshotResponse {
            snapshot: cache_of(vec![
                live(&x, 1, PIN_ENDPOINT, true),
                live(&z, 1, "http://127.0.0.1:19005", true),
            ]),
        };
        let report =
            store.apply_pinned_seed_response(PinnedTlsProof::for_test(&pin_of(&x)), &answer, NOW);
        assert_eq!(report.inserted, 1);
        assert!(has(&store, &x));
        assert!(
            !has(&store, &z),
            "a third party stays an untrusted candidate"
        );
    }

    #[test]
    fn two_different_descriptors_for_the_pin_in_one_answer_are_both_refused() {
        let x = identity(1);
        let store = pinned_store(pin_of(&x), &[]);
        let a = live(&x, 1, PIN_ENDPOINT, true);
        let b = live(&x, 2, PIN_ENDPOINT, true);
        let answer = NodeDiscoveryMessage::SnapshotResponse {
            snapshot: cache_of(vec![a.clone(), b, a]),
        };
        let report =
            store.apply_pinned_seed_response(PinnedTlsProof::for_test(&pin_of(&x)), &answer, NOW);
        assert_eq!((report.inserted, report.rejected), (0, 2));
        assert!(!has(&store, &x));
        let dup = NodeDiscoveryMessage::SnapshotResponse {
            snapshot: cache_of(vec![
                live(&x, 1, PIN_ENDPOINT, true),
                live(&x, 1, PIN_ENDPOINT, true),
            ]),
        };
        let report =
            store.apply_pinned_seed_response(PinnedTlsProof::for_test(&pin_of(&x)), &dup, NOW);
        assert_eq!(report.inserted, 1, "identical copies are one descriptor");
    }

    // ── directory and URL ──────────────────────────────────────────────────

    #[test]
    fn the_pinned_url_comes_from_one_directory_and_refuses_ambiguity() {
        let x = identity(1);
        let other = identity(6);
        let pin = pin_of(&x);
        // No descriptor at all: the pin alone claims its address.
        let only_pins =
            peer_tls::PeerTlsDirectory::from_descriptors_and_pins(std::iter::empty(), &[pin]);
        let url = pinned_gossip_url(&pin, &only_pins).unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(
            peer_tls::decode_peer_tls_host(url.host_str().unwrap()),
            Some((x.public_key_bytes(), pin.addr.ip()))
        );
        // Rebuilt after an unrelated peer changes: the pin is still a claimant.
        let unrelated = live(&identity(7), 1, "http://127.0.0.1:19007", true);
        let rebuilt = peer_tls::PeerTlsDirectory::from_descriptors_and_pins([&unrelated], &[pin]);
        assert!(pinned_gossip_url(&pin, &rebuilt).is_ok());
        // A directory that pins the address to another identity is reported
        // as not pinned (the audit bucket), before any rewrite is attempted.
        let foreign = PinnedSeed {
            node_id: other.public_key_bytes(),
            addr: pin.addr,
        };
        let foreign_only =
            peer_tls::PeerTlsDirectory::from_descriptors_and_pins(std::iter::empty(), &[foreign]);
        assert_eq!(
            pinned_gossip_url(&pin, &foreign_only),
            Err(PinnedFetchError::NotPinned)
        );
        let unclaimed =
            peer_tls::PeerTlsDirectory::from_descriptors_and_pins(std::iter::empty(), &[]);
        assert_eq!(
            pinned_gossip_url(&pin, &unclaimed),
            Err(PinnedFetchError::NotPinned)
        );
        // Another identity claims the pinned address: refuse, no URL.
        let squatter = live(&other, 1, PIN_ENDPOINT, true);
        let ambiguous = peer_tls::PeerTlsDirectory::from_descriptors_and_pins([&squatter], &[pin]);
        assert_eq!(
            pinned_gossip_url(&pin, &ambiguous),
            Err(PinnedFetchError::NotPinned)
        );
        let plain_squatter = live(&other, 1, PIN_ENDPOINT, false);
        let ambiguous =
            peer_tls::PeerTlsDirectory::from_descriptors_and_pins([&plain_squatter], &[pin]);
        assert_eq!(
            pinned_gossip_url(&pin, &ambiguous),
            Err(PinnedFetchError::NotPinned)
        );
    }

    // ── quarantine file ────────────────────────────────────────────────────

    #[derive(Default)]
    struct FaultyIo {
        fail_at: Option<&'static str>,
        read_error: Option<io::ErrorKind>,
        sync_dir_calls: std::sync::atomic::AtomicUsize,
    }

    impl QuarantineIo for FaultyIo {
        fn lock(&self, path: &Path) -> io::Result<File> {
            FsQuarantineIo.lock(path)
        }
        fn open_existing(&self, path: &Path) -> io::Result<File> {
            FsQuarantineIo.open_existing(path)
        }
        fn read_bounded(&self, path: &Path, max: u64) -> io::Result<Vec<u8>> {
            match self.read_error {
                Some(kind) => Err(io::Error::from(kind)),
                None => FsQuarantineIo.read_bounded(path, max),
            }
        }
        fn write_new(&self, path: &Path, bytes: &[u8]) -> io::Result<File> {
            self.step("write", || FsQuarantineIo.write_new(path, bytes))
        }
        fn sync_file(&self, file: &File) -> io::Result<()> {
            self.step("sync_file", || FsQuarantineIo.sync_file(file))
        }
        fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            self.step("rename", || FsQuarantineIo.rename(from, to))
        }
        fn sync_dir(&self, dir: &Path) -> io::Result<()> {
            self.sync_dir_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.step("sync_dir", || FsQuarantineIo.sync_dir(dir))
        }
    }

    impl FaultyIo {
        fn step<T>(&self, name: &str, op: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
            if self.fail_at == Some(name) {
                return Err(io::Error::other(format!("injected {name} failure")));
            }
            op()
        }
    }

    fn read_ids(path: &Path) -> HashSet<[u8; 32]> {
        parse_quarantine(&std::fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn quarantine_starts_empty_only_when_absent_and_keeps_a_sorted_union() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.json");
        let a = pin_of(&identity(1));
        let b = PinnedSeed {
            node_id: identity(2).public_key_bytes(),
            addr: "127.0.0.1:2".parse().unwrap(),
        };
        assert_eq!(
            load_and_extend_quarantine(&FsQuarantineIo, &path, &[])
                .unwrap()
                .len(),
            0
        );
        assert!(!path.exists(), "nothing to add: nothing written");
        load_and_extend_quarantine(&FsQuarantineIo, &path, &[a]).unwrap();
        let ids = load_and_extend_quarantine(&FsQuarantineIo, &path, &[b]).unwrap();
        assert_eq!(ids, read_ids(&path));
        assert!(
            ids.contains(&a.node_id) && ids.contains(&b.node_id),
            "removing a pin never shrinks Q"
        );
    }

    #[test]
    fn every_quarantine_error_other_than_absence_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.json");
        let pin = pin_of(&identity(1));
        for kind in [io::ErrorKind::PermissionDenied, io::ErrorKind::Other] {
            let io = FaultyIo {
                read_error: Some(kind),
                ..FaultyIo::default()
            };
            assert!(
                load_and_extend_quarantine(&io, &path, &[pin]).is_err(),
                "{kind:?}"
            );
        }
        let too_many: Vec<String> = (0..=MAX_PIN_QUARANTINE_IDS)
            .map(|i| format!("{i:064x}"))
            .collect();
        for body in [
            b"not json".to_vec(),
            br#"{"v":2,"ids":[]}"#.to_vec(),
            br#"{"v":1,"ids":["zz"]}"#.to_vec(),
            br#"{"v":1,"ids":[],"extra":1}"#.to_vec(),
            serde_json::to_vec(&QuarantineFile {
                v: 1,
                ids: too_many,
            })
            .unwrap(),
            vec![b' '; PIN_QUARANTINE_MAX_BYTES as usize + 1],
        ] {
            std::fs::write(&path, &body).unwrap();
            assert!(load_and_extend_quarantine(&FsQuarantineIo, &path, &[pin]).is_err());
        }
    }

    #[test]
    fn a_failure_at_any_write_step_refuses_and_a_restart_recovers() {
        let pin = pin_of(&identity(1));
        for step in ["write", "sync_file", "rename", "sync_dir"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("q.json");
            let io = FaultyIo {
                fail_at: Some(step),
                ..FaultyIo::default()
            };
            assert!(
                load_and_extend_quarantine(&io, &path, &[pin]).is_err(),
                "{step}"
            );
            let leftovers: Vec<_> = std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
                .collect();
            assert!(leftovers.is_empty(), "{step}: no temp file left behind");
            let ids = load_and_extend_quarantine(&FsQuarantineIo, &path, &[pin]).unwrap();
            assert!(
                ids.contains(&pin.node_id) && read_ids(&path).contains(&pin.node_id),
                "{step}"
            );
        }
    }

    #[test]
    fn two_starts_at_once_never_lose_each_others_ids() {
        for _ in 0..50 {
            let dir = tempfile::tempdir().unwrap();
            let path = Arc::new(dir.path().join("q.json"));
            let start = Arc::new(Barrier::new(2));
            let workers: Vec<_> = [1u8, 2]
                .into_iter()
                .map(|seed| {
                    let (path, start) = (path.clone(), start.clone());
                    std::thread::spawn(move || {
                        let pin = PinnedSeed {
                            node_id: identity(seed).public_key_bytes(),
                            addr: "127.0.0.1:1".parse().unwrap(),
                        };
                        start.wait();
                        load_and_extend_quarantine(&FsQuarantineIo, &path, &[pin]).unwrap();
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
            assert_eq!(read_ids(&path).len(), 2);
        }
    }

    // Review of 6515223: a rename that landed before a failed directory sync
    // must be completed by the next start, not waved through as "unchanged".
    #[test]
    fn a_failed_directory_sync_is_completed_by_a_later_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.json");
        let pin = pin_of(&identity(1));
        let failing = || FaultyIo {
            fail_at: Some("sync_dir"),
            ..FaultyIo::default()
        };
        assert!(load_and_extend_quarantine(&failing(), &path, &[pin]).is_err());
        assert!(read_ids(&path).contains(&pin.node_id), "renamed into place");
        assert!(
            load_and_extend_quarantine(&failing(), &path, &[pin]).is_err(),
            "an unchanged Q still owes its directory sync"
        );
        assert!(
            load_and_extend_quarantine(&failing(), &path, &[]).is_err(),
            "also when the pins were removed"
        );
        let healthy = FaultyIo::default();
        assert!(load_and_extend_quarantine(&healthy, &path, &[pin]).is_ok());
        assert_eq!(
            healthy
                .sync_dir_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the completing start syncs the directory"
        );
    }

    // Review of 6515223: a node without pins gains no new way to fail start.
    #[test]
    fn without_pins_an_absent_quarantine_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing/deeper/q.json");
        let io = FaultyIo::default();
        assert!(load_and_extend_quarantine(&io, &path, &[])
            .unwrap()
            .is_empty());
        assert!(
            !dir.path().join("missing").exists(),
            "no directory, lock or file"
        );
        assert_eq!(
            io.sync_dir_calls.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        // An existing Q is still read strictly without pins.
        let existing = dir.path().join("q.json");
        std::fs::write(&existing, b"not json").unwrap();
        assert!(load_and_extend_quarantine(&FsQuarantineIo, &existing, &[]).is_err());
        let denied = FaultyIo {
            read_error: Some(io::ErrorKind::PermissionDenied),
            ..FaultyIo::default()
        };
        assert!(load_and_extend_quarantine(&denied, &existing, &[]).is_err());
    }
}
