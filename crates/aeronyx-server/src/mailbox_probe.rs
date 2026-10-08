// ============================================
// File: crates/aeronyx-server/src/mailbox_probe.rs
// ============================================
//! # Client-sourced anonymous mailbox probe
//!
//! ## Creation Reason
//! [MAILBOX-PROBE 2026-10-09 by Claude] Every existing mailbox test peels the
//! onion and calls the terminal directly; none sends a request the way a phone
//! does. This command is the reference client: it acts as its own onion source
//! over plain HTTP, against real nodes, with no VPN session and no node
//! identity, and runs the whole custody lifecycle end to end.
//!
//! ## Main Functionality
//! - Collects signed descriptors from one or more seed nodes and lets
//!   `VerifiedOnionRoute` verify them; nothing unsigned selects a hop.
//! - Builds each terminal frame exactly as a client would: source carrier with
//!   a one-shot reply key, signed route request, one- or two-hop onion.
//! - POSTs to the entry node's public `/api/chat/peer/blind-relay`, opens the
//!   source-sealed reply, and verifies the terminal's signature against the
//!   exact request.
//! - Runs ticket → lease → put → pull → ack → pull and checks that the pulled
//!   bytes are the deposited bytes and that the mailbox is empty afterwards.
//!
//! ## Dependencies
//! - `aeronyx-core::protocol::{anonymous_mailbox, onion, memchain, discovery}`
//! - `aeronyx_server::api::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse}`
//!
//! ## Important Notes for Next Developer
//! - Every key here is ephemeral and generated per run. Never accept or print
//!   real user keys, mailbox ids, item ids, or payload bytes.
//! - The deposited bytes are random. The terminal treats a sealed envelope as
//!   opaque, so this proves custody, not recipient decryption.
//! - The Dart client must reproduce `exchange` byte for byte; keep this file
//!   free of shortcuts a phone could not take.
//!
//! Last Modified: v1.0.0 - Initial client-sourced lifecycle probe.
// ============================================

use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, ensure, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use rand::RngCore;
use serde::Serialize;
use sha2::{Digest, Sha256};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxAckV1, AnonymousMailboxAdmissionTicketV1, AnonymousMailboxLeaseCreateV1,
    AnonymousMailboxOperationV1, AnonymousMailboxOutcomeV1, AnonymousMailboxPullOneV1,
    AnonymousMailboxPullResultV1, AnonymousMailboxPutV1, AnonymousMailboxRouteRequestV1,
    AnonymousMailboxSourceTerminalCarrierV1, AnonymousMailboxTerminalFrameV1,
    AnonymousMailboxTerminalResponseV1, AnonymousMailboxTicketIssueV1,
};
use aeronyx_core::protocol::discovery::SignedNodeDescriptor;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{OnionRoutePurpose, VerifiedOnionRoute};
use aeronyx_server::api::chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse};

const BLIND_RELAY_PATH: &str = "/api/chat/peer/blind-relay";
const CANDIDATES_PATH: &str = "/api/discovery/onion-candidates";
/// Signed snapshot; unlike the candidate list it also carries the seed's own
/// descriptor and does not wait for route-health quorum.
const SNAPSHOT_PATH: &str = "/api/discovery/snapshot";
/// Purposes queried so both relay-capable entries and mailbox terminals are seen.
const CANDIDATE_PURPOSES: [&str; 2] = ["message_relay", "anonymous_mailbox_v1"];
const TICKET_TTL_SECS: u64 = 240;
const LEASE_TTL_SECS: u64 = 60 * 60;
const ITEM_TTL_SECS: u64 = 60 * 60;
const LEASE_MAX_ITEMS: u16 = 16;
const LEASE_MAX_BYTES: u64 = 1024 * 1024;
const PROBE_ITEM_BYTES: usize = 1024;

/// Inputs for one probe run.
pub struct MailboxProbeOptions {
    /// Seed node base URLs whose candidate lists supply signed descriptors.
    pub seeds: Vec<String>,
    /// Hex node id of the custody (terminal) node.
    pub target: String,
    /// Optional hex node id of a distinct entry node for a two-hop route.
    pub entry: Option<String>,
    /// Proof-of-work bits for the ticket request; must be at least the
    /// target's configured `ticket_issue_work_bits`.
    pub work_bits: u8,
    /// Per-request HTTP timeout.
    pub timeout: Duration,
}

/// Aggregate-only result; contains no identities, ids, or payload bytes.
#[derive(Debug, Serialize)]
pub struct MailboxProbeReport {
    pub status: &'static str,
    pub hops: usize,
    pub steps: Vec<MailboxProbeStep>,
    pub pulled_bytes_match: bool,
    pub empty_after_ack: bool,
}

/// One lifecycle step.
#[derive(Debug, Serialize)]
pub struct MailboxProbeStep {
    pub operation: &'static str,
    pub outcome: String,
    pub elapsed_ms: u128,
}

/// Runs the full client-sourced lifecycle once.
pub async fn run(options: MailboxProbeOptions) -> Result<MailboxProbeReport> {
    let http = reqwest::Client::builder()
        .timeout(options.timeout)
        .build()
        .context("build HTTP client")?;
    let descriptors = collect_descriptors(&http, &options.seeds).await?;
    let target_id = parse_node_id(&options.target)?;
    let target = descriptors
        .get(&target_id)
        .cloned()
        .ok_or_else(|| anyhow!("target descriptor not offered by any seed"))?;
    let mut path = Vec::with_capacity(2);
    if let Some(entry) = &options.entry {
        let entry_id = parse_node_id(entry)?;
        ensure!(entry_id != target_id, "entry and target must differ");
        path.push(
            descriptors
                .get(&entry_id)
                .cloned()
                .ok_or_else(|| anyhow!("entry descriptor not offered by any seed"))?,
        );
    }
    path.push(target);

    let probe = Probe {
        http,
        path,
        target_id,
        source: IdentityKeyPair::generate(),
    };
    let reader = IdentityKeyPair::generate();
    let depositor = IdentityKeyPair::generate();
    let mailbox_id: [u8; 32] = random_bytes();
    let mut steps = Vec::new();

    // 1. Ticket: proof of work bound to the exact lease claims.
    let now = unix_now()?;
    let lease_issued_at = now;
    let lease_expires_at = now + LEASE_TTL_SECS;
    let claims = AnonymousMailboxLeaseCreateV1::lease_claims_commitment(
        &mailbox_id,
        &depositor.public_key_bytes(),
        &reader.public_key_bytes(),
        LEASE_MAX_ITEMS,
        LEASE_MAX_BYTES,
        lease_issued_at,
        lease_expires_at,
    );
    let ticket_request = solve_ticket_request(target_id, claims, now, options.work_bits)?;
    let started = Instant::now();
    let ticket = match probe
        .exchange(AnonymousMailboxTerminalFrameV1::TicketIssue(ticket_request.clone()))
        .await?
    {
        AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response) => {
            response
                .verify_for_request(&ticket_request, &target_id)
                .map_err(|error| anyhow!("ticket response signature: {error:?}"))?;
            steps.push(step("ticket_issue", response.outcome, started));
            ensure_accepted("ticket_issue", response.outcome)?;
            response
                .ticket
                .ok_or_else(|| anyhow!("accepted ticket response carried no ticket"))?
        }
        _ => bail!("ticket_issue: unexpected response frame"),
    };
    verify_ticket(&ticket, &target_id, claims)?;

    // 2. Lease.
    let lease = AnonymousMailboxLeaseCreateV1::new(
        mailbox_id,
        depositor.public_key_bytes(),
        LEASE_MAX_ITEMS,
        LEASE_MAX_BYTES,
        lease_issued_at,
        lease_expires_at,
        ticket,
        &reader,
    )
    .map_err(|error| anyhow!("lease request: {error:?}"))?;
    let lease_commitment = lease
        .request_commitment()
        .map_err(|error| anyhow!("lease commitment: {error:?}"))?;
    let lease_request_id = lease.admission.ticket_id;
    let response = probe
        .terminal(
            "lease_create",
            AnonymousMailboxTerminalFrameV1::LeaseCreate(lease),
            AnonymousMailboxOperationV1::LeaseCreate,
            lease_request_id,
            lease_commitment,
            &mut steps,
        )
        .await?;
    ensure_accepted("lease_create", response.outcome)?;

    // 3. Put: what a sender does with a deposit grant.
    let mut sealed = vec![0u8; PROBE_ITEM_BYTES];
    rand::thread_rng().fill_bytes(&mut sealed);
    let now = unix_now()?;
    let put = AnonymousMailboxPutV1::new(
        mailbox_id,
        random_bytes(),
        sealed.clone(),
        now,
        // The terminal answers `Expired` for an item that would outlive its
        // lease, so a depositor must cap the item at the lease expiry.
        (now + ITEM_TTL_SECS).min(lease_expires_at),
        &depositor,
    )
    .map_err(|error| anyhow!("put request: {error:?}"))?;
    let (put_id, put_commitment) = (
        put.item_id,
        put.request_commitment()
            .map_err(|error| anyhow!("put commitment: {error:?}"))?,
    );
    let response = probe
        .terminal(
            "put",
            AnonymousMailboxTerminalFrameV1::Put(put),
            AnonymousMailboxOperationV1::Put,
            put_id,
            put_commitment,
            &mut steps,
        )
        .await?;
    ensure_accepted("put", response.outcome)?;

    // 4. Pull: what the recipient does.
    let pulled = probe.pull(mailbox_id, &reader, &mut steps).await?;
    let pulled = pulled.ok_or_else(|| anyhow!("pull returned an empty mailbox after put"))?;
    let pulled_bytes_match = pulled.item_id == put_id && pulled.sealed_item == sealed;
    ensure!(pulled_bytes_match, "pulled item differs from the deposited item");

    // 5. Ack, then confirm the mailbox is empty.
    let now = unix_now()?;
    let ack = AnonymousMailboxAckV1::new(
        mailbox_id,
        random_bytes(),
        pulled.item_id,
        Sha256::digest(&pulled.sealed_item).into(),
        now,
        &reader,
    )
    .map_err(|error| anyhow!("ack request: {error:?}"))?;
    let (ack_id, ack_commitment) = (
        ack.request_id,
        ack.request_commitment()
            .map_err(|error| anyhow!("ack commitment: {error:?}"))?,
    );
    let response = probe
        .terminal(
            "ack",
            AnonymousMailboxTerminalFrameV1::Ack(ack),
            AnonymousMailboxOperationV1::Ack,
            ack_id,
            ack_commitment,
            &mut steps,
        )
        .await?;
    ensure_accepted("ack", response.outcome)?;
    let empty_after_ack = probe.pull(mailbox_id, &reader, &mut steps).await?.is_none();
    ensure!(empty_after_ack, "mailbox still returned an item after ack");

    Ok(MailboxProbeReport {
        status: "verified",
        hops: probe.path.len(),
        steps,
        pulled_bytes_match,
        empty_after_ack,
    })
}

struct Probe {
    http: reqwest::Client,
    /// Entry first, terminal last.
    path: Vec<SignedNodeDescriptor>,
    target_id: [u8; 32],
    /// Ephemeral onion source; never a node identity.
    source: IdentityKeyPair,
}

impl Probe {
    /// Sends one terminal frame through the onion and returns the opened,
    /// decoded response frame.
    async fn exchange(
        &self,
        frame: AnonymousMailboxTerminalFrameV1,
    ) -> Result<AnonymousMailboxTerminalFrameV1> {
        let now = unix_now()?;
        let route_id: [u8; 16] = random_bytes();
        let terminal_frame = encode_anonymous_mailbox_terminal_frame(&frame)
            .map_err(|error| anyhow!("encode terminal frame: {error:?}"))?;
        let (carrier, mut session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
            route_id,
            self.target_id,
            terminal_frame,
        )
        .map_err(|error| anyhow!("source carrier: {error:?}"))?;
        let route_request = AnonymousMailboxRouteRequestV1::signed(
            route_id,
            self.target_id,
            carrier
                .encode()
                .map_err(|error| anyhow!("carrier bytes: {error:?}"))?,
            now,
            &self.source,
        )
        .map_err(|error| anyhow!("route request: {error:?}"))?;
        let payload = encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(route_request))
            .context("encode route request")?;
        let route = VerifiedOnionRoute::from_signed_descriptors(
            self.source.public_key_bytes(),
            self.path.iter(),
            OnionRoutePurpose::AnonymousMailboxV1,
            now,
        )
        .map_err(|error| anyhow!("onion route: {error:?}"))?;
        ensure!(
            route.terminal_node_id() == self.target_id,
            "verified route terminal differs from the requested target"
        );
        let envelope = route
            .build_envelope(&payload, route_id, now, &self.source)
            .map_err(|error| anyhow!("onion envelope: {error:?}"))?;
        let request = PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: self.source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        };
        let entry = self.path[0]
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or_else(|| anyhow!("entry descriptor has no public endpoint"))?;
        let url = format!("{}{}", entry.trim_end_matches('/'), BLIND_RELAY_PATH);
        let http = self
            .http
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("POST blind relay")?;
        let status = http.status();
        let body = http.bytes().await.context("read blind relay response")?;
        let response: PeerBlindRelayResponse = serde_json::from_slice(&body).with_context(|| {
            format!("blind relay HTTP {status}: undecodable response body")
        })?;
        ensure!(
            status.is_success() && response.accepted,
            "blind relay HTTP {status} rejected: {}",
            response.reason.as_deref().unwrap_or("no_reason")
        );
        let sealed = response
            .opaque_terminal_response_b64
            .ok_or_else(|| anyhow!("accepted response carried no sealed terminal reply"))?;
        let sealed = BASE64.decode(sealed).context("sealed reply base64")?;
        let opened = session
            .open(&sealed)
            .map_err(|error| anyhow!("open sealed reply: {error:?}"))?;
        decode_anonymous_mailbox_terminal_frame(&opened)
            .map_err(|error| anyhow!("decode terminal reply: {error:?}"))
    }

    /// Sends one generic operation and verifies the terminal signature over
    /// the exact request before trusting the outcome.
    async fn terminal(
        &self,
        name: &'static str,
        frame: AnonymousMailboxTerminalFrameV1,
        operation: AnonymousMailboxOperationV1,
        request_id: [u8; 16],
        commitment: [u8; 32],
        steps: &mut Vec<MailboxProbeStep>,
    ) -> Result<AnonymousMailboxTerminalResponseV1> {
        let started = Instant::now();
        let response = match self.exchange(frame).await? {
            AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response)
            | AnonymousMailboxTerminalFrameV1::PutResponse(response)
            | AnonymousMailboxTerminalFrameV1::PullOneResponse(response)
            | AnonymousMailboxTerminalFrameV1::AckResponse(response) => response,
            _ => bail!("{name}: unexpected response frame"),
        };
        response
            .verify_for_request(operation, &request_id, &commitment, &self.target_id)
            .map_err(|error| anyhow!("{name} response signature: {error:?}"))?;
        steps.push(step(name, response.outcome, started));
        Ok(response)
    }

    async fn pull(
        &self,
        mailbox_id: [u8; 32],
        reader: &IdentityKeyPair,
        steps: &mut Vec<MailboxProbeStep>,
    ) -> Result<Option<AnonymousMailboxPullResultV1>> {
        let pull = AnonymousMailboxPullOneV1::new(
            mailbox_id,
            random_bytes(),
            Vec::new(),
            unix_now()?,
            reader,
        )
        .map_err(|error| anyhow!("pull request: {error:?}"))?;
        let (request_id, commitment) = (
            pull.request_id,
            pull.request_commitment()
                .map_err(|error| anyhow!("pull commitment: {error:?}"))?,
        );
        let response = self
            .terminal(
                "pull_one",
                AnonymousMailboxTerminalFrameV1::PullOne(pull),
                AnonymousMailboxOperationV1::PullOne,
                request_id,
                commitment,
                steps,
            )
            .await?;
        ensure_accepted("pull_one", response.outcome)?;
        if response.sealed_payload.is_empty() {
            return Ok(None);
        }
        AnonymousMailboxPullResultV1::decode(&response.sealed_payload)
            .map(Some)
            .map_err(|error| anyhow!("pull result: {error:?}"))
    }
}

/// Fetches signed descriptors from every seed's candidate lists and signed
/// snapshot. Unsigned fields are ignored; `VerifiedOnionRoute` re-verifies the
/// signature and the mailbox protocol features of every descriptor it uses.
async fn collect_descriptors(
    http: &reqwest::Client,
    seeds: &[String],
) -> Result<BTreeMap<[u8; 32], SignedNodeDescriptor>> {
    ensure!(!seeds.is_empty(), "at least one --seed is required");
    let mut descriptors = BTreeMap::new();
    for seed in seeds {
        for purpose in CANDIDATE_PURPOSES {
            let url = format!(
                "{}{}?purpose={purpose}",
                seed.trim_end_matches('/'),
                CANDIDATES_PATH
            );
            let body: serde_json::Value = http
                .get(&url)
                .send()
                .await
                .with_context(|| format!("GET candidates from seed ({purpose})"))?
                .error_for_status()
                .context("candidates HTTP status")?
                .json()
                .await
                .context("candidates JSON")?;
            let Some(candidates) = body.get("candidates").and_then(|value| value.as_array())
            else {
                continue;
            };
            for candidate in candidates {
                let Some(signed) = candidate.get("signed_descriptor") else {
                    continue;
                };
                let Ok(signed) = serde_json::from_value::<SignedNodeDescriptor>(signed.clone())
                else {
                    continue;
                };
                descriptors.insert(signed.descriptor.node_id, signed);
            }
        }
        let url = format!("{}{}", seed.trim_end_matches('/'), SNAPSHOT_PATH);
        let snapshot: serde_json::Value = http
            .get(&url)
            .send()
            .await
            .context("GET snapshot from seed")?
            .error_for_status()
            .context("snapshot HTTP status")?
            .json()
            .await
            .context("snapshot JSON")?;
        for peer in snapshot
            .get("peers")
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
        {
            if let Ok(signed) = serde_json::from_value::<SignedNodeDescriptor>(peer.clone()) {
                descriptors.insert(signed.descriptor.node_id, signed);
            }
        }
    }
    Ok(descriptors)
}

/// Grinds the ticket proof of work. The digest covers the full request, so a
/// fresh request id and ticket id are fixed before the nonce search starts.
fn solve_ticket_request(
    target: [u8; 32],
    claims: [u8; 32],
    now: u64,
    work_bits: u8,
) -> Result<AnonymousMailboxTicketIssueV1> {
    let (request_id, ticket_id) = (random_bytes(), random_bytes());
    for nonce in 0..u64::MAX {
        let request = AnonymousMailboxTicketIssueV1::new(
            request_id,
            ticket_id,
            target,
            claims,
            now,
            now + TICKET_TTL_SECS,
            nonce,
        )
        .map_err(|error| anyhow!("ticket request: {error:?}"))?;
        // `verify_for_target` checks the same leading-zero rule the
        // terminal applies, so the probe cannot drift from the server.
        if request.verify_for_target(&target, now, work_bits).is_ok() {
            return Ok(request);
        }
    }
    bail!("ticket proof of work exhausted the nonce space")
}

fn verify_ticket(
    ticket: &AnonymousMailboxAdmissionTicketV1,
    target: &[u8; 32],
    claims: [u8; 32],
) -> Result<()> {
    ensure!(
        &ticket.target_node_id == target && ticket.lease_claims_commitment == claims,
        "ticket is not bound to the requested target and claims"
    );
    ticket
        .verify_at(target, &claims, unix_now()?)
        .map_err(|error| anyhow!("ticket signature: {error:?}"))
}

fn ensure_accepted(name: &str, outcome: AnonymousMailboxOutcomeV1) -> Result<()> {
    ensure!(
        outcome == AnonymousMailboxOutcomeV1::Accepted,
        "{name}: terminal answered {outcome:?}"
    );
    Ok(())
}

fn step(
    operation: &'static str,
    outcome: AnonymousMailboxOutcomeV1,
    started: Instant,
) -> MailboxProbeStep {
    MailboxProbeStep {
        operation,
        outcome: format!("{outcome:?}"),
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn parse_node_id(hex_id: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hex_id.trim()).context("node id must be hex")?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("node id must be 32 bytes"))
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
}

fn unix_now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_secs())
}
