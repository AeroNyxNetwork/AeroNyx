// ============================================
// File: crates/aeronyx-server/src/main.rs
// ============================================
//! # AeroNyx Server Entry Point
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Subcommand implementations moved to
//! `commands/`; bodies unchanged.
//! This file keeps only the CLI argument types and the `main` dispatcher.
//! - `commands.rs` + `commands/`: one module per subcommand plus shared
//!   helpers (see the `commands` module docs for the full layout)
//! - `mailbox_probe.rs`, `relay_smoke.rs`: live-proof protocol clients
//!
//! ## Modification Reason
//! - Added MemChain status display in `status` command (AOF file size,
//!   mode, API address).
//! - Added MemChain config display in `validate` command.
//! - v2.5.3+Security / v1.0.0-MultiTenant: Server::new() gains third
//!   argument `config_path: Option<PathBuf>` for auto-generated secret
//!   persistence. cmd_start passes Some(config_path.clone()) so that
//!   api_secret and jwt_secret are written back to disk on first startup.
//! - v1.1.0-SectionSafeAuth: resolve and inject `memchain.api_secret` before
//!   constructing Server, closing the first-start unauthenticated window.
//! - v1.2.0-DirectoryReplicaQuarantineResolution: add host-local incident
//!   inspection and node-identity-signed compare-and-swap resolution commands.
//! - v1.3.0-AofIntegrityCommand: add a read-only, privacy-safe MemChain AOF
//!   verification command for framing, semantic, Merkle, and ancestry checks.
//! - v1.4.0-DirectoryCarrierSmoke: add a bounded local API client that proves
//!   explicit signed mirror-carrier recovery without importing evidence.
//! - v1.5.0-PortableObservationCertificateVerifier: add a bounded offline
//!   verifier for exact Directory observation-certificate frames.
//! - v1.6.0-PortableObservationCertificateImport: add a host-local durable
//!   import command backed by the signed schema-v10 certificate history.
//! - v1.7.0-AuthenticatedCertificatePull: add explicit pinned-source network
//!   retrieval with a strict certificate-age gate before durable import.
//! - [NODE-REGISTRATION-PROFILE 2026-08-02 by Codex] Bind the validated VPN
//!   listener port and optional operator name/region/public policy during the
//!   one-time registration request instead of accepting stale CMS defaults.
//! - [REGISTRATION-CODE-STDIN 2026-08-02 by Codex] Accept bounded registration
//!   codes from standard input so installers do not expose one-time credentials
//!   through process command lines; the legacy `--code` flag remains compatible.
//! - [MANAGEMENT-CLIENT-STARTUP 2026-08-12 by Codex] Propagate management HTTP
//!   client initialization errors from node registration instead of panicking.
//! - [LIVE-RELAY-SMOKE 2026-08-15 by Codex] Add a host-local operator command
//!   that proves the production authenticated UDP, E2E relay, terminal receipt,
//!   mailbox pull, and ACK path without exposing protocol secrets.
//! - [V1-COMPATIBILITY-SMOKE 2026-09-13 by Codex] Add a host-local frozen
//!   v0x01 handshake/keepalive/close proof without changing protocol defaults.
//! - [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Add host-local custody
//!   retention audit and confirmation-gated prune commands with aggregate-only
//!   output; no management-plane or HTTP mutation endpoint is introduced.
//! - [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] Add a non-destructive
//!   latest-backup restore preflight with path-free aggregate output.
//! - [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Add short-lived,
//!   node-secret-authenticated restore plans bound to private storage state.
//! - [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Add bounded, host-local
//!   verification for the private HMAC-chained custody maintenance history.
//! - [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] Surface aggregate-only
//!   immutable segment/checkpoint and interrupted-rotation status.
//! - [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] Add create-new export and
//!   fail-closed offline verification for exact node-signed custody anchors.
//! - [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Add durable independent-node
//!   countersigning and exact offline verification for custody audit anchors.
//! - [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Close the air-gapped
//!   producer workflow with a bounded host-local signed-receipt import.
//! - [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Re-audit current-anchor
//!   witness policy locally after restart without scheduling network traffic.
//! - [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] Collect and
//!   durably re-audit current-anchor witness receipts through one explicit,
//!   snapshot-pinned operator command without enabling a background scheduler.
//! - [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] Report the exact aggregate
//!   threshold lifetime and a local renewal recommendation without networking.
//!
//! ## Last Modified
//! v0.1.0 - Initial CLI implementation
//! v0.2.0 - Added register command, simplified user flow
//! v0.3.0 - Added MemChain status and config display
//! v1.0.0-MultiTenant - Pass config_path to Server::new() (3rd argument)
//! v0.3.0-DiscoveryBootstrap - Show discovery bootstrap config in validate
//! v1.1.0-SectionSafeAuth - Resolve/migrate API secret before server startup
//! v1.2.0-DirectoryReplicaQuarantineResolution - Add audited host-local
//! quarantine inspection and resolution without exposing a mutation API
//! v1.3.0-AofIntegrityCommand - Add aggregate-only `memchain verify-aof`
//! v1.4.0-DirectoryCarrierSmoke - Add read-only `directory-replica carrier-smoke`
//! v1.5.0-PortableObservationCertificateVerifier - Add fail-closed offline
//! certificate verification with exact frame SHA-256 binding
//! v1.6.0-PortableObservationCertificateImport - Add bounded, pinned,
//! hash-linked third-party certificate persistence and restart audit
//! v1.7.0-AuthenticatedCertificatePull - Add hardened pinned-source certificate
//! pull, exact response verification, local trust policy, and freshness gate
//! v1.8.0-NodeOnboarding - Add policy-safe node registration metadata
//! v1.9.0-RegistrationCodeStdin - Add bounded secret-safe registration input
//! v1.10.0-ManagementClientStartup - Fail registration cleanly when the
//! management HTTP client cannot initialize
//! v1.11.0-LiveRelaySmoke - Add a bounded authenticated live relay smoke
//! v1.12.0-CustodyBackupPrune - Add host-local relay custody maintenance
//! v1.13.0-CustodyRestoreReadiness - Add read-only recovery preflight
//! v1.14.0-CustodyRestorePlan - Add authenticated host-local recovery plans
//! v1.15.0-CustodyAuditVerify - Add aggregate maintenance-chain verification
//! v1.16.0-CustodyAuditRotation - Report authenticated audit segment state
//! v1.17.0-CustodyAuditAnchor - Export and verify portable checkpoint anchors
//! v1.18.0-CustodyAuditWitness - Persist and verify independent witness receipts
//! v1.19.0-CustodyWitnessReceiptImport - Import pinned receipts into the
//! producer's fully re-audited bounded evidence vault
//! v1.20.0-CustodyWitnessVaultAudit - Audit current-anchor receipt readiness
//! locally with an optional fail-closed operator health gate
//! v1.21.0-CustodyWitnessOperatorCollect - Explicitly collect current-anchor
//! witness receipts from signed snapshot-pinned peers and re-audit durability
//! v1.22.0-CustodyWitnessAtomicReadiness - Use one typed SQLite snapshot for
//! vault audit and current-anchor readiness across startup and operator tools
//! v1.23.0-CustodyQuorumExpiry - Add privacy-safe quorum validity and renewal
//! window fields to custody import, collection, and vault audit reports

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::error;

use commands::{
    cmd_directory_replica, cmd_mailbox_probe, cmd_memchain, cmd_pubkey, cmd_register,
    cmd_relay_custody, cmd_relay_smoke, cmd_start, cmd_status, cmd_v1_compatibility_smoke,
    cmd_validate, init_logging, resolve_registration_code,
};

mod commands;
mod mailbox_probe;
mod relay_smoke;

// ============================================
// CLI Definition
// ============================================

/// AeroNyx Privacy Network Server
#[derive(Parser, Debug)]
#[command(name = "aeronyx-server")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Register this node with AeroNyx network
    Register {
        /// Registration code from dashboard (e.g., NYX-1234-ABCDE)
        #[arg(
            short = 'C',
            long,
            required_unless_present = "code_stdin",
            conflicts_with = "code_stdin"
        )]
        code: Option<String>,

        /// Read the registration code from one bounded line on standard input
        #[arg(long, required_unless_present = "code", conflicts_with = "code")]
        code_stdin: bool,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// CMS API URL (usually not needed, uses default)
        #[arg(long, hide = true)]
        cms_url: Option<String>,

        /// Operator-facing node name shown in nodeboard and the VPN pool
        #[arg(long, value_name = "NAME")]
        node_name: Option<String>,

        /// ISO 3166-1 alpha-2 deployment region (for example TW or KR)
        #[arg(long, value_name = "CC")]
        region: Option<String>,

        /// Publish this VPN node in the authenticated public node pool
        #[arg(long)]
        public_vpn: bool,
    },

    /// Start the server
    Start {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Check node registration status
    Status {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Validate configuration file
    Validate {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Inspect MemChain persistence without exposing memory contents
    #[command(subcommand)]
    Memchain(MemchainCommands),

    /// Show node public key (for troubleshooting)
    #[command(hide = true)]
    Pubkey {
        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Output format: base64 (default), hex
        #[arg(long, default_value = "hex")]
        format: String,
    },

    /// Inspect, verify, or resolve Directory Replica state locally
    #[command(subcommand)]
    DirectoryReplica(DirectoryReplicaCommands),

    /// Inspect or explicitly prune private relay-custody recovery artifacts
    #[command(subcommand)]
    RelayCustody(RelayCustodyCommands),

    /// Prove one host-local authenticated multi-hop ciphertext relay
    RelaySmoke {
        /// Running node UDP listener; only loopback addresses are accepted
        #[arg(long, default_value = "127.0.0.1:51820")]
        server: std::net::SocketAddr,

        /// Running node aggregate health URL; only loopback HTTP is accepted
        #[arg(long, default_value = "http://127.0.0.1:8421/api/vpn/health")]
        health_url: String,

        /// Path to the running node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Total bounded proof window in seconds
        #[arg(
            long,
            default_value_t = 30,
            value_parser = clap::value_parser!(u64).range(5..=120)
        )]
        timeout_seconds: u64,

        /// Confirm creation of two ephemeral test sessions and one ciphertext
        #[arg(long)]
        confirm_live_relay_smoke: bool,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Prove a client can run the anonymous mailbox lifecycle over the onion
    ///
    /// [MAILBOX-PROBE 2026-10-09 by Claude] Acts as a phone would: ephemeral
    /// keys, its own onion source, plain HTTP to the entry node, no VPN.
    MailboxProbe {
        /// Seed node base URL supplying signed descriptors (repeatable)
        #[arg(long = "seed", required = true)]
        seeds: Vec<String>,

        /// Hex node id of the custody node holding the mailbox
        #[arg(long)]
        target: String,

        /// Hex node id of a distinct entry node (two-hop route)
        #[arg(long)]
        entry: Option<String>,

        /// Ticket proof-of-work bits; at least the target's configured value
        #[arg(
            long,
            default_value_t = aeronyx_server::config_chat_relay::DEFAULT_ANONYMOUS_MAILBOX_TICKET_ISSUE_WORK_BITS
        )]
        work_bits: u8,

        /// Per-request HTTP timeout in seconds
        #[arg(long, default_value_t = 20)]
        timeout_seconds: u64,

        /// Confirm creating one short-lived mailbox and one test item
        #[arg(long)]
        confirm_live_mailbox_probe: bool,

        /// Emit the aggregate JSON report
        #[arg(long)]
        json: bool,
    },

    /// Prove the running node still serves the frozen v0x01 transport
    V1CompatibilitySmoke {
        /// Running node UDP listener; only loopback addresses are accepted
        #[arg(long, default_value = "127.0.0.1:51820")]
        server: std::net::SocketAddr,

        /// Running node aggregate health URL; only loopback HTTP is accepted
        #[arg(long, default_value = "http://127.0.0.1:8421/api/vpn/health")]
        health_url: String,

        /// Path to the running node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Total bounded proof window, including the scheduled probe wait
        #[arg(
            long,
            default_value_t = 90,
            value_parser = clap::value_parser!(u64).range(65..=180)
        )]
        timeout_seconds: u64,

        /// Confirm creation of one ephemeral frozen-v1 session
        #[arg(long)]
        confirm_v1_compatibility_smoke: bool,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum MemchainCommands {
    /// Verify AOF framing, content IDs, Merkle roots, and Block ancestry
    VerifyAof {
        /// Optional AOF path override
        #[arg(long)]
        path: Option<PathBuf>,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum RelayCustodyCommands {
    /// Verify and report aggregate private backup retention state
    Audit {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Authenticate the complete private custody-maintenance audit chain
    VerifyAudit {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Export the latest immutable custody checkpoint as a signed binary anchor
    CreateAuditAnchor {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// New path for the canonical binary anchor; existing files are refused
        #[arg(long)]
        output: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Verify one exact custody anchor offline against local trust pins
    VerifyAuditAnchor {
        /// Path to the canonical binary anchor frame
        #[arg(long)]
        input: PathBuf,

        /// Expected SHA-256 of the exact binary frame
        #[arg(long)]
        expected_sha256: String,

        /// Trusted producer node identity (64 hexadecimal characters)
        #[arg(long)]
        expected_node: String,

        /// Lowest checkpoint generation already trusted by this verifier
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
        minimum_checkpoint_generation: u64,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Persist and countersign one producer anchor on an independent node
    WitnessAuditAnchor {
        /// Path to this witness node's local configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Path to the producer's canonical binary anchor frame
        #[arg(long)]
        input: PathBuf,

        /// Expected SHA-256 of the exact producer anchor frame
        #[arg(long)]
        expected_sha256: String,

        /// Pinned producer node identity (64 hexadecimal characters)
        #[arg(long)]
        expected_producer: String,

        /// Lowest producer generation accepted on first witness observation
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
        minimum_checkpoint_generation: u64,

        /// New path for the signed witness receipt; existing files are refused
        #[arg(long)]
        output: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Verify one accepted independent witness receipt against its exact anchor
    VerifyAuditWitness {
        /// Path to the producer's canonical binary anchor frame
        #[arg(long)]
        anchor: PathBuf,

        /// Expected SHA-256 of the exact producer anchor frame
        #[arg(long)]
        anchor_sha256: String,

        /// Path to the canonical binary witness receipt
        #[arg(long)]
        receipt: PathBuf,

        /// Expected SHA-256 of the exact witness receipt frame
        #[arg(long)]
        receipt_sha256: String,

        /// Pinned producer node identity (64 hexadecimal characters)
        #[arg(long)]
        expected_producer: String,

        /// Pinned independent witness identity (64 hexadecimal characters)
        #[arg(long)]
        expected_witness: String,

        /// Lowest checkpoint generation trusted by this verifier
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
        minimum_checkpoint_generation: u64,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Import one pinned witness receipt into this producer's durable vault
    ImportAuditWitness {
        /// Path to this producer node's local configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Path to this producer's canonical binary anchor frame
        #[arg(long)]
        anchor: PathBuf,

        /// Expected SHA-256 of the exact producer anchor frame
        #[arg(long)]
        anchor_sha256: String,

        /// Path to the canonical binary witness receipt
        #[arg(long)]
        receipt: PathBuf,

        /// Expected SHA-256 of the exact witness receipt frame
        #[arg(long)]
        receipt_sha256: String,

        /// Configured independent witness identity (64 hexadecimal characters)
        #[arg(long)]
        expected_witness: String,

        /// Maximum accepted age of the signed witness observation
        #[arg(
            long,
            default_value_t = 7200,
            value_parser = clap::value_parser!(u64).range(60..=604800)
        )]
        max_age_seconds: u64,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Re-audit current-checkpoint witness receipts without network activity
    AuditWitnessVault {
        /// Path to this producer node's local configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Maximum accepted age of signed witness observations
        #[arg(
            long,
            default_value_t = 7200,
            value_parser = clap::value_parser!(u64).range(60..=604800)
        )]
        max_age_seconds: u64,

        /// Return failure unless the configured current-checkpoint policy is ready
        #[arg(long)]
        require_ready: bool,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Explicitly collect and persist current-checkpoint witness receipts
    CollectAuditWitnesses {
        /// Path to this producer node's local configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Signed discovery snapshot containing the configured witness descriptors
        #[arg(long)]
        discovery_snapshot: PathBuf,

        /// Complete per-witness request timeout
        #[arg(
            long,
            default_value_t = 15,
            value_parser = clap::value_parser!(u64).range(1..=60)
        )]
        timeout_seconds: u64,

        /// Maximum accepted age of signed witness observations
        #[arg(
            long,
            default_value_t = 7200,
            value_parser = clap::value_parser!(u64).range(60..=604800)
        )]
        max_age_seconds: u64,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Verify latest-backup restore readiness without changing storage
    RestoreReadiness {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Create a ten-minute state-bound restore plan without changing storage
    RestorePlan {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable path-free JSON plan contract
        #[arg(long)]
        json: bool,
    },

    /// Re-verify one private restore plan against current node storage state
    VerifyRestorePlan {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Owner-private JSON plan emitted by `restore-plan --json`
        #[arg(long)]
        plan_file: PathBuf,

        /// Emit a minimal verification result
        #[arg(long)]
        json: bool,
    },

    /// Dry-run retention by default; delete only after explicit confirmation
    Prune {
        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Delete verified policy candidates instead of only planning them
        #[arg(
            long,
            requires_all = ["confirm_node_stopped", "confirm_prune"]
        )]
        execute: bool,

        /// Confirm the serving node process has been stopped
        #[arg(long, requires = "execute")]
        confirm_node_stopped: bool,

        /// Must exactly equal PRUNE-VERIFIED-RELAY-BACKUPS
        #[arg(long, requires = "execute", value_name = "PHRASE")]
        confirm_prune: Option<String>,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum DirectoryReplicaCommands {
    /// Verify one exact portable observation certificate without network access
    VerifyObservationCertificate {
        /// Path to the canonical binary certificate frame
        #[arg(long)]
        input: PathBuf,

        /// Expected SHA-256 of the exact binary frame
        #[arg(long)]
        expected_sha256: String,

        /// Trusted observer node identity (64 hexadecimal characters)
        #[arg(long)]
        expected_observer: String,

        /// Trusted witness node identity; repeat for every allowed witness
        #[arg(long = "allowed-witness", required = true)]
        allowed_witnesses: Vec<String>,

        /// Locally required count of distinct trusted witness receipts
        #[arg(long)]
        minimum_witnesses: u16,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Verify and durably import one third-party observation certificate
    ImportObservationCertificate {
        /// Path to the canonical binary certificate frame
        #[arg(long)]
        input: PathBuf,

        /// Expected SHA-256 of the exact binary frame
        #[arg(long)]
        expected_sha256: String,

        /// Trusted external observer node identity (64 hexadecimal characters)
        #[arg(long)]
        expected_observer: String,

        /// Trusted witness node identity; repeat for every allowed witness
        #[arg(long = "allowed-witness", required = true)]
        allowed_witnesses: Vec<String>,

        /// Locally required count of distinct trusted witness receipts
        #[arg(long)]
        minimum_witnesses: u16,

        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Fetch, verify, and durably import a fresh certificate from a pinned node
    PullObservationCertificate {
        /// Public node endpoint serving authenticated Directory peer frames
        #[arg(long)]
        source_endpoint: String,

        /// Expected source and certificate-observer identity (64 hex characters)
        #[arg(long)]
        expected_observer: String,

        /// Trusted witness node identity; repeat for every allowed witness
        #[arg(long = "allowed-witness", required = true)]
        allowed_witnesses: Vec<String>,

        /// Locally required count of distinct trusted witness receipts
        #[arg(long)]
        minimum_witnesses: u16,

        /// Maximum accepted checkpoint age for network retrieval
        #[arg(long, default_value_t = 900)]
        max_age_seconds: u64,

        /// Path to the local node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Verify one explicit signed mirror carrier without importing evidence
    CarrierSmoke {
        /// Path to the running node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Prove an empty replica can bootstrap through an explicit public carrier
    CarrierColdBootstrapSmoke {
        /// Path to the running node configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,

        /// Emit the stable aggregate JSON contract
        #[arg(long)]
        json: bool,
    },

    /// Verify an incident and print its exact compare-and-swap state
    InspectIncident {
        /// Content-addressed incident digest (64 hexadecimal characters)
        #[arg(long)]
        digest: String,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },

    /// Resume one exact accepted prefix after explicit operator review
    ResolveQuarantine {
        /// Content-addressed incident digest
        #[arg(long)]
        digest: String,

        /// Quarantined producer identity
        #[arg(long)]
        producer: String,

        /// Accepted prefix height printed by `inspect-incident`
        #[arg(long)]
        expected_tip_height: u64,

        /// Accepted prefix hash printed by `inspect-incident`
        #[arg(long)]
        expected_tip_hash: String,

        /// Quarantine kind printed by `inspect-incident`
        #[arg(long)]
        expected_kind: String,

        /// Previous linked resolution digest, when one exists
        #[arg(long)]
        expected_previous_resolution_digest: Option<String>,

        /// Must exactly repeat `--digest` to prevent accidental execution
        #[arg(long)]
        confirm_incident: String,

        /// Path to configuration file
        #[arg(short, long, default_value = "/etc/aeronyx/server.toml")]
        config: PathBuf,
    },
}

// ============================================
// Main
// ============================================

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_logging("info");

    let result = match cli.command {
        Commands::Register {
            code,
            code_stdin,
            config,
            cms_url,
            node_name,
            region,
            public_vpn,
        } => match resolve_registration_code(code, code_stdin) {
            Ok(code) => cmd_register(code, config, cms_url, node_name, region, public_vpn).await,
            Err(error) => Err(error),
        },
        Commands::Start { config } => cmd_start(config).await,
        Commands::Status { config } => cmd_status(config).await,
        Commands::Validate { config } => cmd_validate(config).await,
        Commands::Memchain(command) => cmd_memchain(command).await,
        Commands::Pubkey { config, format } => cmd_pubkey(config, format).await,
        Commands::DirectoryReplica(command) => cmd_directory_replica(command).await,
        Commands::RelayCustody(command) => cmd_relay_custody(command).await,
        Commands::RelaySmoke {
            server,
            health_url,
            config,
            timeout_seconds,
            confirm_live_relay_smoke,
            json,
        } => {
            cmd_relay_smoke(
                server,
                health_url,
                config,
                timeout_seconds,
                confirm_live_relay_smoke,
                json,
            )
            .await
        }
        Commands::MailboxProbe {
            seeds,
            target,
            entry,
            work_bits,
            timeout_seconds,
            confirm_live_mailbox_probe,
            json,
        } => {
            cmd_mailbox_probe(
                mailbox_probe::MailboxProbeOptions {
                    seeds,
                    target,
                    entry,
                    work_bits,
                    timeout: std::time::Duration::from_secs(timeout_seconds),
                },
                confirm_live_mailbox_probe,
                json,
            )
            .await
        }
        Commands::V1CompatibilitySmoke {
            server,
            health_url,
            config,
            timeout_seconds,
            confirm_v1_compatibility_smoke,
            json,
        } => {
            cmd_v1_compatibility_smoke(
                server,
                health_url,
                config,
                timeout_seconds,
                confirm_v1_compatibility_smoke,
                json,
            )
            .await
        }
    };

    if let Err(e) = result {
        error!("{}", e);
        std::process::exit(1);
    }
}
