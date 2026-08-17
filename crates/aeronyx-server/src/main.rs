// ============================================
// File: crates/aeronyx-server/src/main.rs
// ============================================
//! # AeroNyx Server Entry Point
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

use std::fs::{File, OpenOptions};
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use clap::{Parser, Subcommand};
use rand::RngCore;
use sha2::{Digest, Sha256};
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::{
    decode_custody_audit_anchor, decode_custody_audit_witness_receipt, encode_custody_audit_anchor,
    encode_custody_audit_witness_receipt, CustodyAuditAnchorV1, CustodyAuditWitnessReceiptV1,
    CUSTODY_AUDIT_WITNESS_ADVANCED_V1, CUSTODY_AUDIT_WITNESS_CONFLICT_V1,
    CUSTODY_AUDIT_WITNESS_GAP_V1, CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
    CUSTODY_AUDIT_WITNESS_STALE_V1, MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
    MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
};
use aeronyx_core::protocol::discovery::MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES;
use aeronyx_server::api::auth::ensure_api_secret;
use aeronyx_server::api::directory_replica_sync::{
    build_directory_certificate_exchange_http_client, fetch_authenticated_observation_certificate,
};
use aeronyx_server::management::models::{NodeRegistrationProfile, StoredNodeInfo};
use aeronyx_server::services::chat_relay::ChatRelayCustodyAuditAnchorGuard;
use aeronyx_server::services::directory_replica::{
    verify_directory_observation_certificate_frame as verify_portable_observation_certificate_frame,
    DirectoryObservationCertificateTrustPolicy,
};
use aeronyx_server::services::memchain::{
    derive_record_key, CustodyAuditAnchorWitnessOutcome, CustodyAuditWitnessReceiptPolicyEvidence,
    MemoryStorage,
};
use aeronyx_server::services::{
    derive_node_secret, AofWriter, ChatRelayBackupPruneRequest, ChatRelayRestorePlanReceipt,
    ChatRelayRestoreReadinessReceipt, ChatRelayService, DirectoryReplicaResolutionCommand,
    DirectoryReplicaStore, DirectoryReplicaTip, CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
    CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS,
};
use aeronyx_server::{ManagementClient, Server, ServerConfig};

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
    };

    if let Err(e) = result {
        error!("{}", e);
        std::process::exit(1);
    }
}

// ============================================
// Commands
// ============================================

/// Maximum accepted one-time registration-code length after trimming.
const MAX_REGISTRATION_CODE_BYTES: usize = 128;

/// Runs one explicit, aggregate-only proof against the node on this host.
async fn cmd_relay_smoke(
    server_addr: std::net::SocketAddr,
    health_url: String,
    config_path: PathBuf,
    timeout_seconds: u64,
    confirmed: bool,
    emit_json: bool,
) -> anyhow::Result<()> {
    // [LIVE-RELAY-SMOKE 2026-08-15 by Codex] This command creates protocol
    // traffic and two ephemeral sessions. Requiring an explicit confirmation
    // prevents an operator from confusing a read-only health check with a live
    // end-to-end proof.
    anyhow::ensure!(confirmed, "relay smoke requires --confirm-live-relay-smoke");
    let config = ServerConfig::load(&config_path)
        .await
        .with_context(|| format!("load node config {}", config_path.display()))?;
    anyhow::ensure!(
        server_addr.port() == config.listen_addr().port(),
        "relay smoke UDP port does not match the configured node listener"
    );
    let key_path = PathBuf::from(&config.server_key.key_file);
    let expected_server_key = relay_smoke::load_expected_server_public_key(&key_path).await?;
    let report = relay_smoke::run(relay_smoke::RelaySmokeOptions {
        server_addr,
        health_url,
        expected_server_key,
        timeout: std::time::Duration::from_secs(timeout_seconds),
    })
    .await?;

    if emit_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("AeroNyx authenticated live relay smoke");
        println!("  Status:                         {}", report.status);
        println!("  Transport:                      {}", report.transport);
        println!(
            "  Verified client deliveries:     {} -> {}",
            report.verified_client_deliveries_before, report.verified_client_deliveries_after
        );
        println!(
            "  Terminal receipt observed:      {}",
            report.terminal_receipt_observed
        );
        println!(
            "  Entry mailbox round trip:       {}",
            report.entry_mailbox_round_trip_verified
        );
        println!(
            "  Entry mailbox ACK:              {}",
            report.entry_mailbox_ack_verified
        );
        println!(
            "  E2E ciphertext verified:        {}",
            report.e2e_ciphertext_verified
        );
        println!(
            "  Ephemeral sessions:             {}",
            report.ephemeral_sessions_created
        );
        println!(
            "  Session cleanup:                {}",
            report.session_cleanup
        );
        println!(
            "  Terminal replica cleanup:       {}",
            report.terminal_replica_cleanup
        );
        println!(
            "  Evidence scope:                 {}",
            report.evidence_scope
        );
        println!("  Elapsed:                        {} ms", report.elapsed_ms);
        println!(
            "  Privacy boundary:               {}",
            report.privacy_boundary
        );
    }
    Ok(())
}

/// Validates a registration code without logging or retaining surrounding input.
fn normalize_registration_code(value: &str) -> anyhow::Result<String> {
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "registration code cannot be empty");
    anyhow::ensure!(
        value.len() <= MAX_REGISTRATION_CODE_BYTES,
        "registration code cannot exceed {MAX_REGISTRATION_CODE_BYTES} bytes"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "registration code cannot contain control characters"
    );
    Ok(value.to_string())
}

/// Reads at most one bounded registration-code line from an anonymous stream.
fn read_registration_code<R: BufRead>(reader: R) -> anyhow::Result<String> {
    // [REGISTRATION-CODE-STDIN 2026-08-02 by Codex] `take` bounds allocation
    // before UTF-8 validation. A malformed or unbounded pipe must fail closed
    // without copying secret material into logs or process arguments.
    let mut bounded = reader.take((MAX_REGISTRATION_CODE_BYTES + 2) as u64);
    let mut line = String::new();
    bounded
        .read_line(&mut line)
        .context("failed to read registration code from standard input")?;
    normalize_registration_code(&line)
}

/// Resolves the backward-compatible CLI value or the secret-safe stdin mode.
fn resolve_registration_code(code: Option<String>, code_stdin: bool) -> anyhow::Result<String> {
    match (code, code_stdin) {
        (Some(code), false) => normalize_registration_code(&code),
        (None, true) => read_registration_code(std::io::stdin().lock()),
        _ => anyhow::bail!("provide exactly one of --code or --code-stdin"),
    }
}

/// Validates an optional node name before it reaches the one-time bind API.
fn normalize_registration_name(value: Option<String>) -> anyhow::Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "node name cannot be empty");
    anyhow::ensure!(
        value.chars().count() <= 100,
        "node name cannot exceed 100 characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "node name cannot contain control characters"
    );
    Ok(Some(value.to_string()))
}

/// Normalizes an optional ISO 3166-1 alpha-2 region code.
fn normalize_registration_region(value: Option<String>) -> anyhow::Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    anyhow::ensure!(
        value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_alphabetic()),
        "region must be a two-letter ISO 3166-1 alpha-2 code"
    );
    Ok(Some(value.to_ascii_uppercase()))
}

/// Registers node with CMS.
async fn cmd_register(
    code: String,
    config_path: PathBuf,
    cms_url_override: Option<String>,
    node_name: Option<String>,
    region: Option<String>,
    public_vpn: bool,
) -> anyhow::Result<()> {
    println!("🚀 AeroNyx Node Registration");
    println!("════════════════════════════════════════");
    println!();

    let config = load_or_default_config(&config_path).await;
    let key_path = PathBuf::from(&config.server_key.key_file);
    let node_info_path = &config.management.node_info_path;

    if std::path::Path::new(node_info_path).exists() {
        if let Ok(info) = StoredNodeInfo::load(node_info_path) {
            println!("⚠️  This node is already registered!");
            println!();
            println!("   Node ID:  {}", info.node_id);
            println!("   Name:     {}", info.name);
            println!("   Owner:    {}", info.owner_wallet);
            println!();
            println!("If you want to re-register, delete the file:");
            println!("   rm {node_info_path}");
            return Ok(());
        }
    }

    let identity = if key_path.exists() {
        info!("Loading existing node key...");
        load_key(&key_path).await?
    } else {
        info!("Generating secure node key...");
        let identity = IdentityKeyPair::generate();
        save_key(&identity, &key_path).await?;
        identity
    };

    let mut mgmt_config = config.management.clone();
    if let Some(url) = cms_url_override {
        mgmt_config.cms_url = url;
    }

    // [MANAGEMENT-CLIENT-STARTUP 2026-08-12 by Codex] Registration is a CLI
    // transaction: connector initialization must return an actionable error
    // before any remote request or local registration record is created.
    let client = ManagementClient::new(mgmt_config.clone(), identity)
        .context("Failed to initialize management HTTP client")?;
    let registration_profile = NodeRegistrationProfile {
        name: normalize_registration_name(node_name)?,
        port: Some(config.listen_addr().port()),
        region_code: normalize_registration_region(region)?,
        visibility: public_vpn.then(|| "public".to_string()),
        is_vpn_node: Some(true),
    };

    println!("📡 Connecting to AeroNyx network...");
    println!();

    match client
        .register_node_with_profile(&code, registration_profile)
        .await
    {
        Ok(node_info) => {
            let stored = StoredNodeInfo {
                node_id: node_info.id.clone(),
                owner_wallet: node_info.owner_wallet.clone(),
                name: node_info.name.clone(),
                registered_at: node_info.created_at.clone(),
            };
            stored.save(&mgmt_config.node_info_path)?;

            println!("✅ Registration successful!");
            println!();
            println!("════════════════════════════════════════");
            println!("   Node ID:  {}", node_info.id);
            println!("   Name:     {}", node_info.name);
            println!("   Owner:    {}", node_info.owner_wallet);
            println!("════════════════════════════════════════");
            println!();
            println!("🎉 Your node is ready! Start it with:");
            println!();
            println!("   aeronyx-server start");
            println!();
        }
        Err(e) => {
            println!("❌ Registration failed: {e}");
            println!();
            println!("Please check:");
            println!("  • Is the registration code correct?");
            println!("  • Has the code expired? (codes expire in 15 minutes)");
            println!("  • Is there network connectivity?");
            println!();
            println!("Get a new code from: https://app.aeronyx.network");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Starts the server.
///
/// v1.0.0-MultiTenant: passes `Some(config_path.clone())` to Server::new()
/// so auto-generated api_secret and jwt_secret are persisted to the config
/// file on first SaaS startup.
async fn cmd_start(config_path: PathBuf) -> anyhow::Result<()> {
    info!("Starting AeroNyx server...");

    let mut config = if config_path.exists() {
        ServerConfig::load(&config_path).await?
    } else {
        info!("Config file not found, using defaults");
        ServerConfig::default()
    };

    init_logging(&config.logging.level);

    // Resolve before constructing Server so first-start admin routes receive
    // the same secret that is persisted to `[memchain]`. The legacy writer ran
    // after config loading and left the current process unauthenticated.
    if config.memchain.is_enabled() {
        let persisted_path = config_path.exists().then_some(config_path.as_path());
        let api_secret = ensure_api_secret(config.memchain.effective_api_secret(), persisted_path)
            .map_err(anyhow::Error::msg)?;
        config.memchain.api_secret = Some(api_secret);
    }

    let key_path = PathBuf::from(&config.server_key.key_file);
    let node_info_path = &config.management.node_info_path;

    if !std::path::Path::new(node_info_path).exists() {
        println!();
        println!("❌ Node is not registered!");
        println!();
        println!("All nodes must be registered to join the AeroNyx network.");
        println!();
        println!("To register your node:");
        println!("  1. Get a registration code from https://app.aeronyx.network");
        println!("  2. Pipe it privately: printf '%s\\n' '<YOUR_CODE>' | aeronyx-server register --code-stdin");
        println!("     Legacy compatibility: aeronyx-server register --code <YOUR_CODE>");
        println!();
        std::process::exit(1);
    }

    let node_info = match StoredNodeInfo::load(node_info_path) {
        Ok(info) => info,
        Err(e) => {
            error!("Failed to load registration info: {}", e);
            println!();
            println!("❌ Registration data is corrupted.");
            println!();
            println!("Please re-register your node:");
            println!("  rm {node_info_path}");
            println!("  aeronyx-server register --code <YOUR_CODE>");
            std::process::exit(1);
        }
    };

    let identity = if key_path.exists() {
        load_key(&key_path).await?
    } else {
        println!();
        println!("❌ Server key not found!");
        println!();
        println!("The key file is missing. Please re-register your node:");
        println!("  aeronyx-server register --code <YOUR_CODE>");
        std::process::exit(1);
    };

    info!("════════════════════════════════════════");
    info!("Node ID:    {}", node_info.node_id);
    info!("Node Name:  {}", node_info.name);
    info!("Owner:      {}", node_info.owner_wallet);
    info!("════════════════════════════════════════");

    // v1.0.0-MultiTenant: pass config_path so auto-generated secrets
    // (api_secret, jwt_secret) are written back to disk on first startup.
    let server = Server::new(config, identity, Some(config_path.clone()));
    server.run().await?;

    Ok(())
}

/// Shows node registration status + MemChain status.
async fn cmd_status(config_path: PathBuf) -> anyhow::Result<()> {
    let config = load_or_default_config(&config_path).await;
    let node_info_path = &config.management.node_info_path;
    let key_path = PathBuf::from(&config.server_key.key_file);

    println!();
    println!("AeroNyx Node Status");
    println!("════════════════════════════════════════");
    println!();

    // Check registration
    match StoredNodeInfo::load(node_info_path) {
        Ok(info) => {
            println!("Registration:  ✅ Registered");
            println!();
            println!("   Node ID:       {}", info.node_id);
            println!("   Name:          {}", info.name);
            println!("   Owner:         {}", info.owner_wallet);
            println!("   Registered:    {}", info.registered_at);
        }
        Err(_) => {
            println!("Registration:  ❌ Not registered");
            println!();
            println!("Run this command to register:");
            println!("   aeronyx-server register --code <YOUR_CODE>");
            return Ok(());
        }
    }

    println!();

    // Check key file
    if key_path.exists() {
        match load_key(&key_path).await {
            Ok(identity) => {
                println!("Server Key:    ✅ Valid");
                println!(
                    "   Public Key:    {}",
                    hex::encode(identity.public_key_bytes())
                );
            }
            Err(_) => {
                println!("Server Key:    ⚠️  Invalid or corrupted");
            }
        }
    } else {
        println!("Server Key:    ❌ Missing");
    }

    println!();

    // MemChain Status
    println!("MemChain:");
    println!("   Mode:          {:?}", config.memchain.mode);

    if config.memchain.is_enabled() {
        println!("   API Address:   {}", config.memchain.api_listen_addr);
        println!("   AOF Path:      {}", config.memchain.aof_path);

        let aof_path = std::path::Path::new(&config.memchain.aof_path);
        if aof_path.exists() {
            match std::fs::metadata(aof_path) {
                Ok(meta) => {
                    let size_kb = meta.len() as f64 / 1024.0;
                    if size_kb < 1024.0 {
                        println!("   AOF Size:      {size_kb:.1} KB");
                    } else {
                        println!("   AOF Size:      {:.2} MB", size_kb / 1024.0);
                    }
                }
                Err(_) => {
                    println!("   AOF Size:      ⚠️  Could not read");
                }
            }
        } else {
            println!("   AOF File:      (not yet created — will be created on first write)");
        }
    } else {
        println!("   Status:        Disabled");
    }

    println!();
    println!("════════════════════════════════════════");
    println!();

    Ok(())
}

/// Validates configuration file + shows MemChain config.
async fn cmd_validate(config_path: PathBuf) -> anyhow::Result<()> {
    if !config_path.exists() {
        println!("⚠️  Config file not found: {}", config_path.display());
        println!("   Server will use default values.");
        return Ok(());
    }

    let config = ServerConfig::load(&config_path).await?;

    println!("✅ Configuration is valid");
    println!();
    println!("Network:");
    println!("   Listen:     {}", config.listen_addr());
    if let Some(ep) = &config.network.public_endpoint {
        println!("   Public:     {ep}");
    }
    println!();
    println!("AeroNyx Privacy Protocol:");
    println!("   IP Range:   {}", config.ip_range());
    println!("   Gateway:    {}", config.gateway_ip());
    println!();
    println!("TUN:");
    println!("   Device:     {}", config.device_name());
    println!("   MTU:        {}", config.mtu());
    println!();
    println!("Limits:");
    println!("   Max Connections:  {}", config.max_sessions());
    println!("   Session Timeout:  {}s", config.session_timeout_secs());
    println!();
    println!("MemChain:");
    println!("   Mode:             {:?}", config.memchain.mode);
    if config.memchain.is_enabled() {
        println!("   API Listen:       {}", config.memchain.api_listen_addr);
        println!("   AOF Path:         {}", config.memchain.aof_path);
    }
    println!();
    println!("Relay custody:");
    println!(
        "   Enabled:          {}",
        config.memchain.chat_relay.enabled
    );
    if config.memchain.chat_relay.enabled {
        // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Surface policy during
        // config validation without exposing artifact names or audit contents.
        println!(
            "   Retained backups: {}",
            config
                .memchain
                .chat_relay
                .custody_backup_retention_target_artifacts
        );
        println!(
            "   Retained bytes:   {}",
            config
                .memchain
                .chat_relay
                .custody_backup_retention_target_bytes
        );
        println!(
            "   Partial grace:    {}s",
            config.memchain.chat_relay.custody_backup_partial_grace_secs
        );
    }
    println!();
    println!("Discovery:");
    println!("   Enabled:          {}", config.discovery.enabled);
    if let Some(path) = &config.discovery.bootstrap_snapshot_path {
        println!("   Snapshot Path:    {path}");
    }
    if let Some(url) = &config.discovery.bootstrap_snapshot_url {
        println!("   Snapshot URL:     {url}");
        println!(
            "   Fetch Timeout:    {}s",
            config.discovery.fetch_timeout_secs
        );
    }
    println!();

    Ok(())
}

/// Runs host-local relay custody maintenance without opening an HTTP surface.
async fn cmd_relay_custody(command: RelayCustodyCommands) -> anyhow::Result<()> {
    // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Keep this boundary local:
    // it reads node-owned config/key material and never calls CMS or HTTP.
    // Plans and readiness checks cannot replace or remove custody data.
    match command {
        RelayCustodyCommands::Audit { config, json } => {
            let server_config = load_relay_custody_config(&config).await?;
            let receipt = ChatRelayService::audit_verified_backup_retention_for_config(
                &server_config.memchain.chat_relay,
            )
            .map_err(|error| anyhow::anyhow!("relay custody audit failed: {error}"))?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                println!("Relay custody retention audit");
                println!("════════════════════════════════════════");
                println!("Retained backups:   {}", receipt.retained_count);
                println!("Retained bytes:     {}", receipt.retained_bytes);
                println!("Excess backups:     {}", receipt.excess_count);
                println!("Excess bytes:       {}", receipt.excess_bytes);
                println!("Interrupted files:  {}", receipt.partial_count);
                println!("Interrupted bytes:  {}", receipt.partial_bytes);
                println!("Budget exceeded:    {}", receipt.budget_exceeded);
            }
        }
        RelayCustodyCommands::VerifyAudit { config, json } => {
            cmd_relay_verify_audit(&config, json).await?;
        }
        RelayCustodyCommands::CreateAuditAnchor {
            config,
            output,
            json,
        } => {
            cmd_relay_create_audit_anchor(&config, &output, json).await?;
        }
        RelayCustodyCommands::VerifyAuditAnchor {
            input,
            expected_sha256,
            expected_node,
            minimum_checkpoint_generation,
            json,
        } => {
            cmd_relay_verify_audit_anchor(
                &input,
                &expected_sha256,
                &expected_node,
                minimum_checkpoint_generation,
                json,
            )?;
        }
        RelayCustodyCommands::WitnessAuditAnchor {
            config,
            input,
            expected_sha256,
            expected_producer,
            minimum_checkpoint_generation,
            output,
            json,
        } => {
            cmd_relay_witness_audit_anchor(
                &config,
                &input,
                &expected_sha256,
                &expected_producer,
                minimum_checkpoint_generation,
                &output,
                json,
            )
            .await?;
        }
        RelayCustodyCommands::VerifyAuditWitness {
            anchor,
            anchor_sha256,
            receipt,
            receipt_sha256,
            expected_producer,
            expected_witness,
            minimum_checkpoint_generation,
            json,
        } => {
            cmd_relay_verify_audit_witness(
                &anchor,
                &anchor_sha256,
                &receipt,
                &receipt_sha256,
                &expected_producer,
                &expected_witness,
                minimum_checkpoint_generation,
                json,
            )?;
        }
        RelayCustodyCommands::ImportAuditWitness {
            config,
            anchor,
            anchor_sha256,
            receipt,
            receipt_sha256,
            expected_witness,
            max_age_seconds,
            json,
        } => {
            cmd_relay_import_audit_witness(
                &config,
                &anchor,
                &anchor_sha256,
                &receipt,
                &receipt_sha256,
                &expected_witness,
                max_age_seconds,
                json,
            )
            .await?;
        }
        RelayCustodyCommands::AuditWitnessVault {
            config,
            max_age_seconds,
            require_ready,
            json,
        } => {
            cmd_relay_audit_witness_vault(&config, max_age_seconds, require_ready, json).await?;
        }
        RelayCustodyCommands::RestoreReadiness { config, json } => {
            let server_config = load_relay_custody_config(&config).await?;
            let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(
                &server_config.memchain.chat_relay,
            )
            .map_err(|error| anyhow::anyhow!("relay custody restore preflight failed: {error}"))?;
            print_relay_restore_readiness(&receipt, json)?;
        }
        RelayCustodyCommands::RestorePlan { config, json } => {
            cmd_relay_restore_plan(&config, json).await?;
        }
        RelayCustodyCommands::VerifyRestorePlan {
            config,
            plan_file,
            json,
        } => {
            cmd_relay_verify_restore_plan(&config, &plan_file, json).await?;
        }
        RelayCustodyCommands::Prune {
            config,
            execute,
            confirm_node_stopped,
            confirm_prune,
            json,
        } => {
            let server_config = load_relay_custody_config(&config).await?;
            let node_secret = load_relay_custody_node_secret(&server_config, "prune").await?;
            let request = ChatRelayBackupPruneRequest {
                execute,
                confirmation: confirm_prune,
                node_stopped_confirmed: confirm_node_stopped,
            };
            let receipt = ChatRelayService::prune_verified_backup_retention_for_config(
                &server_config.memchain.chat_relay,
                &node_secret,
                &request,
            )
            .map_err(|error| anyhow::anyhow!("relay custody prune failed: {error}"))?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                println!(
                    "Relay custody retention {}",
                    if receipt.executed { "prune" } else { "dry-run" }
                );
                println!("════════════════════════════════════════");
                println!("Planned backups:    {}", receipt.planned_backup_count);
                println!("Planned bytes:      {}", receipt.planned_backup_bytes);
                println!("Planned partials:   {}", receipt.planned_partial_count);
                println!("Partial bytes:      {}", receipt.planned_partial_bytes);
                println!("Deleted backups:    {}", receipt.deleted_backup_count);
                println!("Deleted bytes:      {}", receipt.deleted_backup_bytes);
                println!("Deleted partials:   {}", receipt.deleted_partial_count);
                println!("Partial bytes freed: {}", receipt.deleted_partial_bytes);
                println!("Remaining backups:  {}", receipt.remaining.retained_count);
                println!("Remaining excess:   {}", receipt.remaining.excess_count);
                if !receipt.executed {
                    println!();
                    println!("Dry-run only; no recovery artifact was deleted.");
                    println!(
                        "Execution requires --execute --confirm-node-stopped --confirm-prune {CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION}"
                    );
                }
            }
        }
    }
    Ok(())
}

// [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Verification owns no network
// client and emits only fixed aggregate fields. Keep node-secret loading and
// chain authentication outside the general command dispatcher.
async fn cmd_relay_verify_audit(config_path: &Path, json: bool) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let node_secret =
        load_relay_custody_node_secret(&server_config, "maintenance audit verification").await?;
    let receipt = ChatRelayService::verify_backup_maintenance_audit_for_config(
        &server_config.memchain.chat_relay,
        &node_secret,
    )
    .map_err(|error| anyhow::anyhow!("relay custody audit verification failed: {error}"))?;
    if json {
        println!("{}", serde_json::to_string(&receipt)?);
    } else {
        println!("Relay custody maintenance audit");
        println!("════════════════════════════════════════");
        println!("Verified:            {}", receipt.verified);
        println!("Records:             {}", receipt.record_count);
        println!(
            "Last recorded at:    {}",
            receipt.last_recorded_at.unwrap_or(0)
        );
        println!("Dry runs:            {}", receipt.dry_run_count);
        println!("Planned executions:  {}", receipt.planned_count);
        println!("Completed:           {}", receipt.completed_count);
        println!("Failed:              {}", receipt.failed_count);
        println!("Verified bytes:      {}", receipt.verified_bytes);
        println!("Checkpoints:         {}", receipt.checkpoint_count);
        println!("Archived records:    {}", receipt.archived_record_count);
        println!("Active records:      {}", receipt.active_record_count);
        println!("Archived bytes:      {}", receipt.archived_bytes);
        println!("Rotation pending:    {}", receipt.rotation_pending);
        println!();
        println!("Read-only verification; no audit or custody data was changed.");
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditAnchorReport {
    contract_version: &'static str,
    status: &'static str,
    protocol_version: u8,
    producer_node_id: String,
    checkpoint_generation: u64,
    archived_record_count: u64,
    archived_bytes: u64,
    anchor_digest: String,
    frame_sha256: String,
    frame_bytes: usize,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

// [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] Export remains host-local and
// create-new. The exact frame digest is intended for a separately administered
// retainer; the producer must not silently replace evidence already retained.
async fn cmd_relay_create_audit_anchor(
    config_path: &Path,
    output_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let identity = load_relay_custody_identity(&server_config, "audit anchor export").await?;
    let anchor = ChatRelayService::create_backup_maintenance_audit_anchor_for_config(
        &server_config.memchain.chat_relay,
        &identity,
    )
    .map_err(|error| anyhow::anyhow!("relay custody audit anchor export failed: {error}"))?;
    let frame = encode_custody_audit_anchor(&anchor)
        .map_err(|_| anyhow::anyhow!("unable to encode relay custody audit anchor"))?;
    anyhow::ensure!(
        !frame.is_empty() && frame.len() <= MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "encoded relay custody audit anchor violates its protocol bound"
    );
    let frame_sha256: [u8; 32] = Sha256::digest(&frame).into();
    write_new_relay_custody_anchor(output_path, &frame)?;
    print_relay_custody_audit_anchor(&anchor, &frame_sha256, frame.len(), "created", json)
}

fn cmd_relay_verify_audit_anchor(
    input_path: &Path,
    expected_sha256_hex: &str,
    expected_node_hex: &str,
    minimum_checkpoint_generation: u64,
    json: bool,
) -> anyhow::Result<()> {
    let expected_sha256 = parse_hex32(expected_sha256_hex, "audit anchor SHA-256")?;
    let expected_node = parse_hex32(expected_node_hex, "expected producer node identity")?;
    let frame = read_bounded_relay_custody_anchor(input_path)?;
    let anchor = verify_relay_custody_anchor_frame(
        &frame,
        &expected_sha256,
        &expected_node,
        minimum_checkpoint_generation,
    )?;
    print_relay_custody_audit_anchor(&anchor, &expected_sha256, frame.len(), "verified", json)
}

fn verify_relay_custody_anchor_frame(
    frame: &[u8],
    expected_sha256: &[u8; 32],
    expected_node: &[u8; 32],
    minimum_checkpoint_generation: u64,
) -> anyhow::Result<CustodyAuditAnchorV1> {
    anyhow::ensure!(
        !frame.is_empty() && frame.len() <= MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "relay custody audit anchor violates its complete-frame bound"
    );
    let actual_sha256: [u8; 32] = Sha256::digest(frame).into();
    anyhow::ensure!(
        &actual_sha256 == expected_sha256,
        "relay custody audit anchor SHA-256 does not match the explicit pin"
    );
    let anchor = decode_custody_audit_anchor(frame)
        .map_err(|_| anyhow::anyhow!("relay custody audit anchor is malformed"))?;
    let canonical = encode_custody_audit_anchor(&anchor)
        .map_err(|_| anyhow::anyhow!("relay custody audit anchor cannot be canonicalized"))?;
    anyhow::ensure!(
        canonical == frame,
        "relay custody audit anchor is not canonically encoded"
    );
    anchor
        .verify_expected(expected_node, minimum_checkpoint_generation)
        .map_err(|_| anyhow::anyhow!("relay custody audit anchor trust policy failed"))?;
    Ok(anchor)
}

fn print_relay_custody_audit_anchor(
    anchor: &CustodyAuditAnchorV1,
    frame_sha256: &[u8; 32],
    frame_bytes: usize,
    status: &'static str,
    json: bool,
) -> anyhow::Result<()> {
    let report = RelayCustodyAuditAnchorReport {
        contract_version: "relay_custody_audit_anchor.v1",
        status,
        protocol_version: anchor.version,
        producer_node_id: hex::encode(anchor.producer_node_id),
        checkpoint_generation: anchor.checkpoint_generation,
        archived_record_count: anchor.archived_record_count,
        archived_bytes: anchor.archived_bytes,
        anchor_digest: hex::encode(anchor.anchor_digest),
        frame_sha256: hex::encode(frame_sha256),
        frame_bytes,
        security_model: "producer-signed opaque checkpoint commitment with explicit identity, exact-frame digest, and verifier-owned rollback floor; not an independent witness receipt, validator vote, consensus, or global finality",
        privacy_boundary: "checkpoint generation and aggregate archived record/byte counts only; no private HMAC, path, operation id, message id, endpoint, route, identity owner, payload, ciphertext, memory, destination, DNS, or social graph metadata",
    };
    if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("Relay custody audit anchor");
        println!("════════════════════════════════════════");
        println!("Status:               {}", report.status);
        println!("Producer node:        {}", report.producer_node_id);
        println!("Checkpoint generation: {}", report.checkpoint_generation);
        println!("Archived records:     {}", report.archived_record_count);
        println!("Archived bytes:       {}", report.archived_bytes);
        println!("Anchor digest:        {}", report.anchor_digest);
        println!("Frame SHA-256:        {}", report.frame_sha256);
        println!("Frame bytes:          {}", report.frame_bytes);
        println!();
        println!("Security model: {}", report.security_model);
        println!("Privacy: {}", report.privacy_boundary);
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessReport {
    contract_version: &'static str,
    status: &'static str,
    accepted: bool,
    outcome: &'static str,
    producer_node_id: String,
    witness_node_id: String,
    checkpoint_generation: u64,
    observed_at: u64,
    retained_checkpoint_generation: u64,
    anchor_frame_sha256: String,
    retained_frame_sha256: String,
    receipt_sha256: String,
    receipt_bytes: usize,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessImportReport {
    contract_version: &'static str,
    status: &'static str,
    import_disposition: &'static str,
    receipt_outcome: &'static str,
    checkpoint_generation: u64,
    observed_at: u64,
    vault_records: usize,
    vault_accepted_records: usize,
    vault_adverse_records: usize,
    configured_witnesses: usize,
    fresh_verified: usize,
    accepted: usize,
    adverse: usize,
    missing: usize,
    minimum_verified: usize,
    policy_ready: bool,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct RelayCustodyAuditWitnessVaultReport {
    contract_version: &'static str,
    status: &'static str,
    evaluated_at: u64,
    checkpoint_generation: u64,
    max_age_seconds: u64,
    vault_records: usize,
    vault_accepted_records: usize,
    vault_adverse_records: usize,
    configured_witnesses: usize,
    fresh_verified: usize,
    accepted: usize,
    adverse: usize,
    missing: usize,
    minimum_verified: usize,
    policy_ready: bool,
    required_ready: bool,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

/// Current producer custody context protected from concurrent checkpoint change.
///
/// [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Both import and audit use
/// this single boundary so identity, configured pins, current anchor generation,
/// and canonical frame digest cannot be checked under different policies.
struct CurrentRelayCustodyAuditWitnessContext {
    config: ServerConfig,
    identity: IdentityKeyPair,
    producer: [u8; 32],
    configured_witnesses: Vec<[u8; 32]>,
    anchor_guard: ChatRelayCustodyAuditAnchorGuard,
    anchor_sha256: [u8; 32],
}

/// Fully verified pre-persistence context for one producer receipt import.
///
/// [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] File, identity, pin,
/// signature, and exact-current-anchor checks complete before this value can
/// exist. The value keeps the cross-process maintenance lock alive, while
/// receipt-vault mutation remains a separate phase in the command handler.
struct VerifiedRelayCustodyAuditWitnessImport {
    current: CurrentRelayCustodyAuditWitnessContext,
    receipt: CustodyAuditWitnessReceiptV1,
}

#[allow(clippy::too_many_arguments)]
async fn cmd_relay_witness_audit_anchor(
    config_path: &Path,
    input_path: &Path,
    expected_sha256_hex: &str,
    expected_producer_hex: &str,
    minimum_checkpoint_generation: u64,
    output_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let config = ServerConfig::load(config_path).await?;
    anyhow::ensure!(
        config.memchain.is_enabled() && config.memchain.db_path.trim() != ":memory:",
        "custody audit witnessing requires persistent local MemChain storage"
    );
    let witness_identity = load_relay_custody_identity(&config, "audit witness").await?;
    let expected_sha256 = parse_hex32(expected_sha256_hex, "audit anchor SHA-256")?;
    let expected_producer = parse_hex32(expected_producer_hex, "expected producer node identity")?;
    anyhow::ensure!(
        witness_identity.public_key_bytes() != expected_producer,
        "custody audit anchor must be witnessed by an independent node identity"
    );

    let anchor_frame = read_bounded_relay_custody_artifact(
        input_path,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )?;
    let anchor = verify_relay_custody_anchor_frame(
        &anchor_frame,
        &expected_sha256,
        &expected_producer,
        minimum_checkpoint_generation,
    )?;
    let observed_at = unix_timestamp_now()?;
    anyhow::ensure!(observed_at > 0, "custody audit witness clock is invalid");

    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Persist the witness
    // high-water decision before signing or publishing its receipt. A failed
    // output write is safely retryable as an idempotent observation.
    let record_key = derive_record_key(&witness_identity.to_bytes());
    let storage = MemoryStorage::open(&config.memchain.db_path, Some(record_key))
        .map_err(|_| anyhow::anyhow!("unable to open custody audit witness storage"))?;
    let decision = storage
        .witness_custody_audit_anchor(
            &expected_producer,
            anchor.checkpoint_generation,
            &expected_sha256,
            observed_at,
        )
        .await
        .map_err(|_| anyhow::anyhow!("unable to persist custody audit witness decision"))?;
    let (outcome, retained_generation, retained_sha256) =
        custody_audit_witness_decision_fields(decision);
    let receipt = CustodyAuditWitnessReceiptV1::signed(
        expected_producer,
        anchor.checkpoint_generation,
        expected_sha256,
        observed_at,
        retained_generation,
        retained_sha256,
        outcome,
        &witness_identity,
    )
    .map_err(|_| anyhow::anyhow!("unable to sign custody audit witness receipt"))?;
    let receipt_frame = encode_custody_audit_witness_receipt(&receipt)
        .map_err(|_| anyhow::anyhow!("unable to encode custody audit witness receipt"))?;
    let receipt_sha256: [u8; 32] = Sha256::digest(&receipt_frame).into();
    write_new_relay_custody_artifact(
        output_path,
        &receipt_frame,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )?;
    print_relay_custody_audit_witness(
        &receipt,
        &receipt_sha256,
        receipt_frame.len(),
        "created",
        json,
    )?;
    anyhow::ensure!(
        receipt.accepted(),
        "custody audit witness rejected the producer anchor; retain the signed negative receipt"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_relay_verify_audit_witness(
    anchor_path: &Path,
    anchor_sha256_hex: &str,
    receipt_path: &Path,
    receipt_sha256_hex: &str,
    expected_producer_hex: &str,
    expected_witness_hex: &str,
    minimum_checkpoint_generation: u64,
    json: bool,
) -> anyhow::Result<()> {
    let anchor_sha256 = parse_hex32(anchor_sha256_hex, "audit anchor SHA-256")?;
    let receipt_sha256 = parse_hex32(receipt_sha256_hex, "audit witness receipt SHA-256")?;
    let expected_producer = parse_hex32(expected_producer_hex, "expected producer node identity")?;
    let expected_witness = parse_hex32(expected_witness_hex, "expected witness node identity")?;
    anyhow::ensure!(
        expected_producer != expected_witness,
        "producer and independent witness identities must differ"
    );

    let anchor_frame = read_bounded_relay_custody_artifact(
        anchor_path,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )?;
    let anchor = verify_relay_custody_anchor_frame(
        &anchor_frame,
        &anchor_sha256,
        &expected_producer,
        minimum_checkpoint_generation,
    )?;
    let receipt_frame = read_bounded_relay_custody_artifact(
        receipt_path,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )?;
    let receipt = verify_relay_custody_witness_receipt_frame(
        &receipt_frame,
        &receipt_sha256,
        &anchor,
        &anchor_sha256,
        &expected_producer,
        &expected_witness,
        minimum_checkpoint_generation,
    )?;
    anyhow::ensure!(
        receipt.accepted(),
        "custody audit witness receipt does not prove accepted custody"
    );
    print_relay_custody_audit_witness(
        &receipt,
        &receipt_sha256,
        receipt_frame.len(),
        "verified",
        json,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_relay_custody_witness_receipt_frame(
    receipt_frame: &[u8],
    expected_receipt_sha256: &[u8; 32],
    anchor: &CustodyAuditAnchorV1,
    anchor_sha256: &[u8; 32],
    expected_producer: &[u8; 32],
    expected_witness: &[u8; 32],
    minimum_checkpoint_generation: u64,
) -> anyhow::Result<CustodyAuditWitnessReceiptV1> {
    anyhow::ensure!(
        !receipt_frame.is_empty()
            && receipt_frame.len() <= MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "custody audit witness receipt violates its complete-frame bound"
    );
    let actual_receipt_sha256: [u8; 32] = Sha256::digest(receipt_frame).into();
    anyhow::ensure!(
        &actual_receipt_sha256 == expected_receipt_sha256,
        "custody audit witness receipt SHA-256 does not match the explicit pin"
    );
    let receipt = decode_custody_audit_witness_receipt(receipt_frame)
        .map_err(|_| anyhow::anyhow!("custody audit witness receipt is malformed"))?;
    let canonical = encode_custody_audit_witness_receipt(&receipt)
        .map_err(|_| anyhow::anyhow!("custody audit witness receipt cannot be canonicalized"))?;
    anyhow::ensure!(
        canonical == receipt_frame,
        "custody audit witness receipt is not canonically encoded"
    );
    receipt
        .verify_for_anchor(
            anchor,
            anchor_sha256,
            expected_producer,
            expected_witness,
            minimum_checkpoint_generation,
        )
        .map_err(|_| anyhow::anyhow!("custody audit witness receipt trust policy failed"))?;
    Ok(receipt)
}

async fn load_current_relay_custody_audit_witness_context(
    config_path: &Path,
    operation: &'static str,
) -> anyhow::Result<CurrentRelayCustodyAuditWitnessContext> {
    let config = load_relay_custody_config(config_path).await?;
    anyhow::ensure!(
        config.memchain.is_enabled() && config.memchain.db_path.trim() != ":memory:",
        "custody witness receipt policy requires persistent local MemChain storage"
    );
    let identity = load_relay_custody_identity(&config, operation).await?;
    let producer = identity.public_key_bytes();
    let configured_witnesses = config.discovery.custody_audit_witness_node_id_bytes();
    anyhow::ensure!(
        !configured_witnesses.is_empty(),
        "custody witness receipt policy has no configured independent witnesses"
    );

    // [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Keep the exact
    // checkpoint immutable through every local policy query. A concurrent
    // backup-maintenance process cannot make the final report stale mid-command.
    let anchor_guard = ChatRelayService::hold_backup_maintenance_audit_anchor_for_config(
        &config.memchain.chat_relay,
        &identity,
    )
    .map_err(|_| anyhow::anyhow!("unable to hold current relay custody anchor"))?;
    let current_anchor_frame = encode_custody_audit_anchor(anchor_guard.anchor())
        .map_err(|_| anyhow::anyhow!("unable to encode current relay custody anchor"))?;
    let anchor_sha256: [u8; 32] = Sha256::digest(&current_anchor_frame).into();
    Ok(CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
async fn verify_relay_custody_audit_witness_import(
    config_path: &Path,
    anchor_path: &Path,
    anchor_sha256_hex: &str,
    receipt_path: &Path,
    receipt_sha256_hex: &str,
    expected_witness_hex: &str,
) -> anyhow::Result<VerifiedRelayCustodyAuditWitnessImport> {
    let current =
        load_current_relay_custody_audit_witness_context(config_path, "audit witness import")
            .await?;
    let expected_witness = parse_hex32(expected_witness_hex, "expected witness node identity")?;
    anyhow::ensure!(
        expected_witness != current.producer,
        "producer and independent witness identities must differ"
    );
    anyhow::ensure!(
        current.configured_witnesses.contains(&expected_witness),
        "witness identity is not pinned by discovery.custody_audit_witness_node_ids"
    );

    let anchor_sha256 = parse_hex32(anchor_sha256_hex, "audit anchor SHA-256")?;
    let anchor_frame = read_bounded_relay_custody_artifact(
        anchor_path,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )?;
    let anchor =
        verify_relay_custody_anchor_frame(&anchor_frame, &anchor_sha256, &current.producer, 1)?;
    anyhow::ensure!(
        current.anchor_guard.anchor().checkpoint_generation == anchor.checkpoint_generation
            && current.anchor_sha256 == anchor_sha256,
        "custody witness receipt anchor is not the current local checkpoint"
    );

    let receipt_sha256 = parse_hex32(receipt_sha256_hex, "audit witness receipt SHA-256")?;
    let receipt_frame = read_bounded_relay_custody_artifact(
        receipt_path,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )?;
    let receipt = verify_relay_custody_witness_receipt_frame(
        &receipt_frame,
        &receipt_sha256,
        &anchor,
        &anchor_sha256,
        &current.producer,
        &expected_witness,
        anchor.checkpoint_generation,
    )?;
    Ok(VerifiedRelayCustodyAuditWitnessImport { current, receipt })
}

fn print_relay_custody_audit_witness_import(
    report: &RelayCustodyAuditWitnessImportReport,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(report)?);
        return Ok(());
    }
    println!("Relay custody witness receipt import");
    println!("════════════════════════════════════════");
    println!("Status:               {}", report.status);
    println!("Import disposition:   {}", report.import_disposition);
    println!("Receipt outcome:      {}", report.receipt_outcome);
    println!("Checkpoint generation: {}", report.checkpoint_generation);
    println!("Vault records:        {}", report.vault_records);
    println!("Configured witnesses: {}", report.configured_witnesses);
    println!("Fresh verified:       {}", report.fresh_verified);
    println!(
        "Accepted / adverse:   {} / {}",
        report.accepted, report.adverse
    );
    println!("Missing:              {}", report.missing);
    println!("Minimum verified:     {}", report.minimum_verified);
    println!("Policy ready:         {}", report.policy_ready);
    println!();
    println!("Security model: {}", report.security_model);
    println!("Privacy: {}", report.privacy_boundary);
    Ok(())
}

fn open_relay_custody_witness_storage(
    config: &ServerConfig,
    identity: &IdentityKeyPair,
) -> anyhow::Result<MemoryStorage> {
    let record_key = derive_record_key(&identity.to_bytes());
    MemoryStorage::open(&config.memchain.db_path, Some(record_key))
        .map_err(|_| anyhow::anyhow!("unable to open custody witness receipt vault"))
}

const fn custody_audit_witness_policy_status(
    policy: &CustodyAuditWitnessReceiptPolicyEvidence,
) -> &'static str {
    if policy.adverse > 0 {
        "adverse"
    } else if policy.quorum_satisfied {
        "ready"
    } else {
        "collecting"
    }
}

fn print_relay_custody_audit_witness_vault(
    report: &RelayCustodyAuditWitnessVaultReport,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(report)?);
        return Ok(());
    }
    println!("Relay custody witness vault audit");
    println!("════════════════════════════════════════");
    println!("Status:               {}", report.status);
    println!("Evaluated at:         {}", report.evaluated_at);
    println!("Checkpoint generation: {}", report.checkpoint_generation);
    println!("Freshness window:     {}s", report.max_age_seconds);
    println!("Vault records:        {}", report.vault_records);
    println!(
        "Vault accepted/adverse: {} / {}",
        report.vault_accepted_records, report.vault_adverse_records
    );
    println!("Configured witnesses: {}", report.configured_witnesses);
    println!("Fresh verified:       {}", report.fresh_verified);
    println!(
        "Accepted / adverse:   {} / {}",
        report.accepted, report.adverse
    );
    println!("Missing:              {}", report.missing);
    println!("Minimum verified:     {}", report.minimum_verified);
    println!("Policy ready:         {}", report.policy_ready);
    println!("Ready required:       {}", report.required_ready);
    println!();
    println!("Security model: {}", report.security_model);
    println!("Privacy: {}", report.privacy_boundary);
    Ok(())
}

async fn cmd_relay_audit_witness_vault(
    config_path: &Path,
    max_age_seconds: u64,
    require_ready: bool,
    json: bool,
) -> anyhow::Result<()> {
    let current =
        load_current_relay_custody_audit_witness_context(config_path, "audit witness vault")
            .await?;
    let CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    } = current;
    let evaluated_at = unix_timestamp_now()?;
    let storage = open_relay_custody_witness_storage(&config, &identity)?;
    let vault = storage
        .audit_custody_audit_witness_receipt_evidence()
        .await
        .map_err(|_| anyhow::anyhow!("custody witness receipt vault audit failed closed"))?;
    let policy = storage
        .evaluate_custody_audit_witness_receipt_policy(
            &producer,
            anchor_guard.anchor().checkpoint_generation,
            &anchor_sha256,
            &configured_witnesses,
            config.discovery.custody_audit_witness_min_verified,
            evaluated_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness receipt policy audit failed closed"))?;
    let status = custody_audit_witness_policy_status(&policy);
    let policy_ready = status == "ready";
    let report = RelayCustodyAuditWitnessVaultReport {
        contract_version: "relay_custody_audit_witness_vault.v1",
        status,
        evaluated_at,
        checkpoint_generation: anchor_guard.anchor().checkpoint_generation,
        max_age_seconds,
        vault_records: vault.records,
        vault_accepted_records: vault.accepted_records,
        vault_adverse_records: vault.adverse_records,
        configured_witnesses: policy.configured,
        fresh_verified: policy.fresh_verified,
        accepted: policy.accepted,
        adverse: policy.adverse,
        missing: policy.missing,
        minimum_verified: policy.minimum_verified,
        policy_ready,
        required_ready: require_ready,
        security_model: "host-local current-checkpoint receipt re-audit under an exclusive maintenance guard; no witness contact, consensus, voting, fork choice, or global finality",
        privacy_boundary: "aggregate vault and current policy counts only; no node identities, hashes, signatures, paths, endpoints, messages, users, routes, payloads, memory, destinations, DNS, IP addresses, or social graph metadata",
    };
    print_relay_custody_audit_witness_vault(&report, json)?;
    anyhow::ensure!(
        !require_ready || policy_ready,
        "current custody witness policy is not ready"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_relay_import_audit_witness(
    config_path: &Path,
    anchor_path: &Path,
    anchor_sha256_hex: &str,
    receipt_path: &Path,
    receipt_sha256_hex: &str,
    expected_witness_hex: &str,
    max_age_seconds: u64,
    json: bool,
) -> anyhow::Result<()> {
    let verified = verify_relay_custody_audit_witness_import(
        config_path,
        anchor_path,
        anchor_sha256_hex,
        receipt_path,
        receipt_sha256_hex,
        expected_witness_hex,
    )
    .await?;
    let VerifiedRelayCustodyAuditWitnessImport { current, receipt } = verified;
    let CurrentRelayCustodyAuditWitnessContext {
        config,
        identity,
        producer,
        configured_witnesses,
        anchor_guard,
        anchor_sha256,
    } = current;
    let anchor = anchor_guard.anchor();
    let imported_at = unix_timestamp_now()?;
    let storage = open_relay_custody_witness_storage(&config, &identity)?;
    let disposition = storage
        .import_custody_audit_witness_receipt(
            &receipt,
            &producer,
            anchor.checkpoint_generation,
            &anchor_sha256,
            imported_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness receipt import failed closed"))?;
    let vault = storage
        .audit_custody_audit_witness_receipt_evidence()
        .await
        .map_err(|_| anyhow::anyhow!("custody witness receipt vault audit failed closed"))?;
    let policy = storage
        .evaluate_custody_audit_witness_receipt_policy(
            &producer,
            anchor.checkpoint_generation,
            &anchor_sha256,
            &configured_witnesses,
            config.discovery.custody_audit_witness_min_verified,
            imported_at,
            max_age_seconds,
        )
        .await
        .map_err(|_| anyhow::anyhow!("custody witness receipt policy audit failed closed"))?;
    let status = custody_audit_witness_policy_status(&policy);
    let report = RelayCustodyAuditWitnessImportReport {
        contract_version: "relay_custody_audit_witness_import.v1",
        status,
        import_disposition: disposition.as_str(),
        receipt_outcome: custody_audit_witness_outcome_label(receipt.outcome),
        checkpoint_generation: anchor.checkpoint_generation,
        observed_at: receipt.observed_at,
        vault_records: vault.records,
        vault_accepted_records: vault.accepted_records,
        vault_adverse_records: vault.adverse_records,
        configured_witnesses: policy.configured,
        fresh_verified: policy.fresh_verified,
        accepted: policy.accepted,
        adverse: policy.adverse,
        missing: policy.missing,
        minimum_verified: policy.minimum_verified,
        policy_ready: status == "ready",
        security_model: "host-local exact-current-anchor import; canonical signed receipts are re-audited before policy evaluation and do not establish consensus or global finality",
        privacy_boundary: "aggregate vault and exact-anchor policy counts only; no message, user, route, endpoint, IP, payload, memory, DNS, destination, or social graph metadata",
    };
    print_relay_custody_audit_witness_import(&report, json)?;
    anyhow::ensure!(
        receipt.accepted(),
        "signed adverse custody witness evidence was retained for operator review"
    );
    Ok(())
}

const fn custody_audit_witness_decision_fields(
    decision: CustodyAuditAnchorWitnessOutcome,
) -> (u8, u64, [u8; 32]) {
    match decision {
        CustodyAuditAnchorWitnessOutcome::Advanced {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_ADVANCED_V1, generation, anchor_digest),
        CustodyAuditAnchorWitnessOutcome::Idempotent {
            generation,
            anchor_digest,
        } => (
            CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
            generation,
            anchor_digest,
        ),
        CustodyAuditAnchorWitnessOutcome::Stale {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_STALE_V1, generation, anchor_digest),
        CustodyAuditAnchorWitnessOutcome::Conflict {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_CONFLICT_V1, generation, anchor_digest),
        CustodyAuditAnchorWitnessOutcome::Gap {
            generation,
            anchor_digest,
        } => (CUSTODY_AUDIT_WITNESS_GAP_V1, generation, anchor_digest),
    }
}

const fn custody_audit_witness_outcome_label(outcome: u8) -> &'static str {
    match outcome {
        CUSTODY_AUDIT_WITNESS_ADVANCED_V1 => "advanced",
        CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1 => "idempotent",
        CUSTODY_AUDIT_WITNESS_STALE_V1 => "stale",
        CUSTODY_AUDIT_WITNESS_CONFLICT_V1 => "conflict",
        CUSTODY_AUDIT_WITNESS_GAP_V1 => "gap",
        _ => "invalid",
    }
}

fn print_relay_custody_audit_witness(
    receipt: &CustodyAuditWitnessReceiptV1,
    receipt_sha256: &[u8; 32],
    receipt_bytes: usize,
    status: &'static str,
    json: bool,
) -> anyhow::Result<()> {
    let report = RelayCustodyAuditWitnessReport {
        contract_version: "relay_custody_audit_witness.v1",
        status,
        accepted: receipt.accepted(),
        outcome: custody_audit_witness_outcome_label(receipt.outcome),
        producer_node_id: hex::encode(receipt.producer_node_id),
        witness_node_id: hex::encode(receipt.witness_node_id),
        checkpoint_generation: receipt.requested_checkpoint_generation,
        observed_at: receipt.observed_at,
        retained_checkpoint_generation: receipt.retained_checkpoint_generation,
        anchor_frame_sha256: hex::encode(receipt.requested_frame_sha256),
        retained_frame_sha256: hex::encode(receipt.retained_frame_sha256),
        receipt_sha256: hex::encode(receipt_sha256),
        receipt_bytes,
        security_model: "independent node signature over a durable producer-scoped monotonic decision; accepted receipts detect later rollback only while the witness high-water state and verifier pins remain available; not consensus or global finality",
        privacy_boundary: "producer and witness node identities, checkpoint generation, exact opaque frame digests, witness time, and coarse outcome only; no private HMAC, custody path, message, route, endpoint, payload, ciphertext, memory, destination, DNS, or social graph metadata",
    };
    if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("Relay custody independent witness receipt");
        println!("════════════════════════════════════════");
        println!("Status:               {}", report.status);
        println!("Accepted:             {}", report.accepted);
        println!("Outcome:              {}", report.outcome);
        println!("Producer node:        {}", report.producer_node_id);
        println!("Witness node:         {}", report.witness_node_id);
        println!("Checkpoint generation: {}", report.checkpoint_generation);
        println!("Witness observed at:  {}", report.observed_at);
        println!(
            "Retained generation:   {}",
            report.retained_checkpoint_generation
        );
        println!("Anchor frame SHA-256: {}", report.anchor_frame_sha256);
        println!("Receipt SHA-256:      {}", report.receipt_sha256);
        println!("Receipt bytes:        {}", report.receipt_bytes);
        println!();
        println!("Security model: {}", report.security_model);
        println!("Privacy: {}", report.privacy_boundary);
    }
    Ok(())
}

// [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Keep credential issuance and
// verification isolated from the command dispatcher and all network surfaces.
async fn cmd_relay_restore_plan(config_path: &Path, json: bool) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let node_secret = load_relay_custody_node_secret(&server_config, "restore plan").await?;
    let plan = ChatRelayService::create_latest_restore_plan_for_config(
        &server_config.memchain.chat_relay,
        &node_secret,
    )
    .map_err(|error| anyhow::anyhow!("relay custody restore planning failed: {error}"))?;
    print_relay_restore_plan(&plan, json)
}

async fn cmd_relay_verify_restore_plan(
    config_path: &Path,
    plan_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let server_config = load_relay_custody_config(config_path).await?;
    let node_secret =
        load_relay_custody_node_secret(&server_config, "restore-plan verification").await?;
    let plan = load_private_restore_plan(plan_path)?;
    ChatRelayService::verify_latest_restore_plan_for_config(
        &server_config.memchain.chat_relay,
        &node_secret,
        &plan,
    )
    .map_err(|error| anyhow::anyhow!("relay custody restore-plan verification failed: {error}"))?;
    if json {
        println!(
            "{}",
            serde_json::json!({"valid": true, "expires_at": plan.expires_at})
        );
    } else {
        println!(
            "Relay custody restore plan is valid until {}.",
            plan.expires_at
        );
        println!("Verification is read-only and does not authorize restoration.");
    }
    Ok(())
}

fn print_relay_restore_readiness(
    receipt: &ChatRelayRestoreReadinessReceipt,
    json: bool,
) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(receipt)?);
        return Ok(());
    }

    println!("Relay custody restore readiness");
    println!("════════════════════════════════════════");
    println!("Ready:               {}", receipt.ready);
    println!("Verified backups:    {}", receipt.verified_backup_count);
    println!("Selected bytes:      {}", receipt.selected_backup_bytes);
    println!("Active DB present:   {}", receipt.active_database_present);
    println!("Active DB bytes:     {}", receipt.active_database_bytes);
    println!("Active sidecars:     {}", receipt.active_sidecars_present);
    println!(
        "Blocker:              {}",
        receipt.blocker.unwrap_or("none")
    );
    println!();
    println!("Read-only preflight; no custody data was replaced.");
    Ok(())
}

fn print_relay_restore_plan(plan: &ChatRelayRestorePlanReceipt, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(plan)?);
        return Ok(());
    }

    println!("Relay custody authenticated restore plan");
    println!("════════════════════════════════════════");
    println!("Version:             {}", plan.version);
    println!("Issued at:           {}", plan.issued_at);
    println!("Expires at:          {}", plan.expires_at);
    println!("Validity:            {CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS}s");
    println!("Verified backups:    {}", plan.verified_backup_count);
    println!("Selected bytes:      {}", plan.selected_backup_bytes);
    println!("Active DB present:   {}", plan.active_database_present);
    println!("Active DB bytes:     {}", plan.active_database_bytes);
    println!("Nonce:               {}", plan.nonce);
    println!("Commitment:          {}", plan.commitment);
    println!();
    println!("Preflight credential only; this does not authorize or execute restoration.");
    Ok(())
}

fn write_new_relay_custody_anchor(path: &Path, frame: &[u8]) -> anyhow::Result<()> {
    write_new_relay_custody_artifact(
        path,
        frame,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES,
        "audit anchor",
    )
}

fn write_new_relay_custody_artifact(
    path: &Path,
    frame: &[u8],
    max_frame_bytes: usize,
    artifact_label: &'static str,
) -> anyhow::Result<()> {
    use std::io::Write as _;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;

    anyhow::ensure!(
        !frame.is_empty() && frame.len() <= max_frame_bytes,
        "relay custody {artifact_label} violates its write bound"
    );
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("unable to create new relay custody {artifact_label}"))?;

    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Anchors and witness receipts
    // share one create-new, no-final-symlink, file+directory durability
    // boundary. If the exact frame is not durable, remove only the inode this
    // invocation created; a retry can safely reproduce or re-witness it.
    let write_result = (|| -> anyhow::Result<()> {
        file.write_all(frame)
            .map_err(|_| anyhow::anyhow!("unable to write relay custody {artifact_label}"))?;
        file.sync_all()
            .map_err(|_| anyhow::anyhow!("unable to sync relay custody {artifact_label}"))?;
        let final_len = file
            .metadata()
            .map_err(|_| anyhow::anyhow!("unable to inspect relay custody {artifact_label}"))?
            .len();
        anyhow::ensure!(
            final_len == frame.len() as u64,
            "relay custody {artifact_label} changed during publication"
        );
        Ok(())
    })();
    drop(file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(path);
        return Err(error);
    }

    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|candidate| !candidate.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let directory = File::open(parent)
            .map_err(|_| anyhow::anyhow!("unable to open custody artifact parent directory"))?;
        directory
            .sync_all()
            .map_err(|_| anyhow::anyhow!("unable to sync custody artifact parent directory"))?;
    }
    Ok(())
}

fn read_bounded_relay_custody_anchor(path: &Path) -> anyhow::Result<Vec<u8>> {
    read_bounded_relay_custody_artifact(path, MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES, "audit anchor")
}

fn read_bounded_relay_custody_artifact(
    path: &Path,
    max_frame_bytes: usize,
    artifact_label: &'static str,
) -> anyhow::Result<Vec<u8>> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("unable to open relay custody {artifact_label}"))?;
    let metadata = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("unable to inspect relay custody {artifact_label}"))?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= max_frame_bytes as u64,
        "relay custody {artifact_label} has an invalid file boundary"
    );

    let capacity = usize::try_from(metadata.len())
        .map_err(|_| anyhow::anyhow!("relay custody {artifact_label} exceeds platform capacity"))?;
    let mut frame = Vec::with_capacity(capacity);
    file.by_ref()
        .take(max_frame_bytes as u64 + 1)
        .read_to_end(&mut frame)
        .map_err(|_| anyhow::anyhow!("unable to read relay custody {artifact_label}"))?;
    let final_len = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("unable to re-inspect relay custody {artifact_label}"))?
        .len();
    anyhow::ensure!(
        frame.len() as u64 == metadata.len()
            && final_len == metadata.len()
            && frame.len() <= max_frame_bytes,
        "relay custody {artifact_label} changed during bounded read"
    );
    Ok(frame)
}

async fn load_relay_custody_config(config_path: &Path) -> anyhow::Result<ServerConfig> {
    let config = ServerConfig::load(config_path).await?;
    if !config.memchain.is_chat_relay_enabled() {
        anyhow::bail!("relay custody maintenance requires chat relay to be enabled");
    }
    if config.memchain.chat_relay.db_path == ":memory:" {
        anyhow::bail!("in-memory relay custody has no recoverable backup boundary");
    }
    Ok(config)
}

async fn load_relay_custody_node_secret(
    config: &ServerConfig,
    operation: &str,
) -> anyhow::Result<[u8; 32]> {
    let identity = load_relay_custody_identity(config, operation).await?;
    Ok(derive_node_secret(&identity.to_bytes()))
}

async fn load_relay_custody_identity(
    config: &ServerConfig,
    operation: &str,
) -> anyhow::Result<IdentityKeyPair> {
    let identity_path = PathBuf::from(&config.server_key.key_file);
    load_key(&identity_path)
        .await
        .map_err(|_| anyhow::anyhow!("relay custody {operation} requires the node identity key"))
}

fn load_private_restore_plan(path: &Path) -> anyhow::Result<ChatRelayRestorePlanReceipt> {
    #[cfg(unix)]
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    const MAX_RESTORE_PLAN_BYTES: u64 = 4096;

    // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Treat the plan as a local
    // maintenance credential. Never follow its final symlink or include its
    // path/content in an error returned to an operator surface.
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| anyhow::anyhow!("unable to open private relay restore plan"))?;
    let metadata = file
        .metadata()
        .map_err(|_| anyhow::anyhow!("unable to inspect private relay restore plan"))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_RESTORE_PLAN_BYTES {
        anyhow::bail!("private relay restore plan has an invalid file boundary");
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o077 != 0 {
        anyhow::bail!("private relay restore plan must be owner-private");
    }

    let capacity = usize::try_from(metadata.len())
        .map_err(|_| anyhow::anyhow!("private relay restore plan exceeds platform capacity"))?;
    let mut encoded = Vec::with_capacity(capacity);
    file.by_ref()
        .take(MAX_RESTORE_PLAN_BYTES + 1)
        .read_to_end(&mut encoded)
        .map_err(|_| anyhow::anyhow!("unable to read private relay restore plan"))?;
    if encoded.len() as u64 != metadata.len() || encoded.len() as u64 > MAX_RESTORE_PLAN_BYTES {
        anyhow::bail!("private relay restore plan changed during bounded read");
    }
    serde_json::from_slice(&encoded)
        .map_err(|_| anyhow::anyhow!("private relay restore plan is malformed"))
}

/// Runs read-only `MemChain` operator commands.
async fn cmd_memchain(command: MemchainCommands) -> anyhow::Result<()> {
    match command {
        MemchainCommands::VerifyAof { path, config } => {
            cmd_memchain_verify_aof(&config, path).await
        }
    }
}

/// Verifies only aggregate AOF integrity and never prints record contents.
async fn cmd_memchain_verify_aof(
    config_path: &PathBuf,
    path_override: Option<PathBuf>,
) -> anyhow::Result<()> {
    let path = if let Some(path) = path_override {
        path
    } else {
        let config = if config_path.exists() {
            ServerConfig::load(config_path).await?
        } else {
            ServerConfig::default()
        };
        PathBuf::from(config.memchain.aof_path)
    };
    let report = AofWriter::verify(&path)
        .await
        .with_context(|| format!("verify MemChain AOF {}", path.display()))?;

    // [AOF-INTEGRITY-CLI 2026-07-24 by Codex] This output is deliberately
    // aggregate-only. Never add Fact values, identities, hashes, signatures,
    // or record-level offsets to the operator command.
    println!("MemChain AOF integrity");
    println!("  path: {}", path.display());
    println!("  file_bytes: {}", report.file_bytes);
    println!("  valid_bytes: {}", report.valid_bytes);
    println!("  fact_records: {}", report.fact_records);
    println!("  block_records: {}", report.block_records);
    println!("  last_block_height: {}", report.last_block_height);
    println!("  torn_tail_bytes: {}", report.torn_tail_bytes);
    println!(
        "  status: {}",
        if report.is_clean() {
            "verified"
        } else {
            "torn_tail_detected"
        }
    );
    println!("  privacy: aggregate integrity metadata only; no record contents or identities");

    anyhow::ensure!(
        report.is_clean(),
        "AOF has an incomplete physical tail; start the node through the guarded recovery path"
    );
    Ok(())
}

/// Shows node public key (hidden command for troubleshooting).
async fn cmd_pubkey(config_path: PathBuf, format: String) -> anyhow::Result<()> {
    let config = load_or_default_config(&config_path).await;
    let key_path = PathBuf::from(&config.server_key.key_file);

    if !key_path.exists() {
        println!("❌ Node key not found. Register first:");
        println!("   aeronyx-server register --code <YOUR_CODE>");
        std::process::exit(1);
    }

    let identity = load_key(&key_path).await?;

    match format.as_str() {
        "base64" => println!("{}", identity.public_key()),
        _ => println!("{}", hex::encode(identity.public_key_bytes())),
    }

    Ok(())
}

/// Runs privileged Directory Replica operations without a network endpoint.
async fn cmd_directory_replica(command: DirectoryReplicaCommands) -> anyhow::Result<()> {
    match command {
        DirectoryReplicaCommands::VerifyObservationCertificate {
            input,
            expected_sha256,
            expected_observer,
            allowed_witnesses,
            minimum_witnesses,
            json,
        } => cmd_directory_replica_verify_observation_certificate(
            &input,
            &expected_sha256,
            &expected_observer,
            &allowed_witnesses,
            minimum_witnesses,
            json,
        ),
        DirectoryReplicaCommands::ImportObservationCertificate {
            input,
            expected_sha256,
            expected_observer,
            allowed_witnesses,
            minimum_witnesses,
            config,
            json,
        } => {
            cmd_directory_replica_import_observation_certificate(
                &config,
                &input,
                &expected_sha256,
                &expected_observer,
                &allowed_witnesses,
                minimum_witnesses,
                json,
            )
            .await
        }
        DirectoryReplicaCommands::PullObservationCertificate {
            source_endpoint,
            expected_observer,
            allowed_witnesses,
            minimum_witnesses,
            max_age_seconds,
            config,
            json,
        } => {
            cmd_directory_replica_pull_observation_certificate(
                &config,
                &source_endpoint,
                &expected_observer,
                &allowed_witnesses,
                minimum_witnesses,
                max_age_seconds,
                json,
            )
            .await
        }
        DirectoryReplicaCommands::CarrierSmoke { config, json } => {
            cmd_directory_replica_carrier_smoke(&config, json).await
        }
        DirectoryReplicaCommands::CarrierColdBootstrapSmoke { config, json } => {
            cmd_directory_replica_carrier_cold_bootstrap_smoke(&config, json).await
        }
        DirectoryReplicaCommands::InspectIncident { digest, config } => {
            cmd_directory_replica_inspect(&config, &digest).await
        }
        DirectoryReplicaCommands::ResolveQuarantine {
            digest,
            producer,
            expected_tip_height,
            expected_tip_hash,
            expected_kind,
            expected_previous_resolution_digest,
            confirm_incident,
            config,
        } => {
            let request = DirectoryReplicaResolveRequest {
                digest,
                producer,
                expected_tip_height,
                expected_tip_hash,
                expected_kind,
                expected_previous_resolution_digest,
                confirm_incident,
            };
            cmd_directory_replica_resolve(&config, &request).await
        }
    }
}

/// Stable aggregate output from the offline certificate verifier.
///
/// Full observer and witness public keys remain inside the caller-supplied
/// certificate frame. The CLI emits only a short observer fingerprint because
/// operators need useful provenance without casually copying the complete
/// witness set into logs or automation output.
#[derive(Debug, serde::Serialize)]
struct DirectoryObservationCertificateVerificationReport {
    contract_version: &'static str,
    status: &'static str,
    protocol_version: u16,
    chain_id: String,
    certificate_id: String,
    certificate_sha256: String,
    frame_bytes: usize,
    checkpoint_sequence: u64,
    checkpoint_hash: String,
    checkpoint_observed_at: u64,
    checkpoint_age_seconds: u64,
    observer_fingerprint: String,
    trust_policy_status: &'static str,
    policy_minimum_witnesses: u16,
    policy_allowed_witnesses: usize,
    certificate_minimum_witnesses: u16,
    witness_receipts: usize,
    verified_at: u64,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

fn parse_observation_certificate_trust_policy(
    expected_observer_hex: &str,
    allowed_witness_hex: &[String],
    minimum_witnesses: u16,
) -> anyhow::Result<DirectoryObservationCertificateTrustPolicy> {
    let expected_observer = parse_hex32(expected_observer_hex, "expected observer identity")?;
    let allowed_witnesses = allowed_witness_hex
        .iter()
        .map(|value| parse_hex32(value, "allowed witness identity"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    DirectoryObservationCertificateTrustPolicy::new(
        expected_observer,
        allowed_witnesses,
        minimum_witnesses,
    )
    .map_err(Into::into)
}

/// Verifies one exact portable observation-certificate frame offline.
///
/// [PORTABLE-CERTIFICATE-VERIFIER 2026-07-26 by Codex] The external frame
/// digest is checked before decoding. The canonical re-encoding check then
/// ensures that every accepted byte sequence has one stable representation
/// before observer and witness signatures are trusted.
fn cmd_directory_replica_verify_observation_certificate(
    input_path: &Path,
    expected_sha256_hex: &str,
    expected_observer_hex: &str,
    allowed_witness_hex: &[String],
    minimum_witnesses: u16,
    emit_json: bool,
) -> anyhow::Result<()> {
    let expected_sha256 = parse_hex32(expected_sha256_hex, "certificate SHA-256")?;
    let trust_policy = parse_observation_certificate_trust_policy(
        expected_observer_hex,
        allowed_witness_hex,
        minimum_witnesses,
    )?;
    let frame = read_bounded_observation_certificate_frame(input_path)?;
    let verified_at = unix_timestamp_now()?;
    let report = verify_directory_observation_certificate_frame(
        &frame,
        &expected_sha256,
        &trust_policy,
        verified_at,
    )?;

    if emit_json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("AeroNyx Directory observation certificate");
        println!("  status: {}", report.status);
        println!("  certificate_id: {}", report.certificate_id);
        println!("  certificate_sha256: {}", report.certificate_sha256);
        println!("  frame_bytes: {}", report.frame_bytes);
        println!("  checkpoint_sequence: {}", report.checkpoint_sequence);
        println!("  checkpoint_hash: {}", report.checkpoint_hash);
        println!(
            "  checkpoint_observed_at: {}",
            report.checkpoint_observed_at
        );
        println!(
            "  checkpoint_age_seconds: {}",
            report.checkpoint_age_seconds
        );
        println!("  observer_fingerprint: {}", report.observer_fingerprint);
        println!(
            "  trusted_witnesses: {}/{} required ({} allowed)",
            report.witness_receipts,
            report.policy_minimum_witnesses,
            report.policy_allowed_witnesses
        );
        println!("  trust_policy: {}", report.trust_policy_status);
        println!("  security_model: {}", report.security_model);
        println!("  privacy: {}", report.privacy_boundary);
    }
    Ok(())
}

/// Reads a certificate through the protocol-owned complete-frame bound.
fn read_bounded_observation_certificate_frame(path: &Path) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(path)
        .with_context(|| format!("open observation certificate {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect observation certificate {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "observation certificate input must be a regular file"
    );

    let maximum = u64::try_from(MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES)
        .context("certificate frame bound does not fit u64")?;
    anyhow::ensure!(
        metadata.len() <= maximum,
        "observation certificate frame exceeds {MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES} bytes"
    );

    let read_limit = maximum.saturating_add(1);
    let initial_capacity =
        usize::try_from(metadata.len()).context("certificate frame length does not fit usize")?;
    let mut frame = Vec::with_capacity(initial_capacity);
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut frame)
        .with_context(|| format!("read observation certificate {}", path.display()))?;
    anyhow::ensure!(!frame.is_empty(), "observation certificate frame is empty");
    anyhow::ensure!(
        frame.len() <= MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES,
        "observation certificate changed while reading or exceeds {MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES} bytes"
    );
    Ok(frame)
}

/// Applies exact-frame, canonical-codec, chain, time, and signature checks.
fn verify_directory_observation_certificate_frame(
    frame: &[u8],
    expected_sha256: &[u8; 32],
    trust_policy: &DirectoryObservationCertificateTrustPolicy,
    verified_at: u64,
) -> anyhow::Result<DirectoryObservationCertificateVerificationReport> {
    let verified = verify_portable_observation_certificate_frame(
        frame,
        expected_sha256,
        trust_policy,
        verified_at,
    )?;
    let certificate = &verified.certificate;
    let checkpoint_hash = certificate.checkpoint.hash();
    Ok(DirectoryObservationCertificateVerificationReport {
        contract_version: "directory_observation_certificate_verification.v1",
        status: "verified",
        protocol_version: certificate.protocol_version,
        chain_id: hex::encode(certificate.chain_id),
        certificate_id: hex::encode(verified.certificate_id),
        certificate_sha256: hex::encode(verified.certificate_sha256),
        frame_bytes: frame.len(),
        checkpoint_sequence: certificate.checkpoint.sequence,
        checkpoint_hash: hex::encode(checkpoint_hash),
        checkpoint_observed_at: certificate.checkpoint.observed_at,
        checkpoint_age_seconds: verified_at.saturating_sub(certificate.checkpoint.observed_at),
        observer_fingerprint: hex::encode(&certificate.checkpoint.observer[..6]),
        trust_policy_status: "matched",
        policy_minimum_witnesses: trust_policy.minimum_witnesses(),
        policy_allowed_witnesses: trust_policy.allowed_witnesses().len(),
        certificate_minimum_witnesses: certificate.minimum_witnesses,
        witness_receipts: certificate.receipts.len(),
        verified_at,
        security_model: "pinned observer plus locally pinned witness policy over independent signatures; no validator set, voting weight, fork choice, consensus, global finality, transaction inclusion, or proof of user content",
        privacy_boundary: "aggregate directory observation evidence only; no endpoints, routes, client IPs, message ids, payloads, ciphertext, memory records, DNS contents, destinations, private keys, wallet traffic, or social graph metadata",
    })
}

#[derive(Debug, serde::Serialize)]
struct DirectoryObservationCertificateImportCliReport {
    contract_version: &'static str,
    status: &'static str,
    inserted: bool,
    import_sequence: u64,
    import_digest: String,
    certificate_id: String,
    certificate_sha256: String,
    observer_fingerprint: String,
    checkpoint_sequence: u64,
    checkpoint_hash: String,
    retained_certificates: u64,
    verified_at: u64,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

/// Verifies and appends one exact third-party certificate to local evidence.
///
/// [PORTABLE-CERTIFICATE-IMPORT 2026-07-26 by Codex] This command deliberately
/// has no network mutation endpoint. It requires host access, the exact frame
/// digest, explicit pins, and the local node key before the schema-v10 store
/// signs a new hash-linked row.
#[allow(clippy::too_many_arguments)]
async fn cmd_directory_replica_import_observation_certificate(
    config_path: &PathBuf,
    input_path: &Path,
    expected_sha256_hex: &str,
    expected_observer_hex: &str,
    allowed_witness_hex: &[String],
    minimum_witnesses: u16,
    emit_json: bool,
) -> anyhow::Result<()> {
    let expected_sha256 = parse_hex32(expected_sha256_hex, "certificate SHA-256")?;
    let trust_policy = parse_observation_certificate_trust_policy(
        expected_observer_hex,
        allowed_witness_hex,
        minimum_witnesses,
    )?;
    let frame = read_bounded_observation_certificate_frame(input_path)?;
    let verified_at = unix_timestamp_now()?;
    let (store, identity) = open_directory_replica_store(config_path).await?;
    let report = store.import_observation_certificate(
        &frame,
        &expected_sha256,
        &trust_policy,
        &identity,
        verified_at,
    )?;
    let output = DirectoryObservationCertificateImportCliReport {
        contract_version: "directory_observation_certificate_import.v1",
        status: if report.inserted {
            "imported"
        } else {
            "unchanged"
        },
        inserted: report.inserted,
        import_sequence: report.import_sequence,
        import_digest: hex::encode(report.import_digest),
        certificate_id: hex::encode(report.certificate_id),
        certificate_sha256: hex::encode(report.certificate_sha256),
        observer_fingerprint: hex::encode(&report.observer[..6]),
        checkpoint_sequence: report.checkpoint_sequence,
        checkpoint_hash: hex::encode(report.checkpoint_hash),
        retained_certificates: report.retained_certificates,
        verified_at: report.verified_at,
        security_model: "local node signed append-only evidence over exact third-party certificate bytes and operator-pinned trust policy; no validator set, voting, fork choice, consensus, global finality, transaction inclusion, or proof of user content",
        privacy_boundary: "host-local aggregate Directory evidence only; no endpoints, routes, client IPs, message ids, payloads, ciphertext, memory records, DNS contents, destinations, private keys, wallet traffic, or social graph metadata",
    };

    if emit_json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("AeroNyx Directory observation certificate import");
        println!("  status: {}", output.status);
        println!("  import_sequence: {}", output.import_sequence);
        println!("  import_digest: {}", output.import_digest);
        println!("  certificate_id: {}", output.certificate_id);
        println!("  certificate_sha256: {}", output.certificate_sha256);
        println!("  observer_fingerprint: {}", output.observer_fingerprint);
        println!("  checkpoint_sequence: {}", output.checkpoint_sequence);
        println!("  checkpoint_hash: {}", output.checkpoint_hash);
        println!("  retained_certificates: {}", output.retained_certificates);
        println!("  security_model: {}", output.security_model);
        println!("  privacy: {}", output.privacy_boundary);
    }
    Ok(())
}

const MAX_NETWORK_OBSERVATION_CERTIFICATE_AGE_SECONDS: u64 = 3_600;

#[derive(Debug, serde::Serialize)]
struct DirectoryObservationCertificatePullCliReport {
    contract_version: &'static str,
    status: &'static str,
    source_authenticated: bool,
    inserted: bool,
    import_sequence: u64,
    import_digest: String,
    certificate_id: String,
    certificate_sha256: String,
    observer_fingerprint: String,
    checkpoint_sequence: u64,
    checkpoint_age_seconds: u64,
    max_age_seconds: u64,
    retained_certificates: u64,
    verified_at: u64,
    security_model: &'static str,
    privacy_boundary: &'static str,
}

/// Pulls one fresh certificate through the pinned Directory peer protocol.
///
/// [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] Transport authentication,
/// certificate trust, and freshness remain three independent gates. A valid
/// HTTP response can never bypass observer/witness pins or the network replay
/// age bound before the node signs a durable schema-v10 import row.
#[allow(clippy::too_many_arguments)]
async fn cmd_directory_replica_pull_observation_certificate(
    config_path: &PathBuf,
    source_endpoint: &str,
    expected_observer_hex: &str,
    allowed_witness_hex: &[String],
    minimum_witnesses: u16,
    max_age_seconds: u64,
    emit_json: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        (1..=MAX_NETWORK_OBSERVATION_CERTIFICATE_AGE_SECONDS).contains(&max_age_seconds),
        "--max-age-seconds must be between 1 and {MAX_NETWORK_OBSERVATION_CERTIFICATE_AGE_SECONDS}"
    );
    let expected_observer = parse_hex32(expected_observer_hex, "expected observer identity")?;
    let trust_policy = parse_observation_certificate_trust_policy(
        expected_observer_hex,
        allowed_witness_hex,
        minimum_witnesses,
    )?;
    let (store, identity) = open_directory_replica_store(config_path).await?;
    let client = build_directory_certificate_exchange_http_client().map_err(anyhow::Error::msg)?;
    let authenticated = fetch_authenticated_observation_certificate(
        &client,
        source_endpoint,
        &identity,
        &expected_observer,
    )
    .await
    .map_err(anyhow::Error::msg)?;
    let verified_at = unix_timestamp_now()?;
    let verified = verify_portable_observation_certificate_frame(
        &authenticated.frame,
        &authenticated.certificate_sha256,
        &trust_policy,
        verified_at,
    )?;
    let checkpoint_age_seconds =
        verified_at.saturating_sub(verified.certificate.checkpoint.observed_at);
    anyhow::ensure!(
        checkpoint_age_seconds <= max_age_seconds,
        "observation_certificate_checkpoint_stale"
    );
    let report = store.import_observation_certificate(
        &authenticated.frame,
        &authenticated.certificate_sha256,
        &trust_policy,
        &identity,
        verified_at,
    )?;
    let output = DirectoryObservationCertificatePullCliReport {
        contract_version: "directory_observation_certificate_pull.v1",
        status: if report.inserted {
            "imported"
        } else {
            "unchanged"
        },
        source_authenticated: true,
        inserted: report.inserted,
        import_sequence: report.import_sequence,
        import_digest: hex::encode(report.import_digest),
        certificate_id: hex::encode(report.certificate_id),
        certificate_sha256: hex::encode(report.certificate_sha256),
        observer_fingerprint: hex::encode(&report.observer[..6]),
        checkpoint_sequence: report.checkpoint_sequence,
        checkpoint_age_seconds,
        max_age_seconds,
        retained_certificates: report.retained_certificates,
        verified_at: report.verified_at,
        security_model: "authenticated pinned source transport plus exact-frame digest, pinned observer, locally pinned witness threshold, bounded checkpoint age, and node-signed append-only import evidence; no voting, fork choice, consensus, or global finality",
        privacy_boundary: "host-local aggregate Directory evidence only; source endpoint is neither logged nor persisted; no routes, client IPs, message ids, payloads, ciphertext, memory records, DNS contents, destinations, private keys, wallet traffic, or social graph metadata",
    };

    if emit_json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("AeroNyx Directory observation certificate pull");
        println!("  status: {}", output.status);
        println!("  source_authenticated: {}", output.source_authenticated);
        println!("  import_sequence: {}", output.import_sequence);
        println!("  import_digest: {}", output.import_digest);
        println!("  certificate_id: {}", output.certificate_id);
        println!("  certificate_sha256: {}", output.certificate_sha256);
        println!("  observer_fingerprint: {}", output.observer_fingerprint);
        println!("  checkpoint_sequence: {}", output.checkpoint_sequence);
        println!(
            "  checkpoint_age_seconds: {}/{} maximum",
            output.checkpoint_age_seconds, output.max_age_seconds
        );
        println!("  retained_certificates: {}", output.retained_certificates);
        println!("  security_model: {}", output.security_model);
        println!("  privacy: {}", output.privacy_boundary);
    }
    Ok(())
}

fn unix_timestamp_now() -> anyhow::Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")
        .map(|duration| duration.as_secs())
}

/// Calls the running node's local-only read-only carrier verification route.
async fn cmd_directory_replica_carrier_smoke(
    config_path: &PathBuf,
    emit_json: bool,
) -> anyhow::Result<()> {
    cmd_directory_replica_operator_smoke(
        config_path,
        emit_json,
        "carrier-smoke",
        "Directory Mirror carrier smoke",
        &[
            "retained_producers",
            "eligible_retained_producers",
            "explicit_carrier_candidates",
            "attempted_carriers",
            "verified_blocks",
            "verified_descriptor_objects",
            "storage_effect",
        ],
    )
    .await
}

/// Proves that an empty isolated replica can bootstrap without the producer.
async fn cmd_directory_replica_carrier_cold_bootstrap_smoke(
    config_path: &PathBuf,
    emit_json: bool,
) -> anyhow::Result<()> {
    cmd_directory_replica_operator_smoke(
        config_path,
        emit_json,
        "carrier-cold-bootstrap-smoke",
        "Directory carrier cold-bootstrap smoke",
        &[
            "configured_producers",
            "eligible_producers",
            "explicit_carrier_candidates",
            "attempted_carriers",
            "imported_blocks",
            "imported_commitments",
            "bootstrapped_tip_height",
            "live_store_effect",
        ],
    )
    .await
}

/// Calls one bounded local-only Directory Replica smoke endpoint.
async fn cmd_directory_replica_operator_smoke(
    config_path: &PathBuf,
    emit_json: bool,
    operation: &'static str,
    title: &'static str,
    fields: &[&str],
) -> anyhow::Result<()> {
    const MAX_SMOKE_RESPONSE_BYTES: usize = 64 * 1024;

    let config = ServerConfig::load(config_path)
        .await
        .with_context(|| format!("load node config {}", config_path.display()))?;
    let url = directory_replica_operator_smoke_url(&config, operation);
    // [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] Both smoke commands stay
    // on loopback, ignore proxy settings, reject redirects, and stream into a
    // hard response limit. `Content-Length` is never trusted as the bound.
    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(2))
        .timeout(std::time::Duration::from_secs(45))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("initialize local Directory Replica smoke HTTP client")?;
    let mut response = client
        .post(url)
        .send()
        .await
        .context("contact the running node Directory Replica smoke endpoint")?;
    let http_status = response.status();
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read bounded Directory Replica smoke response")?
    {
        anyhow::ensure!(
            body.len().saturating_add(chunk.len()) <= MAX_SMOKE_RESPONSE_BYTES,
            "Directory Replica smoke response exceeded its local protocol bound"
        );
        body.extend_from_slice(&chunk);
    }
    let report: serde_json::Value =
        serde_json::from_slice(&body).context("decode Directory Replica smoke response")?;
    if emit_json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!("{title}");
        println!(
            "  status: {}",
            report["status"].as_str().unwrap_or("unavailable")
        );
        for field in fields {
            let value = &report[*field];
            if let Some(value) = value.as_str() {
                println!("  {field}: {value}");
            } else if let Some(value) = value.as_u64() {
                println!("  {field}: {value}");
            } else if let Some(value) = value.as_bool() {
                println!("  {field}: {value}");
            }
        }
        if let Some(reason) = report["failure_reason"].as_str() {
            println!("  failure_reason: {reason}");
        }
        println!(
            "  privacy: {}",
            report["privacy_boundary"]
                .as_str()
                .unwrap_or("aggregate verification metadata only")
        );
    }
    anyhow::ensure!(
        http_status.is_success() && report["success"].as_bool() == Some(true),
        "Directory Replica smoke was not verified"
    );
    Ok(())
}

/// Resolve the loopback operator API independently from the UDP tunnel socket.
///
/// [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] `network.listen_addr` is the
/// privacy tunnel's UDP endpoint and cannot accept this HTTP request. Reusing
/// only the configured operator API port preserves custom deployments while
/// ensuring the CLI never follows a non-loopback bind address.
fn directory_replica_operator_smoke_url(config: &ServerConfig, operation: &str) -> String {
    format!(
        "http://127.0.0.1:{}/api/discovery/directory/{operation}",
        config.memchain.api_listen_addr.port(),
    )
}

#[derive(Debug)]
struct DirectoryReplicaResolveRequest {
    digest: String,
    producer: String,
    expected_tip_height: u64,
    expected_tip_hash: String,
    expected_kind: String,
    expected_previous_resolution_digest: Option<String>,
    confirm_incident: String,
}

async fn cmd_directory_replica_inspect(
    config_path: &PathBuf,
    digest_hex: &str,
) -> anyhow::Result<()> {
    let digest = parse_hex32(digest_hex, "incident digest")?;
    let (store, _identity) = open_directory_replica_store(config_path).await?;
    let evidence = store
        .incident_evidence(&digest)?
        .with_context(|| format!("incident {} was not found", hex::encode(digest)))?;
    let tip = store.producer_tip(&evidence.summary.producer)?;
    print_directory_replica_incident(&evidence, &tip);
    Ok(())
}

async fn cmd_directory_replica_resolve(
    config_path: &PathBuf,
    request: &DirectoryReplicaResolveRequest,
) -> anyhow::Result<()> {
    let digest = parse_hex32(&request.digest, "incident digest")?;
    let confirmation = parse_hex32(&request.confirm_incident, "confirmed incident digest")?;
    anyhow::ensure!(
        confirmation == digest,
        "--confirm-incident must exactly repeat --digest"
    );
    let producer = parse_hex32(&request.producer, "producer identity")?;
    let expected_tip_hash = parse_hex32(&request.expected_tip_hash, "expected tip hash")?;
    let previous_resolution_digest = request
        .expected_previous_resolution_digest
        .as_deref()
        .map(|value| parse_hex32(value, "previous resolution digest"))
        .transpose()?;
    let (store, identity) = open_directory_replica_store(config_path).await?;
    let evidence = store
        .incident_evidence(&digest)?
        .with_context(|| format!("incident {} was not found", hex::encode(digest)))?;
    let tip = store.producer_tip(&producer)?;
    validate_resolution_request(
        request,
        digest,
        producer,
        expected_tip_hash,
        &evidence,
        &tip,
    )?;
    anyhow::ensure!(
        tip.last_resolution_digest == previous_resolution_digest,
        "previous resolution digest changed; inspect the incident again"
    );

    let mut command_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut command_id);
    let now = unix_timestamp()?;
    let command = DirectoryReplicaResolutionCommand::sign(
        &identity,
        command_id,
        digest,
        producer,
        request.expected_tip_height,
        expected_tip_hash,
        request.expected_kind.clone(),
        previous_resolution_digest,
        now,
    )?;
    let report = store.resolve_quarantine(&command, now)?;
    println!("Directory Replica quarantine resolved");
    println!(
        "  resolution_digest: {}",
        hex::encode(report.resolution_digest)
    );
    println!("  command_id: {}", hex::encode(report.command_id));
    println!("  producer: {}", hex::encode(report.producer));
    println!("  retained_tip_height: {}", report.retained_tip_height);
    println!(
        "  retained_tip_hash: {}",
        hex::encode(report.retained_tip_hash)
    );
    println!("  resolved_at: {}", report.resolved_at);
    println!("  action: resume_existing_prefix");
    Ok(())
}

fn validate_resolution_request(
    request: &DirectoryReplicaResolveRequest,
    digest: [u8; 32],
    producer: [u8; 32],
    expected_tip_hash: [u8; 32],
    evidence: &aeronyx_server::services::DirectoryReplicaIncidentEvidence,
    tip: &DirectoryReplicaTip,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        evidence.summary.incident_digest == digest,
        "incident digest changed"
    );
    anyhow::ensure!(
        evidence.summary.producer == producer,
        "incident producer mismatch"
    );
    anyhow::ensure!(
        evidence.summary.subject_node_id == producer,
        "incident is not a producer quarantine"
    );
    anyhow::ensure!(tip.quarantined, "producer is not quarantined");
    anyhow::ensure!(
        tip.active_incident_digest == Some(digest),
        "incident is not the active quarantine"
    );
    anyhow::ensure!(
        tip.tip_height == request.expected_tip_height,
        "accepted tip height changed"
    );
    anyhow::ensure!(
        tip.tip_hash == expected_tip_hash,
        "accepted tip hash changed"
    );
    anyhow::ensure!(
        tip.quarantine_kind.as_deref() == Some(request.expected_kind.as_str()),
        "quarantine kind changed"
    );
    Ok(())
}

fn print_directory_replica_incident(
    evidence: &aeronyx_server::services::DirectoryReplicaIncidentEvidence,
    tip: &DirectoryReplicaTip,
) {
    println!("Directory Replica incident (verified, read-only)");
    println!(
        "  incident_digest: {}",
        hex::encode(evidence.summary.incident_digest)
    );
    println!("  producer: {}", hex::encode(evidence.summary.producer));
    println!("  kind: {}", evidence.summary.kind);
    println!("  incident_height: {}", evidence.summary.height);
    println!("  local_hash: {}", hex::encode(evidence.summary.local_hash));
    println!(
        "  remote_hash: {}",
        hex::encode(evidence.summary.remote_hash)
    );
    println!(
        "  evidence_sha256: {}",
        hex::encode(evidence.evidence_sha256)
    );
    println!("  observed_at: {}", evidence.summary.observed_at);
    println!("  quarantined: {}", tip.quarantined);
    println!("  accepted_tip_height: {}", tip.tip_height);
    println!("  accepted_tip_hash: {}", hex::encode(tip.tip_hash));
    println!(
        "  previous_resolution_digest: {}",
        tip.last_resolution_digest
            .map_or_else(|| "none".to_string(), hex::encode)
    );
    println!("No block, incident, or evidence was modified.");
    if tip.quarantined
        && tip.active_incident_digest == Some(evidence.summary.incident_digest)
        && evidence.summary.subject_node_id == evidence.summary.producer
    {
        println!("Exact resolution command after independent evidence review:");
        print!(
            "  aeronyx-server directory-replica resolve-quarantine --digest {} \
--producer {} --expected-tip-height {} --expected-tip-hash {} \
--expected-kind {}",
            hex::encode(evidence.summary.incident_digest),
            hex::encode(evidence.summary.producer),
            tip.tip_height,
            hex::encode(tip.tip_hash),
            evidence.summary.kind,
        );
        if let Some(previous) = tip.last_resolution_digest {
            print!(
                " --expected-previous-resolution-digest {}",
                hex::encode(previous)
            );
        }
        println!(
            " --confirm-incident {}",
            hex::encode(evidence.summary.incident_digest)
        );
    } else {
        println!("Resolution command unavailable: this is not the active producer quarantine.");
    }
}

async fn open_directory_replica_store(
    config_path: &PathBuf,
) -> anyhow::Result<(DirectoryReplicaStore, IdentityKeyPair)> {
    anyhow::ensure!(
        config_path.exists(),
        "configuration file not found: {}",
        config_path.display()
    );
    let config = ServerConfig::load(config_path).await?;
    let database_path = config
        .discovery
        .directory_chain_path
        .as_deref()
        .context("discovery.directory_chain_path is not configured")?;
    let key_path = PathBuf::from(&config.server_key.key_file);
    anyhow::ensure!(
        key_path.exists(),
        "node identity key not found: {}",
        key_path.display()
    );
    let identity = load_key(&key_path).await?;
    let now = unix_timestamp()?;
    let (store, audit) =
        DirectoryReplicaStore::open(database_path, identity.public_key_bytes(), now)?;
    info!(
        producers = audit.producers,
        quarantined_producers = audit.quarantined_producers,
        blocks = audit.blocks,
        incidents = audit.incidents,
        resolutions = audit.resolutions,
        "host-local Directory Replica audit passed"
    );
    Ok((store, identity))
}

// ============================================
// Helper Functions
// ============================================

fn init_logging(level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .try_init()
        .ok();
}

fn parse_hex32(value: &str, field: &str) -> anyhow::Result<[u8; 32]> {
    anyhow::ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{field} must contain exactly 64 hexadecimal characters"
    );
    let decoded = hex::decode(value).with_context(|| format!("invalid {field}"))?;
    decoded
        .try_into()
        .map_err(|_| anyhow::anyhow!("{field} must decode to exactly 32 bytes"))
}

fn unix_timestamp() -> anyhow::Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

async fn load_or_default_config(path: &PathBuf) -> ServerConfig {
    if path.exists() {
        ServerConfig::load(path).await.unwrap_or_default()
    } else {
        ServerConfig::default()
    }
}

async fn load_key(path: &PathBuf) -> anyhow::Result<IdentityKeyPair> {
    let content = tokio::fs::read_to_string(path).await?;
    let key_data: KeyFile = serde_json::from_str(&content)?;

    let private_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &key_data.private_key,
    )?;

    let identity = IdentityKeyPair::from_bytes(&private_bytes)?;
    Ok(identity)
}

async fn save_key(identity: &IdentityKeyPair, path: &PathBuf) -> anyhow::Result<()> {
    use base64::Engine;

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let key_data = KeyFile {
        version: "1.0".to_string(),
        key_type: "ed25519".to_string(),
        public_key: base64::engine::general_purpose::STANDARD.encode(identity.public_key_bytes()),
        private_key: base64::engine::general_purpose::STANDARD.encode(identity.to_bytes()),
        created_at: chrono_lite_timestamp(),
    };

    let content = serde_json::to_string_pretty(&key_data)?;
    tokio::fs::write(path, content).await?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(path).await?.permissions();
        perms.set_mode(0o600);
        tokio::fs::set_permissions(path, perms).await?;
    }

    Ok(())
}

fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    format!("{}Z", duration.as_secs())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct KeyFile {
    version: String,
    key_type: String,
    public_key: String,
    private_key: String,
    created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::protocol::discovery::{
        directory_observation_witness_response_signing_bytes,
        encode_directory_observation_certificate, DirectoryObservationCertificateV1,
        DirectoryObservationCheckpointV1, DirectoryObservationTipV1,
        DirectoryObservationWitnessReceiptV1, DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    };
    use sha2::{Digest, Sha256};

    #[test]
    fn registration_cli_keeps_legacy_code_and_accepts_stdin_mode() {
        let legacy =
            Cli::try_parse_from(["aeronyx-server", "register", "--code", "NYX-LEGACY-123"])
                .unwrap();
        let Commands::Register {
            code, code_stdin, ..
        } = legacy.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(code.as_deref(), Some("NYX-LEGACY-123"));
        assert!(!code_stdin);

        let stdin = Cli::try_parse_from(["aeronyx-server", "register", "--code-stdin"]).unwrap();
        let Commands::Register {
            code, code_stdin, ..
        } = stdin.command
        else {
            panic!("unexpected CLI command")
        };
        assert!(code.is_none());
        assert!(code_stdin);

        assert!(Cli::try_parse_from(["aeronyx-server", "register"]).is_err());
        assert!(Cli::try_parse_from([
            "aeronyx-server",
            "register",
            "--code",
            "NYX-123",
            "--code-stdin",
        ])
        .is_err());
    }

    #[test]
    fn relay_smoke_cli_is_local_bounded_and_explicit() {
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "relay-smoke",
            "--confirm-live-relay-smoke",
            "--json",
        ])
        .unwrap();
        let Commands::RelaySmoke {
            server,
            health_url,
            config,
            timeout_seconds,
            confirm_live_relay_smoke,
            json,
        } = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(server, "127.0.0.1:51820".parse().unwrap());
        assert_eq!(health_url, "http://127.0.0.1:8421/api/vpn/health");
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert_eq!(timeout_seconds, 30);
        assert!(confirm_live_relay_smoke);
        assert!(json);

        assert!(Cli::try_parse_from(
            ["aeronyx-server", "relay-smoke", "--timeout-seconds", "121",]
        )
        .is_err());
    }

    #[test]
    fn relay_custody_cli_defaults_to_dry_run_and_gates_execution() {
        // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] The destructive form
        // must be impossible to express accidentally through a single flag.
        let audit =
            Cli::try_parse_from(["aeronyx-server", "relay-custody", "audit", "--json"]).unwrap();
        let Commands::RelayCustody(RelayCustodyCommands::Audit { config, json }) = audit.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert!(json);

        // [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Integrity
        // verification is a separate, read-only command because retention
        // inspection alone does not authenticate prior prune decisions.
        let verify_audit =
            Cli::try_parse_from(["aeronyx-server", "relay-custody", "verify-audit", "--json"])
                .expect("maintenance audit verification form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::VerifyAudit { config, json }) =
            verify_audit.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert!(json);

        // [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] Creation requires a new
        // explicit output, while offline verification requires all three local
        // trust pins: exact bytes, producer identity, and rollback floor.
        let create_anchor = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "create-audit-anchor",
            "--output",
            "/root/custody-anchor.bin",
            "--json",
        ])
        .expect("audit anchor creation form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::CreateAuditAnchor {
            config,
            output,
            json,
        }) = create_anchor.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert_eq!(output, PathBuf::from("/root/custody-anchor.bin"));
        assert!(json);

        let node = "81".repeat(32);
        let digest = "82".repeat(32);
        let verify_anchor = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "verify-audit-anchor",
            "--input",
            "/root/custody-anchor.bin",
            "--expected-sha256",
            &digest,
            "--expected-node",
            &node,
            "--minimum-checkpoint-generation",
            "7",
            "--json",
        ])
        .expect("audit anchor verification form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::VerifyAuditAnchor {
            input,
            expected_sha256,
            expected_node,
            minimum_checkpoint_generation,
            json,
        }) = verify_anchor.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(input, PathBuf::from("/root/custody-anchor.bin"));
        assert_eq!(expected_sha256, digest);
        assert_eq!(expected_node, node);
        assert_eq!(minimum_checkpoint_generation, 7);
        assert!(json);
        assert!(Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "verify-audit-anchor",
            "--input",
            "/root/custody-anchor.bin",
            "--expected-sha256",
            &"83".repeat(32),
            "--expected-node",
            &"84".repeat(32),
            "--minimum-checkpoint-generation",
            "0",
        ])
        .is_err());

        // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Countersigning requires
        // a producer pin, exact frame pin, first-observation floor, and a new
        // output. Offline acceptance additionally pins the witness and exact
        // receipt bytes.
        let witness = "85".repeat(32);
        let witness_anchor = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "witness-audit-anchor",
            "--input",
            "/root/custody-anchor.bin",
            "--expected-sha256",
            &digest,
            "--expected-producer",
            &node,
            "--minimum-checkpoint-generation",
            "7",
            "--output",
            "/root/custody-witness.bin",
            "--json",
        ])
        .expect("audit witness creation form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::WitnessAuditAnchor {
            config,
            input,
            expected_sha256,
            expected_producer,
            minimum_checkpoint_generation,
            output,
            json,
        }) = witness_anchor.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert_eq!(input, PathBuf::from("/root/custody-anchor.bin"));
        assert_eq!(expected_sha256, digest);
        assert_eq!(expected_producer, node);
        assert_eq!(minimum_checkpoint_generation, 7);
        assert_eq!(output, PathBuf::from("/root/custody-witness.bin"));
        assert!(json);

        let verify_witness = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "verify-audit-witness",
            "--anchor",
            "/root/custody-anchor.bin",
            "--anchor-sha256",
            &"86".repeat(32),
            "--receipt",
            "/root/custody-witness.bin",
            "--receipt-sha256",
            &"87".repeat(32),
            "--expected-producer",
            &node,
            "--expected-witness",
            &witness,
            "--minimum-checkpoint-generation",
            "7",
            "--json",
        ])
        .expect("audit witness verification form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::VerifyAuditWitness {
            anchor,
            anchor_sha256,
            receipt,
            receipt_sha256,
            expected_producer,
            expected_witness,
            minimum_checkpoint_generation,
            json,
        }) = verify_witness.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(anchor, PathBuf::from("/root/custody-anchor.bin"));
        assert_eq!(anchor_sha256, "86".repeat(32));
        assert_eq!(receipt, PathBuf::from("/root/custody-witness.bin"));
        assert_eq!(receipt_sha256, "87".repeat(32));
        assert_eq!(expected_producer, node);
        assert_eq!(expected_witness, witness);
        assert_eq!(minimum_checkpoint_generation, 7);
        assert!(json);

        // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Import is an
        // explicit host-local operation with exact anchor, receipt, and
        // configured witness pins plus a parser-bounded freshness window.
        let import_witness = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "import-audit-witness",
            "--anchor",
            "/root/custody-anchor.bin",
            "--anchor-sha256",
            &"86".repeat(32),
            "--receipt",
            "/root/custody-witness.bin",
            "--receipt-sha256",
            &"87".repeat(32),
            "--expected-witness",
            &witness,
            "--max-age-seconds",
            "3600",
            "--json",
        ])
        .expect("audit witness import form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::ImportAuditWitness {
            config,
            anchor,
            anchor_sha256,
            receipt,
            receipt_sha256,
            expected_witness,
            max_age_seconds,
            json,
        }) = import_witness.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert_eq!(anchor, PathBuf::from("/root/custody-anchor.bin"));
        assert_eq!(anchor_sha256, "86".repeat(32));
        assert_eq!(receipt, PathBuf::from("/root/custody-witness.bin"));
        assert_eq!(receipt_sha256, "87".repeat(32));
        assert_eq!(expected_witness, witness);
        assert_eq!(max_age_seconds, 3600);
        assert!(json);
        assert!(Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "import-audit-witness",
            "--anchor",
            "/root/custody-anchor.bin",
            "--anchor-sha256",
            &"86".repeat(32),
            "--receipt",
            "/root/custody-witness.bin",
            "--receipt-sha256",
            &"87".repeat(32),
            "--expected-witness",
            &"85".repeat(32),
            "--max-age-seconds",
            "604801",
        ])
        .is_err());

        // [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Local re-audit is
        // independently invokable after restart. Strict readiness is explicit,
        // while the parser prevents an unbounded stale-evidence window.
        let audit_witness_vault = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "audit-witness-vault",
            "--max-age-seconds",
            "1800",
            "--require-ready",
            "--json",
        ])
        .expect("custody witness vault audit form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::AuditWitnessVault {
            config,
            max_age_seconds,
            require_ready,
            json,
        }) = audit_witness_vault.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert_eq!(max_age_seconds, 1800);
        assert!(require_ready);
        assert!(json);
        for invalid_age in ["59", "604801"] {
            assert!(Cli::try_parse_from([
                "aeronyx-server",
                "relay-custody",
                "audit-witness-vault",
                "--max-age-seconds",
                invalid_age,
            ])
            .is_err());
        }

        let readiness = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "restore-readiness",
            "--json",
        ])
        .expect("read-only restore readiness form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::RestoreReadiness { config, json }) =
            readiness.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert!(json);

        let plan =
            Cli::try_parse_from(["aeronyx-server", "relay-custody", "restore-plan", "--json"])
                .expect("authenticated restore plan form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::RestorePlan { config, json }) =
            plan.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert!(json);

        let verify_plan = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "verify-restore-plan",
            "--plan-file",
            "/root/relay-restore-plan.json",
            "--json",
        ])
        .expect("restore-plan verification form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::VerifyRestorePlan {
            config,
            plan_file,
            json,
        }) = verify_plan.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
        assert_eq!(plan_file, PathBuf::from("/root/relay-restore-plan.json"));
        assert!(json);

        let dry_run = Cli::try_parse_from(["aeronyx-server", "relay-custody", "prune"])
            .expect("dry-run form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::Prune {
            execute,
            confirm_node_stopped,
            confirm_prune,
            ..
        }) = dry_run.command
        else {
            panic!("unexpected CLI command")
        };
        assert!(!execute);
        assert!(!confirm_node_stopped);
        assert!(confirm_prune.is_none());

        let execute = Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "prune",
            "--execute",
            "--confirm-node-stopped",
            "--confirm-prune",
            CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION,
        ])
        .expect("fully-confirmed execution form must parse");
        let Commands::RelayCustody(RelayCustodyCommands::Prune {
            execute,
            confirm_node_stopped,
            confirm_prune,
            ..
        }) = execute.command
        else {
            panic!("unexpected CLI command")
        };
        assert!(execute);
        assert!(confirm_node_stopped);
        assert_eq!(
            confirm_prune.as_deref(),
            Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION)
        );

        assert!(
            Cli::try_parse_from(["aeronyx-server", "relay-custody", "prune", "--execute",])
                .is_err()
        );
        assert!(Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "prune",
            "--confirm-node-stopped",
        ])
        .is_err());
    }

    #[test]
    fn custody_witness_vault_status_is_stable_and_fail_closed() {
        // [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Monitoring labels
        // must not depend on count ordering: readiness wins only when the full
        // policy says so, while any unresolved adverse evidence is explicit.
        let collecting = CustodyAuditWitnessReceiptPolicyEvidence::default();
        assert_eq!(
            custody_audit_witness_policy_status(&collecting),
            "collecting"
        );

        let adverse = CustodyAuditWitnessReceiptPolicyEvidence {
            adverse: 1,
            ..CustodyAuditWitnessReceiptPolicyEvidence::default()
        };
        assert_eq!(custody_audit_witness_policy_status(&adverse), "adverse");

        let inconsistent = CustodyAuditWitnessReceiptPolicyEvidence {
            adverse: 1,
            quorum_satisfied: true,
            ..CustodyAuditWitnessReceiptPolicyEvidence::default()
        };
        assert_eq!(
            custody_audit_witness_policy_status(&inconsistent),
            "adverse"
        );

        let ready = CustodyAuditWitnessReceiptPolicyEvidence {
            quorum_satisfied: true,
            ..CustodyAuditWitnessReceiptPolicyEvidence::default()
        };
        assert_eq!(custody_audit_witness_policy_status(&ready), "ready");
    }

    #[test]
    fn relay_custody_anchor_file_boundary_is_exact_and_no_overwrite() {
        let directory = tempfile::tempdir().expect("audit anchor CLI directory");
        let path = directory.path().join("anchor.bin");
        let producer = IdentityKeyPair::from_bytes(&[0x85; 32]).expect("anchor producer");
        let anchor =
            CustodyAuditAnchorV1::signed(4, 65_540, 128 * 1024 * 1024, [0x86; 32], &producer)
                .expect("sign CLI anchor fixture");
        let frame = encode_custody_audit_anchor(&anchor).expect("encode CLI anchor fixture");
        let frame_sha256: [u8; 32] = Sha256::digest(&frame).into();

        write_new_relay_custody_anchor(&path, &frame).expect("publish exact CLI anchor");
        assert!(write_new_relay_custody_anchor(&path, &frame).is_err());
        let loaded = read_bounded_relay_custody_anchor(&path).expect("read bounded CLI anchor");
        let verified = verify_relay_custody_anchor_frame(
            &loaded,
            &frame_sha256,
            &producer.public_key_bytes(),
            4,
        )
        .expect("verify exact CLI anchor");
        assert_eq!(verified, anchor);

        let mut wrong_sha256 = frame_sha256;
        wrong_sha256[0] ^= 1;
        assert!(verify_relay_custody_anchor_frame(
            &loaded,
            &wrong_sha256,
            &producer.public_key_bytes(),
            4,
        )
        .is_err());
        assert!(verify_relay_custody_anchor_frame(
            &loaded,
            &frame_sha256,
            &producer.public_key_bytes(),
            5,
        )
        .is_err());

        let oversized_path = directory.path().join("oversized.bin");
        std::fs::write(
            &oversized_path,
            vec![0u8; MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES + 1],
        )
        .expect("write oversized CLI anchor fixture");
        assert!(read_bounded_relay_custody_anchor(&oversized_path).is_err());

        #[cfg(unix)]
        {
            let symlink_path = directory.path().join("anchor-link.bin");
            std::os::unix::fs::symlink(&path, &symlink_path)
                .expect("create audit anchor symlink fixture");
            assert!(read_bounded_relay_custody_anchor(&symlink_path).is_err());
        }
    }

    #[test]
    fn relay_custody_witness_receipt_file_boundary_and_anchor_binding_are_exact() {
        let directory = tempfile::tempdir().expect("audit witness CLI directory");
        let receipt_path = directory.path().join("witness.bin");
        let producer = IdentityKeyPair::from_bytes(&[0x88; 32]).expect("anchor producer");
        let witness = IdentityKeyPair::from_bytes(&[0x89; 32]).expect("anchor witness");
        let anchor =
            CustodyAuditAnchorV1::signed(5, 65_541, 256 * 1024 * 1024, [0x8a; 32], &producer)
                .expect("sign anchor fixture");
        let anchor_frame = encode_custody_audit_anchor(&anchor).expect("encode anchor fixture");
        let anchor_sha256: [u8; 32] = Sha256::digest(&anchor_frame).into();
        let receipt = CustodyAuditWitnessReceiptV1::signed(
            producer.public_key_bytes(),
            5,
            anchor_sha256,
            1_787_200_100,
            5,
            anchor_sha256,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .expect("sign witness fixture");
        let receipt_frame =
            encode_custody_audit_witness_receipt(&receipt).expect("encode witness fixture");

        write_new_relay_custody_artifact(
            &receipt_path,
            &receipt_frame,
            MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
            "audit witness receipt",
        )
        .expect("publish witness receipt");
        assert!(write_new_relay_custody_artifact(
            &receipt_path,
            &receipt_frame,
            MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
            "audit witness receipt",
        )
        .is_err());
        let loaded = read_bounded_relay_custody_artifact(
            &receipt_path,
            MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
            "audit witness receipt",
        )
        .expect("read witness receipt");
        let receipt_sha256: [u8; 32] = Sha256::digest(&loaded).into();
        let decoded = verify_relay_custody_witness_receipt_frame(
            &loaded,
            &receipt_sha256,
            &anchor,
            &anchor_sha256,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            5,
        )
        .expect("verify canonical witness receipt frame");
        decoded
            .verify_accepted_for_anchor(
                &anchor,
                &anchor_sha256,
                &producer.public_key_bytes(),
                &witness.public_key_bytes(),
                5,
            )
            .expect("verify witness receipt binding");

        let wrong_anchor_sha256 = [0x8b; 32];
        assert!(decoded
            .verify_accepted_for_anchor(
                &anchor,
                &wrong_anchor_sha256,
                &producer.public_key_bytes(),
                &witness.public_key_bytes(),
                5,
            )
            .is_err());
        assert!(verify_relay_custody_witness_receipt_frame(
            &loaded,
            &[0x8c; 32],
            &anchor,
            &anchor_sha256,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            5,
        )
        .is_err());

        let oversized_path = directory.path().join("oversized-witness.bin");
        std::fs::write(
            &oversized_path,
            vec![0u8; MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES + 1],
        )
        .expect("write oversized receipt fixture");
        assert!(read_bounded_relay_custody_artifact(
            &oversized_path,
            MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
            "audit witness receipt",
        )
        .is_err());
    }

    #[test]
    fn private_restore_plan_loader_is_bounded_and_strict() {
        // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] Credential loading
        // rejects permission drift, symlinks, and schema extension smuggling.
        let directory = tempfile::tempdir().expect("private plan directory");
        let path = directory.path().join("restore-plan.json");
        let plan = ChatRelayRestorePlanReceipt {
            version: 1,
            issued_at: 1_800_000_000,
            expires_at: 1_800_000_600,
            verified_backup_count: 2,
            selected_backup_bytes: 4096,
            active_database_present: true,
            active_database_bytes: 8192,
            nonce: "11".repeat(16),
            commitment: "22".repeat(32),
        };
        std::fs::write(
            &path,
            serde_json::to_vec(&plan).expect("encode private plan"),
        )
        .expect("write private plan");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .expect("secure private plan permissions");
        }
        assert_eq!(
            load_private_restore_plan(&path).expect("load strict private plan"),
            plan
        );

        let mut extended = serde_json::to_value(&plan).expect("encode extended plan");
        extended.as_object_mut().expect("plan JSON object").insert(
            "selected_backup_path".to_string(),
            serde_json::json!("secret"),
        );
        std::fs::write(
            &path,
            serde_json::to_vec(&extended).expect("encode extended plan JSON"),
        )
        .expect("write extended plan");
        assert!(load_private_restore_plan(&path).is_err());

        std::fs::write(&path, vec![b' '; 4097]).expect("write oversized private plan");
        assert!(load_private_restore_plan(&path).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::{symlink, PermissionsExt};

            std::fs::write(
                &path,
                serde_json::to_vec(&plan).expect("re-encode private plan"),
            )
            .expect("restore private plan");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
                .expect("make plan non-private");
            assert!(load_private_restore_plan(&path).is_err());
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                .expect("restore plan privacy");
            let link = directory.path().join("restore-plan-link.json");
            symlink(&path, &link).expect("create plan symlink");
            assert!(load_private_restore_plan(&link).is_err());
        }
    }

    #[test]
    fn registration_code_stdin_is_trimmed_bounded_and_private_by_contract() {
        let code = read_registration_code(std::io::Cursor::new("  NYX-STDIN-123  \n")).unwrap();
        assert_eq!(code, "NYX-STDIN-123");

        let oversized = format!("{}\n", "A".repeat(MAX_REGISTRATION_CODE_BYTES + 1));
        assert!(read_registration_code(std::io::Cursor::new(oversized)).is_err());
        assert!(normalize_registration_code("NYX-123\0hidden").is_err());
        assert!(normalize_registration_code("   ").is_err());
    }

    #[test]
    fn registration_profile_normalizes_operator_metadata() {
        assert_eq!(
            normalize_registration_name(Some("  TW1  ".to_string())).unwrap(),
            Some("TW1".to_string())
        );
        assert_eq!(
            normalize_registration_region(Some("tw".to_string())).unwrap(),
            Some("TW".to_string())
        );
    }

    #[test]
    fn registration_profile_rejects_unsafe_metadata() {
        assert!(normalize_registration_name(Some("TW1\nadmin".to_string())).is_err());
        assert!(normalize_registration_name(Some(" ".to_string())).is_err());
        assert!(normalize_registration_region(Some("taiwan".to_string())).is_err());
        assert!(normalize_registration_region(Some("T1".to_string())).is_err());
    }

    fn portable_observation_certificate_fixture() -> (Vec<u8>, [u8; 32], [u8; 32], Vec<[u8; 32]>) {
        let observer = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
        let witness_a = IdentityKeyPair::from_bytes(&[0x34; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0x35; 32]).unwrap();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            7,
            1_700_000_700,
            [0x36; 32],
            2,
            vec![
                DirectoryObservationTipV1 {
                    producer: producer_a.public_key_bytes(),
                    tip_height: 41,
                    tip_hash: [0x37; 32],
                },
                DirectoryObservationTipV1 {
                    producer: producer_b.public_key_bytes(),
                    tip_height: 42,
                    tip_hash: [0x38; 32],
                },
            ],
            [0x39; 32],
            &observer,
        )
        .unwrap();

        let receipt = |witness: &IdentityKeyPair, request_id: [u8; 16], response_timestamp: u64| {
            let checkpoint_hash = checkpoint.hash();
            let signing_bytes = directory_observation_witness_response_signing_bytes(
                &checkpoint.chain_id,
                &request_id,
                &checkpoint.observer,
                checkpoint.sequence,
                &checkpoint_hash,
                &witness.public_key_bytes(),
                response_timestamp,
                DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            );
            DirectoryObservationWitnessReceiptV1 {
                chain_id: checkpoint.chain_id,
                request_id,
                observer: checkpoint.observer,
                checkpoint_sequence: checkpoint.sequence,
                checkpoint_hash,
                responder: witness.public_key_bytes(),
                response_timestamp,
                outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                signature: witness.sign(&signing_bytes),
            }
        };
        let receipt_b = receipt(&witness_b, [0x3a; 16], 1_700_000_702);
        let receipt_a = receipt(&witness_a, [0x3b; 16], 1_700_000_701);
        let certificate = DirectoryObservationCertificateV1::new_verified(
            checkpoint,
            2,
            vec![receipt_b, receipt_a],
            1_700_000_702,
        )
        .unwrap();
        let frame = encode_directory_observation_certificate(&certificate).unwrap();
        let frame_sha256 = Sha256::digest(&frame).into();
        (
            frame,
            frame_sha256,
            observer.public_key_bytes(),
            vec![witness_a.public_key_bytes(), witness_b.public_key_bytes()],
        )
    }

    #[test]
    fn directory_observation_certificate_verifier_cli_requires_explicit_trust_policy() {
        let expected_sha256 = "a5".repeat(32);
        let expected_observer = "b6".repeat(32);
        let witness_a = "c7".repeat(32);
        let witness_b = "d8".repeat(32);
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "verify-observation-certificate",
            "--input",
            "/tmp/observation.certificate",
            "--expected-sha256",
            &expected_sha256,
            "--expected-observer",
            &expected_observer,
            "--allowed-witness",
            &witness_a,
            "--allowed-witness",
            &witness_b,
            "--minimum-witnesses",
            "2",
            "--json",
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::VerifyObservationCertificate {
            input,
            expected_sha256: parsed_sha256,
            expected_observer: parsed_observer,
            allowed_witnesses,
            minimum_witnesses,
            json,
        }) = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(input, PathBuf::from("/tmp/observation.certificate"));
        assert_eq!(parsed_sha256, expected_sha256);
        assert_eq!(parsed_observer, expected_observer);
        assert_eq!(allowed_witnesses, vec![witness_a, witness_b]);
        assert_eq!(minimum_witnesses, 2);
        assert!(json);
    }

    #[test]
    fn directory_observation_certificate_import_cli_requires_local_store_and_pins() {
        let expected_sha256 = "a6".repeat(32);
        let expected_observer = "b7".repeat(32);
        let witness = "c8".repeat(32);
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "import-observation-certificate",
            "--input",
            "/tmp/third-party.certificate",
            "--expected-sha256",
            &expected_sha256,
            "--expected-observer",
            &expected_observer,
            "--allowed-witness",
            &witness,
            "--minimum-witnesses",
            "1",
            "--config",
            "/tmp/server.toml",
            "--json",
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::ImportObservationCertificate {
            input,
            expected_sha256: parsed_sha256,
            expected_observer: parsed_observer,
            allowed_witnesses,
            minimum_witnesses,
            config,
            json,
        }) = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(input, PathBuf::from("/tmp/third-party.certificate"));
        assert_eq!(parsed_sha256, expected_sha256);
        assert_eq!(parsed_observer, expected_observer);
        assert_eq!(allowed_witnesses, vec![witness]);
        assert_eq!(minimum_witnesses, 1);
        assert_eq!(config, PathBuf::from("/tmp/server.toml"));
        assert!(json);
    }

    #[test]
    fn directory_observation_certificate_pull_cli_requires_source_pins_and_age() {
        let expected_observer = "b8".repeat(32);
        let witness = "c9".repeat(32);
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "pull-observation-certificate",
            "--source-endpoint",
            "https://203.0.113.9:8422",
            "--expected-observer",
            &expected_observer,
            "--allowed-witness",
            &witness,
            "--minimum-witnesses",
            "1",
            "--max-age-seconds",
            "600",
            "--config",
            "/tmp/server.toml",
            "--json",
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::PullObservationCertificate {
            source_endpoint,
            expected_observer: parsed_observer,
            allowed_witnesses,
            minimum_witnesses,
            max_age_seconds,
            config,
            json,
        }) = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(source_endpoint, "https://203.0.113.9:8422");
        assert_eq!(parsed_observer, expected_observer);
        assert_eq!(allowed_witnesses, vec![witness]);
        assert_eq!(minimum_witnesses, 1);
        assert_eq!(max_age_seconds, 600);
        assert_eq!(config, PathBuf::from("/tmp/server.toml"));
        assert!(json);
    }

    #[test]
    fn directory_observation_certificate_verifier_is_bounded_and_fail_closed() {
        // [PORTABLE-CERTIFICATE-VERIFIER 2026-07-26 by Codex] The server-side
        // adapter must preserve the core verifier's canonical and signature
        // checks rather than treating a matching transport digest as trust.
        let (frame, frame_sha256, observer, witnesses) = portable_observation_certificate_fixture();
        let allowed_witness_hex = witnesses.iter().map(hex::encode).collect::<Vec<_>>();
        let trust_policy = parse_observation_certificate_trust_policy(
            &hex::encode(observer),
            &allowed_witness_hex,
            2,
        )
        .unwrap();
        let report = verify_directory_observation_certificate_frame(
            &frame,
            &frame_sha256,
            &trust_policy,
            1_700_000_702,
        )
        .unwrap();
        assert_eq!(report.status, "verified");
        assert_eq!(report.checkpoint_sequence, 7);
        assert_eq!(report.trust_policy_status, "matched");
        assert_eq!(report.policy_minimum_witnesses, 2);
        assert_eq!(report.policy_allowed_witnesses, 2);
        assert_eq!(report.certificate_minimum_witnesses, 2);
        assert_eq!(report.witness_receipts, 2);
        assert_eq!(report.frame_bytes, frame.len());
        assert_eq!(report.observer_fingerprint.len(), 12);

        let mut wrong_sha256 = frame_sha256;
        wrong_sha256[0] ^= 0x80;
        assert!(verify_directory_observation_certificate_frame(
            &frame,
            &wrong_sha256,
            &trust_policy,
            1_700_000_702,
        )
        .is_err());

        let mut tampered = frame.clone();
        let final_byte = tampered.last_mut().unwrap();
        *final_byte ^= 0x01;
        let tampered_sha256 = Sha256::digest(&tampered).into();
        assert!(verify_directory_observation_certificate_frame(
            &tampered,
            &tampered_sha256,
            &trust_policy,
            1_700_000_702,
        )
        .is_err());

        let oversized =
            vec![0u8; MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES.saturating_add(1)];
        let oversized_sha256 = Sha256::digest(&oversized).into();
        assert!(verify_directory_observation_certificate_frame(
            &oversized,
            &oversized_sha256,
            &trust_policy,
            1_700_000_702,
        )
        .is_err());

        let wrong_observer_policy =
            parse_observation_certificate_trust_policy(&"e9".repeat(32), &allowed_witness_hex, 2)
                .unwrap();
        assert!(verify_directory_observation_certificate_frame(
            &frame,
            &frame_sha256,
            &wrong_observer_policy,
            1_700_000_702,
        )
        .is_err());

        let untrusted_witness_hex = vec![hex::encode(witnesses[0]), "ea".repeat(32)];
        let untrusted_witness_policy = parse_observation_certificate_trust_policy(
            &hex::encode(observer),
            &untrusted_witness_hex,
            1,
        )
        .unwrap();
        assert!(verify_directory_observation_certificate_frame(
            &frame,
            &frame_sha256,
            &untrusted_witness_policy,
            1_700_000_702,
        )
        .is_err());
        assert!(parse_observation_certificate_trust_policy(
            &hex::encode(observer),
            &[hex::encode(witnesses[0]), hex::encode(witnesses[0])],
            1,
        )
        .is_err());
    }

    #[test]
    fn directory_replica_resolution_cli_requires_explicit_cas_fields() {
        let digest = "11".repeat(32);
        let producer = "22".repeat(32);
        let tip_hash = "33".repeat(32);
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "resolve-quarantine",
            "--digest",
            &digest,
            "--producer",
            &producer,
            "--expected-tip-height",
            "7",
            "--expected-tip-hash",
            &tip_hash,
            "--expected-kind",
            "signed_tip_fork",
            "--confirm-incident",
            &digest,
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::ResolveQuarantine {
            expected_tip_height,
            expected_previous_resolution_digest,
            ..
        }) = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert_eq!(expected_tip_height, 7);
        assert_eq!(expected_previous_resolution_digest, None);
    }

    #[test]
    fn directory_replica_carrier_smoke_cli_is_read_only_and_json_capable() {
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "carrier-smoke",
            "--json",
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::CarrierSmoke { json, config }) =
            cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert!(json);
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
    }

    #[test]
    fn directory_replica_carrier_smoke_targets_operator_api_not_udp_tunnel() {
        let mut config = ServerConfig::default();
        config.memchain.api_listen_addr = "0.0.0.0:19421".parse().unwrap();

        assert_eq!(
            directory_replica_operator_smoke_url(&config, "carrier-smoke"),
            "http://127.0.0.1:19421/api/discovery/directory/carrier-smoke"
        );
    }

    #[test]
    fn directory_replica_cold_bootstrap_smoke_cli_is_read_only_and_json_capable() {
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "directory-replica",
            "carrier-cold-bootstrap-smoke",
            "--json",
        ])
        .unwrap();
        let Commands::DirectoryReplica(DirectoryReplicaCommands::CarrierColdBootstrapSmoke {
            json,
            config,
        }) = cli.command
        else {
            panic!("unexpected CLI command")
        };
        assert!(json);
        assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
    }

    #[test]
    fn strict_hex_parser_rejects_ambiguous_or_unbounded_identifiers() {
        assert_eq!(parse_hex32(&"a5".repeat(32), "test").unwrap(), [0xa5; 32]);
        assert!(parse_hex32(&"a5".repeat(31), "test").is_err());
        assert!(parse_hex32(&format!("{}gg", "a5".repeat(31)), "test").is_err());
    }

    #[test]
    fn memchain_verify_aof_cli_accepts_explicit_read_only_path() {
        let cli = Cli::try_parse_from([
            "aeronyx-server",
            "memchain",
            "verify-aof",
            "--path",
            "/tmp/test.memchain",
        ])
        .unwrap();
        let Commands::Memchain(MemchainCommands::VerifyAof { path, .. }) = cli.command else {
            panic!("unexpected CLI command")
        };
        assert_eq!(path, Some(PathBuf::from("/tmp/test.memchain")));
    }
}
