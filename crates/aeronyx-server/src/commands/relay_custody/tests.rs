// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/tests.rs
// ============================================
//! # Tests: `relay-custody` commands
//!
//! Unit tests for `relay-custody` commands, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use std::path::PathBuf;

use clap::Parser;

use crate::{Cli, Commands};

#[test]
fn relay_custody_cli_defaults_to_dry_run_and_gates_execution() {
    // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] The destructive form
    // must be impossible to express accidentally through a single flag.
    let audit =
        Cli::try_parse_from(["aeronyx-server", "relay-custody", "audit", "--json"]).unwrap();
    let Commands::RelayCustody(RelayCustodyCommands::Audit { config, json }) = audit.command else {
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

    // [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] Network
    // collection is explicit, snapshot-bound, time-bounded, and always
    // re-audits receipts under a bounded freshness policy.
    let collect_witnesses = Cli::try_parse_from([
        "aeronyx-server",
        "relay-custody",
        "collect-audit-witnesses",
        "--discovery-snapshot",
        "/root/witness-snapshot.json",
        "--timeout-seconds",
        "12",
        "--max-age-seconds",
        "3600",
        "--json",
    ])
    .expect("custody witness collection form must parse");
    let Commands::RelayCustody(RelayCustodyCommands::CollectAuditWitnesses {
        config,
        discovery_snapshot,
        timeout_seconds,
        max_age_seconds,
        json,
    }) = collect_witnesses.command
    else {
        panic!("unexpected CLI command")
    };
    assert_eq!(config, PathBuf::from("/etc/aeronyx/server.toml"));
    assert_eq!(
        discovery_snapshot,
        PathBuf::from("/root/witness-snapshot.json")
    );
    assert_eq!(timeout_seconds, 12);
    assert_eq!(max_age_seconds, 3600);
    assert!(json);
    for invalid_timeout in ["0", "61"] {
        assert!(Cli::try_parse_from([
            "aeronyx-server",
            "relay-custody",
            "collect-audit-witnesses",
            "--discovery-snapshot",
            "/root/witness-snapshot.json",
            "--timeout-seconds",
            invalid_timeout,
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

    let plan = Cli::try_parse_from(["aeronyx-server", "relay-custody", "restore-plan", "--json"])
        .expect("authenticated restore plan form must parse");
    let Commands::RelayCustody(RelayCustodyCommands::RestorePlan { config, json }) = plan.command
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
        Cli::try_parse_from(["aeronyx-server", "relay-custody", "prune", "--execute",]).is_err()
    );
    assert!(Cli::try_parse_from([
        "aeronyx-server",
        "relay-custody",
        "prune",
        "--confirm-node-stopped",
    ])
    .is_err());
}
