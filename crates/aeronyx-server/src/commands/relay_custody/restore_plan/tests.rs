// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/restore_plan/tests.rs
// ============================================
//! # Tests: Relay custody restore plans
//!
//! Unit tests for the relay custody restore plans, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

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
