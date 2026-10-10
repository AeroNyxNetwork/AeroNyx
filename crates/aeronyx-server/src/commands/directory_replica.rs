// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica.rs
// ============================================
//! # `directory-replica` commands
//!
//! Owns the privileged, host-local `directory-replica` dispatcher. Observation
//! certificates, operator smoke calls and quarantine resolution live in the
//! child modules below.
//!
//! ## Module Layout
//! - `directory_replica/observation_certificate.rs`: certificate verify / import / pull
//! - `directory_replica/operator_smoke.rs`: loopback operator API smoke calls
//! - `directory_replica/quarantine.rs`: incident inspection, resolution, store opening
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use crate::DirectoryReplicaCommands;
use observation_certificate::{
    cmd_directory_replica_import_observation_certificate,
    cmd_directory_replica_pull_observation_certificate,
    cmd_directory_replica_verify_observation_certificate,
};
use operator_smoke::{
    cmd_directory_replica_carrier_cold_bootstrap_smoke, cmd_directory_replica_carrier_smoke,
};
use quarantine::{
    cmd_directory_replica_inspect, cmd_directory_replica_resolve, DirectoryReplicaResolveRequest,
};

mod observation_certificate;
mod operator_smoke;
mod quarantine;

/// Runs privileged Directory Replica operations without a network endpoint.
pub async fn cmd_directory_replica(command: DirectoryReplicaCommands) -> anyhow::Result<()> {
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
