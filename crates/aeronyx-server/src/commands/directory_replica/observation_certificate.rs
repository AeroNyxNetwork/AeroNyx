// ============================================
// File: crates/aeronyx-server/src/commands/directory_replica/observation_certificate.rs
// ============================================
//! # Directory observation certificates
//!
//! Owns portable observation-certificate handling: the offline exact-frame
//! verifier, host-local durable import, and authenticated pinned-source pull
//! with its freshness gate.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Context;

use aeronyx_core::protocol::discovery::MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES;
use aeronyx_server::api::directory_replica_sync::{
    build_directory_certificate_exchange_http_client, fetch_authenticated_observation_certificate,
};
use aeronyx_server::services::directory_replica::{
    verify_directory_observation_certificate_frame as verify_portable_observation_certificate_frame,
    DirectoryObservationCertificateTrustPolicy,
};

use super::quarantine::open_directory_replica_store;
use crate::commands::helpers::{parse_hex32, unix_timestamp_now};

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
pub(super) fn cmd_directory_replica_verify_observation_certificate(
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
pub(super) async fn cmd_directory_replica_import_observation_certificate(
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
pub(super) async fn cmd_directory_replica_pull_observation_certificate(
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

#[cfg(test)]
mod tests;
