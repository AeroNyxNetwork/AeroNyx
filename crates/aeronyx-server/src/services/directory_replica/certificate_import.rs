// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/certificate_import.rs
// ============================================
//! # Portable observation certificate import
//!
//! Owns the operator-pinned certificate trust policy, frame verification, the
//! import report, and the node-signed, hash-linked import-history entry.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    decode_directory_observation_certificate, encode_directory_observation_certificate, Digest,
    DirectoryObservationCertificateV1, DirectoryReplicaStoreError, IdentityKeyPair, Sha256,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    DIRECTORY_OBSERVATION_CERTIFICATE_IMPORT_TIMESTAMP_SKEW_SECS,
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES, MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1,
};

/// Operator-owned trust anchors for one portable observation certificate.
///
/// [PORTABLE-CERTIFICATE-IMPORT 2026-07-26 by Codex] Valid signatures prove
/// authorship, not authority. Every verifier and importer therefore supplies a
/// pinned observer, a bounded witness allowlist, and a local minimum. The
/// certificate's self-declared threshold can never weaken this local policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryObservationCertificateTrustPolicy {
    expected_observer: [u8; 32],
    allowed_witnesses: Vec<[u8; 32]>,
    minimum_witnesses: u16,
}

impl DirectoryObservationCertificateTrustPolicy {
    /// Builds one canonical pinned trust policy.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError::Request`] for zero identities,
    /// duplicate witnesses, observer/witness overlap, or an invalid threshold.
    pub fn new(
        expected_observer: [u8; 32],
        mut allowed_witnesses: Vec<[u8; 32]>,
        minimum_witnesses: u16,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        if expected_observer == [0u8; 32]
            || allowed_witnesses.is_empty()
            || allowed_witnesses.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryReplicaStoreError::Request(
                "portable certificate trust policy identity set is invalid".to_string(),
            ));
        }
        allowed_witnesses.sort_unstable();
        if allowed_witnesses
            .iter()
            .any(|witness| *witness == [0u8; 32] || *witness == expected_observer)
            || allowed_witnesses
                .windows(2)
                .any(|witnesses| witnesses[0] == witnesses[1])
            || minimum_witnesses == 0
            || usize::from(minimum_witnesses) > allowed_witnesses.len()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "portable certificate trust policy is not canonical".to_string(),
            ));
        }
        Ok(Self {
            expected_observer,
            allowed_witnesses,
            minimum_witnesses,
        })
    }

    /// Pinned observer node identity.
    #[must_use]
    pub const fn expected_observer(&self) -> [u8; 32] {
        self.expected_observer
    }

    /// Canonically sorted pinned witness identities.
    #[must_use]
    pub fn allowed_witnesses(&self) -> &[[u8; 32]] {
        &self.allowed_witnesses
    }

    /// Locally required distinct witness count.
    #[must_use]
    pub const fn minimum_witnesses(&self) -> u16 {
        self.minimum_witnesses
    }

    fn verify(
        &self,
        certificate: &DirectoryObservationCertificateV1,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if certificate.checkpoint.observer != self.expected_observer {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate observer does not match the pinned observer".to_string(),
            ));
        }
        if certificate.receipts.iter().any(|receipt| {
            self.allowed_witnesses
                .binary_search(&receipt.responder)
                .is_err()
        }) {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate contains a witness outside the allowed set".to_string(),
            ));
        }
        if certificate.receipts.len() < usize::from(self.minimum_witnesses) {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate does not satisfy the local witness threshold".to_string(),
            ));
        }
        Ok(())
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationCertificateTrustPolicy-v1");
        hasher.update(AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        hasher.update(self.expected_observer);
        hasher.update(self.minimum_witnesses.to_le_bytes());
        hasher.update(
            u64::try_from(self.allowed_witnesses.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for witness in &self.allowed_witnesses {
            hasher.update(witness);
        }
        hasher.finalize().into()
    }
}

/// Fully verified portable observation certificate and transport bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedDirectoryObservationCertificate {
    /// Decoded canonical certificate.
    pub certificate: DirectoryObservationCertificateV1,
    /// Stable certificate identity over checkpoint and exact receipts.
    pub certificate_id: [u8; 32],
    /// SHA-256 of the exact canonical frame supplied by the operator.
    pub certificate_sha256: [u8; 32],
    /// Stable digest of the local pinned trust policy.
    pub policy_digest: [u8; 32],
    /// Host time used for signature and timestamp validation.
    pub verified_at: u64,
}

/// Verifies exact bytes, canonical encoding, signatures, bindings, and pins.
///
/// # Errors
/// Returns [`DirectoryReplicaStoreError::Request`] for a malformed, oversized,
/// non-canonical, mistimed, incorrectly signed, or locally untrusted frame.
pub fn verify_directory_observation_certificate_frame(
    frame: &[u8],
    expected_sha256: &[u8; 32],
    trust_policy: &DirectoryObservationCertificateTrustPolicy,
    verified_at: u64,
) -> Result<VerifiedDirectoryObservationCertificate, DirectoryReplicaStoreError> {
    if verified_at == 0
        || frame.is_empty()
        || frame.len() > MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES
    {
        return Err(DirectoryReplicaStoreError::Request(
            "observation certificate frame size or verification time is invalid".to_string(),
        ));
    }
    let certificate_sha256: [u8; 32] = Sha256::digest(frame).into();
    if &certificate_sha256 != expected_sha256 {
        return Err(DirectoryReplicaStoreError::Request(
            "observation certificate frame SHA-256 mismatch".to_string(),
        ));
    }
    let certificate = decode_directory_observation_certificate(frame).map_err(|error| {
        DirectoryReplicaStoreError::Request(format!(
            "observation certificate decode failed: {error}"
        ))
    })?;
    let canonical_frame =
        encode_directory_observation_certificate(&certificate).map_err(|error| {
            DirectoryReplicaStoreError::Request(format!(
                "observation certificate canonical encoding failed: {error}"
            ))
        })?;
    if canonical_frame != frame {
        return Err(DirectoryReplicaStoreError::Request(
            "observation certificate frame is not canonically encoded".to_string(),
        ));
    }
    certificate
        .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, verified_at)
        .map_err(|error| {
            DirectoryReplicaStoreError::Request(format!(
                "observation certificate signature verification failed: {error}"
            ))
        })?;
    trust_policy.verify(&certificate)?;
    let certificate_id = certificate.hash();
    Ok(VerifiedDirectoryObservationCertificate {
        certificate,
        certificate_id,
        certificate_sha256,
        policy_digest: trust_policy.digest(),
        verified_at,
    })
}

/// Result of one durable host-local certificate import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryObservationCertificateImportReport {
    /// True only when a new signed import row was appended.
    pub inserted: bool,
    /// Local append-only import sequence.
    pub import_sequence: u64,
    /// Hash-linked digest of the local signed import row.
    pub import_digest: [u8; 32],
    /// Stable identity of the imported portable certificate.
    pub certificate_id: [u8; 32],
    /// SHA-256 of the exact imported certificate frame.
    pub certificate_sha256: [u8; 32],
    /// External observer represented by this certificate.
    pub observer: [u8; 32],
    /// External observer checkpoint sequence.
    pub checkpoint_sequence: u64,
    /// External observer checkpoint hash.
    pub checkpoint_hash: [u8; 32],
    /// Number of certificates retained after the operation.
    pub retained_certificates: u64,
    /// Host time at which the frame and local pins were verified.
    pub verified_at: u64,
}

/// One local-node-signed link in the imported certificate history.
///
/// The signature does not promote foreign evidence into consensus. It proves
/// only which exact bytes and local trust policy this node accepted, and where
/// that decision sits in this node's append-only history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DirectoryObservationCertificateImportEntry {
    pub(super) import_sequence: u64,
    pub(super) previous_import_digest: [u8; 32],
    pub(super) certificate_id: [u8; 32],
    pub(super) observer: [u8; 32],
    pub(super) checkpoint_sequence: u64,
    pub(super) checkpoint_hash: [u8; 32],
    pub(super) checkpoint_observed_at: u64,
    pub(super) certificate_sha256: [u8; 32],
    pub(super) policy_digest: [u8; 32],
    pub(super) verified_at: u64,
    pub(super) importer_node_id: [u8; 32],
    pub(super) signature: [u8; 64],
}

impl DirectoryObservationCertificateImportEntry {
    pub(super) fn sign(
        identity: &IdentityKeyPair,
        import_sequence: u64,
        previous_import_digest: [u8; 32],
        verified: &VerifiedDirectoryObservationCertificate,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let checkpoint = &verified.certificate.checkpoint;
        let mut entry = Self {
            import_sequence,
            previous_import_digest,
            certificate_id: verified.certificate_id,
            observer: checkpoint.observer,
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash: checkpoint.hash(),
            checkpoint_observed_at: checkpoint.observed_at,
            certificate_sha256: verified.certificate_sha256,
            policy_digest: verified.policy_digest,
            verified_at: verified.verified_at,
            importer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        entry.validate_unsigned_fields()?;
        entry.signature = identity.sign(&entry.signing_bytes());
        Ok(entry)
    }

    pub(super) fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.import_sequence == 0
            || self.certificate_id == [0u8; 32]
            || self.observer == [0u8; 32]
            || self.checkpoint_sequence == 0
            || self.checkpoint_hash == [0u8; 32]
            || self.checkpoint_observed_at == 0
            || self.certificate_sha256 == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.verified_at == 0
            || self.importer_node_id == [0u8; 32]
            || self.checkpoint_observed_at
                > self
                    .verified_at
                    .saturating_add(DIRECTORY_OBSERVATION_CERTIFICATE_IMPORT_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation certificate import contains an invalid sentinel".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(360);
        bytes.extend_from_slice(b"AeroNyx-DirectoryObservationCertificateImport-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.import_sequence.to_le_bytes());
        bytes.extend_from_slice(&self.previous_import_digest);
        bytes.extend_from_slice(&self.certificate_id);
        bytes.extend_from_slice(&self.observer);
        bytes.extend_from_slice(&self.checkpoint_sequence.to_le_bytes());
        bytes.extend_from_slice(&self.checkpoint_hash);
        bytes.extend_from_slice(&self.checkpoint_observed_at.to_le_bytes());
        bytes.extend_from_slice(&self.certificate_sha256);
        bytes.extend_from_slice(&self.policy_digest);
        bytes.extend_from_slice(&self.verified_at.to_le_bytes());
        bytes.extend_from_slice(&self.importer_node_id);
        bytes
    }

    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct ObservationCertificateImportAudit {
    pub(super) imports: u64,
    pub(super) head: [u8; 32],
}
