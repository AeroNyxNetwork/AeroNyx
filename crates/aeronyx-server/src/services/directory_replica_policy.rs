// ============================================================================
// File: crates/aeronyx-server/src/services/directory_replica_policy.rs
// ============================================================================
//! Signed policy-history reconciliation and restart audit for the replica store.
//!
//! [DIRECTORY-REPLICA-POLICY-MODULE 2026-09-24 by Codex] This private sibling
//! owns the witness, route-domain, and attestor policy state transitions plus
//! remote anchor receipt verification. It does not change the persisted schema,
//! canonical wire bytes, public API, or failure classification.

use super::{
    bytes32, bytes64, nonnegative_i64_to_u64, positive_i64_to_u64, u64_to_i64,
    DirectoryObservationWitnessPolicyAnchor, DirectoryObservationWitnessPolicyAnchorDecision,
    DirectoryObservationWitnessPolicyEpoch, DirectoryObservationWitnessPolicyReconcileReport,
    DirectoryReplicaStore, DirectoryReplicaStoreError, DirectoryRouteDomainAttestorPolicyEpoch,
    DirectoryRouteDomainAttestorPolicyReconcileReport, DirectoryRouteDomainPolicyEpoch,
    DirectoryRouteDomainPolicyReconcileReport, ObservationWitnessPolicyAudit,
    RouteDomainAttestorPolicyAudit, RouteDomainPolicyAudit,
    StoredObservationWitnessPolicyAnchorReceiptRow, StoredObservationWitnessPolicyRow,
    StoredRouteDomainAttestorPolicyRow, StoredRouteDomainPolicyRow,
    MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS, MAX_DIRECTORY_POLICY_ANCHOR_BYTES,
    MAX_DIRECTORY_ROUTE_DOMAIN_ATTESTOR_POLICY_MEMBERS,
    MAX_DIRECTORY_ROUTE_DOMAIN_POLICY_ASSIGNMENTS, RESPONSE_TIMESTAMP_SKEW_SECS,
};
use crate::config::PinnedRouteDomainAssignment;
use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_policy_anchor_request_signing_bytes,
    directory_policy_anchor_response_signing_bytes, encode_directory_sync_message,
    DirectorySyncMessage, AERONYX_DIRECTORY_MAINNET_CHAIN_ID, DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use std::collections::HashMap;

impl DirectoryReplicaStore {
    /// Reconciles validated runtime witness pins into a signed local policy
    /// epoch without rewriting historical policy or witness receipts.
    ///
    /// Reordered pins are canonicalized and therefore idempotent. A pin-set or
    /// threshold change appends exactly one node-identity-signed, hash-linked
    /// epoch and advances the metadata head in the same immediate transaction.
    /// This history describes only this node operator's corroboration policy;
    /// it does not define network membership, voting weight, consensus, fork
    /// choice, or finality.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid pins/thresholds,
    /// identity mismatch, malformed prior history, counter exhaustion, or an
    /// atomic SQLite compare-and-swap failure.
    pub(crate) fn reconcile_observation_witness_policy(
        &self,
        identity: &IdentityKeyPair,
        witness_node_ids: &[[u8; 32]],
        minimum_witnesses: usize,
        activated_at: u64,
    ) -> Result<DirectoryObservationWitnessPolicyReconcileReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id || activated_at == 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy identity or timestamp is invalid".to_string(),
            ));
        }
        let canonical_witnesses =
            canonical_observation_witness_policy_members(witness_node_ids, minimum_witnesses)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        let previous = Self::audit_observation_witness_policies(&transaction, &self.local_node_id)?;
        if let Some(current) = previous.current.as_ref() {
            if current.witness_node_ids == canonical_witnesses
                && current.minimum_witnesses == minimum_witnesses
            {
                transaction.commit()?;
                return Ok(DirectoryObservationWitnessPolicyReconcileReport {
                    appended: false,
                    epoch: current.epoch,
                    policy_digest: previous.current_digest,
                    activated_at: current.activated_at,
                    witness_members: u64::try_from(current.witness_node_ids.len()).map_err(
                        |_| {
                            DirectoryReplicaStoreError::Integrity(
                                "observation witness policy member count exceeds u64".to_string(),
                            )
                        },
                    )?,
                    minimum_witnesses: u64::try_from(current.minimum_witnesses).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "observation witness policy threshold exceeds u64".to_string(),
                        )
                    })?,
                });
            }
        }

        let epoch = previous.epochs.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness policy epoch exhausted".to_string(),
            )
        })?;
        let previous_policy_digest = previous.current_digest;
        let activated_at = previous
            .current
            .as_ref()
            .map_or(activated_at, |policy| activated_at.max(policy.activated_at));
        let policy = DirectoryObservationWitnessPolicyEpoch::sign(
            identity,
            epoch,
            previous_policy_digest,
            activated_at,
            canonical_witnesses,
            minimum_witnesses,
        )?;
        let policy_digest = policy.digest();
        let witness_node_ids = policy
            .witness_node_ids
            .iter()
            .flat_map(|node_id| node_id.iter().copied())
            .collect::<Vec<_>>();
        transaction.execute(
            "INSERT INTO directory_observation_witness_policies
                (epoch, policy_digest, previous_policy_digest, activated_at,
                 witness_threshold, witness_count, witness_node_ids,
                 signer_node_id, signature)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                u64_to_i64(policy.epoch, "observation witness policy epoch")?,
                policy_digest.as_slice(),
                policy.previous_policy_digest.as_slice(),
                u64_to_i64(
                    policy.activated_at,
                    "observation witness policy activation timestamp"
                )?,
                u64_to_i64(
                    u64::try_from(policy.minimum_witnesses).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "observation witness policy threshold exceeds u64".to_string(),
                        )
                    })?,
                    "observation witness policy threshold"
                )?,
                u64_to_i64(
                    u64::try_from(policy.witness_node_ids.len()).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "observation witness policy member count exceeds u64".to_string(),
                        )
                    })?,
                    "observation witness policy member count"
                )?,
                witness_node_ids,
                policy.signer_node_id.as_slice(),
                policy.signature.as_slice(),
            ],
        )?;
        let previous_head = previous
            .current
            .as_ref()
            .map(|_| previous.current_digest.to_vec());
        let changed = transaction.execute(
            "UPDATE directory_replica_meta
             SET witness_policy_epoch = ?1, witness_policy_head = ?2
             WHERE singleton = 1 AND witness_policy_epoch = ?3
               AND ((?4 IS NULL AND witness_policy_head IS NULL)
                    OR witness_policy_head = ?4)",
            params![
                u64_to_i64(epoch, "observation witness policy epoch")?,
                policy_digest.as_slice(),
                u64_to_i64(previous.epochs, "previous observation witness policy epoch")?,
                previous_head,
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy head compare-and-swap failed".to_string(),
            ));
        }
        let audited = Self::audit_observation_witness_policies(&transaction, &self.local_node_id)?;
        if audited.epochs != epoch || audited.current_digest != policy_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy post-append audit diverged".to_string(),
            ));
        }
        transaction.commit()?;
        Ok(DirectoryObservationWitnessPolicyReconcileReport {
            appended: true,
            epoch,
            policy_digest,
            activated_at: policy.activated_at,
            witness_members: u64::try_from(policy.witness_node_ids.len()).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy member count exceeds u64".to_string(),
                )
            })?,
            minimum_witnesses: u64::try_from(policy.minimum_witnesses).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy threshold exceeds u64".to_string(),
                )
            })?,
        })
    }

    /// Reconciles validated runtime route-domain pins into signed local history.
    ///
    /// The first disabled empty configuration remains epoch zero. After the
    /// first non-empty policy, later changes, including disabling and clearing
    /// all assignments, append a new epoch so operators cannot erase policy
    /// transitions by editing configuration. A full signature/link audit runs
    /// before and after each append in one immediate transaction.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for malformed assignments,
    /// identity mismatch, corrupt history, timestamp/counter exhaustion, or
    /// an atomic persistence failure.
    pub(crate) fn reconcile_route_domain_policy(
        &self,
        identity: &IdentityKeyPair,
        assignments: &[PinnedRouteDomainAssignment],
        strict_required: bool,
        activated_at: u64,
    ) -> Result<DirectoryRouteDomainPolicyReconcileReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id || activated_at == 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy identity or timestamp is invalid".to_string(),
            ));
        }
        let canonical_assignments =
            canonical_route_domain_policy_assignments(assignments, strict_required)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        let previous = Self::audit_route_domain_policies(&transaction, &self.local_node_id)?;
        if let Some(current) = previous.current.as_ref() {
            if current.assignments == canonical_assignments
                && current.strict_required == strict_required
            {
                transaction.commit()?;
                return Ok(route_domain_policy_report(
                    false,
                    current,
                    previous.current_digest,
                )?);
            }
        } else if canonical_assignments.is_empty() && !strict_required {
            transaction.commit()?;
            return Ok(DirectoryRouteDomainPolicyReconcileReport::default());
        }

        let epoch = previous.epochs.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity("route-domain policy epoch exhausted".to_string())
        })?;
        let activated_at = previous
            .current
            .as_ref()
            .map_or(activated_at, |policy| activated_at.max(policy.activated_at));
        let policy = DirectoryRouteDomainPolicyEpoch::sign(
            identity,
            epoch,
            previous.current_digest,
            activated_at,
            strict_required,
            canonical_assignments,
        )?;
        let policy_digest = policy.digest();
        let assignments_blob = policy
            .assignments
            .iter()
            .flat_map(|assignment| {
                assignment
                    .node_id
                    .iter()
                    .chain(assignment.route_domain.iter())
                    .copied()
            })
            .collect::<Vec<_>>();
        transaction.execute(
            "INSERT INTO directory_route_domain_policies
                (epoch, policy_digest, previous_policy_digest, activated_at,
                 strict_required, assignment_count, assignments,
                 signer_node_id, signature)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                u64_to_i64(policy.epoch, "route-domain policy epoch")?,
                policy_digest.as_slice(),
                policy.previous_policy_digest.as_slice(),
                u64_to_i64(
                    policy.activated_at,
                    "route-domain policy activation timestamp"
                )?,
                i64::from(policy.strict_required),
                u64_to_i64(
                    u64::try_from(policy.assignments.len()).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "route-domain policy assignment count exceeds u64".to_string(),
                        )
                    })?,
                    "route-domain policy assignment count"
                )?,
                assignments_blob,
                policy.signer_node_id.as_slice(),
                policy.signature.as_slice(),
            ],
        )?;
        let previous_head = previous
            .current
            .as_ref()
            .map(|_| previous.current_digest.to_vec());
        let changed = transaction.execute(
            "UPDATE directory_replica_meta
             SET route_domain_policy_epoch = ?1, route_domain_policy_head = ?2
             WHERE singleton = 1 AND route_domain_policy_epoch = ?3
               AND ((?4 IS NULL AND route_domain_policy_head IS NULL)
                    OR route_domain_policy_head = ?4)",
            params![
                u64_to_i64(epoch, "route-domain policy epoch")?,
                policy_digest.as_slice(),
                u64_to_i64(previous.epochs, "previous route-domain policy epoch")?,
                previous_head,
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy head compare-and-swap failed".to_string(),
            ));
        }
        let audited = Self::audit_route_domain_policies(&transaction, &self.local_node_id)?;
        if audited.epochs != epoch || audited.current_digest != policy_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy post-append audit diverged".to_string(),
            ));
        }
        transaction.commit()?;
        route_domain_policy_report(true, &policy, policy_digest)
    }

    /// Reconciles validated route-domain attestor pins into signed local history.
    ///
    /// The initial disabled policy remains epoch zero. Once enabled, every
    /// pin-set, threshold, strictness, or disable transition appends a signed,
    /// hash-linked epoch. This prevents a later configuration edit from
    /// silently rewriting the local trust-root history used by route selection.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for malformed pins or thresholds,
    /// identity mismatch, corrupt history, counter exhaustion, or an atomic
    /// persistence failure.
    pub(crate) fn reconcile_route_domain_attestor_policy(
        &self,
        identity: &IdentityKeyPair,
        attestor_node_ids: &[[u8; 32]],
        minimum_attestors: usize,
        strict_required: bool,
        activated_at: u64,
    ) -> Result<DirectoryRouteDomainAttestorPolicyReconcileReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id || activated_at == 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy identity or timestamp is invalid".to_string(),
            ));
        }
        let canonical_attestors = canonical_route_domain_attestor_policy_members(
            attestor_node_ids,
            minimum_attestors,
            strict_required,
        )?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        let previous =
            Self::audit_route_domain_attestor_policies(&transaction, &self.local_node_id)?;
        if let Some(current) = previous.current.as_ref() {
            if current.attestor_node_ids == canonical_attestors
                && current.minimum_attestors == minimum_attestors
                && current.strict_required == strict_required
            {
                transaction.commit()?;
                return route_domain_attestor_policy_report(
                    false,
                    current,
                    previous.current_digest,
                );
            }
        } else if canonical_attestors.is_empty() && !strict_required && minimum_attestors == 1 {
            transaction.commit()?;
            return Ok(DirectoryRouteDomainAttestorPolicyReconcileReport::default());
        }

        let epoch = previous.epochs.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy epoch exhausted".to_string(),
            )
        })?;
        let activated_at = previous
            .current
            .as_ref()
            .map_or(activated_at, |policy| activated_at.max(policy.activated_at));
        let policy = DirectoryRouteDomainAttestorPolicyEpoch::sign(
            identity,
            epoch,
            previous.current_digest,
            activated_at,
            strict_required,
            canonical_attestors,
            minimum_attestors,
        )?;
        let policy_digest = policy.digest();
        let attestor_node_ids = policy
            .attestor_node_ids
            .iter()
            .flat_map(|node_id| node_id.iter().copied())
            .collect::<Vec<_>>();
        transaction.execute(
            "INSERT INTO directory_route_domain_attestor_policies
                (epoch, policy_digest, previous_policy_digest, activated_at,
                 strict_required, attestor_threshold, attestor_count,
                 attestor_node_ids, signer_node_id, signature)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                u64_to_i64(policy.epoch, "route-domain attestor policy epoch")?,
                policy_digest.as_slice(),
                policy.previous_policy_digest.as_slice(),
                u64_to_i64(
                    policy.activated_at,
                    "route-domain attestor policy activation timestamp"
                )?,
                i64::from(policy.strict_required),
                u64_to_i64(
                    u64::try_from(policy.minimum_attestors).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "route-domain attestor threshold exceeds u64".to_string(),
                        )
                    })?,
                    "route-domain attestor policy threshold"
                )?,
                u64_to_i64(
                    u64::try_from(policy.attestor_node_ids.len()).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "route-domain attestor count exceeds u64".to_string(),
                        )
                    })?,
                    "route-domain attestor policy member count"
                )?,
                attestor_node_ids,
                policy.signer_node_id.as_slice(),
                policy.signature.as_slice(),
            ],
        )?;
        let previous_head = previous
            .current
            .as_ref()
            .map(|_| previous.current_digest.to_vec());
        let changed = transaction.execute(
            "UPDATE directory_replica_meta
             SET route_domain_attestor_policy_epoch = ?1,
                 route_domain_attestor_policy_head = ?2
             WHERE singleton = 1 AND route_domain_attestor_policy_epoch = ?3
               AND ((?4 IS NULL AND route_domain_attestor_policy_head IS NULL)
                    OR route_domain_attestor_policy_head = ?4)",
            params![
                u64_to_i64(epoch, "route-domain attestor policy epoch")?,
                policy_digest.as_slice(),
                u64_to_i64(
                    previous.epochs,
                    "previous route-domain attestor policy epoch"
                )?,
                previous_head,
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy head compare-and-swap failed".to_string(),
            ));
        }
        let audited =
            Self::audit_route_domain_attestor_policies(&transaction, &self.local_node_id)?;
        if audited.epochs != epoch || audited.current_digest != policy_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy post-append audit diverged".to_string(),
            ));
        }
        transaction.commit()?;
        route_domain_attestor_policy_report(true, &policy, policy_digest)
    }

    /// Verifies that the metadata-anchored signed policy head contains the
    /// exact canonical runtime pin set and threshold.
    ///
    /// This comparison returns only a boolean to callers. It never widens the
    /// public status boundary to include policy member identities or digests.
    pub(crate) fn observation_witness_policy_matches(
        &self,
        witness_node_ids: &[[u8; 32]],
        minimum_witnesses: usize,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let canonical_witnesses =
            canonical_observation_witness_policy_members(witness_node_ids, minimum_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let current =
            Self::load_current_observation_witness_policy(&connection, &self.local_node_id)?;
        Ok(current.current.is_some_and(|policy| {
            policy.witness_node_ids == canonical_witnesses
                && policy.minimum_witnesses == minimum_witnesses
        }))
    }

    /// Returns the current opaque policy head after verifying its local chain.
    ///
    /// Policy member identities remain in the local signed policy row and are
    /// deliberately not included in this export object.
    pub(crate) fn current_observation_witness_policy_anchor(
        &self,
    ) -> Result<Option<DirectoryObservationWitnessPolicyAnchor>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let current =
            Self::load_current_observation_witness_policy(&connection, &self.local_node_id)?;
        Ok(current
            .current
            .map(|policy| DirectoryObservationWitnessPolicyAnchor {
                epoch: policy.epoch,
                previous_policy_digest: policy.previous_policy_digest,
                policy_digest: current.current_digest,
            }))
    }

    /// Evaluates and durably retains one authenticated foreign policy head.
    ///
    /// The first head for an observer is a signed trust-on-first-observation
    /// anchor. Later heads must be exact retries or the immediately linked next
    /// epoch. Rollback, same-epoch conflict, and gaps never mutate persistence.
    /// Member identities are never transmitted or stored by this operation.
    pub(crate) fn persist_remote_observation_witness_policy_anchor(
        &self,
        request: &DirectorySyncMessage,
        observed_at: u64,
    ) -> Result<DirectoryObservationWitnessPolicyAnchorDecision, DirectoryReplicaStoreError> {
        let request_blob = encode_directory_sync_message(request)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if request_blob.len() > MAX_DIRECTORY_POLICY_ANCHOR_BYTES {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor request exceeds size bound".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
            chain_id,
            request_id,
            requester,
            request_timestamp,
            policy_epoch,
            previous_policy_digest,
            policy_digest,
            signature,
        } = request
        else {
            return Err(DirectoryReplicaStoreError::Request(
                "unexpected observation policy anchor request".to_string(),
            ));
        };
        let position_valid = (*policy_epoch == 1 && *previous_policy_digest == [0u8; 32])
            || (*policy_epoch > 1 && *previous_policy_digest != [0u8; 32]);
        if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || *requester == [0u8; 32]
            || *requester == self.local_node_id
            || *policy_digest == [0u8; 32]
            || !position_valid
            || *request_timestamp == 0
            || observed_at.abs_diff(*request_timestamp) > RESPONSE_TIMESTAMP_SKEW_SECS
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor request contract mismatch".to_string(),
            ));
        }
        let signing_bytes = directory_policy_anchor_request_signing_bytes(
            chain_id,
            request_id,
            requester,
            *request_timestamp,
            *policy_epoch,
            previous_policy_digest,
            policy_digest,
        );
        IdentityPublicKey::from_bytes(requester)
            .and_then(|key| key.verify(&signing_bytes, signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Request(
                    "observation policy anchor request signature is invalid".to_string(),
                )
            })?;

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let latest = transaction
            .query_row(
                "SELECT policy_epoch, policy_digest
                 FROM directory_observation_remote_policy_anchors
                 WHERE observer = ?1 ORDER BY policy_epoch DESC LIMIT 1",
                params![requester.as_slice()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?
            .map(|(epoch, digest)| {
                Ok::<_, DirectoryReplicaStoreError>((
                    positive_i64_to_u64(epoch, "remote policy anchor epoch")?,
                    bytes32(&digest, "remote policy anchor digest")?,
                ))
            })
            .transpose()?;
        if let Some((latest_epoch, latest_digest)) = latest {
            if *policy_epoch < latest_epoch {
                return Ok(DirectoryObservationWitnessPolicyAnchorDecision::Rollback);
            }
            if *policy_epoch == latest_epoch {
                return Ok(if *policy_digest == latest_digest {
                    DirectoryObservationWitnessPolicyAnchorDecision::Accepted
                } else {
                    DirectoryObservationWitnessPolicyAnchorDecision::Conflict
                });
            }
            if *policy_epoch != latest_epoch.saturating_add(1)
                || *previous_policy_digest != latest_digest
            {
                return Ok(DirectoryObservationWitnessPolicyAnchorDecision::HistoryGap);
            }
        }
        transaction.execute(
            "INSERT INTO directory_observation_remote_policy_anchors
                (observer, policy_epoch, previous_policy_digest, policy_digest,
                 request_timestamp, request_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                requester.as_slice(),
                u64_to_i64(*policy_epoch, "remote policy anchor epoch")?,
                previous_policy_digest.as_slice(),
                policy_digest.as_slice(),
                u64_to_i64(*request_timestamp, "remote policy anchor timestamp")?,
                request_blob,
            ],
        )?;
        transaction.commit()?;
        Ok(DirectoryObservationWitnessPolicyAnchorDecision::Accepted)
    }

    /// Persists one accepted external receipt for an exact local policy head.
    /// Exact retries are idempotent; a witness cannot replace its receipt for
    /// the same epoch with another digest.
    pub(crate) fn persist_observation_witness_policy_anchor_receipt(
        &self,
        response: &DirectorySyncMessage,
        observed_at: u64,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let response_blob = encode_directory_sync_message(response)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if response_blob.len() > MAX_DIRECTORY_POLICY_ANCHOR_BYTES {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor response exceeds size bound".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id,
            request_id,
            observer,
            policy_epoch,
            policy_digest,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Request(
                "unexpected observation policy anchor response".to_string(),
            ));
        };
        if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || *observer != self.local_node_id
            || *responder == [0u8; 32]
            || *responder == self.local_node_id
            || *policy_epoch == 0
            || *policy_digest == [0u8; 32]
            || *outcome != DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1
            || *response_timestamp == 0
            || observed_at.abs_diff(*response_timestamp) > RESPONSE_TIMESTAMP_SKEW_SECS
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor response contract mismatch".to_string(),
            ));
        }
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            chain_id,
            request_id,
            observer,
            *policy_epoch,
            policy_digest,
            responder,
            *response_timestamp,
            *outcome,
        );
        IdentityPublicKey::from_bytes(responder)
            .and_then(|key| key.verify(&signing_bytes, signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Request(
                    "observation policy anchor response signature is invalid".to_string(),
                )
            })?;

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current =
            Self::load_current_observation_witness_policy(&transaction, &self.local_node_id)?;
        let Some(current_policy) = current.current else {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor receipt references an unknown policy".to_string(),
            ));
        };
        if current_policy.epoch != *policy_epoch
            || current.current_digest != *policy_digest
            || !current_policy.witness_node_ids.contains(responder)
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor receipt is outside the current local policy".to_string(),
            ));
        }
        let existing: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT policy_digest FROM directory_observation_policy_anchor_receipts
                 WHERE policy_epoch = ?1 AND witness_node_id = ?2",
                params![
                    u64_to_i64(*policy_epoch, "policy anchor receipt epoch")?,
                    responder.as_slice()
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(existing) = existing {
            if bytes32(&existing, "existing policy anchor receipt digest")? != *policy_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "policy anchor witness signed conflicting digests at one epoch".to_string(),
                ));
            }
            return Ok(false);
        }
        transaction.execute(
            "INSERT INTO directory_observation_policy_anchor_receipts
                (policy_epoch, policy_digest, observer, witness_node_id,
                 witnessed_at, response_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                u64_to_i64(*policy_epoch, "policy anchor receipt epoch")?,
                policy_digest.as_slice(),
                observer.as_slice(),
                responder.as_slice(),
                u64_to_i64(*response_timestamp, "policy anchor receipt timestamp")?,
                response_blob,
            ],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    /// Counts verified current-pin receipts for one exact local policy head.
    pub(crate) fn verified_observation_witness_policy_anchor_count_for_pins(
        &self,
        policy_epoch: u64,
        policy_digest: &[u8; 32],
        eligible_witnesses: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        if policy_epoch == 0 || *policy_digest == [0u8; 32] || eligible_witnesses.is_empty() {
            return Ok(0);
        }
        let eligible = Self::validate_observation_witness_eligibility(eligible_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let receipts = Self::audit_current_observation_policy_anchor_receipts(
            &connection,
            &self.local_node_id,
            policy_epoch,
            policy_digest,
            observed_at,
        )?;
        u64::try_from(
            receipts
                .into_iter()
                .filter(|(epoch, digest, witness)| {
                    *epoch == policy_epoch && digest == policy_digest && eligible.contains(witness)
                })
                .count(),
        )
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt count exceeds u64".to_string(),
            )
        })
    }

    /// Returns current pins with verified receipts for one exact policy head.
    pub(crate) fn verified_observation_witness_policy_anchor_witnesses_for_pins(
        &self,
        policy_epoch: u64,
        policy_digest: &[u8; 32],
        eligible_witnesses: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
        if policy_epoch == 0 || *policy_digest == [0u8; 32] || eligible_witnesses.is_empty() {
            return Ok(Vec::new());
        }
        let eligible = Self::validate_observation_witness_eligibility(eligible_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let mut witnesses = Self::audit_current_observation_policy_anchor_receipts(
            &connection,
            &self.local_node_id,
            policy_epoch,
            policy_digest,
            observed_at,
        )?
        .into_iter()
        .filter_map(|(epoch, digest, witness)| {
            (epoch == policy_epoch && digest == *policy_digest && eligible.contains(&witness))
                .then_some(witness)
        })
        .collect::<Vec<_>>();
        witnesses.sort_unstable();
        Ok(witnesses)
    }

    fn decode_observation_witness_policy(
        row: StoredObservationWitnessPolicyRow,
    ) -> Result<([u8; 32], DirectoryObservationWitnessPolicyEpoch), DirectoryReplicaStoreError>
    {
        let epoch = positive_i64_to_u64(row.epoch, "observation witness policy epoch")?;
        let activated_at = positive_i64_to_u64(
            row.activated_at,
            "observation witness policy activation timestamp",
        )?;
        let minimum_witnesses = usize::try_from(positive_i64_to_u64(
            row.witness_threshold,
            "observation witness policy threshold",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness policy threshold exceeds usize".to_string(),
            )
        })?;
        let witness_count = usize::try_from(nonnegative_i64_to_u64(
            row.witness_count,
            "observation witness policy member count",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness policy member count exceeds usize".to_string(),
            )
        })?;
        if witness_count > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS
            || row.witness_node_ids.len() != witness_count.saturating_mul(32)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy member blob is malformed".to_string(),
            ));
        }
        let witness_node_ids = row
            .witness_node_ids
            .chunks_exact(32)
            .map(|node_id| bytes32(node_id, "observation witness policy member"))
            .collect::<Result<Vec<_>, _>>()?;
        let policy = DirectoryObservationWitnessPolicyEpoch {
            epoch,
            previous_policy_digest: bytes32(
                &row.previous_policy_digest,
                "observation witness previous policy digest",
            )?,
            activated_at,
            witness_node_ids,
            minimum_witnesses,
            signer_node_id: bytes32(&row.signer_node_id, "observation witness policy signer")?,
            signature: bytes64(&row.signature, "observation witness policy signature")?,
        };
        policy.validate_unsigned_fields()?;
        let stored_digest = bytes32(&row.policy_digest, "observation witness policy digest")?;
        Ok((stored_digest, policy))
    }

    pub(super) fn audit_observation_witness_policies(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<ObservationWitnessPolicyAudit, DirectoryReplicaStoreError> {
        let (metadata_epoch, metadata_head) = connection.query_row(
            "SELECT witness_policy_epoch, witness_policy_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_epoch =
            nonnegative_i64_to_u64(metadata_epoch, "replica metadata witness policy epoch")?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata witness policy head"))
            .transpose()?;

        let mut statement = connection.prepare(
            "SELECT epoch, policy_digest, previous_policy_digest, activated_at,
                    witness_threshold, witness_count, witness_node_ids,
                    signer_node_id, signature
             FROM directory_observation_witness_policies ORDER BY epoch ASC",
        )?;
        let mut rows = statement.query([])?;
        let mut audit = ObservationWitnessPolicyAudit::default();
        let mut expected_previous_digest = [0u8; 32];
        let mut previous_activated_at = 0u64;
        while let Some(row) = rows.next()? {
            let stored = StoredObservationWitnessPolicyRow {
                epoch: row.get(0)?,
                policy_digest: row.get(1)?,
                previous_policy_digest: row.get(2)?,
                activated_at: row.get(3)?,
                witness_threshold: row.get(4)?,
                witness_count: row.get(5)?,
                witness_node_ids: row.get(6)?,
                signer_node_id: row.get(7)?,
                signature: row.get(8)?,
            };
            let (stored_digest, policy) = Self::decode_observation_witness_policy(stored)?;
            let expected_epoch = audit.epochs.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy epoch exhausted".to_string(),
                )
            })?;
            if policy.epoch != expected_epoch
                || policy.previous_policy_digest != expected_previous_digest
                || policy.activated_at < previous_activated_at
                || policy.signer_node_id != *local_node_id
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation witness policy history is not canonical".to_string(),
                ));
            }
            IdentityPublicKey::from_bytes(&policy.signer_node_id)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy signer is invalid".to_string(),
                    )
                })?
                .verify(&policy.signing_bytes(), &policy.signature)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy signature is invalid".to_string(),
                    )
                })?;
            if policy.digest() != stored_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation witness policy digest is invalid".to_string(),
                ));
            }
            audit.epochs = expected_epoch;
            expected_previous_digest = stored_digest;
            previous_activated_at = policy.activated_at;
            audit.current_digest = stored_digest;
            audit.current = Some(policy);
        }
        drop(rows);
        drop(statement);
        if audit.epochs != metadata_epoch
            || (audit.epochs == 0 && metadata_head.is_some())
            || (audit.epochs > 0 && metadata_head != Some(audit.current_digest))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy head does not match audited history".to_string(),
            ));
        }
        Ok(audit)
    }

    fn decode_route_domain_policy(
        row: StoredRouteDomainPolicyRow,
    ) -> Result<([u8; 32], DirectoryRouteDomainPolicyEpoch), DirectoryReplicaStoreError> {
        let epoch = positive_i64_to_u64(row.epoch, "route-domain policy epoch")?;
        let activated_at =
            positive_i64_to_u64(row.activated_at, "route-domain policy activation timestamp")?;
        let strict_required = match row.strict_required {
            0 => false,
            1 => true,
            _ => {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "route-domain policy strict flag is malformed".to_string(),
                ));
            }
        };
        let assignment_count = usize::try_from(nonnegative_i64_to_u64(
            row.assignment_count,
            "route-domain policy assignment count",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain policy assignment count exceeds usize".to_string(),
            )
        })?;
        if assignment_count > MAX_DIRECTORY_ROUTE_DOMAIN_POLICY_ASSIGNMENTS
            || row.assignments.len() != assignment_count.saturating_mul(48)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy assignment blob is malformed".to_string(),
            ));
        }
        let assignments = row
            .assignments
            .chunks_exact(48)
            .map(|assignment| {
                Ok(PinnedRouteDomainAssignment {
                    node_id: bytes32(&assignment[..32], "route-domain policy node identity")?,
                    route_domain: assignment[32..].try_into().map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "route-domain policy token is malformed".to_string(),
                        )
                    })?,
                })
            })
            .collect::<Result<Vec<_>, DirectoryReplicaStoreError>>()?;
        let policy = DirectoryRouteDomainPolicyEpoch {
            epoch,
            previous_policy_digest: bytes32(
                &row.previous_policy_digest,
                "previous route-domain policy digest",
            )?,
            activated_at,
            strict_required,
            assignments,
            signer_node_id: bytes32(&row.signer_node_id, "route-domain policy signer")?,
            signature: bytes64(&row.signature, "route-domain policy signature")?,
        };
        policy.validate_unsigned_fields()?;
        let stored_digest = bytes32(&row.policy_digest, "route-domain policy digest")?;
        Ok((stored_digest, policy))
    }

    pub(super) fn audit_route_domain_policies(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<RouteDomainPolicyAudit, DirectoryReplicaStoreError> {
        let (metadata_epoch, metadata_head) = connection.query_row(
            "SELECT route_domain_policy_epoch, route_domain_policy_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_epoch =
            nonnegative_i64_to_u64(metadata_epoch, "replica metadata route-domain policy epoch")?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata route-domain policy head"))
            .transpose()?;
        let mut statement = connection.prepare(
            "SELECT epoch, policy_digest, previous_policy_digest, activated_at,
                    strict_required, assignment_count, assignments,
                    signer_node_id, signature
             FROM directory_route_domain_policies ORDER BY epoch ASC",
        )?;
        let mut rows = statement.query([])?;
        let mut audit = RouteDomainPolicyAudit::default();
        let mut expected_previous_digest = [0u8; 32];
        let mut previous_activated_at = 0u64;
        while let Some(row) = rows.next()? {
            let stored = StoredRouteDomainPolicyRow {
                epoch: row.get(0)?,
                policy_digest: row.get(1)?,
                previous_policy_digest: row.get(2)?,
                activated_at: row.get(3)?,
                strict_required: row.get(4)?,
                assignment_count: row.get(5)?,
                assignments: row.get(6)?,
                signer_node_id: row.get(7)?,
                signature: row.get(8)?,
            };
            let (stored_digest, policy) = Self::decode_route_domain_policy(stored)?;
            let expected_epoch = audit.epochs.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "route-domain policy epoch exhausted".to_string(),
                )
            })?;
            if policy.epoch != expected_epoch
                || policy.previous_policy_digest != expected_previous_digest
                || policy.activated_at < previous_activated_at
                || policy.signer_node_id != *local_node_id
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "route-domain policy history is not canonical".to_string(),
                ));
            }
            IdentityPublicKey::from_bytes(&policy.signer_node_id)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain policy signer is invalid".to_string(),
                    )
                })?
                .verify(&policy.signing_bytes(), &policy.signature)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain policy signature is invalid".to_string(),
                    )
                })?;
            if policy.digest() != stored_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "route-domain policy digest is invalid".to_string(),
                ));
            }
            audit.epochs = expected_epoch;
            expected_previous_digest = stored_digest;
            previous_activated_at = policy.activated_at;
            audit.current_digest = stored_digest;
            audit.current = Some(policy);
        }
        drop(rows);
        drop(statement);
        if audit.epochs != metadata_epoch
            || (audit.epochs == 0 && metadata_head.is_some())
            || (audit.epochs > 0 && metadata_head != Some(audit.current_digest))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain policy head does not match audited history".to_string(),
            ));
        }
        Ok(audit)
    }

    fn decode_route_domain_attestor_policy(
        row: StoredRouteDomainAttestorPolicyRow,
    ) -> Result<([u8; 32], DirectoryRouteDomainAttestorPolicyEpoch), DirectoryReplicaStoreError>
    {
        let epoch = positive_i64_to_u64(row.epoch, "route-domain attestor policy epoch")?;
        let activated_at = positive_i64_to_u64(
            row.activated_at,
            "route-domain attestor policy activation timestamp",
        )?;
        let strict_required = match row.strict_required {
            0 => false,
            1 => true,
            _ => {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "route-domain attestor policy strict flag is malformed".to_string(),
                ));
            }
        };
        let minimum_attestors = usize::try_from(positive_i64_to_u64(
            row.attestor_threshold,
            "route-domain attestor policy threshold",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain attestor threshold exceeds usize".to_string(),
            )
        })?;
        let attestor_count = usize::try_from(nonnegative_i64_to_u64(
            row.attestor_count,
            "route-domain attestor policy member count",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain attestor count exceeds usize".to_string(),
            )
        })?;
        if attestor_count > MAX_DIRECTORY_ROUTE_DOMAIN_ATTESTOR_POLICY_MEMBERS
            || row.attestor_node_ids.len() != attestor_count.saturating_mul(32)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy identity blob is malformed".to_string(),
            ));
        }
        let attestor_node_ids = row
            .attestor_node_ids
            .chunks_exact(32)
            .map(|node_id| bytes32(node_id, "route-domain attestor identity"))
            .collect::<Result<Vec<_>, _>>()?;
        let policy = DirectoryRouteDomainAttestorPolicyEpoch {
            epoch,
            previous_policy_digest: bytes32(
                &row.previous_policy_digest,
                "previous route-domain attestor policy digest",
            )?,
            activated_at,
            strict_required,
            attestor_node_ids,
            minimum_attestors,
            signer_node_id: bytes32(&row.signer_node_id, "route-domain attestor policy signer")?,
            signature: bytes64(&row.signature, "route-domain attestor policy signature")?,
        };
        policy.validate_unsigned_fields()?;
        let stored_digest = bytes32(&row.policy_digest, "route-domain attestor policy digest")?;
        Ok((stored_digest, policy))
    }

    pub(super) fn audit_route_domain_attestor_policies(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<RouteDomainAttestorPolicyAudit, DirectoryReplicaStoreError> {
        let (metadata_epoch, metadata_head) = connection.query_row(
            "SELECT route_domain_attestor_policy_epoch, route_domain_attestor_policy_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_epoch = nonnegative_i64_to_u64(
            metadata_epoch,
            "replica metadata route-domain attestor policy epoch",
        )?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata route-domain attestor policy head"))
            .transpose()?;
        let mut statement = connection.prepare(
            "SELECT epoch, policy_digest, previous_policy_digest, activated_at,
                    strict_required, attestor_threshold, attestor_count,
                    attestor_node_ids, signer_node_id, signature
             FROM directory_route_domain_attestor_policies ORDER BY epoch ASC",
        )?;
        let mut rows = statement.query([])?;
        let mut audit = RouteDomainAttestorPolicyAudit::default();
        let mut expected_previous_digest = [0u8; 32];
        let mut previous_activated_at = 0u64;
        while let Some(row) = rows.next()? {
            let stored = StoredRouteDomainAttestorPolicyRow {
                epoch: row.get(0)?,
                policy_digest: row.get(1)?,
                previous_policy_digest: row.get(2)?,
                activated_at: row.get(3)?,
                strict_required: row.get(4)?,
                attestor_threshold: row.get(5)?,
                attestor_count: row.get(6)?,
                attestor_node_ids: row.get(7)?,
                signer_node_id: row.get(8)?,
                signature: row.get(9)?,
            };
            let (stored_digest, policy) = Self::decode_route_domain_attestor_policy(stored)?;
            let expected_epoch = audit.epochs.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "route-domain attestor policy epoch exhausted".to_string(),
                )
            })?;
            if policy.epoch != expected_epoch
                || policy.previous_policy_digest != expected_previous_digest
                || policy.activated_at < previous_activated_at
                || policy.signer_node_id != *local_node_id
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "route-domain attestor policy history is not canonical".to_string(),
                ));
            }
            IdentityPublicKey::from_bytes(&policy.signer_node_id)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain attestor policy signer is invalid".to_string(),
                    )
                })?
                .verify(&policy.signing_bytes(), &policy.signature)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain attestor policy signature is invalid".to_string(),
                    )
                })?;
            if policy.digest() != stored_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "route-domain attestor policy digest is invalid".to_string(),
                ));
            }
            audit.epochs = expected_epoch;
            expected_previous_digest = stored_digest;
            previous_activated_at = policy.activated_at;
            audit.current_digest = stored_digest;
            audit.current = Some(policy);
        }
        drop(rows);
        drop(statement);
        if audit.epochs != metadata_epoch
            || (audit.epochs == 0 && metadata_head.is_some())
            || (audit.epochs > 0 && metadata_head != Some(audit.current_digest))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy head does not match audited history".to_string(),
            ));
        }
        Ok(audit)
    }

    /// Loads and verifies only the metadata-anchored policy head for bounded
    /// runtime status. Complete history verification remains a startup and
    /// explicit operator-audit responsibility.
    pub(super) fn load_current_observation_witness_policy(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<ObservationWitnessPolicyAudit, DirectoryReplicaStoreError> {
        let (metadata_epoch, metadata_head) = connection.query_row(
            "SELECT witness_policy_epoch, witness_policy_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_epoch =
            nonnegative_i64_to_u64(metadata_epoch, "replica metadata witness policy epoch")?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata witness policy head"))
            .transpose()?;
        if metadata_epoch == 0 {
            if metadata_head.is_some() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "empty observation witness policy has a non-empty head".to_string(),
                ));
            }
            return Ok(ObservationWitnessPolicyAudit::default());
        }
        let stored = connection
            .query_row(
                "SELECT epoch, policy_digest, previous_policy_digest, activated_at,
                        witness_threshold, witness_count, witness_node_ids,
                        signer_node_id, signature
                 FROM directory_observation_witness_policies WHERE epoch = ?1",
                params![u64_to_i64(
                    metadata_epoch,
                    "replica metadata witness policy epoch"
                )?],
                |row| {
                    Ok(StoredObservationWitnessPolicyRow {
                        epoch: row.get(0)?,
                        policy_digest: row.get(1)?,
                        previous_policy_digest: row.get(2)?,
                        activated_at: row.get(3)?,
                        witness_threshold: row.get(4)?,
                        witness_count: row.get(5)?,
                        witness_node_ids: row.get(6)?,
                        signer_node_id: row.get(7)?,
                        signature: row.get(8)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy head row is missing".to_string(),
                )
            })?;
        let (stored_digest, policy) = Self::decode_observation_witness_policy(stored)?;
        if policy.epoch != metadata_epoch
            || policy.signer_node_id != *local_node_id
            || metadata_head != Some(stored_digest)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy head is inconsistent".to_string(),
            ));
        }
        IdentityPublicKey::from_bytes(&policy.signer_node_id)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy signer is invalid".to_string(),
                )
            })?
            .verify(&policy.signing_bytes(), &policy.signature)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy signature is invalid".to_string(),
                )
            })?;
        if policy.digest() != stored_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy digest is invalid".to_string(),
            ));
        }
        Ok(ObservationWitnessPolicyAudit {
            epochs: metadata_epoch,
            current: Some(policy),
            current_digest: stored_digest,
        })
    }

    pub(super) fn audit_remote_observation_policy_anchors(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT observer, policy_epoch, previous_policy_digest, policy_digest,
                    request_timestamp, request_blob
             FROM directory_observation_remote_policy_anchors
             ORDER BY observer ASC, policy_epoch ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Vec<u8>>(5)?,
            ))
        })?;
        let mut heads = HashMap::<[u8; 32], (u64, [u8; 32])>::new();
        let mut count = 0u64;
        for row in rows {
            let (observer, epoch, previous_digest, digest, request_timestamp, request_blob) = row?;
            let observer = bytes32(&observer, "remote policy anchor observer")?;
            let epoch = positive_i64_to_u64(epoch, "remote policy anchor epoch")?;
            let previous_digest =
                bytes32(&previous_digest, "remote policy anchor previous digest")?;
            let digest = bytes32(&digest, "remote policy anchor digest")?;
            let request_timestamp =
                positive_i64_to_u64(request_timestamp, "remote policy anchor timestamp")?;
            let request = decode_directory_sync_message(&request_blob)
                .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
            if encode_directory_sync_message(&request)
                .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
                != request_blob
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor request is noncanonical".to_string(),
                ));
            }
            let DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
                chain_id,
                request_id,
                requester,
                request_timestamp: signed_at,
                policy_epoch,
                previous_policy_digest,
                policy_digest,
                signature,
            } = request
            else {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor row contains an unexpected frame".to_string(),
                ));
            };
            let position_valid = (epoch == 1 && previous_digest == [0u8; 32])
                || (epoch > 1 && previous_digest != [0u8; 32]);
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                || requester != observer
                || requester == *local_node_id
                || policy_epoch != epoch
                || previous_policy_digest != previous_digest
                || policy_digest != digest
                || signed_at != request_timestamp
                || digest == [0u8; 32]
                || !position_valid
                || request_timestamp > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor row violates its signed contract".to_string(),
                ));
            }
            let signing_bytes = directory_policy_anchor_request_signing_bytes(
                &chain_id,
                &request_id,
                &requester,
                signed_at,
                policy_epoch,
                &previous_policy_digest,
                &policy_digest,
            );
            IdentityPublicKey::from_bytes(&requester)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "remote policy anchor signature is invalid".to_string(),
                    )
                })?;
            if let Some((previous_epoch, previous_head)) = heads.get(&observer) {
                if epoch != previous_epoch.saturating_add(1) || previous_digest != *previous_head {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "remote policy anchor history is not contiguous".to_string(),
                    ));
                }
            }
            heads.insert(observer, (epoch, digest));
            count = count.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor count overflow".to_string(),
                )
            })?;
        }
        Ok(count)
    }

    fn verify_observation_policy_anchor_receipt_row(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
        row: StoredObservationWitnessPolicyAnchorReceiptRow,
    ) -> Result<(u64, [u8; 32], [u8; 32]), DirectoryReplicaStoreError> {
        let epoch = positive_i64_to_u64(row.policy_epoch, "policy anchor receipt epoch")?;
        let digest = bytes32(&row.policy_digest, "policy anchor receipt digest")?;
        let observer = bytes32(&row.observer, "policy anchor receipt observer")?;
        let witness = bytes32(&row.witness_node_id, "policy anchor receipt witness")?;
        let witnessed_at =
            positive_i64_to_u64(row.witnessed_at, "policy anchor receipt timestamp")?;
        let response = decode_directory_sync_message(&row.response_blob)
            .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
        if encode_directory_sync_message(&response)
            .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
            != row.response_blob
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt is noncanonical".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id,
            request_id,
            observer: signed_observer,
            policy_epoch,
            policy_digest,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt contains an unexpected frame".to_string(),
            ));
        };
        if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || observer != *local_node_id
            || signed_observer != observer
            || policy_epoch != epoch
            || policy_digest != digest
            || responder != witness
            || response_timestamp != witnessed_at
            || outcome != DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1
            || witnessed_at > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt violates its signed contract".to_string(),
            ));
        }
        let policy_members: (Vec<u8>, Vec<u8>) = connection.query_row(
            "SELECT policy_digest, witness_node_ids
             FROM directory_observation_witness_policies WHERE epoch = ?1",
            params![u64_to_i64(epoch, "policy anchor receipt epoch")?],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if policy_members.1.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS * 32
            || policy_members.1.len() % 32 != 0
            || bytes32(&policy_members.0, "policy anchor local policy digest")? != digest
            || !policy_members
                .1
                .chunks_exact(32)
                .any(|member| member == witness.as_slice())
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt is not admitted by its policy epoch".to_string(),
            ));
        }
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            &chain_id,
            &request_id,
            &signed_observer,
            policy_epoch,
            &policy_digest,
            &responder,
            response_timestamp,
            outcome,
        );
        IdentityPublicKey::from_bytes(&responder)
            .and_then(|key| key.verify(&signing_bytes, &signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "policy anchor receipt signature is invalid".to_string(),
                )
            })?;
        Ok((epoch, digest, witness))
    }

    pub(super) fn audit_observation_policy_anchor_receipts(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<Vec<(u64, [u8; 32], [u8; 32])>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT policy_epoch, policy_digest, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_policy_anchor_receipts
             ORDER BY policy_epoch ASC, witness_node_id ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredObservationWitnessPolicyAnchorReceiptRow {
                policy_epoch: row.get(0)?,
                policy_digest: row.get(1)?,
                observer: row.get(2)?,
                witness_node_id: row.get(3)?,
                witnessed_at: row.get(4)?,
                response_blob: row.get(5)?,
            })
        })?;
        let mut verified = Vec::new();
        for row in rows {
            verified.push(Self::verify_observation_policy_anchor_receipt_row(
                connection,
                local_node_id,
                observed_at,
                row?,
            )?);
        }
        Ok(verified)
    }

    fn audit_current_observation_policy_anchor_receipts(
        connection: &Connection,
        local_node_id: &[u8; 32],
        policy_epoch: u64,
        policy_digest: &[u8; 32],
        observed_at: u64,
    ) -> Result<Vec<(u64, [u8; 32], [u8; 32])>, DirectoryReplicaStoreError> {
        let current = Self::load_current_observation_witness_policy(connection, local_node_id)?;
        let Some(current_policy) = current.current else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt query has no current local policy".to_string(),
            ));
        };
        if current_policy.epoch != policy_epoch || current.current_digest != *policy_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt query does not match the current local policy".to_string(),
            ));
        }
        let row_limit = MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS + 1;
        let mut statement = connection.prepare(
            "SELECT policy_epoch, policy_digest, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_policy_anchor_receipts
             WHERE policy_epoch = ?1 AND policy_digest = ?2
             ORDER BY witness_node_id ASC LIMIT ?3",
        )?;
        let rows = statement.query_map(
            params![
                u64_to_i64(policy_epoch, "policy anchor receipt query epoch")?,
                policy_digest.as_slice(),
                i64::try_from(row_limit).map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "policy anchor receipt query limit exceeds i64".to_string(),
                    )
                })?
            ],
            |row| {
                Ok(StoredObservationWitnessPolicyAnchorReceiptRow {
                    policy_epoch: row.get(0)?,
                    policy_digest: row.get(1)?,
                    observer: row.get(2)?,
                    witness_node_id: row.get(3)?,
                    witnessed_at: row.get(4)?,
                    response_blob: row.get(5)?,
                })
            },
        )?;
        let mut verified = Vec::new();
        for row in rows {
            let verified_row = Self::verify_observation_policy_anchor_receipt_row(
                connection,
                local_node_id,
                observed_at,
                row?,
            )?;
            if !current_policy.witness_node_ids.contains(&verified_row.2) {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "current policy anchor receipt witness is outside the signed policy"
                        .to_string(),
                ));
            }
            verified.push(verified_row);
        }
        if verified.len() > current_policy.witness_node_ids.len() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "current policy anchor receipts exceed signed policy membership".to_string(),
            ));
        }
        Ok(verified)
    }
}

fn canonical_observation_witness_policy_members(
    witness_node_ids: &[[u8; 32]],
    minimum_witnesses: usize,
) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
    let mut canonical = witness_node_ids.to_vec();
    canonical.sort_unstable();
    let original_len = canonical.len();
    canonical.dedup();
    if canonical.len() != original_len {
        return Err(DirectoryReplicaStoreError::Request(
            "observation witness policy contains duplicate node identities".to_string(),
        ));
    }
    validate_observation_witness_policy_members(&canonical, minimum_witnesses)?;
    Ok(canonical)
}

pub(super) fn validate_observation_witness_policy_members(
    witness_node_ids: &[[u8; 32]],
    minimum_witnesses: usize,
) -> Result<(), DirectoryReplicaStoreError> {
    if witness_node_ids.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS
        || minimum_witnesses == 0
        || minimum_witnesses > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS
        || (witness_node_ids.is_empty() && minimum_witnesses != 1)
        || (!witness_node_ids.is_empty() && minimum_witnesses > witness_node_ids.len())
        || witness_node_ids.iter().any(|node_id| *node_id == [0u8; 32])
        || witness_node_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(DirectoryReplicaStoreError::Request(
            "observation witness policy pins or threshold are invalid".to_string(),
        ));
    }
    Ok(())
}

fn canonical_route_domain_policy_assignments(
    assignments: &[PinnedRouteDomainAssignment],
    strict_required: bool,
) -> Result<Vec<PinnedRouteDomainAssignment>, DirectoryReplicaStoreError> {
    let mut canonical = assignments.to_vec();
    canonical.sort_unstable();
    if canonical
        .windows(2)
        .any(|pair| pair[0].node_id == pair[1].node_id)
    {
        return Err(DirectoryReplicaStoreError::Request(
            "route-domain policy contains duplicate node identities".to_string(),
        ));
    }
    validate_route_domain_policy_assignments(&canonical, strict_required)?;
    Ok(canonical)
}

pub(super) fn validate_route_domain_policy_assignments(
    assignments: &[PinnedRouteDomainAssignment],
    strict_required: bool,
) -> Result<(), DirectoryReplicaStoreError> {
    if assignments.len() > MAX_DIRECTORY_ROUTE_DOMAIN_POLICY_ASSIGNMENTS
        || (strict_required && assignments.is_empty())
        || assignments.iter().any(|assignment| {
            assignment.node_id == [0u8; 32] || assignment.route_domain == [0u8; 16]
        })
        || assignments
            .windows(2)
            .any(|pair| pair[0].node_id >= pair[1].node_id)
    {
        return Err(DirectoryReplicaStoreError::Request(
            "route-domain policy assignments or strict mode are invalid".to_string(),
        ));
    }
    Ok(())
}

fn route_domain_policy_report(
    appended: bool,
    policy: &DirectoryRouteDomainPolicyEpoch,
    policy_digest: [u8; 32],
) -> Result<DirectoryRouteDomainPolicyReconcileReport, DirectoryReplicaStoreError> {
    Ok(DirectoryRouteDomainPolicyReconcileReport {
        appended,
        epoch: policy.epoch,
        policy_digest,
        activated_at: policy.activated_at,
        assignments: u64::try_from(policy.assignments.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain policy assignment count exceeds u64".to_string(),
            )
        })?,
        strict_required: policy.strict_required,
    })
}

fn canonical_route_domain_attestor_policy_members(
    attestor_node_ids: &[[u8; 32]],
    minimum_attestors: usize,
    strict_required: bool,
) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
    let mut canonical = attestor_node_ids.to_vec();
    canonical.sort_unstable();
    if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(DirectoryReplicaStoreError::Request(
            "route-domain attestor policy contains duplicate identities".to_string(),
        ));
    }
    validate_route_domain_attestor_policy_members(&canonical, minimum_attestors, strict_required)?;
    Ok(canonical)
}

pub(super) fn validate_route_domain_attestor_policy_members(
    attestor_node_ids: &[[u8; 32]],
    minimum_attestors: usize,
    strict_required: bool,
) -> Result<(), DirectoryReplicaStoreError> {
    let empty_policy_invalid =
        attestor_node_ids.is_empty() && (strict_required || minimum_attestors != 1);
    let populated_policy_invalid = !attestor_node_ids.is_empty()
        && (minimum_attestors == 0 || minimum_attestors > attestor_node_ids.len());
    if attestor_node_ids.len() > MAX_DIRECTORY_ROUTE_DOMAIN_ATTESTOR_POLICY_MEMBERS
        || empty_policy_invalid
        || populated_policy_invalid
        || attestor_node_ids
            .iter()
            .any(|node_id| *node_id == [0u8; 32])
        || attestor_node_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(DirectoryReplicaStoreError::Request(
            "route-domain attestor policy members, threshold, or strict mode are invalid"
                .to_string(),
        ));
    }
    Ok(())
}

fn route_domain_attestor_policy_report(
    appended: bool,
    policy: &DirectoryRouteDomainAttestorPolicyEpoch,
    policy_digest: [u8; 32],
) -> Result<DirectoryRouteDomainAttestorPolicyReconcileReport, DirectoryReplicaStoreError> {
    Ok(DirectoryRouteDomainAttestorPolicyReconcileReport {
        appended,
        epoch: policy.epoch,
        policy_digest,
        activated_at: policy.activated_at,
        attestors: u64::try_from(policy.attestor_node_ids.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy member count exceeds u64".to_string(),
            )
        })?,
        minimum_attestors: u64::try_from(policy.minimum_attestors).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "route-domain attestor policy threshold exceeds u64".to_string(),
            )
        })?,
        strict_required: policy.strict_required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const NOW: u64 = 1_700_000_100;

    fn policy_anchor_request(
        observer: &IdentityKeyPair,
        anchor: DirectoryObservationWitnessPolicyAnchor,
        request_seed: u8,
    ) -> DirectorySyncMessage {
        let request_id = [request_seed; 16];
        let requester = observer.public_key_bytes();
        let signing_bytes = directory_policy_anchor_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester,
            NOW + 30,
            anchor.epoch,
            &anchor.previous_policy_digest,
            &anchor.policy_digest,
        );
        DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester,
            request_timestamp: NOW + 30,
            policy_epoch: anchor.epoch,
            previous_policy_digest: anchor.previous_policy_digest,
            policy_digest: anchor.policy_digest,
            signature: observer.sign(&signing_bytes),
        }
    }

    fn accepted_policy_anchor_response(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        anchor: DirectoryObservationWitnessPolicyAnchor,
        request_seed: u8,
    ) -> DirectorySyncMessage {
        let request_id = [request_seed; 16];
        let responder = witness.public_key_bytes();
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            anchor.epoch,
            &anchor.policy_digest,
            &responder,
            NOW + 31,
            DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
        );
        DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            policy_epoch: anchor.epoch,
            policy_digest: anchor.policy_digest,
            responder,
            response_timestamp: NOW + 31,
            outcome: DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        }
    }

    #[test]
    fn witness_policy_epochs_are_canonical_idempotent_and_restart_durable() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
        let witness_a = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0xe3; 32]).unwrap();
        let witness_c = IdentityKeyPair::from_bytes(&[0xe4; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();

        let first = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_b.public_key_bytes(), witness_a.public_key_bytes()],
                2,
                NOW + 20,
            )
            .unwrap();
        assert!(first.appended);
        assert_eq!(first.epoch, 1);
        assert_eq!(first.witness_members, 2);
        assert_eq!(first.minimum_witnesses, 2);

        let reordered = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_a.public_key_bytes(), witness_b.public_key_bytes()],
                2,
                NOW + 21,
            )
            .unwrap();
        assert!(!reordered.appended);
        assert_eq!(reordered.epoch, 1);
        assert_eq!(reordered.policy_digest, first.policy_digest);
        assert_eq!(reordered.activated_at, first.activated_at);

        let threshold_change = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_a.public_key_bytes(), witness_b.public_key_bytes()],
                1,
                NOW + 22,
            )
            .unwrap();
        assert!(threshold_change.appended);
        assert_eq!(threshold_change.epoch, 2);
        assert_ne!(threshold_change.policy_digest, first.policy_digest);

        let rotation = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_b.public_key_bytes(), witness_c.public_key_bytes()],
                2,
                NOW + 23,
            )
            .unwrap();
        assert!(rotation.appended);
        assert_eq!(rotation.epoch, 3);
        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(snapshot.observation_witness_policy_epochs, 3);
        assert_eq!(snapshot.observation_witness_policy_epoch, 3);
        assert_eq!(snapshot.observation_witness_policy_members, 2);
        assert_eq!(snapshot.observation_witness_policy_threshold, 2);
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 24).unwrap();
        assert_eq!(audit.observation_witness_policy_epochs, 3);
        assert_eq!(audit.observation_witness_policy_epoch, 3);
        assert_eq!(audit.observation_witness_policy_activated_at, NOW + 23);
        assert_eq!(audit.observation_witness_policy_members, 2);
        assert_eq!(audit.observation_witness_policy_threshold, 2);
        assert!(reopened
            .observation_witness_policy_matches(
                &[witness_b.public_key_bytes(), witness_c.public_key_bytes()],
                2,
            )
            .unwrap());
        assert!(!reopened
            .observation_witness_policy_matches(
                &[witness_a.public_key_bytes(), witness_c.public_key_bytes()],
                2,
            )
            .unwrap());
        let idempotent_after_restart = reopened
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_c.public_key_bytes(), witness_b.public_key_bytes()],
                2,
                NOW + 25,
            )
            .unwrap();
        assert!(!idempotent_after_restart.appended);
        assert_eq!(idempotent_after_restart.epoch, 3);
        assert_eq!(
            idempotent_after_restart.policy_digest,
            rotation.policy_digest
        );
    }

    #[test]
    fn route_domain_policy_is_canonical_idempotent_and_restart_audited() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let operator = IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap();
        let first = PinnedRouteDomainAssignment {
            node_id: [0x11; 32],
            route_domain: [0xa1; 16],
        };
        let second = PinnedRouteDomainAssignment {
            node_id: [0x22; 32],
            route_domain: [0xb2; 16],
        };
        let (store, empty_audit) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 20).unwrap();
        assert_eq!(empty_audit.route_domain_policy_epochs, 0);
        let disabled = store
            .reconcile_route_domain_policy(&operator, &[], false, NOW + 20)
            .unwrap();
        assert_eq!(
            disabled,
            DirectoryRouteDomainPolicyReconcileReport::default()
        );

        let initial = store
            .reconcile_route_domain_policy(&operator, &[second, first], false, NOW + 21)
            .unwrap();
        assert!(initial.appended);
        assert_eq!(initial.epoch, 1);
        assert_eq!(initial.assignments, 2);
        assert!(!initial.strict_required);
        let reordered = store
            .reconcile_route_domain_policy(&operator, &[first, second], false, NOW + 22)
            .unwrap();
        assert!(!reordered.appended);
        assert_eq!(reordered.policy_digest, initial.policy_digest);
        assert_eq!(reordered.activated_at, initial.activated_at);

        let strict = store
            .reconcile_route_domain_policy(&operator, &[first, second], true, NOW + 23)
            .unwrap();
        assert!(strict.appended);
        assert_eq!(strict.epoch, 2);
        assert!(strict.strict_required);
        let cleared = store
            .reconcile_route_domain_policy(&operator, &[], false, NOW + 24)
            .unwrap();
        assert!(cleared.appended);
        assert_eq!(cleared.epoch, 3);
        assert_eq!(cleared.assignments, 0);
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 25).unwrap();
        assert_eq!(audit.route_domain_policy_epochs, 3);
        assert_eq!(audit.route_domain_policy_epoch, 3);
        assert_eq!(audit.route_domain_policy_activated_at, NOW + 24);
        assert_eq!(audit.route_domain_policy_assignments, 0);
        assert!(!audit.route_domain_policy_strict);
        let idempotent = reopened
            .reconcile_route_domain_policy(&operator, &[], false, NOW + 26)
            .unwrap();
        assert!(!idempotent.appended);
        assert_eq!(idempotent.epoch, 3);
        assert_eq!(idempotent.policy_digest, cleared.policy_digest);
    }

    #[test]
    fn tampered_route_domain_policy_signature_fails_restart_audit() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let operator = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
        let assignment = PinnedRouteDomainAssignment {
            node_id: [0x33; 32],
            route_domain: [0xc3; 16],
        };
        let (store, _) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_route_domain_policy(&operator, &[assignment], true, NOW + 21)
            .unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_route_domain_policies SET signature = ?1 WHERE epoch = 1",
                params![vec![0x44u8; 64]],
            )
            .unwrap();
        drop(connection);
        assert!(DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 22).is_err());
    }

    #[test]
    fn deleted_route_domain_policy_table_cannot_reset_anchored_history() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let operator = IdentityKeyPair::from_bytes(&[0xd3; 32]).unwrap();
        let assignment = PinnedRouteDomainAssignment {
            node_id: [0x55; 32],
            route_domain: [0xe5; 16],
        };
        let (store, _) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_route_domain_policy(&operator, &[assignment], true, NOW + 21)
            .unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("DROP TABLE directory_route_domain_policies;")
            .unwrap();
        drop(connection);
        assert!(DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 22).is_err());
    }

    #[test]
    fn route_domain_attestor_policy_is_canonical_idempotent_and_restart_audited() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let operator = IdentityKeyPair::from_bytes(&[0xd4; 32]).unwrap();
        let first = [0x31; 32];
        let second = [0x42; 32];
        let (store, empty_audit) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 20).unwrap();
        assert_eq!(empty_audit.route_domain_attestor_policy_epochs, 0);
        let disabled = store
            .reconcile_route_domain_attestor_policy(&operator, &[], 1, false, NOW + 20)
            .unwrap();
        assert_eq!(
            disabled,
            DirectoryRouteDomainAttestorPolicyReconcileReport::default()
        );

        let initial = store
            .reconcile_route_domain_attestor_policy(&operator, &[second, first], 1, false, NOW + 21)
            .unwrap();
        assert!(initial.appended);
        assert_eq!(initial.epoch, 1);
        assert_eq!(initial.attestors, 2);
        assert_eq!(initial.minimum_attestors, 1);
        let reordered = store
            .reconcile_route_domain_attestor_policy(&operator, &[first, second], 1, false, NOW + 22)
            .unwrap();
        assert!(!reordered.appended);
        assert_eq!(reordered.policy_digest, initial.policy_digest);
        assert_eq!(reordered.activated_at, initial.activated_at);

        let strict_quorum = store
            .reconcile_route_domain_attestor_policy(&operator, &[first, second], 2, true, NOW + 23)
            .unwrap();
        assert!(strict_quorum.appended);
        assert_eq!(strict_quorum.epoch, 2);
        assert_eq!(strict_quorum.minimum_attestors, 2);
        assert!(strict_quorum.strict_required);
        let cleared = store
            .reconcile_route_domain_attestor_policy(&operator, &[], 1, false, NOW + 24)
            .unwrap();
        assert!(cleared.appended);
        assert_eq!(cleared.epoch, 3);
        assert_eq!(cleared.attestors, 0);
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 25).unwrap();
        assert_eq!(audit.route_domain_attestor_policy_epochs, 3);
        assert_eq!(audit.route_domain_attestor_policy_epoch, 3);
        assert_eq!(audit.route_domain_attestor_policy_activated_at, NOW + 24);
        assert_eq!(audit.route_domain_attestor_policy_members, 0);
        assert_eq!(audit.route_domain_attestor_policy_threshold, 1);
        assert!(!audit.route_domain_attestor_policy_strict);
        let idempotent = reopened
            .reconcile_route_domain_attestor_policy(&operator, &[], 1, false, NOW + 26)
            .unwrap();
        assert!(!idempotent.appended);
        assert_eq!(idempotent.epoch, 3);
        assert_eq!(idempotent.policy_digest, cleared.policy_digest);
    }

    #[test]
    fn route_domain_attestor_policy_rejects_invalid_thresholds_and_duplicates() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let operator = IdentityKeyPair::from_bytes(&[0xd5; 32]).unwrap();
        let attestor = [0x51; 32];
        let (store, _) =
            DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 20).unwrap();
        assert!(store
            .reconcile_route_domain_attestor_policy(&operator, &[attestor], 0, false, NOW + 21)
            .is_err());
        assert!(store
            .reconcile_route_domain_attestor_policy(&operator, &[attestor], 2, true, NOW + 21)
            .is_err());
        assert!(store
            .reconcile_route_domain_attestor_policy(
                &operator,
                &[attestor, attestor],
                1,
                false,
                NOW + 21,
            )
            .is_err());
        assert!(store
            .reconcile_route_domain_attestor_policy(&operator, &[], 1, true, NOW + 21)
            .is_err());
    }

    #[test]
    fn tampered_or_deleted_route_domain_attestor_history_fails_restart_audit() {
        for delete_table in [false, true] {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("directory.db");
            let operator = IdentityKeyPair::from_bytes(&[0xd6; 32]).unwrap();
            let (store, _) =
                DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 20).unwrap();
            store
                .reconcile_route_domain_attestor_policy(&operator, &[[0x61; 32]], 1, true, NOW + 21)
                .unwrap();
            drop(store);

            let connection = Connection::open(&path).unwrap();
            if delete_table {
                connection
                    .execute_batch("DROP TABLE directory_route_domain_attestor_policies;")
                    .unwrap();
            } else {
                connection
                    .execute(
                        "UPDATE directory_route_domain_attestor_policies
                         SET signature = ?1 WHERE epoch = 1",
                        params![vec![0x62u8; 64]],
                    )
                    .unwrap();
            }
            drop(connection);
            assert!(
                DirectoryReplicaStore::open(&path, operator.public_key_bytes(), NOW + 22).is_err()
            );
        }
    }

    #[test]
    fn tampered_witness_policy_signature_or_metadata_head_fails_startup_audit() {
        for tamper_head in [false, true] {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("directory.db");
            let observer = IdentityKeyPair::from_bytes(&[0xe5; 32]).unwrap();
            let witness = IdentityKeyPair::from_bytes(&[0xe6; 32]).unwrap();
            let (store, _) =
                DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
            store
                .reconcile_observation_witness_policy(
                    &observer,
                    &[witness.public_key_bytes()],
                    1,
                    NOW + 20,
                )
                .unwrap();
            drop(store);

            let connection = Connection::open(&path).unwrap();
            if tamper_head {
                connection
                    .execute(
                        "UPDATE directory_replica_meta SET witness_policy_head = ?1
                         WHERE singleton = 1",
                        params![[0x99u8; 32].as_slice()],
                    )
                    .unwrap();
            } else {
                connection
                    .execute(
                        "UPDATE directory_observation_witness_policies SET signature = ?1
                         WHERE epoch = 1",
                        params![[0u8; 64].as_slice()],
                    )
                    .unwrap();
            }
            drop(connection);
            assert!(
                DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 21).is_err()
            );
        }
    }

    #[test]
    fn deleted_witness_policy_table_cannot_reset_anchored_history() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xe7; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xe8; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("DROP TABLE directory_observation_witness_policies;")
            .unwrap();
        drop(connection);
        assert!(DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 21).is_err());
    }

    #[test]
    fn remote_policy_anchor_is_monotonic_idempotent_and_restart_durable() {
        let observer_temp = TempDir::new().unwrap();
        let witness_temp = TempDir::new().unwrap();
        let observer = IdentityKeyPair::from_bytes(&[0xf1; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xf2; 32]).unwrap();
        let replacement = IdentityKeyPair::from_bytes(&[0xf3; 32]).unwrap();
        let (observer_store, _) = DirectoryReplicaStore::open(
            observer_temp.path().join("observer.db"),
            observer.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        observer_store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        let first = observer_store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        let witness_path = witness_temp.path().join("witness.db");
        let (witness_store, _) =
            DirectoryReplicaStore::open(&witness_path, witness.public_key_bytes(), NOW + 20)
                .unwrap();
        let first_request = policy_anchor_request(&observer, first, 0xa1);
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(&first_request, NOW + 30)
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Accepted
        );
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(&first_request, NOW + 30)
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Accepted
        );

        let conflicting = DirectoryObservationWitnessPolicyAnchor {
            policy_digest: [0x44; 32],
            ..first
        };
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(
                    &policy_anchor_request(&observer, conflicting, 0xa2),
                    NOW + 30,
                )
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Conflict
        );
        let gap = DirectoryObservationWitnessPolicyAnchor {
            epoch: 3,
            previous_policy_digest: [0x45; 32],
            policy_digest: [0x46; 32],
        };
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(
                    &policy_anchor_request(&observer, gap, 0xa3),
                    NOW + 30,
                )
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::HistoryGap
        );

        observer_store
            .reconcile_observation_witness_policy(
                &observer,
                &[replacement.public_key_bytes()],
                1,
                NOW + 21,
            )
            .unwrap();
        let second = observer_store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        assert_eq!(second.epoch, 2);
        assert_eq!(second.previous_policy_digest, first.policy_digest);
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(
                    &policy_anchor_request(&observer, second, 0xa4),
                    NOW + 30,
                )
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Accepted
        );
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(&first_request, NOW + 30)
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Rollback
        );
        drop(witness_store);
        let (_, audit) =
            DirectoryReplicaStore::open(&witness_path, witness.public_key_bytes(), NOW + 32)
                .unwrap();
        assert_eq!(audit.observation_witness_remote_policy_anchors, 2);
    }

    #[test]
    fn policy_anchor_receipts_are_pinned_signed_and_tamper_evident() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xf4; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xf5; 32]).unwrap();
        let outsider = IdentityKeyPair::from_bytes(&[0xf6; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        let anchor = store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        let response = accepted_policy_anchor_response(&observer, &witness, anchor, 0xb1);
        assert!(store
            .persist_observation_witness_policy_anchor_receipt(&response, NOW + 31)
            .unwrap());
        assert!(!store
            .persist_observation_witness_policy_anchor_receipt(&response, NOW + 31)
            .unwrap());
        assert_eq!(
            store
                .verified_observation_witness_policy_anchor_count_for_pins(
                    anchor.epoch,
                    &anchor.policy_digest,
                    &[witness.public_key_bytes()],
                    NOW + 31,
                )
                .unwrap(),
            1
        );
        assert!(store
            .persist_observation_witness_policy_anchor_receipt(
                &accepted_policy_anchor_response(&observer, &outsider, anchor, 0xb2),
                NOW + 31,
            )
            .is_err());
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_observation_policy_anchor_receipts
                 SET response_blob = zeroblob(length(response_blob))",
                [],
            )
            .unwrap();
        drop(connection);
        assert!(DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 32).is_err());
    }

    #[test]
    fn policy_anchor_receipt_rejects_a_stale_policy_after_rotation() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xe4; 32]).unwrap();
        let retired_witness = IdentityKeyPair::from_bytes(&[0xe5; 32]).unwrap();
        let replacement_witness = IdentityKeyPair::from_bytes(&[0xe6; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_observation_witness_policy(
                &observer,
                &[retired_witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        let retired_anchor = store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        let stale_response =
            accepted_policy_anchor_response(&observer, &retired_witness, retired_anchor, 0xc1);

        store
            .reconcile_observation_witness_policy(
                &observer,
                &[replacement_witness.public_key_bytes()],
                1,
                NOW + 30,
            )
            .unwrap();
        assert!(matches!(
            store
                .persist_observation_witness_policy_anchor_receipt(&stale_response, NOW + 31)
                .unwrap_err(),
            DirectoryReplicaStoreError::Request(message)
                if message == "observation policy anchor receipt is outside the current local policy"
        ));
        assert_eq!(
            store
                .status_snapshot()
                .unwrap()
                .observation_witness_policy_anchor_receipts,
            0
        );
    }
}
