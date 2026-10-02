// [ARCH-SPLIT 2026-10-02]
// Monotonic verified-delivery and custody-audit anchor witness decisions.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl MemoryStorage {
    /// Atomically witnesses one opaque verified-delivery cache anchor.
    ///
    /// The first observation establishes trust-on-first-use at any positive
    /// generation. Once established, only the exact next generation may
    /// advance the row. This makes missed witness updates explicit instead of
    /// allowing a correctly signed but discontinuous host snapshot to replace
    /// the durable high-water mark.
    ///
    /// The table is bounded to one row per requester and intentionally stores
    /// no delivery count, delivery timestamp, route, message identifier,
    /// payload, endpoint, or user identity.
    ///
    /// # Errors
    ///
    /// Returns an error for zero/out-of-range generations, a zero digest,
    /// malformed durable state, integer conversion failure, or SQLite failure.
    pub async fn witness_verified_delivery_anchor(
        &self,
        requester: &[u8; 32],
        generation: u64,
        anchor_digest: &[u8; 32],
        observed_at: u64,
    ) -> Result<VerifiedDeliveryAnchorWitnessOutcome, String> {
        self.witness_monotonic_anchor(
            MonotonicAnchorWitnessNamespace::VerifiedDelivery,
            requester,
            generation,
            anchor_digest,
            observed_at,
        )
        .await
    }

    /// Atomically witnesses one exact producer-signed custody checkpoint frame.
    ///
    /// [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] This uses the same audited
    /// monotonic transition as delivery evidence but a physically separate
    /// table and producer key. The row stores only producer identity,
    /// checkpoint generation, exact frame SHA-256, and witness observation
    /// time. Custody counts, paths, private HMACs, messages, routes, payloads,
    /// endpoints, and user identities never enter this database boundary.
    ///
    /// # Errors
    /// Returns an error for sentinel/out-of-range state, corrupt durable state,
    /// integer conversion failure, or SQLite transaction failure.
    pub async fn witness_custody_audit_anchor(
        &self,
        producer: &[u8; 32],
        checkpoint_generation: u64,
        frame_sha256: &[u8; 32],
        observed_at: u64,
    ) -> Result<CustodyAuditAnchorWitnessOutcome, String> {
        self.witness_monotonic_anchor(
            MonotonicAnchorWitnessNamespace::CustodyAudit,
            producer,
            checkpoint_generation,
            frame_sha256,
            observed_at,
        )
        .await
    }

    pub(super) async fn witness_monotonic_anchor(
        &self,
        namespace: MonotonicAnchorWitnessNamespace,
        subject: &[u8; 32],
        generation: u64,
        digest: &[u8; 32],
        observed_at: u64,
    ) -> Result<MonotonicAnchorWitnessOutcome, String> {
        let label = namespace.label();
        if subject == &[0u8; 32] {
            return Err(format!("{label} witness subject must be non-zero"));
        }
        if generation == 0 {
            return Err(format!("{label} witness generation must be positive"));
        }
        if digest == &[0u8; 32] {
            return Err(format!("{label} witness digest must be non-zero"));
        }
        if observed_at == 0 {
            return Err(format!("{label} witness time must be positive"));
        }
        let generation_i64 = i64::try_from(generation)
            .map_err(|_| format!("{label} witness generation is outside SQLite range"))?;
        let observed_at_i64 = i64::try_from(observed_at)
            .map_err(|_| format!("{label} witness time is outside SQLite range"))?;

        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin {label} witness transaction: {error}"))?;
        let existing = tx
            .query_row(namespace.select_sql(), params![subject.as_slice()], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .optional()
            .map_err(|error| format!("read {label} witness high-water: {error}"))?;

        let outcome = if let Some((stored_generation, stored_digest)) = existing {
            let stored_generation = u64::try_from(stored_generation)
                .map_err(|_| format!("{label} witness generation is invalid"))?;
            let stored_digest: [u8; 32] = stored_digest.try_into().map_err(|value: Vec<u8>| {
                format!("{label} witness digest length {}", value.len())
            })?;

            if generation < stored_generation {
                MonotonicAnchorWitnessOutcome::Stale {
                    generation: stored_generation,
                    anchor_digest: stored_digest,
                }
            } else if generation == stored_generation {
                if digest == &stored_digest {
                    tx.execute(
                        namespace.refresh_sql(),
                        params![subject.as_slice(), observed_at_i64],
                    )
                    .map_err(|error| format!("refresh {label} witness observation: {error}"))?;
                    MonotonicAnchorWitnessOutcome::Idempotent {
                        generation: stored_generation,
                        anchor_digest: stored_digest,
                    }
                } else {
                    MonotonicAnchorWitnessOutcome::Conflict {
                        generation: stored_generation,
                        anchor_digest: stored_digest,
                    }
                }
            } else if stored_generation.checked_add(1) == Some(generation) {
                tx.execute(
                    namespace.advance_sql(),
                    params![
                        subject.as_slice(),
                        generation_i64,
                        digest.as_slice(),
                        observed_at_i64
                    ],
                )
                .map_err(|error| format!("advance {label} witness high-water: {error}"))?;
                MonotonicAnchorWitnessOutcome::Advanced {
                    generation,
                    anchor_digest: *digest,
                }
            } else {
                MonotonicAnchorWitnessOutcome::Gap {
                    generation: stored_generation,
                    anchor_digest: stored_digest,
                }
            }
        } else {
            tx.execute(
                namespace.insert_sql(),
                params![
                    subject.as_slice(),
                    generation_i64,
                    digest.as_slice(),
                    observed_at_i64
                ],
            )
            .map_err(|error| format!("insert {label} witness high-water: {error}"))?;
            MonotonicAnchorWitnessOutcome::Advanced {
                generation,
                anchor_digest: *digest,
            }
        };

        tx.commit()
            .map_err(|error| format!("commit {label} witness decision: {error}"))?;
        Ok(outcome)
    }
}
