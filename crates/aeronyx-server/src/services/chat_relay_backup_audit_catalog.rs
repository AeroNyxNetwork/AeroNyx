// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_audit_catalog.rs
// ============================================
// Version: 1.0.0-SegmentCatalogDomain
//
// Creation Reason:
//   [CHAT-RELAY-AUDIT-CATALOG-DOMAIN 2026-08-26 by Codex] Isolate immutable
//   segment/checkpoint naming, classification, pairing, and bounds from I/O.
//
// Main Functionality:
//   - Encodes and parses the existing fixed-width segment artifact names.
//   - Pairs one segment and one checkpoint per inclusive sequence range.
//   - Rejects malformed, ambiguous, duplicate, invalid, and excess artifacts.
//   - Keeps catalog state path-free and deterministically range-ordered.
//
// Dependencies:
//   - `chat_relay_backup_audit_rotation.rs` owns validated segment ranges.
//   - `chat_relay.rs` owns directory reads, protected path construction, file
//     type/permission checks, opening, verification, and recovery execution.
//
// Main Logical Flow:
//   1. Classify a UTF-8 directory entry against the two v1 name contracts.
//   2. Parse its fixed-width inclusive sequence range with checked validation.
//   3. Insert it into a bounded ordered catalog without replacing duplicates.
//   4. Return path-free canonical names for service-owned path resolution.
//
// Important Note for Next Developer:
//   - Keep the v1 prefixes, suffixes, and 20-digit widths byte-for-byte stable.
//   - Never add absolute paths, node identity, payload, or message metadata.
//   - Do not perform directory or file I/O in this module.
//   - New artifact kinds require an explicit enum/version migration.
//
// Last Modified:
//   v1.0.0-SegmentCatalogDomain - Initial trait-based extraction
// ============================================

use std::collections::BTreeMap;

use crate::services::chat_relay_backup_audit_rotation::{
    BackupAuditRotationError, ChatRelayBackupAuditSegmentRange,
};

const SEGMENT_PREFIX: &str = ".aeronyx-relay-backup-maintenance-audit.segment-";
const SEGMENT_SUFFIX: &str = ".jsonl";
const CHECKPOINT_PREFIX: &str = ".aeronyx-relay-backup-maintenance-audit.checkpoint-";
const CHECKPOINT_SUFFIX: &str = ".json";
const SEQUENCE_WIDTH: usize = 20;

/// Closed artifact kinds understood by the v1 maintenance-audit catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditCatalogArtifactKind {
    Segment,
    Checkpoint,
}

/// Canonical path-free artifact identity produced by name classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupAuditCatalogArtifact {
    pub(crate) range: ChatRelayBackupAuditSegmentRange,
    pub(crate) kind: BackupAuditCatalogArtifactKind,
    pub(crate) file_name: String,
}

/// Path-free paired names for one immutable sequence range.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BackupAuditCatalogFiles {
    pub(crate) segment_file_name: Option<String>,
    pub(crate) checkpoint_file_name: Option<String>,
}

/// Closed catalog failures mapped to stable service errors by the I/O owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackupAuditCatalogError {
    MalformedName,
    AmbiguousName,
    InvalidRange,
    DuplicateArtifact,
    SegmentLimitReached,
}

/// Replaceable path-free catalog contract for immutable audit artifacts.
pub(crate) trait BackupAuditSegmentCatalog {
    fn segment_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String;

    fn checkpoint_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String;

    fn classify(
        &self,
        file_name: &str,
    ) -> Result<Option<BackupAuditCatalogArtifact>, BackupAuditCatalogError>;

    fn insert_file_name(&mut self, file_name: String) -> Result<(), BackupAuditCatalogError>;

    fn into_files(self) -> BTreeMap<ChatRelayBackupAuditSegmentRange, BackupAuditCatalogFiles>;
}

/// Production catalog with a fixed maximum number of sequence ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundedBackupAuditSegmentCatalog {
    max_segments: usize,
    files: BTreeMap<ChatRelayBackupAuditSegmentRange, BackupAuditCatalogFiles>,
}

impl BoundedBackupAuditSegmentCatalog {
    pub(crate) fn new(max_segments: usize) -> Self {
        Self {
            max_segments,
            files: BTreeMap::new(),
        }
    }

    fn parse_range(
        file_name: &str,
        prefix: &str,
        suffix: &str,
    ) -> Result<Option<ChatRelayBackupAuditSegmentRange>, BackupAuditCatalogError> {
        if !file_name.starts_with(prefix) {
            return Ok(None);
        }
        let body = file_name
            .strip_prefix(prefix)
            .and_then(|name| name.strip_suffix(suffix))
            .ok_or(BackupAuditCatalogError::MalformedName)?;
        let (first, last) = body
            .split_once('-')
            .ok_or(BackupAuditCatalogError::MalformedName)?;
        if first.len() != SEQUENCE_WIDTH
            || last.len() != SEQUENCE_WIDTH
            || !first.bytes().all(|byte| byte.is_ascii_digit())
            || !last.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(BackupAuditCatalogError::MalformedName);
        }
        let first_sequence = first
            .parse::<u64>()
            .map_err(|_| BackupAuditCatalogError::MalformedName)?;
        let last_sequence = last
            .parse::<u64>()
            .map_err(|_| BackupAuditCatalogError::MalformedName)?;
        ChatRelayBackupAuditSegmentRange::new(first_sequence, last_sequence)
            .map(Some)
            .map_err(|error| match error {
                BackupAuditRotationError::InvalidSegmentRange => {
                    BackupAuditCatalogError::InvalidRange
                }
                _ => BackupAuditCatalogError::MalformedName,
            })
    }
}

impl BackupAuditSegmentCatalog for BoundedBackupAuditSegmentCatalog {
    fn segment_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String {
        format!(
            "{SEGMENT_PREFIX}{:020}-{:020}{SEGMENT_SUFFIX}",
            range.first_sequence, range.last_sequence
        )
    }

    fn checkpoint_file_name(&self, range: ChatRelayBackupAuditSegmentRange) -> String {
        format!(
            "{CHECKPOINT_PREFIX}{:020}-{:020}{CHECKPOINT_SUFFIX}",
            range.first_sequence, range.last_sequence
        )
    }

    fn classify(
        &self,
        file_name: &str,
    ) -> Result<Option<BackupAuditCatalogArtifact>, BackupAuditCatalogError> {
        let segment_range = Self::parse_range(file_name, SEGMENT_PREFIX, SEGMENT_SUFFIX)?;
        let checkpoint_range = Self::parse_range(file_name, CHECKPOINT_PREFIX, CHECKPOINT_SUFFIX)?;
        let (range, kind) = match (segment_range, checkpoint_range) {
            (Some(range), None) => (range, BackupAuditCatalogArtifactKind::Segment),
            (None, Some(range)) => (range, BackupAuditCatalogArtifactKind::Checkpoint),
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => return Err(BackupAuditCatalogError::AmbiguousName),
        };
        Ok(Some(BackupAuditCatalogArtifact {
            range,
            kind,
            file_name: file_name.to_string(),
        }))
    }

    fn insert_file_name(&mut self, file_name: String) -> Result<(), BackupAuditCatalogError> {
        let Some(artifact) = self.classify(&file_name)? else {
            return Ok(());
        };
        if !self.files.contains_key(&artifact.range) && self.files.len() >= self.max_segments {
            return Err(BackupAuditCatalogError::SegmentLimitReached);
        }
        let files = self.files.entry(artifact.range).or_default();
        let slot = match artifact.kind {
            BackupAuditCatalogArtifactKind::Segment => &mut files.segment_file_name,
            BackupAuditCatalogArtifactKind::Checkpoint => &mut files.checkpoint_file_name,
        };
        if slot.is_some() {
            return Err(BackupAuditCatalogError::DuplicateArtifact);
        }
        *slot = Some(artifact.file_name);
        Ok(())
    }

    fn into_files(self) -> BTreeMap<ChatRelayBackupAuditSegmentRange, BackupAuditCatalogFiles> {
        self.files
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(first_sequence: u64, last_sequence: u64) -> ChatRelayBackupAuditSegmentRange {
        ChatRelayBackupAuditSegmentRange::new(first_sequence, last_sequence).expect("valid range")
    }

    #[test]
    fn canonical_names_round_trip_without_paths() {
        let catalog = BoundedBackupAuditSegmentCatalog::new(2);
        let expected = range(7, 42);
        let segment = catalog.segment_file_name(expected);
        let checkpoint = catalog.checkpoint_file_name(expected);

        assert_eq!(
            catalog.classify(&segment),
            Ok(Some(BackupAuditCatalogArtifact {
                range: expected,
                kind: BackupAuditCatalogArtifactKind::Segment,
                file_name: segment,
            }))
        );
        assert_eq!(
            catalog.classify(&checkpoint),
            Ok(Some(BackupAuditCatalogArtifact {
                range: expected,
                kind: BackupAuditCatalogArtifactKind::Checkpoint,
                file_name: checkpoint,
            }))
        );
    }

    #[test]
    fn pairs_artifacts_and_orders_ranges_deterministically() {
        let mut catalog = BoundedBackupAuditSegmentCatalog::new(2);
        for current in [range(3, 4), range(1, 2)] {
            catalog
                .insert_file_name(catalog.checkpoint_file_name(current))
                .expect("insert checkpoint");
            catalog
                .insert_file_name(catalog.segment_file_name(current))
                .expect("insert segment");
        }

        let files = catalog.into_files();
        assert_eq!(
            files.keys().copied().collect::<Vec<_>>(),
            [range(1, 2), range(3, 4)]
        );
        assert!(files.values().all(|pair| {
            pair.segment_file_name.is_some() && pair.checkpoint_file_name.is_some()
        }));
    }

    #[test]
    fn ignores_unrelated_and_rejects_malformed_or_invalid_names() {
        let catalog = BoundedBackupAuditSegmentCatalog::new(2);
        assert_eq!(catalog.classify("notes.txt"), Ok(None));
        assert_eq!(
            catalog.classify(".aeronyx-relay-backup-maintenance-audit.segment-1-2.jsonl"),
            Err(BackupAuditCatalogError::MalformedName)
        );
        assert_eq!(
            catalog.classify(
                ".aeronyx-relay-backup-maintenance-audit.segment-00000000000000000000-00000000000000000001.jsonl",
            ),
            Err(BackupAuditCatalogError::InvalidRange)
        );
    }

    #[test]
    fn rejects_duplicates_and_excess_ranges() {
        let mut catalog = BoundedBackupAuditSegmentCatalog::new(1);
        let first = catalog.segment_file_name(range(1, 1));
        catalog
            .insert_file_name(first.clone())
            .expect("insert first segment");
        assert_eq!(
            catalog.insert_file_name(first),
            Err(BackupAuditCatalogError::DuplicateArtifact)
        );
        assert_eq!(
            catalog.insert_file_name(catalog.segment_file_name(range(2, 2))),
            Err(BackupAuditCatalogError::SegmentLimitReached)
        );
        let files = catalog.into_files();
        assert_eq!(files.len(), 1);
        assert_eq!(
            files.get(&range(1, 1)),
            Some(&BackupAuditCatalogFiles {
                segment_file_name: Some(
                    ".aeronyx-relay-backup-maintenance-audit.segment-00000000000000000001-00000000000000000001.jsonl"
                        .to_string(),
                ),
                checkpoint_file_name: None,
            })
        );
    }
}
