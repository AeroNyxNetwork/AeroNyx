// ============================================
// File: crates/aeronyx-server/src/server/memchain_storage_gate.rs
// ============================================
//! # `MemChain` storage dispatch gate
//!
//! Owns the explicit storage-requirement classification and authorization
//! gate that decides which `MemChain` wire variants need which persistence.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `server.rs`; bodies unchanged.

use std::sync::Arc;

use tokio::sync::Mutex as TokioMutex;

use aeronyx_core::protocol::memchain::MemChainMessage;

#[allow(deprecated)]
use crate::services::memchain::{AofWriter, MemPool, MemoryStorage};

// [CHAT-DISPATCH-STORAGE-DECOUPLING 2026-09-02 by Codex] Keep optional
// persistence requirements explicit instead of using the existence of one
// storage tuple as an implicit gate for every MemChain wire variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MemChainStorageRequirement {
    IndependentRuntime,
    FactAof,
    RecordStore,
}

#[derive(Clone, Copy)]
pub(super) enum MemChainStorageAccess<'a> {
    IndependentRuntime,
    FactAof {
        mempool: &'a Arc<MemPool>,
        aof_writer: &'a Arc<TokioMutex<AofWriter>>,
    },
    RecordStore {
        storage: &'a Arc<MemoryStorage>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MemChainDispatchGateError {
    FactAofUnavailable,
    RecordStoreUnavailable,
}

impl MemChainDispatchGateError {
    pub(super) const fn reason_bucket(self) -> &'static str {
        match self {
            Self::FactAofUnavailable => "fact_aof_unavailable",
            Self::RecordStoreUnavailable => "record_store_unavailable",
        }
    }

    pub(super) const fn family_bucket(self) -> &'static str {
        match self {
            Self::FactAofUnavailable => "fact",
            Self::RecordStoreUnavailable => "record",
        }
    }
}

impl MemChainStorageRequirement {
    pub(super) fn for_message(message: &MemChainMessage) -> Self {
        match message {
            MemChainMessage::ChatRelay(_)
            | MemChainMessage::ChatRelayVerifiedSubmitV1(_)
            | MemChainMessage::ChatRelayVerifiedSubmitResponseV1(_)
            | MemChainMessage::ChatPull { .. }
            | MemChainMessage::ChatPullV2 { .. }
            | MemChainMessage::ChatAck { .. }
            | MemChainMessage::DeviceRegister { .. }
            | MemChainMessage::WalletPresence { .. } => Self::IndependentRuntime,
            MemChainMessage::BroadcastRecord(_)
            | MemChainMessage::SyncRecordRequest { .. }
            | MemChainMessage::SyncRecordResponse { .. } => Self::RecordStore,
            // Preserve the historical storage gate for every non-chat variant.
            // New wire variants therefore fail closed until they are explicitly
            // assigned to an independently configured runtime.
            _ => Self::FactAof,
        }
    }

    pub(super) fn authorize<'a>(
        self,
        mempool: Option<&'a Arc<MemPool>>,
        aof_writer: Option<&'a Arc<TokioMutex<AofWriter>>>,
        storage: &'a Option<Arc<MemoryStorage>>,
    ) -> std::result::Result<MemChainStorageAccess<'a>, MemChainDispatchGateError> {
        match self {
            Self::IndependentRuntime => Ok(MemChainStorageAccess::IndependentRuntime),
            Self::FactAof => match (mempool, aof_writer) {
                (Some(mempool), Some(aof_writer)) => Ok(MemChainStorageAccess::FactAof {
                    mempool,
                    aof_writer,
                }),
                _ => Err(MemChainDispatchGateError::FactAofUnavailable),
            },
            Self::RecordStore => {
                // Record messages historically passed through the same
                // MemPool/AOF outer gate before consulting MemoryStorage.
                // Preserve that prerequisite even though their handler only
                // needs the record store after admission.
                if mempool.is_none() || aof_writer.is_none() {
                    return Err(MemChainDispatchGateError::FactAofUnavailable);
                }
                storage
                    .as_ref()
                    .map(|storage| MemChainStorageAccess::RecordStore { storage })
                    .ok_or(MemChainDispatchGateError::RecordStoreUnavailable)
            }
        }
    }
}
