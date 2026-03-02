// ============================================
// File: crates/aeronyx-server/src/miner/reflection.rs
// ============================================
//! # ReflectionMiner — Periodic Block Packing
//!
//! ## Creation Reason
//! A lightweight background task that periodically drains the MemPool,
//! packs pending Facts into a Block with a Merkle root, persists it
//! to the AOF, and broadcasts the BlockHeader to all connected peers.
//!
//! ## Main Functionality
//! - `ReflectionMiner::new()` — configure the miner
//! - `ReflectionMiner::run()` — the async mining loop (call via `tokio::spawn`)
//!
//! ## Mining Cycle
//! ```text
//! Every `interval` seconds:
//!   1. mempool.drain_for_block() → Vec<Fact> (atomic drain)
//!   2. if empty → skip
//!   3. merkle_root(fact_ids) → [u8; 32]
//!   4. aof_writer.last_block_hash() → prev_block_hash
//!   5. aof_writer.last_block_height() + 1 → height
//!   6. Assemble Block { header, facts }
//!   7. aof_writer.append_block(&block) → persist
//!   8. Broadcast BlockAnnounce(header) to all sessions (<100 bytes, MTU safe)
//! ```
//!
//! ## Concurrency Safety
//! `drain_for_block()` atomically removes facts from MemPool.
//! Facts arriving during the drain window go into the NEXT cycle.
//! No facts are lost: MemPool's DashMap + RwLock guarantee this.
//!
//! ## MTU Safety
//! Only `BlockHeader` (~100 bytes) is broadcast over UDP.
//! Full blocks are NEVER sent over the wire directly.
//!
//! ## Dependencies
//! - `MemPool` — drain pending facts
//! - `AofWriter` — persist blocks + chain state
//! - `SessionManager` + `UdpTransport` — broadcast header
//! - `DefaultTransportCrypto` — encrypt outbound packets
//!
//! ## ⚠️ Important Note for Next Developer
//! - The Miner does NOT do proof-of-work. It is a simple timer-based packer.
//! - `drain_for_block()` clears the MemPool. Ensure AOF has all facts
//!   before draining (facts are written to AOF individually as they arrive).
//! - Future: `BLOCK_TYPE_CHECKPOINT` will trigger LLM summarisation.
//!
//! ## Last Modified
//! v0.5.0 - 🌟 Initial ReflectionMiner implementation

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::ledger::{Block, BlockHeader, BLOCK_TYPE_NORMAL, merkle_root};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::DataPacket;
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;

use crate::services::memchain::{AofWriter, MemPool};
use crate::services::SessionManager;

// ============================================
// ReflectionMiner
// ============================================

/// Lightweight background miner that packs Facts into Blocks.
pub struct ReflectionMiner {
    /// Mining interval.
    interval: Duration,
    /// Shared MemPool.
    mempool: Arc<MemPool>,
    /// Shared AOF writer.
    aof_writer: Arc<TokioMutex<AofWriter>>,
    /// Session manager for broadcast.
    sessions: Arc<SessionManager>,
    /// UDP transport for broadcast.
    udp: Arc<UdpTransport>,
}

impl ReflectionMiner {
    /// Creates a new ReflectionMiner.
    ///
    /// # Arguments
    /// * `interval_secs` — seconds between mining cycles (default 3600)
    /// * `mempool` — shared MemPool to drain
    /// * `aof_writer` — shared AOF for persistence
    /// * `sessions` — session manager for peer broadcast
    /// * `udp` — UDP transport for sending announcements
    #[must_use]
    pub fn new(
        interval_secs: u64,
        mempool: Arc<MemPool>,
        aof_writer: Arc<TokioMutex<AofWriter>>,
        sessions: Arc<SessionManager>,
        udp: Arc<UdpTransport>,
    ) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            mempool,
            aof_writer,
            sessions,
            udp,
        }
    }

    /// Runs the mining loop until shutdown.
    ///
    /// This should be called inside `tokio::spawn`.
    pub async fn run(
        self,
        mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    ) {
        info!(
            interval_secs = self.interval.as_secs(),
            "[MINER] ⛏️ ReflectionMiner started"
        );

        let mut timer = tokio::time::interval(self.interval);
        // Skip the first immediate tick
        timer.tick().await;

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("[MINER] Received shutdown signal, stopping");
                    break;
                }
                _ = timer.tick() => {
                    self.mine_cycle().await;
                }
            }
        }

        info!("[MINER] ⛏️ ReflectionMiner stopped");
    }

    /// Executes a single mining cycle.
    async fn mine_cycle(&self) {
        // Step 1: Drain MemPool (atomic — new facts go to next cycle)
        let facts = self.mempool.drain_for_block();

        if facts.is_empty() {
            debug!("[MINER] No pending facts, skipping cycle");
            return;
        }

        let fact_count = facts.len();

        // Step 2: Compute Merkle root
        let leaf_ids: Vec<[u8; 32]> = facts.iter().map(|f| f.fact_id).collect();
        let root = merkle_root(&leaf_ids);

        // Step 3: Get chain state from AOF
        let (prev_hash, prev_height) = {
            let writer = self.aof_writer.lock().await;
            (writer.last_block_hash(), writer.last_block_height())
        };

        let new_height = if prev_hash == [0u8; 32] && prev_height == 0 {
            1 // First block
        } else {
            prev_height + 1
        };

        // Step 4: Generate timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Step 5: Assemble Block
        let header = BlockHeader {
            height: new_height,
            timestamp,
            prev_block_hash: prev_hash,
            merkle_root: root,
            block_type: BLOCK_TYPE_NORMAL,
        };

        let block = Block::new(header.clone(), facts);
        let block_hash_hex = header.hash_hex();

        // Step 6: Persist to AOF
        {
            let mut writer = self.aof_writer.lock().await;
            if let Err(e) = writer.append_block(&block).await {
                error!(
                    height = new_height,
                    error = %e,
                    "[MINER] ❌ Failed to persist block to AOF"
                );
                return;
            }
        }

        info!(
            height = new_height,
            facts = fact_count,
            merkle_root = hex::encode(root),
            block_hash = %block_hash_hex,
            "[MINER] ⛏️✅ Block mined and persisted"
        );

        // Step 7: Broadcast BlockAnnounce to all peers (MTU safe: <100 bytes)
        let announce_msg = MemChainMessage::BlockAnnounce(header);
        let sent = self.broadcast_header(announce_msg).await;

        info!(
            height = new_height,
            peers_notified = sent,
            "[MINER] 📡 BlockAnnounce broadcast complete"
        );
    }

    /// Broadcasts a `BlockAnnounce` message to all connected sessions.
    ///
    /// Returns the number of peers successfully sent to.
    async fn broadcast_header(&self, msg: MemChainMessage) -> usize {
        let plaintext = match encode_memchain(&msg) {
            Ok(p) => p,
            Err(e) => {
                error!("[MINER] ❌ Failed to encode BlockAnnounce: {}", e);
                return 0;
            }
        };

        // Safety check: ensure we're well within MTU
        if plaintext.len() > 1300 {
            error!(
                size = plaintext.len(),
                "[MINER] ❌ BlockAnnounce exceeds 1300 bytes! This should never happen."
            );
            return 0;
        }

        let all_sessions = self.sessions.all_sessions();
        let crypto = DefaultTransportCrypto::new();
        let mut sent = 0usize;

        for session in &all_sessions {
            if !session.is_established() {
                continue;
            }

            let counter = session.next_tx_counter();
            let encrypted_len = plaintext.len() + ENCRYPTION_OVERHEAD;
            let mut encrypted = vec![0u8; encrypted_len];

            let actual_len = match crypto.encrypt(
                &session.session_key,
                counter,
                session.id.as_bytes(),
                &plaintext,
                &mut encrypted,
            ) {
                Ok(len) => len,
                Err(e) => {
                    warn!(
                        session_id = %session.id,
                        error = %e,
                        "[MINER] ⚠️ Encryption failed, skipping"
                    );
                    continue;
                }
            };
            encrypted.truncate(actual_len);

            let data_packet = DataPacket::new(
                *session.id.as_bytes(),
                counter,
                encrypted,
            );
            let packet_bytes = encode_data_packet(&data_packet).to_vec();

            if let Err(e) = self.udp.send(&packet_bytes, &session.client_endpoint).await {
                warn!(
                    session_id = %session.id,
                    error = %e,
                    "[MINER] ⚠️ UDP send failed"
                );
                continue;
            }

            sent += 1;
        }

        sent
    }
}

impl std::fmt::Debug for ReflectionMiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReflectionMiner")
            .field("interval", &self.interval)
            .finish()
    }
}
