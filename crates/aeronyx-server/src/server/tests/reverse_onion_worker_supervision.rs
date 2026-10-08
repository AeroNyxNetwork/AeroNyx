// ============================================
// File: crates/aeronyx-server/src/server/tests/reverse_onion_worker_supervision.rs
// ============================================
//! Focused lifecycle contract tests for the required reverse recipient worker.
//! [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Also covers source
//! custody observation through the real lifecycle and durable source journal.

use super::super::reverse_onion_runtime;

// [PHALA-ORDERED-RUN-STOP 2026-10-08 by Codex] Authored, not run:
// dropping the real public-waiter guard must wake the retained queue owner,
// not stop dependency services while accepted queue work still holds a permit.
#[cfg(unix)]
#[tokio::test]
async fn cancelled_run_waiter_keeps_dependencies_through_actual_queue_drain() {
    use std::sync::{Arc, atomic::Ordering};
    use super::super::{Server, ReverseRunCancellation};
    use super::super::runtime_supervision::{
        ProcessShutdownSignals, ReverseRuntimeDependencies, PreReadyRuntimeDecision,
    };
    use crate::config::ServerConfig;
    use aeronyx_core::crypto::IdentityKeyPair;
    let directory = tempfile::Builder::new().prefix("phala-run-stop-queue-")
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
        .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
    let relay = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
    let recipient = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
    let source = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
    let mut config = ServerConfig::default();
    config.reverse_onion.queue.enabled = true;
    config.reverse_onion.queue.db_path = directory.path().join("queue.sqlite").display().to_string();
    config.reverse_onion.queue.recipient_node_ids = vec![hex::encode(recipient.public_key_bytes())];
    config.reverse_onion.queue.source_node_ids = vec![hex::encode(source.public_key_bytes())];
    config.reverse_onion.queue.max_in_flight = 1;
    let server = Server::new(config, relay.clone(), None);
    let mut dependencies = ReverseRuntimeDependencies::default();
    dependencies.queue = super::super::open_reverse_onion_queue_runtime(
        &server.config.reverse_onion, &relay,
        Arc::new(crate::services::peer_store::PeerStore::new()),
        Some("https://relay.example.net"),
    ).await.unwrap();
    let admission = dependencies.queue.as_ref().unwrap().private_admission.as_ref().unwrap().clone();
    let accepted_work = admission.try_queue_permit().unwrap();
    let mut dependency_stop = server.shutdown_tx.subscribe();
    let (_failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let mut signals = ProcessShutdownSignals::from_waiter(std::future::pending());
    {
        let waiter = server.wait_for_shutdown(&mut failure_rx, &mut signals);
        tokio::pin!(waiter);
        assert!(futures::poll!(waiter.as_mut()).is_pending());
    }
    drop(ReverseRunCancellation {
        stop: server.reverse_run_stop.clone(), source: None, recipient: None,
    });
    assert!(server.wait_for_shutdown(&mut failure_rx, &mut signals).await.is_none());
    assert_eq!(server.reverse_run_stop.before_ready(PreReadyRuntimeDecision::Ready),
        PreReadyRuntimeDecision::Stopped);
    let mut drain = Box::pin(dependencies.drain_reverse_owners(None, None));
    assert!(futures::poll!(drain.as_mut()).is_pending());
    assert!(admission.queue_stop_signal().load(Ordering::Acquire));
    assert!(admission.try_queue_permit().is_err());
    assert!(!server.shutdown.load(Ordering::Acquire));
    assert!(matches!(dependency_stop.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Empty)));
    drop(drain);
    // A cancelled drain waiter cannot convert the pending accepted permit to
    // completion or reopen admission; the retained owner can resume the drain.
    let mut drain = Box::pin(dependencies.drain_reverse_owners(None, None));
    assert!(futures::poll!(drain.as_mut()).is_pending());
    assert!(!server.shutdown.load(Ordering::Acquire));
    drop(accepted_work);
    drain.await.unwrap();
    server.publish_dependency_shutdown();
    assert!(server.shutdown.load(Ordering::Acquire));
    dependency_stop.recv().await.unwrap();
    assert!(admission.try_queue_permit().is_err());
}

// [PHALA-SOURCE-MPI-COMPOSITION 2026-10-08 by Codex] Authored, not run:
// source admission follows actual local MPI + VPN ownership; unrelated peer
// and private-recipient roles need neither an MPI runtime nor a VPN listener.
#[test]
fn reverse_source_api_composition_requires_local_mpi_and_vpn() {
    use crate::api::mpi::Mode;
    for source_enabled in [false, true] {
        for vpn_enabled in [false, true] {
            for mode in [None, Some(Mode::Local), Some(Mode::Saas)] {
                let admitted = super::super::api_runtime::validate_reverse_source_api_composition(
                    source_enabled, vpn_enabled, mode,
                ).is_ok();
                assert_eq!(admitted, !source_enabled || (vpn_enabled && mode == Some(Mode::Local)));
            }
        }
    }
}

// [PHALA-SOURCE-MPI-COMPOSITION 2026-10-08 by Codex] Authored, not run:
// the real Server::run parent gate rejects an unusable source before signal
// ownership, journal creation/migration or starting the reverse lifecycle.
#[cfg(unix)]
#[tokio::test]
async fn source_startup_rejects_incompatible_mpi_before_creating_custody() {
    use crate::config::{ServerConfig, MemChainMode};
    use crate::server::Server;
    use crate::services::reverse_onion_source::{
        tests::Fixture, ReverseOnionSourceJournal, SourceJournalLimits, SourcePhase,
    };
    use std::sync::atomic::Ordering;
    for recovery_only in [false, true] {
        for mode in [MemChainMode::Off, MemChainMode::Saas] {
            let directory = tempfile::Builder::new().prefix("phala-source-mpi-contract-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = directory.path().join("source.sqlite");
            let fixture = Fixture::new_for_runtime();
            let identity = fixture.source_identity();
            let limits = || SourceJournalLimits { max_entries: 1, max_bytes: 64 * 1024 * 1024 };
            let before = if recovery_only {
                let journal = ReverseOnionSourceJournal::open(
                    &path, identity.clone(), limits(), fixture.now(),
                ).unwrap();
                fixture.prepare(&journal, fixture.now());
                journal.arm(fixture.route(), fixture.now() + 1).unwrap();
                drop(journal);
                Some(std::fs::read(&path).unwrap())
            } else { None };
            let (relay, recipient, _) = fixture.policy_parts();
            let mut config = ServerConfig::default();
            config.memchain.mode = mode;
            config.discovery.enabled = true;
            config.discovery.gossip_enabled = true;
            config.reverse_onion.source.enabled = true;
            config.reverse_onion.source.recovery_only = recovery_only;
            config.reverse_onion.source.state_db_path = path.to_string_lossy().into_owned();
            config.reverse_onion.source.relay_node_id = hex::encode(relay.descriptor.node_id);
            config.reverse_onion.source.recipient_node_id = hex::encode(recipient.descriptor.node_id);
            config.reverse_onion.source.relay_endpoint = "https://relay.example.net".into();
            let server = Server::new(config, identity.as_ref().clone(), None);
            assert!(server.run().await.unwrap_err().to_string()
                .contains("source pulls require local or p2p MemChain MPI runtime"));
            if let Some(before) = before {
                assert_eq!(std::fs::read(&path).unwrap(), before);
                let journal = ReverseOnionSourceJournal::open_existing(
                    &path, identity, limits(), fixture.now() + 1,
                ).unwrap();
                assert_eq!(journal.lookup_phase(fixture.route(), fixture.now() + 1).unwrap(),
                    Some(SourcePhase::Armed));
            } else {
                assert!(!path.exists());
            }
            assert!(!server.reverse_run_started.load(Ordering::SeqCst));
        }
    }
}

// [PHALA-SOURCE-CONSTRUCTOR-BOUND 2026-10-08 by Codex] Authored, not run:
// constructor hardening must not normalize an invalid config into an active
// server, create a live journal, or mutate retained recovery custody.
#[cfg(unix)]
#[tokio::test]
async fn source_invalid_capacity_is_rejected_before_startup_or_custody_mutation() {
    use crate::config::{ServerConfig, MemChainMode};
    use crate::server::Server;
    use crate::services::reverse_onion_source::{
        tests::Fixture, ReverseOnionSourceJournal, SourceJournalLimits, SourcePhase,
    };
    use std::sync::atomic::Ordering;
    for recovery_only in [false, true] {
        for requested in [0, 65, usize::MAX] {
            let directory = tempfile::Builder::new().prefix("phala-source-constructor-bound-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let path = directory.path().join("source.sqlite");
            let fixture = Fixture::new_for_runtime();
            let identity = fixture.source_identity();
            let limits = || SourceJournalLimits { max_entries: 1, max_bytes: 64 * 1024 * 1024 };
            let before = if recovery_only {
                let journal = ReverseOnionSourceJournal::open(
                    &path, identity.clone(), limits(), fixture.now(),
                ).unwrap();
                fixture.prepare(&journal, fixture.now());
                journal.arm(fixture.route(), fixture.now() + 1).unwrap();
                drop(journal);
                Some(std::fs::read(&path).unwrap())
            } else { None };
            let (relay, recipient, _) = fixture.policy_parts();
            let mut config = ServerConfig::default();
            config.vpn.enabled = true;
            config.memchain.mode = MemChainMode::Local;
            config.discovery.enabled = true;
            config.discovery.gossip_enabled = true;
            config.reverse_onion.source.enabled = true;
            config.reverse_onion.source.recovery_only = recovery_only;
            config.reverse_onion.source.max_in_flight = requested;
            config.reverse_onion.source.state_db_path = path.to_string_lossy().into_owned();
            config.reverse_onion.source.relay_node_id = hex::encode(relay.descriptor.node_id);
            config.reverse_onion.source.recipient_node_id = hex::encode(recipient.descriptor.node_id);
            config.reverse_onion.source.relay_endpoint = "https://relay.example.net".into();
            let server = Server::new(config, identity.as_ref().clone(), None);
            assert!(server.run().await.unwrap_err().to_string()
                .contains("source limits outside bounded rollout policy"));
            assert!(!server.reverse_run_started.load(Ordering::SeqCst));
            if let Some(before) = before {
                assert_eq!(std::fs::read(&path).unwrap(), before);
                let journal = ReverseOnionSourceJournal::open_existing(
                    &path, identity, limits(), fixture.now() + 1,
                ).unwrap();
                assert_eq!(journal.lookup_phase(fixture.route(), fixture.now() + 1).unwrap(),
                    Some(SourcePhase::Armed));
            } else {
                assert!(!path.exists());
            }
        }
    }
}

// [PHALA-QUEUE-CAPACITY-OPEN 2026-10-08 by Codex] Authored, not run:
// actual recovery startup must reject a future durable clock before API
// publication, preserve the pending schema migration, and leave custody intact.
#[cfg(unix)]
#[tokio::test]
async fn recovery_queue_rollback_prevents_router_publication_without_migrating() {
    use aeronyx_core::crypto::IdentityKeyPair;
    use crate::config_reverse_onion::ReverseOnionConfig;
    use crate::services::reverse_onion_queue::ReverseOnionQueueLimits;
    use crate::services::reverse_onion_queue_db::{ReverseOnionQueueDb, ReverseOnionQueueDbConfig};
    use rusqlite::{params, Connection};
    let relay = IdentityKeyPair::from_bytes(&[91; 32]).unwrap();
    let recipient = IdentityKeyPair::from_bytes(&[92; 32]).unwrap().public_key_bytes();
    let directory = tempfile::Builder::new().prefix("phala-queue-open-clock-")
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
        .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
    let path = directory.path().join("queue.sqlite");
    let now = super::super::unix_now_secs();
    let future = now.checked_add(86_400).unwrap();
    let limits = ReverseOnionQueueLimits::new(4, 2 * 1024 * 1024, 2, 60, 120).unwrap();
    let config = ReverseOnionQueueDbConfig::new(path.clone(), 16 * 1024 * 1024, limits).unwrap();
    drop(ReverseOnionQueueDb::open(config, now).unwrap());
    {
        let c = Connection::open(&path).unwrap();
        c.execute_batch("DROP INDEX idx_reverse_onion_delivery_queue_v1_source_route;
            UPDATE reverse_onion_delivery_queue_v1_meta SET schema_version=3;").unwrap();
        c.execute("UPDATE reverse_onion_delivery_queue_v1_meta SET clock_high_water=?1",
            params![future as i64]).unwrap();
    }
    let mut config = ReverseOnionConfig::default();
    config.queue.enabled = true;
    config.queue.recovery_only = true;
    config.queue.db_path = path.to_string_lossy().into_owned();
    config.queue.recipient_node_ids = vec![hex::encode(recipient)];
    assert!(super::super::open_reverse_onion_recovery_runtime(&config, &relay).await.is_err());
    let c = Connection::open(&path).unwrap();
    let observed: (i64, i64) = c.query_row(
        "SELECT schema_version,clock_high_water FROM reverse_onion_delivery_queue_v1_meta", [],
        |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert_eq!(observed, (3, future as i64));
    let index_exists: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master
        WHERE name='idx_reverse_onion_delivery_queue_v1_source_route')", [], |r| r.get(0)).unwrap();
    assert!(!index_exists);
}

// [PHALA-JOURNAL-OPEN-OWNER 2026-10-08 by Codex] Authored only:
// actual worker startup must fail before readiness, scheduling or terminal
// dispatch for foreign/noncanonical/rollback metadata, then drain its owner.
#[cfg(unix)]
#[tokio::test]
async fn recipient_recovery_open_metadata_failure_never_publishes_ready() {
    use std::sync::{Arc, atomic::Ordering};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use aeronyx_core::crypto::IdentityKeyPair;
    use rusqlite::{params, types::Value, Connection};
    use crate::api::reverse_onion_terminal::ReverseOnionTerminalAdapter;
    use crate::config_reverse_onion::ReverseOnionConfig;
    use crate::services::{peer_store::PeerStore, reverse_onion_recipient::{
        ReverseOnionRecipientJournal, RecipientJournalLimits,
    }};
    use reverse_onion_runtime::{ReverseOnionRecipientWorker, RecipientWorkerError};
    for scenario in 0..4 {
        let directory = tempfile::Builder::new().prefix("phala-recipient-open-owner-")
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
            .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let path = directory.path().join("recipient.sqlite");
        let relay = IdentityKeyPair::from_bytes(&[95; 32]).unwrap().public_key_bytes();
        let recipient = Arc::new(IdentityKeyPair::from_bytes(&[96; 32]).unwrap());
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        drop(ReverseOnionRecipientJournal::open(&path, relay, recipient.public_key_bytes(),
            RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now).unwrap());
        let snapshot = {
            let connection = Connection::open(&path).unwrap();
            match scenario {
                1 => { connection.execute("UPDATE recipient_meta SET recipient=?1",
                    params![IdentityKeyPair::from_bytes(&[97; 32]).unwrap().public_key_bytes().as_slice()]).unwrap(); }
                2 => { connection.execute("UPDATE recipient_meta SET clock=?1", params![i64::MAX - 1]).unwrap(); }
                3 => { connection.execute("UPDATE recipient_meta SET clock=?1", params![now as f64 + 0.5]).unwrap(); }
                _ => {}
            }
            connection.query_row("SELECT relay,recipient,clock FROM recipient_meta", [],
                |r| Ok((r.get::<_, Value>(0)?, r.get::<_, Value>(1)?, r.get::<_, Value>(2)?))).unwrap()
        };
        let mut config = ReverseOnionConfig::default();
        config.recipient.enabled = true;
        config.recipient.recovery_only = true;
        config.recipient.relay_node_id = hex::encode(relay);
        config.recipient.relay_endpoint = "https://1.1.1.1".into();
        config.recipient.state_db_path = path.to_string_lossy().into_owned();
        config.recipient.max_pending_items = 4;
        let adapter = Arc::new(ReverseOnionTerminalAdapter::new(
            axum::Router::new(), recipient.clone(), relay, Duration::from_secs(1),
        ).unwrap());
        let worker = ReverseOnionRecipientWorker::start(
            &config, recipient, Arc::new(PeerStore::new()), adapter.clone(),
        ).unwrap();
        let ready = tokio::time::timeout(Duration::from_secs(5), worker.wait_ready()).await.unwrap();
        if scenario == 0 {
            assert_eq!(ready, Ok(()));
            assert_eq!(worker.shutdown_and_drain().await, Ok(()));
        } else {
            assert_eq!(ready, Err(RecipientWorkerError::Unavailable));
            assert!(adapter.intake_stop_flag().load(Ordering::SeqCst));
            assert_eq!(worker.shutdown_and_drain().await, Err(RecipientWorkerError::Unavailable));
            let connection = Connection::open(&path).unwrap();
            let after = connection.query_row("SELECT relay,recipient,clock FROM recipient_meta", [],
                |r| Ok((r.get::<_, Value>(0)?, r.get::<_, Value>(1)?, r.get::<_, Value>(2)?))).unwrap();
            assert_eq!(after, snapshot);
            let count: i64 = connection.query_row("SELECT count(*) FROM recipient_jobs", [], |r| r.get(0)).unwrap();
            assert_eq!(count, 0);
        }
    }
}

// [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Authored, not run:
// embedded callers must reject bad complete policies before pinning or SQL.
#[cfg(unix)]
#[tokio::test]
async fn live_queue_rejects_identity_and_capacity_failures_before_database_open() {
    use std::sync::Arc;
    use aeronyx_core::crypto::IdentityKeyPair;
    use crate::config_reverse_onion::ReverseOnionConfig;
    use crate::services::peer_store::PeerStore;
    let id = |seed| IdentityKeyPair::from_bytes(&[seed; 32]).unwrap().public_key_bytes();
    let relay = IdentityKeyPair::from_bytes(&[101; 32]).unwrap();
    let recipient = id(102);
    for scenario in 0..9 {
        let directory = tempfile::Builder::new().prefix("phala-queue-pin-startup-")
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
            .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let path = directory.path().join("queue.sqlite");
        let mut config = ReverseOnionConfig::default();
        config.queue.enabled = true;
        config.queue.db_path = path.to_string_lossy().into_owned();
        config.queue.recipient_node_ids = vec![hex::encode(recipient)];
        config.queue.source_node_ids = vec![hex::encode(id(103))];
        let peers = Arc::new(PeerStore::new());
        match scenario {
            0 => config.queue.source_node_ids.clear(),
            1 => config.queue.source_node_ids.push(hex::encode(id(103)).to_uppercase()),
            2 => config.queue.source_node_ids = vec![hex::encode(relay.public_key_bytes())],
            3 => config.queue.source_node_ids = vec![hex::encode(recipient)],
            4 => config.queue.source_node_ids = (1..=65).map(|seed| hex::encode(id(seed))).collect(),
            5 => config.queue.max_in_flight = 65,
            6 => config.queue.route_max_secs = u64::MAX,
            7 => {
                for seed in 1..=64 { peers.pin_private_onion_source_identity(id(seed)).unwrap(); }
            }
            8 => {
                for seed in 1..=64 { peers.pin_private_onion_route_identities(id(seed), recipient).unwrap(); }
            }
            _ => unreachable!(),
        }
        assert!(super::super::open_reverse_onion_queue_runtime(
            &config, &relay, Arc::clone(&peers), Some("https://relay.example.net"),
        ).await.is_err(), "scenario {scenario}");
        assert!(!path.exists(), "no rejected policy may open SQL: {scenario}");
        assert!(!peers.has_private_onion_route_identity_pin(&relay.public_key_bytes(), &recipient));
        assert!(peers.current_private_onion_pull_authority_snapshot(
            &relay.public_key_bytes(), &recipient, 1_700_000_100).is_none());
    }
}

// [PHALA-QUEUE-IDENTITY-PINS 2026-10-08 by Codex] Authored, not run:
// pin-only bootstrap stays usable without treating pins as fresh authority.
#[cfg(unix)]
#[tokio::test]
async fn live_queue_pin_only_startup_reapplies_the_same_full_policy() {
    use std::sync::Arc;
    use aeronyx_core::crypto::IdentityKeyPair;
    use crate::config_reverse_onion::ReverseOnionConfig;
    use crate::services::peer_store::PeerStore;
    let relay = IdentityKeyPair::from_bytes(&[101; 32]).unwrap();
    let recipient = IdentityKeyPair::from_bytes(&[102; 32]).unwrap().public_key_bytes();
    let source = IdentityKeyPair::from_bytes(&[103; 32]).unwrap().public_key_bytes();
    let directory = tempfile::Builder::new().prefix("phala-queue-pin-only-")
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
        .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
    let mut config = ReverseOnionConfig::default();
    config.queue.enabled = true;
    config.queue.db_path = directory.path().join("queue.sqlite").to_string_lossy().into_owned();
    config.queue.recipient_node_ids = vec![hex::encode(recipient).to_uppercase()];
    config.queue.source_node_ids = vec![hex::encode(source).to_uppercase()];
    let peers = Arc::new(PeerStore::new());
    let pins = config.queue.live_identity_pins(relay.public_key_bytes()).unwrap();
    peers.pin_private_onion_queue_identities(&pins).unwrap();
    let runtime = super::super::open_reverse_onion_queue_runtime(
        &config, &relay, Arc::clone(&peers), Some("https://relay.example.net"),
    ).await.unwrap().unwrap();
    let admission = runtime.private_admission.as_ref().unwrap();
    assert_eq!(admission.recipient_node_id(), recipient);
    assert!(admission.source_allowed(source));
    assert!(!admission.source_allowed(relay.public_key_bytes()));
    assert!(peers.is_empty());
    assert!(peers.current_private_onion_pull_authority_snapshot(
        &relay.public_key_bytes(), &recipient, 1_700_000_100).is_none());
    runtime.shutdown_and_drain().await;
}

// [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] Authored, not
// run: actual source resume rejects a known impossible trusted local floor
// before transport or journal mutation. The required-server gate must retain
// this fault even when shutdown races it, and restart preserves the old phase.
#[cfg(unix)]
#[tokio::test]
async fn source_local_clock_floor_failure_prevents_ready_without_replacing_durable_work() {
    use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use axum::body::Bytes;
    use crate::api::PinnedPeerHttpTarget;
    use crate::services::reverse_onion_source::{tests::Fixture, SourcePhase};
    use super::super::{runtime_supervision::{self, PreReadyRuntimeDecision as D},
        reverse_onion_source_runtime::{ReverseOnionSourceRuntime, ReverseOnionSourceLifecycle,
            SourcePinnedRelayPolicy, SourceRuntimeConfig, SourceRuntimeError, SourceTransport,
            SourceTransportOutcome, SourceSendAdmission}};
    struct NoEntryTransport(AtomicUsize);
    #[async_trait::async_trait]
    impl SourceTransport for NoEntryTransport {
        async fn post(&self, _: PinnedPeerHttpTarget, _: Bytes, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            self.0.fetch_add(1, Ordering::SeqCst);
            panic!("source clock fault must precede HTTP entry");
        }
        async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            self.0.fetch_add(1, Ordering::SeqCst);
            panic!("source clock fault must precede evidence HTTP entry");
        }
    }
    for armed in [false, true] {
        let f = Fixture::new_for_runtime();
        let journal = Arc::new(f.open(f.now()));
        f.prepare(&journal, f.now());
        if armed { journal.arm(f.route(), f.now() + 1).unwrap(); }
        let transport = Arc::new(NoEntryTransport(AtomicUsize::new(0)));
        let (relay, recipient, authorization) = f.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy::new(f.source_identity().public_key_bytes(),
            relay, recipient, authorization, f.now()).unwrap());
        let runtime = Arc::new(ReverseOnionSourceRuntime::new(journal.clone(), f.source_identity(),
            policy, transport.clone(), SourceRuntimeConfig::new(1, Duration::from_secs(1), 600,
                Duration::ZERO, Duration::from_millis(250)).unwrap()).unwrap());
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        lifecycle.install(runtime.clone()).unwrap();
        let (_critical_tx, mut critical_rx) = tokio::sync::mpsc::channel(1);
        let shutdown = AtomicBool::new(false);
        assert_eq!(runtime_supervision::pre_ready_runtime_decision(
            &mut critical_rx, &shutdown, Some(&lifecycle), None, false), D::Ready);
        assert_eq!(lifecycle.resume(f.route(), u64::MAX).await.err(), Some(SourceRuntimeError::Unavailable));
        assert!(lifecycle.has_failed());
        assert!(lifecycle.request_admission().is_stopped());
        shutdown.store(true, Ordering::SeqCst);
        let failure = runtime_supervision::reverse_onion_source_failed();
        assert_eq!(runtime_supervision::pre_ready_runtime_decision(
            &mut critical_rx, &shutdown, Some(&lifecycle), None, false), D::Failed(failure.clone()));
        assert_eq!(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)).await, failure);
        assert_eq!(transport.0.load(Ordering::SeqCst), 0);
        lifecycle.shutdown_and_drain().await.unwrap();
        lifecycle.shutdown_and_drain().await.unwrap();
        drop(lifecycle);
        drop(runtime);
        drop(journal);
        let at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let reopened = f.open_existing(at);
        assert_eq!(reopened.lookup_phase(f.route(), at).unwrap(),
            Some(if armed { SourcePhase::Armed } else { SourcePhase::Prepared }));
    }
}

// [PHALA-CONNECTED-REVERSE-LOOP 2026-10-07 by Codex] Authored, not run.
// Source HTTP transport uses Router::oneshot; the fixture drives recipient
// Claim/Result exchanges through real APIs and the worker's Lease verifier.
// Relay ingress, recipient durability, Armed admission, private Blind Vault
// execution and source evidence/opening use production code. This does not
// cover independent-host KEM custody, TLS, DNS or Phala attestation.
// [PHALA-ACTUAL-RECIPIENT-WORKER 2026-10-07 by Codex] Lease execution now
// uses actual recovery-only worker startup/scheduling/drain. Its empty route
// store prevents network entry; the fixture submits the persisted Result to R.
// [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] Additional fresh
// cases let that same worker create Claims and exchange Results itself using
// cfg(test) in-process transport; the recovery/drain cases remain separate.
#[cfg(unix)]
mod connected_reverse_loop {
    use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
    use aeronyx_core::protocol::onion::OnionRoutePurpose;
    // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex] Real
    // signed replacements, not a toggled readiness flag or fabricated frame.
    use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor,
        SignedPrivateOnionRecipientAuthorizationV1};
    // [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] Authored, not run:
    // include actual wallet-signed HTTP input and canonical response decoding.
    use aeronyx_core::protocol::onion::reverse_delivery::{ReverseOnionFrameV1,
        reverse_onion_source_route_id, ReverseOnionSourcePullRequestV1,
        ReverseOnionSourcePullResponseV1, MAX_REVERSE_ONION_SOURCE_PULL_RESPONSE_BYTES};
    use axum::{body::{Body, Bytes, to_bytes}, http::Request, Router};
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Authenticate the
    // connected source through the actual MPI body digest, not an extension.
    use sha2::{Digest, Sha256};
    use tower::ServiceExt;
    use crate::api::{PinnedPeerHttpTarget, chat_peer::{
        build_chat_peer_router_with_private_recipient_admission,
        PrivateBlindRelayAdmission, PeerBlindRelayResponse,
    }, reverse_onion::{ReverseOnionApi, build_reverse_onion_router},
        reverse_onion_terminal::ReverseOnionTerminalAdapter,
        reverse_onion_source::{build_reverse_onion_source_router, SOURCE_PULL_PATH},
        mpi::{MpiState, SessionEmbeddingCache, build_mpi_router_with_reverse_onion_source}};
    use crate::config::ChatRelayConfig;
    use crate::config_reverse_onion::ReverseOnionConfig;
    use crate::services::{ChatRelayService, SessionManager, peer_store::PeerStore,
        reverse_onion_queue::ReverseOnionQueueLimits,
        reverse_onion_queue_db::{ReverseOnionQueueDb, ReverseOnionQueueDbConfig},
        reverse_onion_recipient::{ReverseOnionRecipientJournal, RecipientJournalLimits, RecipientRecovery},
        reverse_onion_source::{SourcePhase, tests::Fixture}};
    use super::super::super::{reverse_onion_runtime::{
        accept_poll_response, PollAttempt, ReverseOnionRecipientWorker,
        RecipientOutboundCarrier, RecipientWorkerError, ReverseOnionExchange,
        ReverseOnionPreflightError,
    }, reverse_onion_source_runtime::{
        ReverseOnionSourceRuntime, ReverseOnionSourceLifecycle, SourcePinnedRelayPolicy,
        SourceRuntimeConfig, SourceRuntimeError, SourceTransport, SourceTransportOutcome,
        SourceSendAdmission,
    }};

    fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() }

    async fn request(router: Router, path: &str, body: Bytes, authorization: Option<Bytes>) -> (u16, Vec<u8>) {
        let mut request = Request::builder().method("POST").uri(path)
            .header("content-type", if authorization.is_some() { "application/json" } else { "application/octet-stream" });
        if let Some(authorization) = authorization {
            request = request.header("x-aeronyx-private-recipient-authorization", STANDARD.encode(authorization));
        }
        let response = router.oneshot(request.body(Body::from(body)).unwrap()).await.unwrap();
        let status = response.status().as_u16();
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap().to_vec();
        (status, body)
    }

    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] In-memory MPI
    // storage exists only to compose the actual auth middleware; no inference
    // engine or production owner state is provided by this synthetic fixture.
    fn source_mpi_state(identity: &IdentityKeyPair) -> Arc<MpiState> {
        use std::collections::HashMap;
        use parking_lot::RwLock;
        use crate::services::memchain::{MemoryStorage, VectorIndex};
        Arc::new(MpiState::local(
            Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
            Arc::new(VectorIndex::new()), identity.clone(),
            RwLock::new(HashMap::new()), AtomicBool::new(true),
            Arc::new(RwLock::new(HashMap::new())), 0.0, false,
            RwLock::new(SessionEmbeddingCache::default()), RwLock::new(None),
            identity.public_key_bytes(), Some("connected-source-secret".to_string()),
            None, true, false, 0, None, false, false, None, None, None,
        ))
    }

    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Real Local Bearer
    // or Remote Ed25519 MPI authentication precedes the signed source body.
    // Router::oneshot still does not establish HTTPS or independent hosts.
    async fn source_request(router: Router, input: &ReverseOnionSourcePullRequestV1,
        identity: &IdentityKeyPair, remote: bool) -> (u16, Vec<u8>) {
        let response = source_http_response(router, input, identity, remote).await;
        let status = response.status().as_u16();
        (status, to_bytes(response.into_body(), MAX_REVERSE_ONION_SOURCE_PULL_RESPONSE_BYTES)
            .await.unwrap().to_vec())
    }

    // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] Keep the
    // real authenticated response available for a disconnected/slow reader.
    async fn source_http_response(router: Router, input: &ReverseOnionSourcePullRequestV1,
        identity: &IdentityKeyPair, remote: bool) -> axum::response::Response {
        let body = input.encode_json().unwrap();
        let mut request = Request::builder().method("POST").uri(SOURCE_PULL_PATH)
            .header("content-type", "application/json");
        if remote {
            let timestamp = now().to_string();
            let mut digest = Sha256::new();
            digest.update(timestamp.as_bytes());
            digest.update(b"POST");
            digest.update(SOURCE_PULL_PATH.as_bytes());
            digest.update(Sha256::digest(&body));
            request = request.header("x-memchain-publickey", hex::encode(identity.public_key_bytes()))
                .header("x-memchain-timestamp", timestamp)
                .header("x-memchain-signature", hex::encode(identity.sign(&digest.finalize())));
        } else {
            request = request.header("authorization", "Bearer connected-source-secret");
        }
        let response = router.oneshot(request.body(Body::from(body)).unwrap()).await.unwrap();
        assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
            "no-store, no-cache, must-revalidate, private");
        assert_eq!(response.headers()[axum::http::header::PRAGMA], "no-cache");
        assert_eq!(response.headers()[axum::http::header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        response
    }

    struct LoopTransport {
        relay_router: Router,
        recipient: Arc<IdentityKeyPair>,
        relay_id: [u8; 32],
        origin: [u8; 32],
        journal_path: std::path::PathBuf,
        worker_config: ReverseOnionConfig,
        result_ready: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<()>>,
        result_gate: Arc<tokio::sync::Semaphore>,
        adapter: Arc<ReverseOnionTerminalAdapter>,
        // [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] A restarted
        // owner gets a new stop gate, never reopens the drained adapter.
        terminal_router: Router,
        delivered: tokio::sync::Mutex<bool>,
        posts: AtomicUsize,
        queries: AtomicUsize,
        executions: Arc<AtomicUsize>,
        lose_ack: bool,
        tamper_next_evidence: AtomicBool,
        // [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] Fresh cases
        // use the worker's own Claim/Result path; recovery cases stay intact.
        fresh_peers: Option<Arc<PeerStore>>,
        // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
        renewal: Option<Arc<ConnectedAuthorityRenewal>>,
    }

    // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex] The new
    // grant is deliberately withheld until exact old custody completes.
    struct ConnectedAuthorityRenewal {
        relay: SignedNodeDescriptor,
        recipient: SignedNodeDescriptor,
        previous_grant_issued_at: u64,
    }

    // [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] Replace only the
    // outbound network with real in-process R routes. The HTTPS echo bit is
    // synthetic transport evidence, NOT TLS/attestation coverage. Save bytes
    // to prove lost responses retry the identical Claim, Lease and Result.
    #[derive(Default)]
    struct FreshRecipientTrace {
        claim: std::sync::Mutex<Option<Vec<u8>>>,
        lease: std::sync::Mutex<Option<Vec<u8>>>,
        result: std::sync::Mutex<Option<Vec<u8>>>,
        claims: AtomicUsize,
        results: AtomicUsize,
        completed: AtomicBool,
        recovery: AtomicBool,
        result_delivered: tokio::sync::Notify,
        // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
        renewals: AtomicUsize,
    }

    struct FreshRecipientCarrier {
        router: Router,
        peers: Arc<PeerStore>,
        relay: [u8; 32],
        recipient: [u8; 32],
        origin: [u8; 32],
        timeout: Duration,
        stopped: Arc<AtomicBool>,
        trace: Arc<FreshRecipientTrace>,
        // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
        renewal: Option<Arc<ConnectedAuthorityRenewal>>,
    }

    #[async_trait::async_trait]
    impl RecipientOutboundCarrier for FreshRecipientCarrier {
        fn binding(&self) -> ([u8; 32], [u8; 32], Duration) {
            (self.relay, self.recipient, self.timeout)
        }
        fn has_current_pull_authority(&self) -> Result<bool, RecipientWorkerError> {
            Ok(!self.stopped.load(Ordering::SeqCst)
                && (!self.trace.completed.load(Ordering::SeqCst) || self.trace.recovery.load(Ordering::SeqCst))
                && self.peers.current_private_onion_pull_authority_snapshot(
                    &self.relay, &self.recipient, now()).is_some())
        }
        async fn prepare_new_claim_origin(&self)
            -> Result<Option<([u8; 32], u64)>, ReverseOnionPreflightError> {
            if !self.has_current_pull_authority().unwrap() { return Ok(None); }
            assert!(self.trace.claim.lock().unwrap().is_none(), "one unresolved Claim fences new polls");
            Ok(Some((self.origin, now())))
        }
        async fn exchange(&self, frame: &ReverseOnionFrameV1, require_live_pull_authority: bool,
            expected_origin: [u8; 32]) -> Result<ReverseOnionExchange, ReverseOnionPreflightError> {
            use aeronyx_core::protocol::onion::reverse_delivery::ReverseOnionKindV1;
            if self.stopped.load(Ordering::SeqCst) { return Ok(ReverseOnionExchange::Deferred); }
            assert_eq!((frame.relay(), frame.immediate_recipient()), (self.relay, self.recipient));
            assert_eq!(expected_origin, self.origin);
            frame.verify_recipient_retry(now()).unwrap();
            let descriptor = if require_live_pull_authority {
                self.peers.current_private_onion_pull_authority_snapshot(
                    &self.relay, &self.recipient, now()).unwrap().relay
            } else {
                self.peers.current_private_onion_relay_descriptor(&self.relay, now()).unwrap()
            };
            assert_eq!(crate::api::reverse_onion_origin_commitment(
                descriptor.descriptor.public_endpoint.as_deref().unwrap()).unwrap(), self.origin);
            let exact = frame.encode();
            let (path, first) = match frame.kind() {
                ReverseOnionKindV1::Claim => {
                    let first = self.trace.claims.fetch_add(1, Ordering::SeqCst) == 0;
                    let mut saved = self.trace.claim.lock().unwrap();
                    if first { *saved = Some(exact.clone()); } else { assert_eq!(saved.as_ref(), Some(&exact)); }
                    assert_eq!(require_live_pull_authority, first);
                    ("/api/chat/peer/reverse-onion/claim", first)
                }
                ReverseOnionKindV1::Result => {
                    assert!(!require_live_pull_authority);
                    let first = self.trace.results.fetch_add(1, Ordering::SeqCst) == 0;
                    let mut saved = self.trace.result.lock().unwrap();
                    if first { *saved = Some(exact.clone()); } else { assert_eq!(saved.as_ref(), Some(&exact)); }
                    ("/api/chat/peer/reverse-onion/result", first)
                }
                _ => panic!("recipient worker must not send a Lease"),
            };
            let Some(seconds) = frame.recipient_retry_deadline().unwrap().checked_sub(now())
                .filter(|seconds| *seconds != 0) else { return Ok(ReverseOnionExchange::Deferred); };
            let remaining = Duration::from_secs(seconds);
            let response = tokio::time::timeout(self.timeout.min(remaining),
                request(self.router.clone(), path, Bytes::from(exact.clone()), None)).await;
            let Ok((status, body)) = response else { return Ok(ReverseOnionExchange::Ambiguous); };
            // [PHALA-CONNECTED-INGRESS-DIAGNOSTIC 2026-10-08 by Codex]
            // Only this synthetic protocol response is printed on failure.
            assert_eq!(status, 200, "recipient carrier: {}", String::from_utf8_lossy(&body));
            if frame.kind() == ReverseOnionKindV1::Claim {
                let mut lease = self.trace.lease.lock().unwrap();
                if first { *lease = Some(body.clone()); } else { assert_eq!(lease.as_ref(), Some(&body)); }
            } else {
                assert_eq!(body, exact);
                if !first {
                    self.trace.completed.store(true, Ordering::SeqCst);
                    self.trace.result_delivered.notify_one();
                }
            }
            // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
            // R has actually armed and persisted this Lease. Lose its reply
            // only after both authenticated descriptors move to a new epoch;
            // do not install the matching new grant yet. Recovery must not
            // depend on fresh Pull authority or change any signed frame.
            if first && frame.kind() == ReverseOnionKindV1::Claim {
                if let Some(renewal) = &self.renewal {
                    for descriptor in [&renewal.relay, &renewal.recipient] {
                        assert!(self.peers.upsert_verified_from_source(
                            descriptor.clone(), now(), "connected_renewal").unwrap());
                        assert_eq!(self.peers.get_valid(&descriptor.node_id(), now()).as_ref(),
                            Some(descriptor));
                    }
                    assert!(self.peers.current_private_onion_pull_authority_snapshot(
                        &self.relay, &self.recipient, now()).is_none(),
                        "the old grant cannot authorize the renewed pair");
                    assert_eq!(self.trace.renewals.fetch_add(1, Ordering::SeqCst), 0);
                }
            }
            // The relay actually accepted each first POST. Lose its reply,
            // never fabricate zero-send or replace a durable frame afterward.
            if first { return Ok(ReverseOnionExchange::Ambiguous); }
            Ok(ReverseOnionExchange::Response { status, body,
                authenticated_https: true, observed_at: now() })
        }
    }

    impl LoopTransport {
        async fn deliver_once(&self) {
            let mut delivered = self.delivered.lock().await;
            if *delivered { return; }
            // [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] No manual
            // journal prepare/Lease acceptance/Result submission in this path.
            if let Some(peers) = &self.fresh_peers {
                self.deliver_fresh(peers.clone()).await;
                *delivered = true;
                return;
            }
            let at = now();
            // [PHALA-ACTUAL-RECIPIENT-WORKER 2026-10-07 by Codex] Close
            // the bootstrap owner before the actual worker acquires its inode.
            let journal = Arc::new(ReverseOnionRecipientJournal::open_existing(&self.journal_path,
                self.relay_id, self.recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, at).unwrap());
            let claim = ReverseOnionFrameV1::claim(self.relay_id, [72; 16], at, at + 30, &self.recipient).unwrap();
            let exact = journal.prepare_poll(&claim, self.origin, at).unwrap();
            let (status, lease) = request(self.relay_router.clone(), "/api/chat/peer/reverse-onion/claim",
                Bytes::from(exact.clone()), None).await;
            // [PHALA-CLAIM-EXECUTION-CAP 2026-10-08 by Codex] Exercise the
            // real signed Lease, not merely HTTP acceptance of this Claim.
            assert_eq!(status, 200, "connected Claim rejected");
            let issued_lease = ReverseOnionFrameV1::decode_for_recovery(&lease).unwrap();
            assert_eq!(issued_lease.expires_at(), issued_lease.issued_at() + 300);
            assert!(issued_lease.expires_at() > claim.expires_at());
            // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] The
            // connected response supplies only its checked local completion.
            assert_eq!(accept_poll_response(&journal, exact.clone(), claim.clone(), self.relay_id,
                self.recipient.public_key_bytes(), lease.clone(), now()).await.unwrap(), PollAttempt::Progressed);
            // Duplicate Claim must replay the exact already-armed relay Lease.
            let (status, retry) = request(self.relay_router.clone(), "/api/chat/peer/reverse-onion/claim",
                Bytes::from(exact), None).await;
            assert_eq!((status, retry), (200, lease));
            drop(journal);
            let worker = ReverseOnionRecipientWorker::start(&self.worker_config, self.recipient.clone(),
                Arc::new(PeerStore::new()), self.adapter.clone()).unwrap();
            worker.wait_ready().await.unwrap();
            // The signal precedes result persistence. Stop must still drain
            // the entered terminal operation and its exact journal commit.
            let mut ready = self.result_ready.lock().await;
            let observed = tokio::time::timeout(Duration::from_secs(30), ready.recv()).await;
            drop(ready);
            worker.request_stop();
            let drain = worker.shutdown_and_drain();
            tokio::pin!(drain);
            if matches!(observed, Ok(Some(()))) {
                assert!(futures::poll!(&mut drain).is_pending(), "result persistence remains owned during stop");
            }
            self.result_gate.add_permits(1);
            tokio::time::timeout(Duration::from_secs(30), drain).await.unwrap().unwrap();
            assert!(matches!(observed, Ok(Some(()))));
            assert_eq!(self.executions.load(Ordering::SeqCst), 1);
            let journal = ReverseOnionRecipientJournal::open_existing(&self.journal_path,
                self.relay_id, self.recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now()).unwrap();
            let page = journal.resume(None, 64, now()).unwrap();
            let [RecipientRecovery::Result { exact_bytes: exact_result, .. }] = page.items.as_slice()
                else { panic!("drained worker must preserve its exact Result") };
            let exact_result = exact_result.clone();
            // Duplicate Result is custody-only and cannot reenter the terminal.
            for _ in 0..2 {
                let (status, echo) = request(self.relay_router.clone(), "/api/chat/peer/reverse-onion/result",
                    Bytes::from(exact_result.clone()), None).await;
                assert_eq!((status, echo), (200, exact_result.clone()));
            }
            assert!(journal.arm(claim.claim_id(), now()).is_err());
            *delivered = true;
        }

        async fn deliver_fresh(&self, peers: Arc<PeerStore>) {
            assert!(!self.worker_config.recipient.recovery_only);
            let trace = Arc::new(FreshRecipientTrace::default());
            let worker = ReverseOnionRecipientWorker::start_with_test_carrier(
                &self.worker_config, self.recipient.clone(), self.adapter.clone(), |stopped| {
                    FreshRecipientCarrier { router: self.relay_router.clone(), peers: peers.clone(),
                        relay: self.relay_id, recipient: self.recipient.public_key_bytes(), origin: self.origin,
                        timeout: Duration::from_secs(self.worker_config.recipient.request_timeout_secs),
                        stopped, trace: trace.clone(), renewal: self.renewal.clone() }
                }).unwrap();
            worker.wait_ready().await.unwrap();
            let observed = {
                let mut ready = self.result_ready.lock().await;
                tokio::time::timeout(Duration::from_secs(30), ready.recv()).await
            };
            // The terminal has entered but its Result is not yet durable;
            // the worker must not send a Result ahead of that commit.
            let results_before_commit = trace.results.load(Ordering::SeqCst);
            self.result_gate.add_permits(1);
            let completed = tokio::time::timeout(Duration::from_secs(30),
                trace.result_delivered.notified()).await;
            worker.request_stop();
            tokio::time::timeout(Duration::from_secs(30), worker.shutdown_and_drain()).await.unwrap().unwrap();
            assert!(matches!(observed, Ok(Some(()))));
            assert!(completed.is_ok());
            assert_eq!(results_before_commit, 0);
            assert_eq!(trace.claims.load(Ordering::SeqCst), 2);
            assert_eq!(trace.results.load(Ordering::SeqCst), 2);
            assert_eq!(self.executions.load(Ordering::SeqCst), 1);
            // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
            // Exact Claim retry and both Result attempts succeeded while no
            // grant matched the current pair. Restore only fresh authority
            // before restart, so recovery_only itself must forbid a new poll.
            assert_eq!(trace.renewals.load(Ordering::SeqCst), usize::from(self.renewal.is_some()));
            if let Some(renewal) = &self.renewal {
                assert!(peers.current_private_onion_pull_authority_snapshot(
                    &self.relay_id, &self.recipient.public_key_bytes(), now()).is_none());
                // The real cache rejects different grants at the same issued
                // second. Await a bounded live sample; do not invent a future
                // timestamp or bypass that anti-equivocation boundary.
                let issued_at = tokio::time::timeout(Duration::from_secs(2), async {
                    loop {
                        let at = now();
                        if at > renewal.previous_grant_issued_at { break at; }
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                }).await.expect("replacement grant requires a later live second");
                let grant = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
                    &renewal.relay, &renewal.recipient, OnionRoutePurpose::BlindVaultPull.as_str(),
                    issued_at, issued_at + 600, &self.recipient).unwrap();
                assert!(peers.import_private_onion_authorization_bundle(grant,
                    renewal.relay.clone(), renewal.recipient.clone(), self.relay_id, now()).unwrap());
                let current = peers.current_private_onion_pull_authority_snapshot(
                    &self.relay_id, &self.recipient.public_key_bytes(), now()).unwrap();
                assert_eq!(current.relay, renewal.relay);
                assert_eq!(current.recipient, renewal.recipient);
            }
            let journal = ReverseOnionRecipientJournal::open_existing(&self.journal_path,
                self.relay_id, self.recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now()).unwrap();
            let page = journal.resume(None, 64, now()).unwrap();
            let [RecipientRecovery::Result { exact_bytes, .. }] = page.items.as_slice()
                else { panic!("fresh worker must preserve its exact Result for restart") };
            assert_eq!(Some(exact_bytes), trace.result.lock().unwrap().as_ref());
            let claim = ReverseOnionFrameV1::decode_for_recovery(
                trace.claim.lock().unwrap().as_ref().unwrap()).unwrap();
            assert!(journal.arm(claim.claim_id(), now()).is_err());
            drop(journal);
            // Result ACK state is intentionally process-local. A real new
            // recovery-only worker resends that same durable Result; it must
            // not poll, rearm a Lease or reenter the terminal after restart.
            let mut recovery_config = self.worker_config.clone();
            recovery_config.recipient.recovery_only = true;
            // Leave current authority present on restart: recovery mode,
            // not a missing route, must prevent a replacement Claim.
            trace.recovery.store(true, Ordering::SeqCst);
            let recovery_adapter = Arc::new(ReverseOnionTerminalAdapter::new(
                self.terminal_router.clone(), self.recipient.clone(), self.relay_id,
                Duration::from_secs(30)).unwrap());
            let recovery = ReverseOnionRecipientWorker::start_with_test_carrier(
                &recovery_config, self.recipient.clone(), recovery_adapter, |stopped| {
                    FreshRecipientCarrier { router: self.relay_router.clone(), peers,
                        relay: self.relay_id, recipient: self.recipient.public_key_bytes(), origin: self.origin,
                        timeout: Duration::from_secs(recovery_config.recipient.request_timeout_secs),
                        stopped, trace: trace.clone(), renewal: self.renewal.clone() }
                }).unwrap();
            recovery.wait_ready().await.unwrap();
            let replayed = tokio::time::timeout(Duration::from_secs(30),
                trace.result_delivered.notified()).await;
            recovery.request_stop();
            tokio::time::timeout(Duration::from_secs(30), recovery.shutdown_and_drain()).await.unwrap().unwrap();
            assert!(replayed.is_ok());
            assert_eq!(trace.claims.load(Ordering::SeqCst), 2);
            assert_eq!(trace.results.load(Ordering::SeqCst), 3);
            assert_eq!(self.executions.load(Ordering::SeqCst), 1);
        }
    }

    #[async_trait::async_trait]
    impl SourceTransport for LoopTransport {
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Local Router
        // transport observes the production owner/deadline admission contract.
        async fn post(&self, target: PinnedPeerHttpTarget, body: Bytes, authorization: Bytes,
            admission: SourceSendAdmission) -> SourceTransportOutcome {
            if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
            assert_eq!(target.url.path(), "/api/chat/peer/blind-relay");
            self.posts.fetch_add(1, Ordering::SeqCst);
            let (status, ack) = request(self.relay_router.clone(), target.url.path(), body.clone(),
                Some(authorization.clone())).await;
            // [PHALA-CONNECTED-INGRESS-DIAGNOSTIC 2026-10-08 by Codex]
            assert_eq!(status, 200, "relay ingress: {}", String::from_utf8_lossy(&ack));
            let receipt: PeerBlindRelayResponse = serde_json::from_slice(&ack).unwrap();
            assert!(receipt.accepted && receipt.forwarded && !receipt.terminal);
            assert!(receipt.opaque_terminal_response_b64.is_none());
            assert_eq!(self.executions.load(Ordering::SeqCst), 0, "custody is not execution");
            // Deliberately inject a duplicate at relay ingress, not another
            // source attempt or a fabricated queue insertion.
            let replay = request(self.relay_router.clone(), target.url.path(), body, Some(authorization)).await;
            assert_eq!(replay, (status, ack.clone()));
            if self.lose_ack { SourceTransportOutcome::Ambiguous }
            else { SourceTransportOutcome::Response { status, body: ack } }
        }
        async fn query(&self, target: PinnedPeerHttpTarget, body: Bytes,
            admission: SourceSendAdmission) -> SourceTransportOutcome {
            if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
            assert_eq!(target.url.path(), "/api/chat/peer/reverse-onion/source-query");
            self.queries.fetch_add(1, Ordering::SeqCst);
            self.deliver_once().await;
            let (status, mut body) = request(self.relay_router.clone(), target.url.path(), body, None).await;
            assert_eq!(status, 200);
            if self.tamper_next_evidence.swap(false, Ordering::SeqCst) {
                *body.last_mut().unwrap() ^= 1;
            }
            SourceTransportOutcome::Response { status, body }
        }
    }

    #[tokio::test]
    async fn actual_private_pull_loop_verifies_ciphertext_and_recovers_without_reexecution() {
        // 0: success; 1: actual custody but lost ACK; 2: altered signed evidence.
        // [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] Preserve direct
        // lifecycle cases; repeat each through signed source HTTP admission.
        // [PHALA-FRESH-RECIPIENT-LOOP 2026-10-07 by Codex] Repeat all
        // six source cases with fresh recipient scheduling and lost replies.
        // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex] Six
        // more cases combine real descriptor renewal after durable Lease
        // acceptance with all source fault/HTTP variants and fresh workers.
        // This is source-only: no test, socket or guest deployment was run.
        for mode in 0..18 {
            let fault = mode % 3;
            let through_api = mode % 6 >= 3;
            let fresh_recipient = mode >= 6;
            let renew_authority = mode >= 12;
            let f = Fixture::new_for_connected_loop();
            let directory = tempfile::Builder::new().prefix("connected-reverse-loop-")
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
                .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
            let relay = f.relay_identity();
            let recipient = f.recipient_identity();
            let (r, p, grant) = f.policy_parts();
            // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
            // Same identities/origin/KEM, newer signed descriptor sequences.
            let renewal = renew_authority.then(|| {
                let (relay, recipient, _) = f.renewed_policy_parts(f.now());
                Arc::new(ConnectedAuthorityRenewal { relay, recipient,
                    previous_grant_issued_at: grant.issued_at() })
            });
            let peers = Arc::new(PeerStore::new());
            peers.pin_private_onion_route_identities(relay.public_key_bytes(), recipient.public_key_bytes()).unwrap();
            // [PHALA-CONNECTED-REVERSE-LOOP 2026-10-07 by Codex] R may
            // import a grant only against its already authenticated pair.
            assert!(peers.import_private_onion_authorization_bundle(grant.clone(), r.clone(), p.clone(),
                relay.public_key_bytes(), now()).is_err());
            for descriptor in [r.clone(), p.clone()] {
                assert!(peers.seed_private_onion_route_descriptor(&relay.public_key_bytes(),
                    &recipient.public_key_bytes(), descriptor, now(), "connected_reverse_loop").unwrap());
            }
            peers.import_private_onion_authorization_bundle(grant.clone(), r.clone(), p.clone(),
                relay.public_key_bytes(), now()).unwrap();
            let origin = crate::api::reverse_onion_origin_commitment(r.descriptor.public_endpoint.as_deref().unwrap()).unwrap();
            let queue = Arc::new(ReverseOnionQueueDb::open(ReverseOnionQueueDbConfig::new(
                // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
                // Physical budget includes rollback pages and metadata headroom.
                directory.path().join("queue.sqlite"), 32 * 1024 * 1024,
                // [PHALA-CONNECTED-QUEUE-HORIZON 2026-10-08 by Codex]
                // Match startup's separate route and execution lifetime caps.
                ReverseOnionQueueLimits::new(4, 8 * 1024 * 1024, 4, 300, 600).unwrap()
                    .with_route_max_secs(600).unwrap(),
            ).unwrap(), now()).unwrap());
            let admission = Arc::new(PrivateBlindRelayAdmission::new(relay.public_key_bytes(), r.clone(), p.clone(),
                grant.clone(), OnionRoutePurpose::BlindVaultPull, vec![f.source_identity().public_key_bytes()],
                queue.clone(), 4, 600, now()).unwrap());
            let api = Arc::new(ReverseOnionApi::new(queue.clone(), relay.clone(), recipient.public_key_bytes(), 4)
                .unwrap().with_live_private_admission(admission.clone(), peers.clone()).unwrap());
            let chat = Arc::new(ChatRelayService::new(ChatRelayConfig {
                enabled: true, db_path: directory.path().join("relay.sqlite").display().to_string(),
                ..ChatRelayConfig::default()
            }, [7; 32]).unwrap());
            // [PHALA-CONNECTED-REVERSE-LOOP 2026-10-07 by Codex] P's
            // real HTTP gate also requires durable replay protection. Keep
            // its store separate from R, matching production role isolation.
            let terminal_chat = Arc::new(ChatRelayService::new(ChatRelayConfig {
                enabled: true, db_path: directory.path().join("terminal-relay.sqlite").display().to_string(),
                ..ChatRelayConfig::default()
            }, [9; 32]).unwrap());
            let (_vault_directory, vault, stored) = crate::api::chat_peer::tests::reverse_onion_private_pull_vault(
                &recipient, now() * 1_000,
            );
            let udp = Arc::new(aeronyx_transport::UdpTransport::bind("127.0.0.1:0").await.unwrap());
            let sessions = Arc::new(SessionManager::new(16, Duration::from_secs(60)));
            let http = Arc::new(reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).build().unwrap());
            let relay_router = build_chat_peer_router_with_private_recipient_admission(
                Some(chat), sessions.clone(), udp.clone(), peers.clone(), relay.clone(), http.clone(),
                None, None, Some(admission.clone()),
            ).merge(build_reverse_onion_router(api.clone()));
            let executions = Arc::new(AtomicUsize::new(0));
            let terminal_entries = executions.clone();
            let terminal_router = build_chat_peer_router_with_private_recipient_admission(
                Some(terminal_chat), sessions, udp, peers.clone(), recipient.clone(), http, Some(vault.clone()), None, None,
            ).layer(axum::middleware::from_fn(move |request: Request<Body>, next: axum::middleware::Next| {
                let entries = terminal_entries.clone();
                async move {
                    entries.fetch_add(1, Ordering::SeqCst);
                    next.run(request).await
                }
            }));
            let (result_ready, result_observed) = tokio::sync::mpsc::channel(1);
            let result_gate = Arc::new(tokio::sync::Semaphore::new(0));
            let adapter = Arc::new(ReverseOnionTerminalAdapter::new(terminal_router.clone(), recipient.clone(),
                relay.public_key_bytes(), Duration::from_secs(30)).unwrap()
                .with_result_persistence_gate(result_ready, result_gate.clone()));
            let recipient_path = directory.path().join("recipient.sqlite");
            let recipient_journal = Arc::new(ReverseOnionRecipientJournal::open(&recipient_path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now()).unwrap());
            drop(recipient_journal);
            let mut worker_config = ReverseOnionConfig::default();
            worker_config.recipient.enabled = true;
            worker_config.recipient.recovery_only = !fresh_recipient;
            worker_config.recipient.poll_interval_ms = 250;
            worker_config.recipient.relay_node_id = hex::encode(relay.public_key_bytes());
            worker_config.recipient.relay_endpoint = r.descriptor.public_endpoint.clone().unwrap();
            worker_config.recipient.state_db_path = recipient_path.display().to_string();
            worker_config.recipient.max_pending_items = 4;
            let transport = Arc::new(LoopTransport {
                relay_router, recipient: recipient.clone(), relay_id: relay.public_key_bytes(), origin,
                journal_path: recipient_path.clone(), worker_config,
                result_ready: tokio::sync::Mutex::new(result_observed),
                result_gate,
                adapter: adapter.clone(), delivered: tokio::sync::Mutex::new(false),
                terminal_router,
                posts: AtomicUsize::new(0), queries: AtomicUsize::new(0), executions,
                lose_ack: fault == 1, tamper_next_evidence: AtomicBool::new(fault == 2),
                fresh_peers: fresh_recipient.then(|| peers.clone()),
                renewal,
            });
            let source_journal = Arc::new(f.open(now()));
            let policy = Arc::new(SourcePinnedRelayPolicy::new(f.source_identity().public_key_bytes(), r.clone(), p, grant, now()).unwrap());
            let config = SourceRuntimeConfig::new(2, Duration::from_secs(30), 600,
                Duration::ZERO, Duration::from_millis(250)).unwrap();
            let runtime = Arc::new(if through_api {
                ReverseOnionSourceRuntime::new_identity_pinned_for_test(source_journal.clone(),
                    f.source_identity(), relay.public_key_bytes(), recipient.public_key_bytes(),
                    r.descriptor.public_endpoint.clone().unwrap(), peers.clone(), transport.clone(), config)
            } else {
                ReverseOnionSourceRuntime::new(source_journal.clone(), f.source_identity(),
                    policy.clone(), transport.clone(), config)
            }.unwrap());
            let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(2));
            lifecycle.install(runtime.clone()).unwrap();
            // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Same
            // dedicated VPN composition as api_runtime, including outer slots.
            let source_router = if through_api {
                build_mpi_router_with_reverse_onion_source(source_mpi_state(&f.source_identity()),
                    Router::new(), lifecycle.clone())
            } else { build_reverse_onion_source_router(lifecycle.clone()) };
            let nonce = [73; 16];
            let input = f.source_api_request(nonce, now());
            let route = if through_api {
                reverse_onion_source_route_id(&f.source_identity().public_key_bytes(), &nonce, &input.pull).unwrap()
            } else { f.route() };
            let remote_source = fault == 2;
            let page = if through_api {
                let wrong_wallet = ReverseOnionSourcePullRequestV1::new_signed(&recipient, nonce,
                    now(), input.pull.clone(), &f.policy_parts().2).unwrap();
                let (status, _) = source_request(source_router.clone(), &wrong_wallet,
                    &f.source_identity(), remote_source).await;
                assert_eq!(status, 401);
                let mut tampered = f.source_api_request(nonce, now());
                tampered.signature_b64 = STANDARD.encode([0; 64]);
                let (status, _) = source_request(source_router.clone(), &tampered,
                    &f.source_identity(), remote_source).await;
                assert_eq!(status, 401);
                let stale_unknown = f.source_api_request([74; 16], now().saturating_sub(3_600));
                let unknown_route = reverse_onion_source_route_id(&f.source_identity().public_key_bytes(),
                    &[74; 16], &stale_unknown.pull).unwrap();
                assert_ne!(unknown_route, route);
                let (status, _) = source_request(source_router.clone(), &stale_unknown,
                    &f.source_identity(), remote_source).await;
                assert_eq!(status, 400, "a stale signature cannot bootstrap a new route");
                assert_eq!(source_journal.lookup_phase(unknown_route, now()).unwrap(), None);
                assert_eq!(source_journal.lookup_phase(route, now()).unwrap(), None);
                assert_eq!(transport.posts.load(Ordering::SeqCst), 0);
                assert_eq!(transport.queries.load(Ordering::SeqCst), 0);
                assert_eq!(transport.executions.load(Ordering::SeqCst), 0);
                let (status, mut body) = source_request(source_router.clone(), &input,
                    &f.source_identity(), remote_source).await;
                if fault == 0 { assert_eq!(status, 200); } else {
                    assert_eq!(status, 409);
                    assert_eq!(source_journal.lookup_phase(route, now()).unwrap(), Some(SourcePhase::DispatchAmbiguous));
                    assert_eq!(transport.executions.load(Ordering::SeqCst), usize::from(fault == 2));
                    // A stale but authentic request only resumes this exact
                    // owner+nonce+Pull; refreshing time cannot create a route.
                    let retry = f.source_api_request(nonce, now().saturating_sub(3_600));
                    assert_eq!(reverse_onion_source_route_id(&f.source_identity().public_key_bytes(),
                        &nonce, &retry.pull).unwrap(), route);
                    let recovered = source_request(source_router.clone(), &retry,
                        &f.source_identity(), remote_source).await;
                    assert_eq!(recovered.0, 200);
                    body = recovered.1;
                }
                serde_json::from_slice::<ReverseOnionSourcePullResponseV1>(&body).unwrap()
                    .decode_completed().unwrap()
            } else {
                let (request, expected, session, terminal) = f.runtime_admission_parts();
                let result = lifecycle.dispatch(request, expected, session, f.now() + 600, terminal, now(), policy).await;
                let completed = if fault == 0 { result.unwrap() } else {
                    assert_eq!(result.err(), Some(SourceRuntimeError::Ambiguous));
                    assert_eq!(source_journal.lookup_phase(route, now()).unwrap(), Some(SourcePhase::DispatchAmbiguous));
                    assert_eq!(transport.executions.load(Ordering::SeqCst), usize::from(fault == 2));
                    lifecycle.resume(route, now()).await.unwrap()
                };
                completed.response().clone()
            };
            page.validate_and_verify(&IdentityPublicKey::from_bytes(&recipient.public_key_bytes()).unwrap()).unwrap();
            assert_eq!(page.lease_id, stored.lease_id);
            assert_eq!(page.node_id, recipient.public_key_bytes());
            assert_eq!(page.objects.len(), 1);
            assert_eq!(page.objects[0].object_id, stored.object_id);
            assert_eq!(page.objects[0].ciphertext, stored.ciphertext);
            assert_eq!(source_journal.lookup_phase(route, now()).unwrap(), Some(SourcePhase::Verified));
            assert_eq!(transport.posts.load(Ordering::SeqCst), 1);
            // [PHALA-CONNECTED-AUTHORITY-RENEWAL 2026-10-08 by Codex]
            // The complete real result path must not rewrite source custody
            // to today's authority. Compare authenticated journal metadata
            // to both signed pairs, not only a final success status.
            if let Some(renewal) = &transport.renewal {
                let metadata = source_journal.recover_metadata(None, 64, now()).unwrap();
                let [item] = metadata.items.as_slice() else { panic!("one retained source route"); };
                let (original_relay, original_recipient, _) = f.policy_parts();
                let commitment = |descriptor: &SignedNodeDescriptor|
                    DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor).unwrap().hash();
                assert_eq!(item.route(), route);
                assert_eq!(item.relay_descriptor_commitment(), commitment(&original_relay));
                assert_eq!(item.recipient_descriptor_commitment(), commitment(&original_recipient));
                assert_ne!(item.relay_descriptor_commitment(), commitment(&renewal.relay));
                assert_ne!(item.recipient_descriptor_commitment(), commitment(&renewal.recipient));
            }
            assert_eq!(transport.executions.load(Ordering::SeqCst), 1);
            let query_count = transport.queries.load(Ordering::SeqCst);
            assert!(query_count >= 3);
            if through_api {
                let retry = f.source_api_request(nonce, now());
                // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex]
                // Discard an actual completed HTTP reply without reading it.
                // The cached result remains Verified and retries perform no
                // POST, evidence query or execution, under Local and Remote MPI.
                let lost = source_http_response(source_router.clone(), &retry,
                    &f.source_identity(), remote_source).await;
                assert_eq!(lost.status(), axum::http::StatusCode::OK);
                let admission = lifecycle.request_admission();
                let spare = admission.try_acquire().unwrap();
                assert!(admission.try_acquire().is_none(), "unread successful response retains its HTTP slot");
                drop(lost);
                assert!(admission.try_acquire().is_some());
                drop(spare);
                assert_eq!(source_journal.lookup_phase(route, now()).unwrap(), Some(SourcePhase::Verified));
                let (status, body) = source_request(source_router.clone(), &retry,
                    &f.source_identity(), remote_source).await;
                assert_eq!(status, 200);
                let cached = serde_json::from_slice::<ReverseOnionSourcePullResponseV1>(&body).unwrap()
                    .decode_completed().unwrap();
                assert_eq!(cached, page);
            } else {
                let cached = lifecycle.resume(route, now()).await.unwrap();
                assert_eq!(cached.response(), &page);
            }
            assert_eq!(transport.queries.load(Ordering::SeqCst), query_count);
            assert_eq!(transport.posts.load(Ordering::SeqCst), 1);
            assert_eq!(transport.executions.load(Ordering::SeqCst), 1);
            let expected_page = page.clone();
            lifecycle.shutdown_and_drain().await.unwrap();
            adapter.shutdown_and_drain().await.unwrap();
            api.shutdown_and_drain().await;
            drop(source_router);
            drop(lifecycle);
            drop(runtime);
            drop(transport);
            drop(source_journal);
            let reopened_source = f.open_existing(now());
            assert_eq!(reopened_source.read_verified(route, now()).unwrap(), expected_page);
            let reopened_recipient = ReverseOnionRecipientJournal::open_existing(&recipient_path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now()).unwrap();
            assert!(matches!(reopened_recipient.resume(None, 64, now()).unwrap().items.as_slice(),
                [RecipientRecovery::Result { .. }]));
            assert_eq!(vault.status(now() * 1_000).unwrap().live_objects, 1);
            assert!(!adapter.has_failed());
        }
    }
}

// [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex] Authored, not run:
// use the real journal transaction, runtime stop flag, lifecycle and required
// supervisor. Stopping between Prepared/Armed is not execution or owner failure;
// a clock regression at the same boundary must still publish a sticky fault.
#[cfg(unix)]
#[tokio::test]
async fn source_post_lock_stop_preserves_prepared_but_clock_regression_fails_owner() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use std::time::Duration;
    use axum::body::Bytes;
    use crate::api::PinnedPeerHttpTarget;
    use crate::services::reverse_onion_source::{SourceJournalError, SourcePhase, tests::Fixture};
    use super::super::{runtime_supervision, reverse_onion_source_runtime::{
        source_admission_now, ReverseOnionSourceLifecycle, ReverseOnionSourceRuntime,
        SourcePinnedRelayPolicy, SourceRuntimeConfig, SourceTransport, SourceTransportOutcome,
        SourceSendAdmission,
    }};
    struct NoSendTransport(AtomicUsize);
    #[async_trait::async_trait]
    impl SourceTransport for NoSendTransport {
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] This fixture must
        // never be reached before durable arming, regardless of admission.
        async fn post(&self, _: PinnedPeerHttpTarget, _: Bytes, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            self.0.fetch_add(1, Ordering::SeqCst);
            panic!("unarmed source cannot post");
        }
        async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            panic!("Prepared is not relay custody");
        }
    }
    for rollback in [false, true] {
        let f = Fixture::new_for_runtime();
        let journal = Arc::new(f.open(f.now()));
        f.prepare(&journal, f.now());
        let (relay, recipient, authorization) = f.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy::new(
            f.source_identity().public_key_bytes(), relay, recipient, authorization, f.now(),
        ).unwrap());
        let transport = Arc::new(NoSendTransport(AtomicUsize::new(0)));
        let runtime = Arc::new(ReverseOnionSourceRuntime::new(
            journal.clone(), f.source_identity(), policy, transport.clone(),
            SourceRuntimeConfig::new(1, Duration::from_secs(1), 600, Duration::ZERO,
                Duration::from_millis(250)).unwrap(),
        ).unwrap());
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        lifecycle.install(runtime.clone()).unwrap();
        let stopped = journal.intake_stop_flag();
        let result = journal.arm_at(f.route(), || source_admission_now(&stopped, f.now() + 1, || {
            lifecycle.request_stop();
            Ok(if rollback { f.now() } else { f.now() + 1 })
        }));
        assert_eq!(result.err(), Some(if rollback { SourceJournalError::ClockRollback }
            else { SourceJournalError::Rejected }));
        assert!(lifecycle.request_admission().is_stopped());
        assert_eq!(lifecycle.has_failed(), rollback);
        assert_eq!(transport.0.load(Ordering::SeqCst), 0);
        let failure = runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle));
        assert_eq!(failure.is_some(), rollback);
        lifecycle.shutdown_and_drain().await.unwrap();
        drop(lifecycle);
        drop(runtime);
        drop(journal);
        let reopened = f.open(f.now() + 2);
        assert_eq!(reopened.lookup_phase(f.route(), f.now() + 2).unwrap(), Some(SourcePhase::Prepared));
    }
}

// [PHALA-SOURCE-JOURNAL-FAULT 2026-10-07 by Codex] Authored, not run:
// cancel a real blocking reply-open waiter before its failed commit. The
// journal itself must close bound intake and notify required-source supervision.
#[cfg(unix)]
#[tokio::test]
async fn cancelled_source_db_waiter_cannot_hide_late_result_open_failure() {
    use std::sync::{Arc, atomic::Ordering};
    use std::time::Duration;
    use axum::body::Bytes;
    use crate::api::PinnedPeerHttpTarget;
    use crate::services::reverse_onion_source::tests::Fixture;
    use super::super::{runtime_supervision, reverse_onion_source_runtime::{
        source_db, ReverseOnionSourceLifecycle, ReverseOnionSourceRuntime,
        SourcePinnedRelayPolicy, SourceRuntimeConfig, SourceRuntimeError,
        SourceTransport, SourceTransportOutcome,
        SourceSendAdmission,
    }};
    struct NeverTransport;
    #[async_trait::async_trait]
    impl SourceTransport for NeverTransport {
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Durable result
        // recovery still has no permitted transport path.
        async fn post(&self, _: PinnedPeerHttpTarget, _: Bytes, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            panic!("durable result recovery must not post");
        }
        async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            panic!("durable result recovery must not query");
        }
    }
    // [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] Authored, not run:
    // cancellation/shutdown cannot hide a post-crypto clock regression either.
    for clock_rollback in [false, true] {
        let fixture = Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open(fixture.now()));
        fixture.ready(&journal);
        let (relay, recipient, authorization) = fixture.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy::new(
            fixture.source_identity().public_key_bytes(), relay, recipient, authorization, fixture.now(),
        ).unwrap());
        let runtime = Arc::new(ReverseOnionSourceRuntime::new(
            journal.clone(), fixture.source_identity(), policy, Arc::new(NeverTransport),
            SourceRuntimeConfig::new(1, Duration::from_secs(1), 600, Duration::ZERO,
                Duration::from_millis(250)).unwrap(),
        ).unwrap());
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        lifecycle.install(runtime.clone()).unwrap();
        let signal = journal.failure_signal();
        let mut failure = signal.subscribe();
        let admission = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = Arc::new(admission.clone().acquire_owned().await.unwrap());
        let lane = journal.blocking_operation_lane();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let owned_journal = journal.clone();
        let route = fixture.route();
        let caller = tokio::spawn(async move {
            source_db(&owned_journal, permit, 0, move |journal, now| {
                let _ = entered_tx.send(());
                release_rx.recv().map_err(|_| SourceRuntimeError::Unavailable)?;
                let mut calls = 0;
                journal.open_result_at(route, now, || {
                    calls += 1;
                    Ok(if clock_rollback && calls == 2 { now - 1 } else { now })
                }).map_err(SourceRuntimeError::from)
            }).await
        });
        entered_rx.await.unwrap();
        caller.abort();
        assert!(caller.await.err().expect("DB caller must be cancelled").is_cancelled());
        assert_eq!(admission.available_permits(), 0);
        assert_eq!(lane.available_permits(), 0);
        // [PHALA-SHUTDOWN-FAULT-RECONCILIATION 2026-10-07 by Codex] A
        // shutdown winner precedes this actual blocking completion fault.
        lifecycle.request_stop();
        assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
            None, Some(&lifecycle), None,
        ), None);
        if !clock_rollback { journal.fail_next_commit_fence(); }
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), failure.changed()).await.unwrap().unwrap();
        assert!(*failure.borrow());
        assert!(journal.intake_stop_flag().load(Ordering::SeqCst));
        assert!(lifecycle.request_admission().is_stopped());
        let expected = runtime_supervision::reverse_onion_source_failed();
        assert_eq!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)), Some(expected.clone()));
        assert_eq!(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)).await, expected.clone());
        assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
            None, Some(&lifecycle), None,
        ), Some(expected.clone()));
        // Await the original blocking owner, not merely its earlier failure signal.
        let completed = lane.clone().acquire_owned().await.unwrap();
        let released = admission.clone().acquire_owned().await.unwrap();
        drop(released);
        drop(completed);
        assert_eq!(admission.available_permits(), 1);
        assert_eq!(runtime.resume(fixture.route(), fixture.now()).await.err(), Some(SourceRuntimeError::Stopped));
        lifecycle.shutdown_and_drain().await.unwrap();
        assert!(lifecycle.has_failed());
        assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
            None, Some(&lifecycle), None,
        ), Some(expected));
    }
}

// [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
// not run: real adapter entry/unwind after caller cancellation reaches the
// same health projection used by the worker. This is no business-success or
// deployed Phala proof; the synthetic router intentionally never replies.
#[cfg(unix)]
#[tokio::test]
async fn cancelled_terminal_caller_cannot_hide_owner_failure_from_worker() {
    use std::sync::{Arc, Mutex, atomic::Ordering};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::blind_vault::{
        encode_blind_vault_frame, BlindVaultFrame, BlindVaultPullRequest,
        BLIND_VAULT_PROTOCOL_VERSION,
    };
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
    use aeronyx_core::protocol::onion::reverse_delivery::ReverseOnionFrameV1;
    use aeronyx_core::protocol::onion_reply::{
        encode_onion_reply_request, OnionReplySession, ONION_REPLY_RESPONSE_SIZE_CLASSES,
    };
    use crate::api::reverse_onion_terminal::{ReverseOnionTerminalAdapter, ReverseOnionTerminalError};
    use crate::services::reverse_onion_recipient::{
        RecipientJournalLimits, RecipientRecovery, ReverseOnionRecipientJournal,
    };
    async fn unwind_after_entry(
        entered: tokio::sync::mpsc::Sender<()>,
        release: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
        rollback: bool,
    ) -> axum::Json<serde_json::Value> {
        let released = release.lock().unwrap().take().unwrap();
        entered.try_send(()).unwrap();
        released.await.unwrap();
        // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] A
        // successful HTTP body reaches the real post-entry clock boundary;
        // the synthetic body is intentionally not an execution-success proof.
        if rollback { return axum::Json(serde_json::json!({})); }
        panic!("synthetic terminal unwind after caller cancellation");
    }
    // The rollback case passed the old preparation-only comparison, but must
    // still wake supervision after the original dispatch caller is cancelled.
    for rollback in [false, true] {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let relay = IdentityKeyPair::from_bytes(&[81; 32]).unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[82; 32]).unwrap());
        let route = [81; 16];
        let payload = encode_blind_vault_frame(&BlindVaultFrame::PullRequest(BlindVaultPullRequest {
            version: BLIND_VAULT_PROTOCOL_VERSION, lease_id: [81; 32], read_capability: [82; 32],
            continuation_cursor: Vec::new(), limit: 1,
        })).unwrap();
        let (reply, _session) = OnionReplySession::prepare_source_sealed(
            route, local.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0], payload,
        ).unwrap();
        let envelope = build_onion_envelope(&[OnionHop {
            node_id: local.public_key_bytes(), kem_pub: crate::services::onion_keys::current_public_key(),
        }], &encode_onion_reply_request(&reply).unwrap(), route, 1, now, &relay).unwrap();
        let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), route, now, now + 30, &local).unwrap();
        let lease = ReverseOnionFrameV1::lease(&claim, &envelope, route, now + 600, now, &relay).unwrap();
        let directory = tempfile::Builder::new().prefix("phala-terminal-fault-")
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
            .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let journal = Arc::new(ReverseOnionRecipientJournal::open(
            &directory.path().join("recipient.sqlite"), relay.public_key_bytes(), local.public_key_bytes(),
            RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, now,
        ).unwrap());
        journal.prepare_poll(&claim, [9; 32], now).unwrap();
        journal.record_lease(claim.claim_id(), &lease, now + 600, now).unwrap();
        let armed = journal.arm(claim.claim_id(), now).unwrap();
        let clock_floor = armed.armed_at;
        let (entered, mut entries) = tokio::sync::mpsc::channel(1);
        let (release, released) = tokio::sync::oneshot::channel();
        let released = Arc::new(Mutex::new(Some(released)));
        let router = axum::Router::new().route("/api/chat/peer/blind-relay", axum::routing::post(
            move || unwind_after_entry(entered.clone(), released.clone(), rollback),
        ));
        let adapter = ReverseOnionTerminalAdapter::new(
            router, local, relay.public_key_bytes(), Duration::from_secs(30),
        ).unwrap();
        let adapter = Arc::new(if rollback {
            adapter.with_clock_samples(&[Ok(clock_floor), Ok(clock_floor + 20), Ok(clock_floor + 10)])
        } else { adapter });
        assert_eq!(reverse_onion_runtime::terminal_owner_health(&adapter), Ok(()));
        let mut observer = Box::pin(adapter.wait_for_failure());
        assert!(futures::poll!(observer.as_mut()).is_pending());
        let caller_adapter = adapter.clone();
        let caller_journal = journal.clone();
        let caller = tokio::spawn(async move { caller_adapter.dispatch(armed, caller_journal).await });
        tokio::time::timeout(Duration::from_secs(1), entries.recv()).await.unwrap().unwrap();
        caller.abort();
        assert!(caller.await.err().expect("caller must be cancelled").is_cancelled());
        drop(observer);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), adapter.wait_for_failure()).await.unwrap();
        assert!(adapter.intake_stop_flag().load(Ordering::SeqCst));
        assert_eq!(reverse_onion_runtime::terminal_owner_health(&adapter),
            Err(reverse_onion_runtime::RecipientWorkerError::Unavailable));
        assert_eq!(adapter.shutdown_and_drain().await, Err(ReverseOnionTerminalError::Ambiguous));
        assert_eq!(adapter.shutdown_and_drain().await, Err(ReverseOnionTerminalError::Ambiguous));
        assert!(matches!(journal.resume(None, 64, clock_floor).unwrap().items.as_slice(),
            [RecipientRecovery::Ambiguous { .. }]));
        assert!(journal.arm(claim.claim_id(), clock_floor).is_err());
    }
}

// [PHALA-RECIPIENT-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored,
// not run: graceful intake closure must not become a worker owner fault.
#[tokio::test]
async fn normal_terminal_stop_is_not_a_recipient_owner_failure() {
    use std::sync::Arc;
    use aeronyx_core::crypto::IdentityKeyPair;
    use crate::api::reverse_onion_terminal::ReverseOnionTerminalAdapter;
    let local = Arc::new(IdentityKeyPair::from_bytes(&[83; 32]).unwrap());
    let relay = IdentityKeyPair::from_bytes(&[84; 32]).unwrap();
    let adapter = ReverseOnionTerminalAdapter::new(
        axum::Router::new(), local, relay.public_key_bytes(), std::time::Duration::from_secs(1),
    ).unwrap();
    adapter.request_stop();
    adapter.shutdown_and_drain().await.unwrap();
    assert_eq!(reverse_onion_runtime::terminal_owner_health(&adapter), Ok(()));
    let mut observer = Box::pin(adapter.wait_for_failure());
    assert!(futures::poll!(observer.as_mut()).is_pending());
}

// [PHALA-RECIPIENT-PREFLIGHT-FAULT 2026-10-07 by Codex] Authored, not
// run: real existing-only worker startup publishes a sticky local failure,
// while an empty audited recovery journal may stop normally without a fault.
// No PeerStore route exists, so neither case enters DNS/HTTP or inference.
#[cfg(unix)]
#[tokio::test]
async fn recipient_worker_startup_fault_outranks_cancellation_but_normal_stop_does_not() {
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use aeronyx_core::crypto::IdentityKeyPair;
    use crate::api::reverse_onion_terminal::ReverseOnionTerminalAdapter;
    use crate::config_reverse_onion::ReverseOnionConfig;
    use crate::services::{peer_store::PeerStore, reverse_onion_recipient::{
        ReverseOnionRecipientJournal, RecipientJournalLimits,
    }};
    use super::super::{reverse_onion_runtime::{
        RecipientServerLifecycle, ReverseOnionRecipientWorker, RecipientWorkerError,
    }, runtime_supervision::{self, PreReadyRuntimeDecision as D}};
    for missing_journal in [false, true] {
        let directory = tempfile::Builder::new().prefix("phala-recipient-owner-fault-")
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
            .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
        let path = directory.path().join("recipient.sqlite");
        let relay = IdentityKeyPair::from_bytes(&[85; 32]).unwrap();
        let recipient = Arc::new(IdentityKeyPair::from_bytes(&[86; 32]).unwrap());
        if !missing_journal {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            let journal = ReverseOnionRecipientJournal::open(&path,
                relay.public_key_bytes(), recipient.public_key_bytes(),
                RecipientJournalLimits { max_entries: 4, max_bytes: 64 * 1024 * 1024 }, now).unwrap();
            drop(journal);
        }
        let mut config = ReverseOnionConfig::default();
        config.recipient.enabled = true;
        config.recipient.recovery_only = true;
        config.recipient.relay_node_id = hex::encode(relay.public_key_bytes());
        config.recipient.relay_endpoint = "https://1.1.1.1".into();
        config.recipient.state_db_path = path.display().to_string();
        config.recipient.max_pending_items = 4;
        let adapter = Arc::new(ReverseOnionTerminalAdapter::new(
            axum::Router::new(), recipient.clone(), relay.public_key_bytes(), Duration::from_secs(1),
        ).unwrap());
        let worker = Arc::new(ReverseOnionRecipientWorker::start(&config, recipient,
            Arc::new(PeerStore::new()), adapter.clone()).unwrap());
        let owner = RecipientServerLifecycle::new();
        owner.active.store(true, Ordering::SeqCst);
        owner.install(worker.clone()).unwrap();
        let ready = tokio::time::timeout(Duration::from_secs(5), owner.verify_ready()).await.unwrap();
        let (_critical_tx, mut critical_rx) = tokio::sync::mpsc::channel(1);
        let shutdown = AtomicBool::new(false);
        if missing_journal {
            assert_eq!(ready, Err(RecipientWorkerError::Unavailable));
            assert!(owner.has_failed());
            assert!(adapter.intake_stop_flag().load(Ordering::SeqCst));
            assert!(!path.exists());
        } else {
            assert_eq!(ready, Ok(()));
            assert!(!owner.has_failed());
            assert_eq!(runtime_supervision::pre_ready_runtime_decision(
                &mut critical_rx, &shutdown, None, Some(&owner), false,
            ), D::Ready);
        }
        // [PHALA-RETAINED-UNWIND-DRAIN 2026-10-07 by Codex] A caught
        // server unwind still uses the actual installed worker's drain path.
        let dependencies = runtime_supervision::ReverseRuntimeDependencies::default();
        let unwound = runtime_supervision::catch_reverse_runtime_unwind(async {
            panic!("synthetic recipient server unwind payload");
        }).await.unwrap_err();
        assert!(matches!(unwound, crate::error::ServerError::RuntimeFailed { task, reason }
            if task == "reverse-owned-server" && reason == "required reverse server owner unwound"));
        owner.request_stop();
        shutdown.store(true, Ordering::SeqCst);
        let expected = if missing_journal { D::Failed(runtime_supervision::reverse_onion_recipient_worker_exited()) }
            else { D::Stopped };
        assert_eq!(runtime_supervision::pre_ready_runtime_decision(
            &mut critical_rx, &shutdown, None, Some(&owner), missing_journal,
        ), expected);
        assert_eq!(tokio::time::timeout(Duration::from_secs(5),
            owner.wait_for_unexpected_worker_failure_or_exit()).await.unwrap(), missing_journal);
        let drain = if missing_journal { Err(RecipientWorkerError::Unavailable) } else { Ok(()) };
        assert_eq!(dependencies.drain_reverse_owners(None, Some(&owner)).await.is_err(), missing_journal);
        assert_eq!(owner.drain().await, drain);
        assert_eq!(owner.drain().await, drain);
        assert_eq!(owner.has_failed(), missing_journal);
        // [PHALA-SHUTDOWN-FAULT-RECONCILIATION 2026-10-07 by Codex]
        // Retained worker faults survive drain; normal cancellation is inert.
        assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
            None, None, Some(&owner),
        ), missing_journal.then(runtime_supervision::reverse_onion_recipient_worker_exited));
        let selected = runtime_supervision::CriticalRuntimeFailure {
            task: "synthetic-required-listener", reason: "synthetic listener failed".into(),
        };
        assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(
            Some(selected.clone()), None, Some(&owner),
        ), Some(selected));
        assert_eq!(runtime_supervision::pre_ready_runtime_decision(
            &mut critical_rx, &shutdown, None, Some(&owner), missing_journal,
        ), expected);
    }
}

// [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored, not
// run: absent optional source is inert; an enabled but uninstalled owner
// cannot satisfy readiness or leave the post-READY supervisor asleep.
#[tokio::test]
async fn source_supervision_distinguishes_disabled_from_missing_owner() {
    use super::super::{runtime_supervision, reverse_onion_source_runtime::ReverseOnionSourceLifecycle};
    // [PHALA-SHUTDOWN-FAULT-RECONCILIATION 2026-10-07 by Codex]
    // An absent optional role remains inert in shutdown reconciliation too.
    assert_eq!(runtime_supervision::reconcile_reverse_onion_runtime_failure(None, None, None), None);
    assert!(runtime_supervision::pre_ready_reverse_onion_source_failure(None).is_none());
    let mut disabled = Box::pin(runtime_supervision::wait_for_reverse_onion_source_failure(None));
    assert!(futures::poll!(disabled.as_mut()).is_pending());
    let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
    let expected = runtime_supervision::reverse_onion_source_failed();
    assert_eq!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)), Some(expected.clone()));
    assert_eq!(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)).await, expected);
    assert!(lifecycle.request_admission().is_stopped());
    lifecycle.shutdown_and_drain().await.unwrap();
}

// [PHALA-SOURCE-POST-OBSERVATION 2026-10-07 by Codex] Authored, not run:
// a signed custody ACK followed by missing result evidence may only query on
// the next call. A failed observation fence stops before queries. This uses
// production dispatch/resume, not a separate phase predicate.
#[cfg(unix)]
#[tokio::test]
async fn custody_observation_prevents_repost_and_fails_closed_on_fence_failure() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use axum::body::Bytes;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::BlindRelaySuccessReceipt;
    use crate::api::{PinnedPeerHttpTarget, chat_peer::{PeerBlindRelayRequest, PeerBlindRelayResponse}};
    use super::super::reverse_onion_source_runtime::{
        ReverseOnionSourceLifecycle, ReverseOnionSourceRuntime, SourcePinnedRelayPolicy,
        SourceRuntimeConfig, SourceRuntimeError, SourceTransport, SourceTransportOutcome,
    };
    use crate::services::reverse_onion_source::{ReverseOnionSourceJournal, SourcePhase, tests::Fixture};
    struct CustodyTransport {
        relay: Arc<IdentityKeyPair>,
        posts: AtomicUsize,
        queries: AtomicUsize,
        fail_observation: Option<Arc<ReverseOnionSourceJournal>>,
    }
    #[async_trait::async_trait]
    impl SourceTransport for CustodyTransport {
        // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Count custody
        // only after the actual source owner/deadline gate accepts this call.
        async fn post(&self, _: PinnedPeerHttpTarget, body: Bytes, _: Bytes,
            admission: super::super::reverse_onion_source_runtime::SourceSendAdmission) -> SourceTransportOutcome {
            if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
            self.posts.fetch_add(1, Ordering::SeqCst);
            let request: PeerBlindRelayRequest = serde_json::from_slice(&body).unwrap();
            let ttl = request.envelope.ttl - 1;
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            let receipt = BlindRelaySuccessReceipt::forwarded(
                &request.envelope, ttl, None, None, None, now, &self.relay,
            );
            let response = PeerBlindRelayResponse {
                accepted: true, terminal: false, forwarded: true, ttl_remaining: ttl,
                reason: None, delivery_receipt: None, success_receipt: Some(receipt),
                failure_receipt: None, opaque_terminal_response_b64: None,
            };
            let body = serde_json::to_vec(&response).unwrap();
            if let Some(journal) = &self.fail_observation { journal.fail_next_commit_fence(); }
            SourceTransportOutcome::Response { status: 200, body }
        }
        async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
            admission: super::super::reverse_onion_source_runtime::SourceSendAdmission) -> SourceTransportOutcome {
            if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
            self.queries.fetch_add(1, Ordering::SeqCst);
            // No signed result is available while evidence service is busy.
            SourceTransportOutcome::Response { status: 503, body: Vec::new() }
        }
    }
    // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] A full quota
    // must still admit the exact accepted Prepared route and keep supervision
    // asleep while its custody/result recovery remains Pending.
    for (fail_observation, full_quota) in [(false, false), (true, false), (false, true)] {
        let fixture = Fixture::new_for_runtime();
        let journal = Arc::new(if full_quota { fixture.open_one_slot(fixture.now()) }
            else { fixture.open(fixture.now()) });
        if full_quota { fixture.prepare(&journal, fixture.now()); }
        let (relay, recipient, authorization) = fixture.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy::new(
            fixture.source_identity().public_key_bytes(), relay, recipient, authorization, fixture.now(),
        ).unwrap());
        let transport = Arc::new(CustodyTransport {
            relay: fixture.relay_identity(), posts: AtomicUsize::new(0), queries: AtomicUsize::new(0),
            fail_observation: fail_observation.then(|| journal.clone()),
        });
        let runtime = Arc::new(ReverseOnionSourceRuntime::new(
            journal.clone(), fixture.source_identity(), policy.clone(), transport.clone(),
            SourceRuntimeConfig::new(2, std::time::Duration::from_secs(1), 600,
                std::time::Duration::ZERO, std::time::Duration::from_millis(250)).unwrap(),
        ).unwrap());
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(2);
        lifecycle.install(runtime.clone()).unwrap();
        // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Register
        // before the real transport/fence failure, then cancel and resubscribe.
        use super::super::runtime_supervision;
        // [PHALA-READY-PUBLICATION 2026-10-07 by Codex] Authored, not run:
        // use the installed source's real stop/fault signals at publication.
        let (_failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
        let shutdown = std::sync::atomic::AtomicBool::new(false);
        use runtime_supervision::PreReadyRuntimeDecision as ReadyDecision;
        assert_eq!(lifecycle.verify_ready_now(), Ok(()));
        assert_eq!(runtime_supervision::pre_ready_runtime_decision(
            &mut failure_rx, &shutdown, Some(&lifecycle), None, false,
        ), ReadyDecision::Ready);
        assert!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)).is_none());
        let mut observer = Box::pin(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)));
        assert!(futures::poll!(observer.as_mut()).is_pending());
        let (request, expected, session, terminal) = fixture.runtime_admission_parts();
        let result = lifecycle.dispatch(request, expected, session, fixture.now() + 600,
            terminal, fixture.now(), policy).await;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        if fail_observation {
            assert_eq!(result.err(), Some(SourceRuntimeError::Unavailable));
            assert_eq!(runtime.resume(fixture.route(), now).await.err(), Some(SourceRuntimeError::Stopped));
            assert_eq!(transport.queries.load(Ordering::SeqCst), 0);
            assert!(lifecycle.request_admission().is_stopped());
            let expected = runtime_supervision::reverse_onion_source_failed();
            assert_eq!(lifecycle.verify_ready_now(), Err(SourceRuntimeError::Unavailable));
            assert_eq!(runtime_supervision::pre_ready_runtime_decision(
                &mut failure_rx, &shutdown, Some(&lifecycle), None, false,
            ), ReadyDecision::Failed(expected.clone()));
            assert_eq!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)), Some(expected.clone()));
            drop(observer);
            let mut replacement = Box::pin(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)));
            assert_eq!(futures::poll!(replacement.as_mut()), std::task::Poll::Ready(expected.clone()));
            assert_eq!(runtime_supervision::wait_for_reverse_onion_source_failure(Some(&lifecycle)).await, expected);
        } else {
            assert_eq!(result.err(), Some(SourceRuntimeError::Pending));
            assert_eq!(journal.lookup_phase(fixture.route(), now).unwrap(), Some(SourcePhase::DispatchAmbiguous));
            assert_eq!(lifecycle.resume(fixture.route(), now).await.err(), Some(SourceRuntimeError::Pending));
            assert_eq!(transport.queries.load(Ordering::SeqCst), 2);
            assert!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)).is_none());
            lifecycle.request_stop();
            assert_eq!(lifecycle.verify_ready_now(), Err(SourceRuntimeError::Stopped));
            assert_eq!(runtime_supervision::pre_ready_runtime_decision(
                &mut failure_rx, &shutdown, Some(&lifecycle), None, false,
            ), ReadyDecision::Stopped);
            assert!(futures::poll!(observer.as_mut()).is_pending());
            drop(observer);
            assert!(runtime_supervision::pre_ready_reverse_onion_source_failure(Some(&lifecycle)).is_none());
        }
        assert_eq!(transport.posts.load(Ordering::SeqCst), 1);
        lifecycle.shutdown_and_drain().await.unwrap();
        // Shutdown must not erase the durable source failure classification.
        shutdown.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(runtime_supervision::pre_ready_runtime_decision(
            &mut failure_rx, &shutdown, Some(&lifecycle), None, false,
        ), if fail_observation { ReadyDecision::Failed(runtime_supervision::reverse_onion_source_failed()) }
            else { ReadyDecision::Stopped });
        // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] Full-slot
        // continuation retains authenticated custody across owner teardown.
        if full_quota {
            drop(runtime);
            drop(journal);
            let reopened = fixture.open_existing(now);
            assert_eq!(reopened.lookup_phase(fixture.route(), now).unwrap(), Some(SourcePhase::DispatchAmbiguous));
            assert!(!*reopened.failure_signal().borrow());
            assert!(!reopened.intake_stop_flag().load(Ordering::SeqCst));
        }
    }
}

// [PHALA-READY-PUBLICATION 2026-10-07 by Codex] Authored, not run.
// These are final-gate decisions only, not a full server-startup acceptance.
#[test]
fn final_ready_gate_rechecks_supervision_shutdown_and_required_ownership() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use super::super::{runtime_supervision::{self, PreReadyRuntimeDecision as D},
        reverse_onion_source_runtime::ReverseOnionSourceLifecycle};

    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let shutdown = AtomicBool::new(false);
    // A disabled role has no required owner or readiness wait to satisfy.
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, None, None, false,
    ), D::Ready);
    shutdown.store(true, Ordering::SeqCst);
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, None, None, false,
    ), D::Stopped);
    let failure = runtime_supervision::reverse_onion_recipient_worker_exited();
    failure_tx.try_send(failure.clone()).unwrap();
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, None, None, false,
    ), D::Failed(failure));
    shutdown.store(false, Ordering::SeqCst);
    let source = ReverseOnionSourceLifecycle::new();
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, Some(&source), None, false,
    ), D::Failed(runtime_supervision::reverse_onion_source_failed()));
    let recipient = reverse_onion_runtime::RecipientServerLifecycle::new();
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, None, Some(&recipient), false,
    ), D::Failed(runtime_supervision::reverse_onion_recipient_worker_exited()));
    recipient.request_stop();
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, None, Some(&recipient), true,
    ), D::Stopped);
    drop(failure_tx);
    assert_eq!(runtime_supervision::pre_ready_runtime_decision(
        &mut failure_rx, &shutdown, None, Some(&recipient), true,
    ), D::Failed(runtime_supervision::required_runtime_supervisor_channel_closed()));
}

// [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Authored, not run:
// drive the real source lifecycle/journal and inject stop at transport entry.
// This covers first-attempt versus crash-window persistence, not live HTTP.
#[cfg(unix)]
#[tokio::test]
async fn transport_no_send_proof_never_reopens_a_historical_armed_attempt() {
    use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
    use std::time::Duration;
    use axum::body::Bytes;
    use crate::api::PinnedPeerHttpTarget;
    use crate::services::reverse_onion_source::{tests::Fixture, SourcePhase};
    use super::super::reverse_onion_source_runtime::{
        ReverseOnionSourceRuntime, ReverseOnionSourceLifecycle, SourcePinnedRelayPolicy,
        SourceRuntimeConfig, SourceTransport, SourceTransportOutcome, SourceSendAdmission, SourceRuntimeError,
    };

    struct StopAtEntry {
        stopped: Arc<AtomicBool>,
        entries: AtomicUsize,
    }
    #[async_trait::async_trait]
    impl SourceTransport for StopAtEntry {
        async fn post(&self, _: PinnedPeerHttpTarget, _: Bytes, _: Bytes,
            admission: SourceSendAdmission) -> SourceTransportOutcome {
            self.entries.fetch_add(1, Ordering::SeqCst);
            self.stopped.store(true, Ordering::SeqCst);
            match admission.check() {
                Err(proof) => SourceTransportOutcome::NotSent(proof),
                Ok(()) => panic!("transport must use the actual owner stop flag"),
            }
        }
        async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
            _: SourceSendAdmission) -> SourceTransportOutcome {
            panic!("stopped transport must not start evidence HTTP");
        }
    }

    for historical in [false, true] {
        let f = Fixture::new_for_runtime();
        let journal = Arc::new(f.open(f.now()));
        if historical {
            f.prepare(&journal, f.now());
            journal.arm(f.route(), f.now()).unwrap();
        }
        let (relay, recipient, authorization) = f.policy_parts();
        let policy = Arc::new(SourcePinnedRelayPolicy::new(f.source_identity().public_key_bytes(),
            relay, recipient, authorization, f.now()).unwrap());
        let transport = Arc::new(StopAtEntry { stopped: journal.intake_stop_flag(), entries: AtomicUsize::new(0) });
        let runtime = Arc::new(ReverseOnionSourceRuntime::new(journal.clone(), f.source_identity(),
            policy.clone(), transport.clone(), SourceRuntimeConfig::new(1, Duration::from_secs(1), 600,
                Duration::ZERO, Duration::from_millis(250)).unwrap()).unwrap());
        let lifecycle = ReverseOnionSourceLifecycle::with_request_limit(1);
        lifecycle.install(runtime.clone()).unwrap();
        let outcome = if historical {
            lifecycle.resume(f.route(), f.now()).await
        } else {
            let (request, expected, session, terminal) = f.runtime_admission_parts();
            lifecycle.dispatch(request, expected, session, f.now() + 600, terminal, f.now(), policy).await
        };
        assert_eq!(outcome.err(), Some(SourceRuntimeError::Stopped));
        assert_eq!(transport.entries.load(Ordering::SeqCst), 1);
        assert!(lifecycle.request_admission().is_stopped());
        assert!(!lifecycle.has_failed());
        let expected = if historical { SourcePhase::Armed } else { SourcePhase::Prepared };
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(journal.lookup_phase(f.route(), at).unwrap(), Some(expected));
        lifecycle.shutdown_and_drain().await.unwrap();
        drop(lifecycle);
        drop(runtime);
        drop(journal);
        let reopened = f.open(at);
        assert_eq!(reopened.lookup_phase(f.route(), at).unwrap(), Some(expected));
    }
}

// [REVERSE-ONION-WORKER-SUPERVISION 2026-10-05 by Codex] Authored, not run:
// dropping the worker-owned sender must wake the server's post-READY monitor.
#[tokio::test]
async fn worker_completion_channel_closure_is_observable() {
    let (sender, receiver) = tokio::sync::watch::channel(());
    assert!(!reverse_onion_runtime::worker_task_has_exited(&receiver));
    let waiter = tokio::spawn(
        reverse_onion_runtime::wait_for_worker_task_exit(receiver),
    );

    drop(sender);

    let (closed_sender, closed_receiver) = tokio::sync::watch::channel(());
    drop(closed_sender);
    assert!(reverse_onion_runtime::worker_task_has_exited(&closed_receiver));

    tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
        .await
        .expect("worker exit signal should wake the supervisor")
        .expect("exit monitor should not panic");
}

// [REVERSE-ONION-LIVE-RECIPIENT-AUTHORITY 2026-10-05 by Codex]
// A live grant controls only creation of a fresh Claim; recovery mode remains
// an independent operator hold even when discovery later renews that grant.
#[test]
fn fresh_claim_gate_tracks_live_authority_without_lifting_recovery_hold() {
    let idle = None;
    assert!(!reverse_onion_runtime::can_start_poll(idle, false, false, false, false));
    assert!(reverse_onion_runtime::can_start_poll(idle, false, false, true, true));
    assert!(!reverse_onion_runtime::can_start_poll(idle, false, false, false, true));
    assert!(!reverse_onion_runtime::can_start_poll(Some([9; 16]), false, false, false, true));
    assert!(!reverse_onion_runtime::can_start_poll(idle, true, false, false, true));
}

// [REVERSE-ONION-ZERO-DISPATCH 2026-10-05 by Codex] Only pre-router
// rejection can make an Armed lease executable again; a generic rejection or
// any ambiguous result must remain custody-only.
#[test]
fn only_capacity_or_proven_zero_dispatch_allows_exact_lease_retry() {
    use crate::api::reverse_onion_terminal::ReverseOnionTerminalError as E;

    assert!(reverse_onion_runtime::may_restore_after_zero_dispatch(E::Busy));
    assert!(reverse_onion_runtime::may_restore_after_zero_dispatch(E::ZeroDispatch));
    assert!(!reverse_onion_runtime::may_restore_after_zero_dispatch(E::Rejected));
    assert!(!reverse_onion_runtime::may_restore_after_zero_dispatch(E::Ambiguous));
    // [PHALA-TERMINAL-CLOCK-CONTINUITY 2026-10-07 by Codex] Authored only.
    assert!(!reverse_onion_runtime::may_restore_after_zero_dispatch(E::Clock));
}

// [PHALA-RECIPIENT-INTAKE-FENCE 2026-10-07 by Codex] Authored, not run:
// stop arrives under the real journal transaction, using the adapter's shared
// worker flag. A restart may arm the preserved Lease once, never an Armed row.
#[cfg(unix)]
#[test]
fn recipient_stop_during_sql_admission_preserves_lease_without_masking_faults() {
    use std::sync::Arc;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
    use aeronyx_core::protocol::onion::reverse_delivery::ReverseOnionFrameV1;
    use crate::api::reverse_onion_terminal::ReverseOnionTerminalAdapter;
    use crate::services::reverse_onion_recipient::{
        RecipientJournalError, RecipientJournalLimits, RecipientRecovery, ReverseOnionRecipientJournal,
    };
    const NOW: u64 = 1_800_000_000;
    let directory = tempfile::Builder::new().prefix("phala-intake-stop-worker-")
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
        .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
    let path = directory.path().join("recipient.sqlite");
    let relay = IdentityKeyPair::from_bytes(&[81; 32]).unwrap();
    let recipient = Arc::new(IdentityKeyPair::from_bytes(&[82; 32]).unwrap());
    let open = |now| ReverseOnionRecipientJournal::open(&path, relay.public_key_bytes(),
        recipient.public_key_bytes(), RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, now).unwrap();
    let journal = open(NOW);
    let (_, kem) = recipient.to_x25519();
    let envelope = build_onion_envelope(&[OnionHop {
        node_id: recipient.public_key_bytes(), kem_pub: kem.to_bytes(),
    }], b"opaque fixture", [2; 16], 1, NOW, &relay).unwrap();
    let claim = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [1; 16], NOW, NOW + 30, &recipient).unwrap();
    let lease = ReverseOnionFrameV1::lease(&claim, &envelope, [3; 16], NOW + 600, NOW, &relay).unwrap();
    journal.prepare_poll(&claim, [9; 32], NOW).unwrap();
    journal.record_lease(claim.claim_id(), &lease, NOW + 600, NOW).unwrap();
    let adapter = ReverseOnionTerminalAdapter::new(axum::Router::new(), recipient.clone(),
        relay.public_key_bytes(), std::time::Duration::from_secs(1)).unwrap();
    let stopped = adapter.intake_stop_flag();
    assert!(reverse_onion_runtime::recipient_intake_open(&stopped).is_ok());
    assert!(reverse_onion_runtime::arm_recovery_lease_with_admission_at(&journal, claim.claim_id(), || {
        adapter.request_stop();
        Ok(NOW + 1)
    }, || reverse_onion_runtime::recipient_intake_open(&stopped)).unwrap().is_none());
    let fresh = ReverseOnionFrameV1::claim(relay.public_key_bytes(), [4; 16], NOW + 1, NOW + 31, &recipient).unwrap();
    assert_eq!(journal.prepare_poll_with_admission_at(&fresh, [9; 32], || Ok(NOW + 1),
        || reverse_onion_runtime::recipient_intake_open(&stopped)).err(), Some(RecipientJournalError::IntakeClosed));
    assert!(matches!(journal.resume(None, 64, NOW + 2).unwrap().items.as_slice(),
        [RecipientRecovery::LeaseReady { .. }]));
    assert_eq!(reverse_onion_runtime::arm_recovery_lease_with_admission_at(&journal, claim.claim_id(),
        || Ok(NOW + 1), || reverse_onion_runtime::recipient_intake_open(&stopped)).err(),
        Some(RecipientJournalError::Rejected));
    assert_eq!(reverse_onion_runtime::arm_recovery_lease_with_admission_at(&journal, claim.claim_id(),
        || Err(RecipientJournalError::Unavailable),
        || reverse_onion_runtime::recipient_intake_open(&stopped)).err(), Some(RecipientJournalError::Unavailable));
    drop(journal);
    let journal = open(NOW + 3);
    assert!(matches!(journal.resume(None, 64, NOW + 3).unwrap().items.as_slice(),
        [RecipientRecovery::LeaseReady { .. }]));
    let restart_gate = std::sync::atomic::AtomicBool::new(false);
    assert!(reverse_onion_runtime::arm_recovery_lease_with_admission_at(&journal, claim.claim_id(),
        || Ok(NOW + 3), || reverse_onion_runtime::recipient_intake_open(&restart_gate)).unwrap().is_some());
    assert_eq!(reverse_onion_runtime::arm_recovery_lease_with_admission_at(&journal, claim.claim_id(),
        || Ok(NOW + 3), || reverse_onion_runtime::recipient_intake_open(&stopped)).err(),
        Some(RecipientJournalError::Ambiguous));
}

// [PHALA-RECIPIENT-EXPIRY-RACE 2026-10-07 by Codex] Authored, not run:
// exercise the worker's actual arm projection against durable journal state,
// not a manufactured worker-completion signal or a synthetic success response.
#[cfg(unix)]
#[test]
fn expired_recovery_page_does_not_fail_worker_arm_or_revive_an_armed_job() {
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
    use aeronyx_core::protocol::onion::reverse_delivery::ReverseOnionFrameV1;
    use crate::services::reverse_onion_recipient::{
        RecipientJournalError, RecipientJournalLimits, RecipientRecovery,
        ReverseOnionRecipientJournal,
    };
    const NOW: u64 = 1_800_000_000;
    let directory = tempfile::Builder::new().prefix("phala-expiry-worker-")
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
        .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
    let path = directory.path().join("recipient.sqlite");
    let relay = IdentityKeyPair::from_bytes(&[61; 32]).unwrap();
    let recipient = IdentityKeyPair::from_bytes(&[62; 32]).unwrap();
    let open = |now| ReverseOnionRecipientJournal::open(
        &path, relay.public_key_bytes(), recipient.public_key_bytes(),
        RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, now,
    ).unwrap();
    let journal = open(NOW);
    let (_, kem) = recipient.to_x25519();
    let hops = [OnionHop { node_id: recipient.public_key_bytes(), kem_pub: kem.to_bytes() }];
    let claim = ReverseOnionFrameV1::claim(
        relay.public_key_bytes(), [1; 16], NOW, NOW + 30, &recipient,
    ).unwrap();
    let envelope = build_onion_envelope(&hops, b"opaque fixture", [2; 16], 1, NOW, &relay).unwrap();
    let lease = ReverseOnionFrameV1::lease(&claim, &envelope, [3; 16], NOW + 10, NOW, &relay).unwrap();
    journal.prepare_poll(&claim, [9; 32], NOW).unwrap();
    journal.record_lease(claim.claim_id(), &lease, NOW + 10, NOW).unwrap();
    assert!(matches!(journal.resume(None, 64, NOW + 9).unwrap().items.as_slice(),
        [RecipientRecovery::LeaseReady { .. }]));
    // [PHALA-RECIPIENT-ADMISSION-CLOCK 2026-10-07 by Codex] The valid
    // page above ages at the actual transaction clock callback, not preflight.
    let mut clock_calls = 0;
    assert!(reverse_onion_runtime::arm_recovery_lease_at(&journal, claim.claim_id(), || {
        clock_calls += 1;
        Ok(NOW + 10)
    }).unwrap().is_none());
    assert_eq!(clock_calls, 1);
    assert!(matches!(journal.resume(None, 64, NOW + 9).unwrap().items.as_slice(),
        [RecipientRecovery::LeaseReady { .. }]));
    assert_eq!(journal.cleanup(64, NOW + 10).unwrap(), 0);
    drop(journal);
    let journal = open(NOW + 11);
    assert!(reverse_onion_runtime::arm_recovery_lease(&journal, claim.claim_id(), NOW + 11).unwrap().is_none());
    assert_eq!(reverse_onion_runtime::arm_recovery_lease(&journal, [99; 16], NOW + 11).err(),
        Some(RecipientJournalError::Rejected));

    let next = ReverseOnionFrameV1::claim(
        relay.public_key_bytes(), [4; 16], NOW + 11, NOW + 41, &recipient,
    ).unwrap();
    let next_envelope = build_onion_envelope(&hops, b"next opaque fixture", [5; 16], 1, NOW + 11, &relay).unwrap();
    let next_lease = ReverseOnionFrameV1::lease(&next, &next_envelope, [6; 16], NOW + 60, NOW + 11, &relay).unwrap();
    journal.prepare_poll(&next, [9; 32], NOW + 11).unwrap();
    journal.record_lease(next.claim_id(), &next_lease, NOW + 60, NOW + 11).unwrap();
    assert!(reverse_onion_runtime::arm_recovery_lease(&journal, next.claim_id(), NOW + 12).unwrap().is_some());
    assert_eq!(reverse_onion_runtime::arm_recovery_lease(&journal, next.claim_id(), NOW + 12).err(),
        Some(RecipientJournalError::Ambiguous));
}

// [PHALA-RECIPIENT-JOURNAL-LANE 2026-10-07 by Codex] Authored, not run:
// cancellation before lane admission has no effect; cancellation after spawn
// cannot let a second owner overtake the real DB operation and its fence.
#[cfg(unix)]
#[tokio::test]
async fn recipient_journal_cancellation_preserves_shared_blocking_ownership() {
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use aeronyx_core::crypto::IdentityKeyPair;
    use crate::services::reverse_onion_recipient::{
        RecipientJournalError, RecipientJournalLimits, ReverseOnionRecipientJournal,
    };
    fn now() -> u64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    }
    let directory = tempfile::Builder::new().prefix("phala-journal-owner-")
        // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
        .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
        .tempdir_in("/Volumes/disk/aeronyx-codex-tmp").unwrap();
    let relay = IdentityKeyPair::from_bytes(&[71; 32]).unwrap();
    let recipient = IdentityKeyPair::from_bytes(&[72; 32]).unwrap();
    let journal = Arc::new(ReverseOnionRecipientJournal::open(
        &directory.path().join("recipient.sqlite"), relay.public_key_bytes(), recipient.public_key_bytes(),
        RecipientJournalLimits { max_entries: 8, max_bytes: 16 * 1024 * 1024 }, now(),
    ).unwrap());
    let lane = journal.blocking_operation_lane();
    assert!(Arc::ptr_eq(&lane, &journal.blocking_operation_lane()));
    let held = Arc::clone(&lane).acquire_owned().await.unwrap();
    let called = Arc::new(AtomicBool::new(false));
    let queued_called = Arc::clone(&called);
    let mut queued = Box::pin(journal.run_blocking(move |_| {
        queued_called.store(true, Ordering::SeqCst);
        Ok(())
    }));
    assert!(futures::poll!(queued.as_mut()).is_pending());
    drop(queued);
    drop(held);
    journal.run_blocking(|_| Ok(())).await.unwrap();
    assert!(!called.load(Ordering::SeqCst));

    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let finished = Arc::new(AtomicBool::new(false));
    let blocking_finished = Arc::clone(&finished);
    let owned_journal = Arc::clone(&journal);
    let caller = tokio::spawn(async move {
        owned_journal.run_blocking(move |journal| {
            let _ = entered_tx.send(());
            release_rx.recv().map_err(|_| RecipientJournalError::Unavailable)?;
            journal.cleanup(1, now())?;
            blocking_finished.store(true, Ordering::SeqCst);
            Ok(())
        }).await
    });
    entered_rx.await.unwrap();
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    assert!(Arc::clone(&lane).try_acquire_owned().is_err());
    let observed_finished = Arc::clone(&finished);
    let mut next = Box::pin(journal.run_blocking(move |journal| {
        assert!(observed_finished.load(Ordering::SeqCst));
        journal.resume(None, 64, now())
    }));
    assert!(futures::poll!(next.as_mut()).is_pending());
    release_tx.send(()).unwrap();
    assert!(next.await.unwrap().items.is_empty());
    assert!(finished.load(Ordering::SeqCst));
    assert_eq!(lane.available_permits(), 1);
}

// [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Authored, not run:
// cancel an actual source DB waiter after blocking entry. Neither source
// admission nor journal ownership can be released ahead of its durable work.
#[cfg(unix)]
#[tokio::test]
async fn source_journal_cancellation_retains_both_owned_permits() {
    use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    use super::super::reverse_onion_source_runtime::{source_db, SourceRuntimeError};
    use crate::services::reverse_onion_source::tests::Fixture;
    let fixture = Fixture::new_for_runtime();
    let journal = Arc::new(fixture.open(fixture.now()));
    let admissions = Arc::new(tokio::sync::Semaphore::new(2));
    let lane = journal.blocking_operation_lane();
    assert!(Arc::ptr_eq(&lane, &journal.blocking_operation_lane()));
    let permit = Arc::new(Arc::clone(&admissions).acquire_owned().await.unwrap());
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let finished = Arc::new(AtomicBool::new(false));
    let blocking_finished = Arc::clone(&finished);
    let owned_journal = Arc::clone(&journal);
    let caller = tokio::spawn(async move {
        source_db(&owned_journal, permit, 0, move |journal, now| {
            let _ = entered_tx.send(());
            release_rx.recv().map_err(|_| SourceRuntimeError::Unavailable)?;
            journal.cleanup(1, now).map_err(SourceRuntimeError::from)?;
            blocking_finished.store(true, Ordering::SeqCst);
            Ok(())
        }).await
    });
    entered_rx.await.unwrap();
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    assert_eq!(admissions.available_permits(), 1);
    assert_eq!(lane.available_permits(), 0);
    let next_permit = Arc::new(Arc::clone(&admissions).acquire_owned().await.unwrap());
    let observed_finished = Arc::clone(&finished);
    let mut next = Box::pin(source_db(&journal, next_permit, 0, move |journal, now| {
        assert!(observed_finished.load(Ordering::SeqCst));
        journal.recover(None, 64, now).map_err(SourceRuntimeError::from)
    }));
    assert!(futures::poll!(next.as_mut()).is_pending());
    assert_eq!(admissions.available_permits(), 0);
    release_tx.send(()).unwrap();
    assert!(next.await.unwrap().items.is_empty());
    assert_eq!(admissions.available_permits(), 2);
    assert_eq!(lane.available_permits(), 1);
}
