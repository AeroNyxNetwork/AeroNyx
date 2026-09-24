// ============================================================================
// File: crates/aeronyx-server/tests/node_join_cli_interop.rs
// ============================================================================
//! Offline operator-join wire interoperability and real HTTP admission gate.
//!
//! [PERMISSIONLESS-NODE-JOIN-INTEROP 2026-09-24 by Codex] The CLI's Python
//! encoder must produce the exact Rust canonical descriptor bytes. A bounded
//! loopback Axum server then exercises the production discovery router;
//! neither a fleet node nor a registration service is contacted.

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{
    NodeBootstrapSnapshot, NodeCapability, NodeCapacity, NodeDescriptor, SignedNodeDescriptor,
};
use aeronyx_server::api::discovery::{build_discovery_router, DiscoveryApiPolicy};
use aeronyx_server::services::peer_store::PeerStore;
use axum::routing::get;
use axum::{Json, Router};
use base64::Engine;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::process::Command;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_secs()
}

fn signed_fixture(identity: &IdentityKeyPair, now: u64, endpoint: &str) -> SignedNodeDescriptor {
    let mut descriptor = NodeDescriptor::new(
        identity.public_key_bytes(),
        7,
        now - 10,
        now + 300,
        "1.0.0+anpf1-pbdr2",
    );
    descriptor.public_endpoint = Some(endpoint.to_owned());
    descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
    descriptor.capacity = NodeCapacity {
        max_sessions: 64,
        max_bps: None,
        max_pps: None,
    };
    SignedNodeDescriptor::sign(descriptor, identity).expect("sign fixture")
}

async fn post_descriptor(
    client: &reqwest::Client,
    base: &str,
    bytes: Vec<u8>,
) -> (u16, Option<Value>) {
    let response = client
        .post(format!("{base}/api/discovery/join"))
        .header("content-type", "application/octet-stream")
        .body(bytes)
        .send()
        .await
        .expect("loopback join response");
    let status = response.status().as_u16();
    let body = response.json::<Value>().await.ok();
    (status, body)
}

#[tokio::test]
async fn cli_canonical_descriptor_is_verified_by_real_join_http_gate() {
    let now = now_secs();
    let identity = IdentityKeyPair::from_bytes(&[7_u8; 32]).expect("fixed test identity");
    let endpoint = "https://9.9.9.9:8422";
    let signed = signed_fixture(&identity, now, endpoint);
    let snapshot = NodeBootstrapSnapshot::new(now, vec![signed.clone()]);
    let status = json!({
        "peer_store": {
            "snapshot": {"valid_peers": 1},
            "runtime": {"last_gossip_at": now}
        },
        "local_capabilities": {
            "status": "ready",
            "capability_config_consistent": true
        }
    });

    let snapshot_value = serde_json::to_value(&snapshot).expect("snapshot JSON");
    let local_router = Router::new()
        .route(
            "/api/discovery/status",
            get(move || {
                let value = status.clone();
                async move { Json(value) }
            }),
        )
        .route(
            "/api/discovery/snapshot",
            get(move || {
                let value = snapshot_value.clone();
                async move { Json(value) }
            }),
        );
    let local_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind isolated local fixture");
    let local_port = local_listener.local_addr().expect("fixture address").port();
    let local_task = tokio::spawn(async move {
        axum::serve(local_listener, local_router)
            .await
            .expect("local fixture server");
    });

    let directory = tempfile::tempdir().expect("private fixture directory");
    let key_path = directory.path().join("key.json");
    let config_path = directory.path().join("server.toml");
    let request_path = directory.path().join("canonical-descriptor.bin");
    let public_key = base64::engine::general_purpose::STANDARD.encode(identity.public_key_bytes());
    std::fs::write(&key_path, json!({"public_key": public_key}).to_string())
        .expect("write public-only key fixture");
    let quoted_key_path = serde_json::to_string(&key_path.to_string_lossy()).expect("quoted path");
    std::fs::write(
        &config_path,
        format!(
            "[server_key]\nkey_file = {quoted_key_path}\n\
             [discovery]\npublic_api_listen_addr = \"127.0.0.1:{local_port}\"\n"
        ),
    )
    .expect("write local config fixture");

    let wrapper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../deploy/node/aeronyx-node.sh")
        .canonicalize()
        .expect("repository-local operator wrapper");
    let cli = Command::new("bash")
        .arg("-c")
        .arg(
            "source \"$1\"; select_join_python; CONFIG_FILE=\"$2\"; \
             JOIN_PUBLIC_ENDPOINT=\"$3\"; JOIN_SEEDS=(\"http://8.8.8.8:8422\"); \
             join_prepare_once \"$4\"",
        )
        .arg("join-interop")
        .arg(&wrapper)
        .arg(&config_path)
        .arg(endpoint)
        .arg(&request_path)
        .env("TMPDIR", directory.path())
        .output()
        .await
        .expect("run local operator CLI");
    assert!(
        cli.status.success(),
        "CLI preparation rejected signed fixture"
    );
    let preparation: Value =
        serde_json::from_slice(&cli.stdout).expect("aggregate CLI preparation");
    assert_eq!(preparation["status"], "ready_to_submit");
    assert_eq!(preparation["signed_descriptor_accepted"], false);
    assert_eq!(preparation["route_ready"], false);
    let bytes = std::fs::read(&request_path).expect("CLI canonical descriptor");
    assert!(bytes.len() <= 16 * 1024);
    let decoded = SignedNodeDescriptor::decode_canonical(&bytes)
        .expect("Rust decodes exact CLI canonical bytes");
    assert_eq!(
        decoded, signed,
        "CLI must encode every signed field exactly"
    );
    decoded
        .verify_at(now_secs())
        .expect("Rust verifies CLI signature and TTL");

    let peer_store = Arc::new(PeerStore::new());
    let join_router =
        build_discovery_router(Arc::clone(&peer_store), DiscoveryApiPolicy::default());
    let join_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind isolated Stage-A gate");
    let join_port = join_listener.local_addr().expect("gate address").port();
    let join_task = tokio::spawn(async move {
        axum::serve(join_listener, join_router)
            .await
            .expect("Stage-A loopback server");
    });
    let base = format!("http://127.0.0.1:{join_port}");
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("loopback client");

    let (status, body) = post_descriptor(&client, &base, bytes.clone()).await;
    assert_eq!(status, 200);
    let body = body.expect("Stage-A aggregate body");
    assert_eq!(body["accepted"], true);
    assert_eq!(body["status"], "candidate_admitted");
    assert_eq!(body["route_authority"], false);
    assert_eq!(
        body["economic_admission"],
        "reserved_future_eth_projection_not_enforced"
    );
    let rendered = body.to_string();
    assert!(!rendered.contains(&hex::encode(identity.public_key_bytes())));
    assert!(!rendered.contains(endpoint));
    assert!(!rendered.contains("signature"));
    assert_eq!(peer_store.len(), 0, "Stage-A must not grant routing");

    let (replay_status, replay_body) = post_descriptor(&client, &base, bytes.clone()).await;
    assert_eq!(replay_status, 200);
    assert_eq!(replay_body.expect("replay body")["status"], "exact_replay");
    assert_eq!(peer_store.len(), 0);

    let mut conflicting = signed.descriptor.clone();
    conflicting.public_endpoint = Some("https://1.1.1.1:8422".to_owned());
    let conflict_bytes = SignedNodeDescriptor::sign(conflicting, &identity)
        .expect("sign conflict")
        .encode_canonical()
        .expect("encode conflict");
    assert_eq!(post_descriptor(&client, &base, conflict_bytes).await.0, 409);

    let mut expired = signed.descriptor.clone();
    expired.sequence = 8;
    expired.issued_at = now - 600;
    expired.expires_at = now - 1;
    let expired_bytes = SignedNodeDescriptor::sign(expired, &identity)
        .expect("sign expired descriptor")
        .encode_canonical()
        .expect("encode expired descriptor");
    assert_eq!(post_descriptor(&client, &base, expired_bytes).await.0, 400);

    let mut tampered = bytes;
    let last = tampered.last_mut().expect("nonempty signature");
    *last ^= 1;
    assert_eq!(post_descriptor(&client, &base, tampered).await.0, 400);
    assert_eq!(
        post_descriptor(&client, &base, vec![0; 16 * 1024 + 1])
            .await
            .0,
        413
    );
    assert_eq!(
        peer_store.len(),
        0,
        "negative cases must remain non-routeable"
    );

    local_task.abort();
    join_task.abort();
}
