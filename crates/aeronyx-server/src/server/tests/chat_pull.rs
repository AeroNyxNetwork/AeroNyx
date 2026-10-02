// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn chat_pull_route_authority_v1_cannot_create() {
    assert_cross_identity_pull_route_authority(false, false).await;
}

#[tokio::test]
async fn chat_pull_route_authority_v2_cannot_create() {
    assert_cross_identity_pull_route_authority(true, false).await;
}

#[tokio::test]
async fn chat_pull_route_authority_v1_cannot_refresh() {
    assert_cross_identity_pull_route_authority(false, true).await;
}

#[tokio::test]
async fn chat_pull_route_authority_v2_cannot_refresh() {
    assert_cross_identity_pull_route_authority(true, true).await;
}

#[tokio::test]
async fn chat_pull_route_authority_same_identity_and_invalid_signature() {
    for v2 in [false, true] {
        let fixture = PullRouteFixture::new(true).await;
        let wallet = fixture.wallet.public_key_bytes();
        let mut invalid = fixture.pull(v2);
        match &mut invalid {
            MemChainMessage::ChatPull { signature, .. }
            | MemChainMessage::ChatPullV2 { signature, .. } => signature[0] ^= 1,
            _ => unreachable!(),
        }
        fixture.dispatch(invalid).await;
        assert!(fixture.relay.wallet_routes.lookup(&wallet).is_empty());
        fixture.dispatch(fixture.pull(v2)).await;
        assert!(fixture.receive_pull(v2).await.is_empty());
        assert!(
            fixture.relay.wallet_routes.lookup(&wallet)
                == vec![(fixture.session.id.clone(), fixture.session.endpoint())]
        );
    }
}

#[tokio::test]
async fn chat_pull_route_authority_session_bound_delegation_still_works() {
    use aeronyx_core::protocol::auth::{
        signed_message_digest, DOMAIN_DEVICE_REGISTER, DOMAIN_WALLET_PRESENCE,
    };
    let fixture = PullRouteFixture::new(false).await;
    let wallet = fixture.wallet.public_key_bytes();
    let timestamp = unix_now_secs();
    let ts = timestamp.to_le_bytes();
    let device_id = [0x74; 16];
    let signature = fixture.wallet.sign(&signed_message_digest(
        DOMAIN_DEVICE_REGISTER,
        &[fixture.session.id.as_bytes(), &device_id, &wallet, &ts],
    ));
    fixture
        .dispatch(MemChainMessage::DeviceRegister {
            device_id,
            device_name: String::new(),
            wallet_pubkey: wallet,
            timestamp,
            signature,
        })
        .await;
    assert!(
        fixture.relay.wallet_routes.lookup(&wallet)
            == vec![(fixture.session.id.clone(), fixture.session.endpoint())]
    );
    assert!(fixture
        .relay
        .wallet_routes
        .remove_route(&wallet, &fixture.session.id));
    let signature = fixture.wallet.sign(&signed_message_digest(
        DOMAIN_WALLET_PRESENCE,
        &[fixture.session.id.as_bytes(), &wallet, &ts],
    ));
    fixture
        .dispatch(MemChainMessage::WalletPresence {
            wallet_pubkey: wallet,
            timestamp,
            signature,
        })
        .await;
    assert!(
        fixture.relay.wallet_routes.lookup(&wallet)
            == vec![(fixture.session.id.clone(), fixture.session.endpoint())]
    );
}
