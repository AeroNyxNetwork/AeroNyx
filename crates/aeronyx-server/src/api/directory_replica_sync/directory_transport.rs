// [ARCH-SPLIT 2026-10-02]
// Hardened directory HTTP client, peer URLs, and bounded frame posts.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn build_hardened_directory_http_client() -> Result<reqwest::Client, &'static str> {
    privacy_safe_peer_http_client_builder()
        // Direct authenticated node relationships must not inherit a process
        // proxy that becomes an unreviewed endpoint-metadata observer.
        .connect_timeout(Duration::from_secs(DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(
            DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS,
        ))
        .pool_max_idle_per_host(1)
        .build()
        .map_err(|_| "directory_sync_http_client_initialization_failed")
}

/// Builds the redirect-free bounded client used for an explicit certificate
/// exchange operation.
///
/// # Errors
/// Returns a stable reason when the HTTP client cannot be initialized.
pub fn build_directory_certificate_exchange_http_client() -> Result<reqwest::Client, String> {
    build_hardened_directory_http_client().map_err(str::to_string)
}

pub(super) fn directory_sync_peer_urls(
    peer_store: &PeerStore,
    producer: &[u8; 32],
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(producer, request_timestamp)
        .ok_or_else(|| "pinned_directory_peer_unavailable".to_string())?;
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_directory_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("pinned_directory_peer_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(endpoint, "/api/discovery/peer/directory/block-range")
        .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    let object_url =
        commitment_peer_url(endpoint, "/api/discovery/peer/directory/descriptor-objects")
            .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

pub(super) fn directory_mirror_peer_urls(
    peer_store: &PeerStore,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(producer, request_timestamp)
        .ok_or_else(|| "directory_mirror_peer_unavailable".to_string())?;
    if descriptor.sequence() != descriptor_sequence
        || !descriptor.descriptor.policy.public_discovery
    {
        return Err("directory_mirror_descriptor_changed".to_string());
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_mirror_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_mirror_peer_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(endpoint, "/api/discovery/peer/directory/block-range")
        .map_err(|_| "directory_mirror_peer_invalid_endpoint".to_string())?;
    let object_url =
        commitment_peer_url(endpoint, "/api/discovery/peer/directory/descriptor-objects")
            .map_err(|_| "directory_mirror_peer_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

pub(super) fn directory_replica_carrier_urls(
    peer_store: &PeerStore,
    carrier: &[u8; 32],
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(carrier, request_timestamp)
        .ok_or_else(|| "pinned_directory_peer_unavailable".to_string())?;
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_directory_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("pinned_directory_peer_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-block-range",
    )
    .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    let object_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-descriptor-objects",
    )
    .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

pub(super) fn directory_mirror_recovery_carrier_urls(
    peer_store: &PeerStore,
    carrier: &[u8; 32],
    descriptor_sequence: u64,
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(carrier, request_timestamp)
        .ok_or_else(|| "directory_mirror_recovery_carrier_unavailable".to_string())?;
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Bind endpoint derivation to the
    // same authenticated descriptor sequence selected by capability policy.
    // A concurrent descriptor change retries through a fresh selection rather
    // than probing or caching an endpoint under the wrong sequence.
    if descriptor.sequence() != descriptor_sequence {
        return Err("directory_mirror_recovery_carrier_descriptor_changed".to_string());
    }
    if !descriptor.descriptor.policy.public_discovery {
        return Err("directory_mirror_recovery_carrier_not_public".to_string());
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_mirror_recovery_carrier_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_mirror_recovery_carrier_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-block-range",
    )
    .map_err(|_| "directory_mirror_recovery_carrier_invalid_endpoint".to_string())?;
    let object_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-descriptor-objects",
    )
    .map_err(|_| "directory_mirror_recovery_carrier_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

pub(super) async fn post_directory_frame(
    client: &reqwest::Client,
    url: reqwest::Url,
    frame: Vec<u8>,
    operation: &'static str,
) -> Result<Vec<u8>, String> {
    post_directory_frame_typed(client, url, frame)
        .await
        .map_err(|error| error.stable_reason(operation))
}

pub(super) async fn post_directory_frame_typed(
    client: &reqwest::Client,
    url: reqwest::Url,
    frame: Vec<u8>,
) -> Result<Vec<u8>, DirectoryFramePostError> {
    post_directory_frame_typed_with_response_limit(
        client,
        url,
        frame,
        MAX_DIRECTORY_SYNC_RESPONSE_BODY_BYTES,
    )
    .await
}

pub(super) async fn post_directory_frame_typed_with_response_limit(
    client: &reqwest::Client,
    url: reqwest::Url,
    frame: Vec<u8>,
    response_body_limit: usize,
) -> Result<Vec<u8>, DirectoryFramePostError> {
    let response = match client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let failure = DirectoryTransportFailure::from_reqwest(&error);
            if let Some(outcome) = failure.outcome() {
                record_directory_sync_transport_outcome(outcome);
            }
            return Err(DirectoryFramePostError::Transport(failure));
        }
    };
    if !response.status().is_success() {
        record_directory_sync_transport_outcome(
            DirectoryReplicaTransportOutcome::HttpStatusFailure,
        );
        let status = response.status().as_u16();
        // [MIRROR-CAPABILITY 2026-07-24 by Codex] A 404 can mean either that
        // an optional route is absent or that this carrier has not retained
        // the requested producer/range yet. Read only the tiny fixed protocol
        // code so temporary lag is never cached as missing software support.
        let peer_code = read_bounded_http_response(response, MAX_DIRECTORY_SYNC_ERROR_BODY_BYTES)
            .await
            .ok()
            .and_then(|body| DirectoryPeerErrorCode::parse(&body));
        return Err(DirectoryFramePostError::HttpStatus { status, peer_code });
    }
    match read_bounded_http_response(response, response_body_limit).await {
        Ok(body) => {
            record_directory_sync_transport_outcome(DirectoryReplicaTransportOutcome::Succeeded);
            Ok(body)
        }
        Err(error) => {
            let outcome = match error {
                BoundedHttpResponseError::TooLarge => {
                    DirectoryReplicaTransportOutcome::ResponseTooLarge
                }
                BoundedHttpResponseError::BodyRead | BoundedHttpResponseError::JsonDecode => {
                    DirectoryReplicaTransportOutcome::ResponseBodyReadFailure
                }
            };
            record_directory_sync_transport_outcome(outcome);
            Err(DirectoryFramePostError::Response(error))
        }
    }
}

pub(super) fn record_directory_sync_transport_outcome(outcome: DirectoryReplicaTransportOutcome) {
    let _ = DIRECTORY_SYNC_TRANSPORT_RUNTIME.try_with(|runtime| {
        runtime.record_directory_sync_transport_outcome(outcome, unix_now_secs());
    });
}

pub(super) fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
