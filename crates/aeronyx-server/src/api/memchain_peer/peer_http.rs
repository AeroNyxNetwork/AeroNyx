// [ARCH-SPLIT 2026-10-02]
// Peer URL construction, bounded reads, and protocol errors shared by the flows above.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Accepts only public IP literals for permissionless witness traffic.
///
/// A signed descriptor proves who advertised an endpoint, not that the target
/// is safe for this host to contact. Domain names are deliberately excluded to
/// prevent DNS rebinding; loopback, private, link-local, CGNAT, benchmark,
/// documentation, multicast, and reserved ranges are also rejected.
pub(crate) fn commitment_peer_endpoint_is_public(endpoint: &str) -> bool {
    // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] MemChain and discovery
    // share one public-host policy so a future range update cannot leave one
    // permissionless transport less protected than another.
    peer_endpoint_is_permitted(endpoint)
}

pub(super) fn commitment_block_range_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/block-range")
}

pub(super) fn commitment_block_announce_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/block-announce")
}

pub(super) fn commitment_checkpoint_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint")
}

pub(super) fn commitment_checkpoint_certificate_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint-certificate")
}

pub(super) fn commitment_coordinator_handover_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-handover")
}

pub(super) fn commitment_coordinator_lease_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-lease")
}

pub(super) fn commitment_coordinator_lease_release_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-lease/release")
}

pub(super) fn verified_delivery_anchor_witness_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(
        endpoint,
        "/api/discovery/peer/verified-delivery-anchor-witness",
    )
}

pub(super) fn custody_audit_anchor_witness_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/custody-audit-anchor-witness")
}

pub(crate) fn commitment_peer_url(endpoint: &str, path: &str) -> Result<Url, String> {
    canonical_peer_http_url(endpoint, path).map_err(|error| match error {
        PeerEndpointUrlError::Missing => "pinned_coordinator_missing_endpoint".to_string(),
        PeerEndpointUrlError::Invalid => "pinned_coordinator_invalid_endpoint".to_string(),
    })
}

pub(super) async fn read_bounded_response(response: reqwest::Response) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BODY_BYTES as u64)
    {
        return Err("response_too_large".to_string());
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| classify_http_error("response_body", &error))?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BODY_BYTES {
            return Err("response_too_large".to_string());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

pub(super) fn classify_http_error(phase: &str, error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_body() {
        "body"
    } else if error.is_decode() {
        "decode"
    } else if error.is_request() {
        "request"
    } else {
        "unknown"
    };
    format!("{phase}_{kind}")
}

pub(super) fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, axum::Json(serde_json::json!({ "error": code }))).into_response()
}

pub(super) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
