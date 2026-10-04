// ============================================
// File: crates/aeronyx-server/src/server/reverse_onion_runtime.rs
// ============================================
//! Bounded outbound carrier for private-recipient reverse onion delivery.
//!
//! [REVERSE-ONION-CARRIER 2026-10-04 by Codex] This carrier never starts a
//! polling loop, peels an onion, changes durable state, or interprets HTTP
//! success as execution. The runtime owner must durably preserve the exact
//! request before calling it, authenticate responses, and perform queue CAS.
//! No redirects, inherited proxies, DNS endpoints, or automatic retries.

use std::time::Duration;

use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, ReverseOnionKindV1, MAX_REVERSE_ONION_FRAME_BYTES,
};

use crate::config_reverse_onion::ReverseOnionConfig;

/// Coarse preflight errors have no network effects or identifying details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionPreflightError {
    Disabled,
    Configuration,
    Frame,
}

/// A response is transport evidence only, never a custody/execution proof.
/// Intentionally omits Debug so opaque bodies cannot enter routine logs.
pub(crate) enum ReverseOnionExchange {
    /// An HTTP response arrived within both the time and body size bounds.
    Response { status: u16, body: Vec<u8> },
    /// A POST may have reached the relay. Preserve the exact durable request;
    /// this must never cause a new claim, lease, target or source reply key.
    Ambiguous,
}

/// Fixed adjacent relay and recipient identity for one outbound worker.
pub(crate) struct ReverseOnionHttpCarrier {
    client: reqwest::Client,
    relay: [u8; 32],
    recipient: [u8; 32],
    claim_url: reqwest::Url,
    result_url: reqwest::Url,
    timeout: Duration,
}

impl ReverseOnionHttpCarrier {
    /// Construct without opening sockets. Endpoint safety is checked before
    /// canonical route composition, including credentials and query rejection.
    pub(crate) fn new(
        config: &ReverseOnionConfig,
        recipient: [u8; 32],
    ) -> Result<Self, ReverseOnionPreflightError> {
        if !config.recipient.enabled {
            return Err(ReverseOnionPreflightError::Disabled);
        }
        config.validate().map_err(|_| ReverseOnionPreflightError::Configuration)?;
        let mut relay = [0u8; 32];
        hex::decode_to_slice(&config.recipient.relay_node_id, &mut relay)
            .map_err(|_| ReverseOnionPreflightError::Configuration)?;
        if recipient == [0; 32] || recipient == relay {
            return Err(ReverseOnionPreflightError::Configuration);
        }
        let timeout = Duration::from_secs(config.recipient.request_timeout_secs);
        let client = crate::api::privacy_safe_peer_http_client_builder()
            .connect_timeout(timeout)
            .timeout(timeout)
            .build()
            .map_err(|_| ReverseOnionPreflightError::Configuration)?;
        let endpoint = &config.recipient.relay_endpoint;
        Ok(Self {
            client,
            relay,
            recipient,
            claim_url: crate::api::canonical_peer_http_url(
                endpoint, "/api/chat/peer/reverse-onion/claim",
            ).map_err(|_| ReverseOnionPreflightError::Configuration)?,
            result_url: crate::api::canonical_peer_http_url(
                endpoint, "/api/chat/peer/reverse-onion/result",
            ).map_err(|_| ReverseOnionPreflightError::Configuration)?,
            timeout,
        })
    }

    /// Submit exactly one already-persisted frame. No retry is performed here.
    /// Even a non-2xx HTTP response requires protocol-specific recovery policy;
    /// a successful status without a verified bound receipt proves nothing.
    pub(crate) async fn exchange(
        &self,
        frame: &ReverseOnionFrameV1,
    ) -> Result<ReverseOnionExchange, ReverseOnionPreflightError> {
        if frame.relay() != self.relay || frame.immediate_recipient() != self.recipient {
            return Err(ReverseOnionPreflightError::Frame);
        }
        let url = match frame.kind() {
            ReverseOnionKindV1::Claim => &self.claim_url,
            ReverseOnionKindV1::Result => &self.result_url,
            ReverseOnionKindV1::Lease => return Err(ReverseOnionPreflightError::Frame),
        };
        let bytes = frame.encode();
        if bytes.len() > MAX_REVERSE_ONION_FRAME_BYTES {
            return Err(ReverseOnionPreflightError::Frame);
        }
        // One absolute timeout covers headers and streaming the complete body.
        // Never retain raw reqwest errors: they may contain the relay URL.
        let attempt = async {
            let mut response = self.client.post(url.clone())
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(bytes)
                .send().await.map_err(|_| ())?;
            if response.content_length().is_some_and(|n| n > MAX_REVERSE_ONION_FRAME_BYTES as u64) {
                return Err(());
            }
            let status = response.status().as_u16();
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
                if chunk.len() > MAX_REVERSE_ONION_FRAME_BYTES.saturating_sub(body.len()) {
                    return Err(());
                }
                body.extend_from_slice(&chunk);
            }
            Ok::<_, ()>(ReverseOnionExchange::Response { status, body })
        };
        Ok(match tokio::time::timeout(self.timeout, attempt).await {
            Ok(Ok(response)) => response,
            _ => ReverseOnionExchange::Ambiguous,
        })
    }
}
