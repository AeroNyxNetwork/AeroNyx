// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/protocol_feature.rs
// ============================================
//! # Signed protocol-feature negotiation
//!
//! Owns `NodeProtocolFeature`: fine-grained peer wire features advertised as
//! exact semantic-version build-metadata tokens inside the signed
//! `software_version` field, so negotiation never changes descriptor schema or
//! capability discriminants.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

// ============================================
// NodeProtocolFeature
// ============================================

/// Fine-grained peer wire features negotiated through signed descriptors.
///
/// [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] These values deliberately do
/// not serialize as `NodeCapability` variants. Each feature maps to an exact,
/// valid SemVer build-metadata identifier inside `software_version`, preserving
/// the schema-v2 bincode layout for old nodes while keeping the advertisement
/// covered by the descriptor signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeProtocolFeature {
    /// Handled blind-relay protocol failures carry an immediate-hop signed
    /// `BlindRelayFailureReceipt` bound to the exact request and reason.
    BlindRelayFailureReceiptV1,
    /// Successful blind-relay responses carry an immediate-hop signed receipt
    /// bound to the exact opaque request and returned evidence surface.
    ///
    /// [BLIND-RELAY-SUCCESS-RECEIPT 2026-08-29 by Codex] Unlike a terminal
    /// delivery receipt, this proves only what the directly contacted peer
    /// accepted. Each relay replaces the downstream receipt with its own, so
    /// no upstream hop learns the final terminal identity.
    BlindRelaySuccessReceiptV1,
    /// The node can return purpose-bound version-2 terminal delivery receipts.
    /// This claim authorizes a probe only; route authority still requires a
    /// successfully verified receipt from the selected terminal.
    PurposeBoundDeliveryReceiptV2,
    /// The node accepts direct encrypted chat relay requests authenticated by
    /// the immediate previous hop's Ed25519 node identity.
    ///
    /// [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] This is advertised through
    /// signed SemVer metadata so upgraded senders can select the authenticated
    /// endpoint without breaking rolling compatibility with legacy nodes.
    DirectPeerRelayAuthV2,
    /// The node returns a target-signed direct relay v2 receipt bound to the
    /// exact authenticated request after durable ciphertext acceptance.
    ///
    /// [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] This is intentionally
    /// separate from request authentication so mixed-version v2 fleets can
    /// upgrade the response contract without treating HTTP claims as trust.
    DirectPeerRelayReceiptV2,
    /// The node accepts direct relay requests whose previous-hop signature is
    /// bound to the exact selected target node identity.
    ///
    /// [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] This prevents one
    /// valid authenticated request from being replayed across different relay
    /// nodes. It remains separately negotiated so v1/v2 peers keep working
    /// during a rolling fleet upgrade.
    DirectPeerRelayTargetBindingV3,
    /// The node supports fixed-size encrypted terminal responses propagated
    /// through blind relay acknowledgements.
    ///
    /// [ONION-REPLY-NEGOTIATION 2026-08-28 by Codex] This token gates the
    /// additive response surface during rolling upgrades; it does not reveal
    /// whether any route carries a request/response workload.
    OnionReplyV1,
    /// Reply-capable routes may keep terminal identity and terminal proof
    /// exclusively inside the source-sealed fixed-size response.
    ///
    /// [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] This feature is
    /// path-wide: every selected hop must also support immediate-hop success
    /// receipts before a source opts into the topology-hiding response mode.
    OnionSourceSealedTerminalProofV1,
    /// The node can propagate the largest fixed-size Blind Vault recovery
    /// response without truncating or rejecting the peer acknowledgement.
    ///
    /// [BLIND-VAULT-LARGE-PULL-NEGOTIATION 2026-08-30 by Codex] This is a
    /// path-wide transport claim, not merely a terminal workload claim. A
    /// source must require it from every selected hop before reserving the
    /// maximum anonymous pull response class.
    OnionBlindVaultLargePullV1,
    /// The node accepts RFC 9474 blind-issued lease admission inside the final
    /// onion layer and returns a request-bound terminal-signed receipt.
    ///
    /// [ONION-BLIND-LEASE-ADMISSION 2026-08-28 by Codex] This remains separate
    /// from `OnionReplyV1`: supporting the generic carrier never implies that a
    /// rolling-upgrade peer executes this sensitive workload.
    OnionBlindLeaseAdmissionV1,
    /// The node accepts an immutable Blind Vault Put inside the final onion
    /// layer and returns a request-bound signed storage receipt.
    ///
    /// [ONION-BLIND-VAULT-PUT-RECEIPT 2026-08-28 by Codex] Legacy one-way Put
    /// remains available under its existing purpose; clients request this
    /// feature only when they require cryptographic custody evidence.
    OnionBlindVaultPutReceiptV1,
    /// The node accepts administration-key retirement of a complete Blind
    /// Vault lease and returns a request-bound signed aggregate receipt.
    ///
    /// [ONION-BLIND-VAULT-LEASE-RETIRE 2026-08-28 by Codex] This token gates
    /// destructive lease-wide mutation independently from generic reply and
    /// object-deletion support during rolling upgrades.
    OnionBlindVaultLeaseRetireV1,
    /// The node consumes a fresh blind credential while atomically extending
    /// one administration-key-controlled live Blind Vault lease.
    OnionBlindVaultLeaseRenewalV1,
    /// The node returns an encrypted terminal-signed observation for one
    /// administration-key-controlled live Blind Vault lease.
    OnionBlindVaultLeaseStatusV1,
    /// The node returns an encrypted terminal-signed commitment to the live
    /// object inventory of one administration-key-controlled lease.
    OnionBlindVaultLeaseInventoryV1,
    /// Valid Blind Vault workload failures are sealed into the same fixed-size
    /// source-only reply instead of escaping through relay-visible status.
    ///
    /// [ONION-BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] This token is
    /// separate from generic `OnionReplyV1` so upgraded sources never assume
    /// typed encrypted failures from a mixed-version terminal.
    OnionBlindVaultEncryptedFailureV1,
    /// The node accepts the core v1 anonymous mailbox terminal codec through
    /// a purpose-bound, source-sealed onion route.
    ///
    /// [ANONYMOUS-MAILBOX-V1 2026-09-02 by Codex] This advertises only protocol
    /// support. It is not a mailbox locator and discloses no tenant activity.
    AnonymousMailboxV1,
    /// The node's `public_endpoint` port also serves TLS 1.3 with a
    /// certificate bound to this descriptor's identity (`aeronyx-node-tls`).
    ///
    /// [NODE-TLS-BINDING 2026-10-10 by Claude] A peer that sees this signed
    /// token connects to an IP-literal endpoint only over identity-pinned TLS
    /// and never falls back to plain HTTP, so a network attacker cannot strip
    /// the upgrade. Nodes without it keep plain HTTP.
    IdentityBoundTlsV1,
}

impl NodeProtocolFeature {
    /// Features understood by this binary, in stable negotiation order.
    pub const ALL: [Self; 18] = [
        Self::BlindRelayFailureReceiptV1,
        Self::BlindRelaySuccessReceiptV1,
        Self::PurposeBoundDeliveryReceiptV2,
        Self::DirectPeerRelayAuthV2,
        Self::DirectPeerRelayReceiptV2,
        Self::DirectPeerRelayTargetBindingV3,
        Self::OnionReplyV1,
        Self::OnionSourceSealedTerminalProofV1,
        Self::OnionBlindVaultLargePullV1,
        Self::OnionBlindLeaseAdmissionV1,
        Self::OnionBlindVaultPutReceiptV1,
        Self::OnionBlindVaultLeaseRetireV1,
        Self::OnionBlindVaultLeaseRenewalV1,
        Self::OnionBlindVaultLeaseStatusV1,
        Self::OnionBlindVaultLeaseInventoryV1,
        Self::OnionBlindVaultEncryptedFailureV1,
        Self::AnonymousMailboxV1,
        Self::IdentityBoundTlsV1,
    ];

    /// Exact SemVer build-metadata identifier used on the signed wire.
    #[must_use]
    pub const fn semver_build_token(self) -> &'static str {
        match self {
            Self::BlindRelayFailureReceiptV1 => "anpf1-brfr1",
            Self::BlindRelaySuccessReceiptV1 => "anpf1-brsr1",
            Self::PurposeBoundDeliveryReceiptV2 => "anpf1-pbdr2",
            Self::DirectPeerRelayAuthV2 => "anpf1-dpra2",
            Self::DirectPeerRelayReceiptV2 => "anpf1-dprr2",
            Self::DirectPeerRelayTargetBindingV3 => "anpf1-dprtb3",
            Self::OnionReplyV1 => "anpf1-or1",
            Self::OnionSourceSealedTerminalProofV1 => "anpf1-osstp1",
            Self::OnionBlindVaultLargePullV1 => "anpf1-oblp1",
            Self::OnionBlindLeaseAdmissionV1 => "anpf1-obla1",
            Self::OnionBlindVaultPutReceiptV1 => "anpf1-obpr1",
            Self::OnionBlindVaultLeaseRetireV1 => "anpf1-oblr1",
            Self::OnionBlindVaultLeaseRenewalV1 => "anpf1-oblw1",
            Self::OnionBlindVaultLeaseStatusV1 => "anpf1-obls1",
            Self::OnionBlindVaultLeaseInventoryV1 => "anpf1-obli1",
            Self::OnionBlindVaultEncryptedFailureV1 => "anpf1-obef1",
            Self::AnonymousMailboxV1 => "anpf1-amb1",
            Self::IdentityBoundTlsV1 => "anpf1-ibt1",
        }
    }
}

#[cfg(test)]
mod tests;
