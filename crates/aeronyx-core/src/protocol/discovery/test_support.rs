// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/test_support.rs
// ============================================
//! # Discovery test fixtures
//!
//! Shared fixtures for the per-module discovery test suites, moved from the
//! former `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use crate::crypto::IdentityKeyPair;

use super::{NodeCapability, NodeCapacity, NodeDescriptor, NodePolicy};

pub(super) fn descriptor_for(kp: &IdentityKeyPair) -> NodeDescriptor {
    let mut descriptor = NodeDescriptor::new(
        kp.public_key_bytes(),
        7,
        1_700_000_000,
        1_700_003_600,
        "test",
    );
    descriptor.public_endpoint = Some("node.example:443".to_string());
    descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
    descriptor.capacity = NodeCapacity {
        max_sessions: 256,
        max_bps: Some(1_000_000_000),
        max_pps: Some(250_000),
    };
    descriptor.policy = NodePolicy {
        allows_public_exit: false,
        public_discovery: true,
        region: Some("test-region".to_string()),
    };
    descriptor
}
