// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn prefix_to_netmask_supports_vpn_pool_expansion() {
    assert_eq!(prefix_to_netmask(22), Ipv4Addr::new(255, 255, 252, 0));
    assert_eq!(prefix_to_netmask(24), Ipv4Addr::new(255, 255, 255, 0));
    assert_eq!(prefix_to_netmask(0), Ipv4Addr::new(0, 0, 0, 0));
}
