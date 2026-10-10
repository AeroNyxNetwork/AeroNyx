// ============================================
// File: crates/aeronyx-server/src/server/systemd_notifier.rs
// ============================================
//! # systemd readiness notifier
//!
//! Owns the process-lifetime bridge to systemd's datagram notification
//! protocol (`Type=notify`).
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `server.rs`; bodies unchanged.

use std::ffi::OsString;

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_os = "linux")]
use std::path::Path;

#[cfg(target_os = "linux")]
use nix::sys::socket::{
    sendto, socket, AddressFamily, MsgFlags, SockFlag, SockProtocol, SockType, UnixAddr,
};

use crate::error::Result;
#[cfg(target_os = "linux")]
use crate::error::ServerError;

/// Process-lifetime bridge to systemd's datagram notification protocol.
///
/// [STARTUP-READINESS 2026-07-29 by Codex] `Type=notify` must reflect the
/// actual listener barrier, not process creation. Manual/non-systemd starts
/// remain backward compatible because an absent `NOTIFY_SOCKET` is a no-op.
/// The implementation uses the existing `nix` dependency and supports both
/// filesystem and Linux abstract namespace notify sockets.
#[derive(Clone, Debug, Default)]
pub(super) struct SystemdNotifier {
    socket: Option<OsString>,
}

impl SystemdNotifier {
    pub(super) fn from_environment() -> Self {
        Self {
            socket: std::env::var_os("NOTIFY_SOCKET"),
        }
    }

    #[cfg(test)]
    pub(super) fn from_socket(socket: Option<OsString>) -> Self {
        Self { socket }
    }

    pub(super) fn status(&self, status: &str) -> Result<bool> {
        self.send(&format!("STATUS={}", Self::sanitize_status(status)))
    }

    pub(super) fn ready(&self, status: &str) -> Result<bool> {
        self.send(&format!(
            "READY=1\nSTATUS={}",
            Self::sanitize_status(status)
        ))
    }

    pub(super) fn stopping(&self, status: &str) -> Result<bool> {
        self.send(&format!(
            "STOPPING=1\nSTATUS={}",
            Self::sanitize_status(status)
        ))
    }

    fn sanitize_status(status: &str) -> String {
        status
            .chars()
            .map(|character| match character {
                '\r' | '\n' => ' ',
                other => other,
            })
            .collect()
    }

    fn send(&self, payload: &str) -> Result<bool> {
        let Some(socket_name) = self.socket.as_ref() else {
            return Ok(false);
        };

        #[cfg(target_os = "linux")]
        {
            let socket_bytes = socket_name.as_os_str().as_bytes();
            if socket_bytes.is_empty() {
                return Err(ServerError::startup_failed(
                    "systemd NOTIFY_SOCKET is empty",
                ));
            }

            let address = if let Some(abstract_name) = socket_bytes.strip_prefix(b"@") {
                if abstract_name.is_empty() {
                    return Err(ServerError::startup_failed(
                        "systemd abstract NOTIFY_SOCKET name is empty",
                    ));
                }
                UnixAddr::new_abstract(abstract_name).map_err(|error| {
                    ServerError::startup_failed(format!(
                        "invalid systemd abstract NOTIFY_SOCKET: {error}"
                    ))
                })?
            } else {
                UnixAddr::new(Path::new(socket_name)).map_err(|error| {
                    ServerError::startup_failed(format!(
                        "invalid systemd filesystem NOTIFY_SOCKET: {error}"
                    ))
                })?
            };
            let datagram = socket(
                AddressFamily::Unix,
                SockType::Datagram,
                SockFlag::SOCK_CLOEXEC,
                None::<SockProtocol>,
            )
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "failed to create systemd notification socket: {error}"
                ))
            })?;
            let sent = sendto(
                datagram.as_raw_fd(),
                payload.as_bytes(),
                &address,
                MsgFlags::empty(),
            )
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "failed to send systemd readiness notification: {error}"
                ))
            })?;
            if sent != payload.len() {
                return Err(ServerError::startup_failed(format!(
                    "partial systemd readiness notification: sent {sent} of {} bytes",
                    payload.len()
                )));
            }
            Ok(true)
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = payload;
            Ok(false)
        }
    }
}
