//! ============================================
//! File: crates/aeronyx-server/src/services/agent/process_stats.rs
//! Path: aeronyx-server/src/services/agent/process_stats.rs
//! ============================================
//!
//! ## Creation Reason
//! Extracted from `agent_manager.rs` to isolate Linux-specific /proc
//! filesystem reading for process CPU and memory statistics.
//!
//! ## Main Functionality
//! - `ProcessStats::collect()`: Aggregates CPU and memory across all
//!   OpenClaw-related processes (main + Node.js worker children)
//!
//! ## Strategy
//! 1. Use `pgrep -f "openclaw"` to find all related PIDs
//! 2. Sum VmRSS from /proc/{pid}/status for memory
//! 3. Sum utime+stime from /proc/{pid}/stat for CPU ticks
//! 4. Calculate CPU% using earliest start time and system uptime
//!
//! ## ⚠️ Important Note for Next Developer
//! - Linux-only: all functions return (None, None) on non-Linux
//! - /proc/stat field indices are relative to after the "(comm)" field
//!   which can contain spaces — use rfind(')') to skip past it
//! - CLK_TCK is hardcoded as 100 (standard for modern Linux kernels)
//! - pgrep may match the aeronyx-server process itself if it has
//!   "openclaw" in its command line — this is acceptable for aggregate stats
//!
//! ## Last Modified
//! v1.4.0 - 🌟 Initial creation (extracted from agent_manager.rs)
//! ============================================

use tokio::process::Command as TokioCommand;

/// Collects CPU and memory usage for the OpenClaw process tree.
pub struct ProcessStats;

impl ProcessStats {
    /// Collects aggregate (cpu_percent, memory_mb) for all OpenClaw processes.
    ///
    /// Returns (None, None) on non-Linux or if no processes found.
    #[cfg(target_os = "linux")]
    pub async fn collect(primary_pid: u32) -> (Option<f32>, Option<u64>) {
        let pids = Self::find_all_pids(primary_pid).await;

        if pids.is_empty() {
            return (None, None);
        }

        let mut total_memory_kb = 0u64;
        let mut total_cpu_ticks = 0u64;
        let mut min_starttime = u64::MAX;
        let mut found_any = false;

        for pid in &pids {
            if let Some(rss_kb) = Self::read_rss(*pid).await {
                total_memory_kb += rss_kb;
                found_any = true;
            }

            if let Some((ticks, starttime)) = Self::read_cpu_ticks(*pid).await {
                total_cpu_ticks += ticks;
                if starttime < min_starttime {
                    min_starttime = starttime;
                }
            }
        }

        if !found_any {
            return (None, None);
        }

        let memory_mb = Some(total_memory_kb / 1024);

        let cpu_pct = if min_starttime < u64::MAX {
            Self::calculate_cpu_percent(total_cpu_ticks, min_starttime).await
        } else {
            None
        };

        (cpu_pct, memory_mb)
    }

    #[cfg(not(target_os = "linux"))]
    pub async fn collect(_primary_pid: u32) -> (Option<f32>, Option<u64>) {
        (None, None)
    }

    /// Finds all PIDs related to OpenClaw (main + child processes).
    #[cfg(target_os = "linux")]
    async fn find_all_pids(primary_pid: u32) -> Vec<u32> {
        let output = TokioCommand::new("pgrep")
            .args(&["-f", "openclaw"])
            .output()
            .await;

        let mut pids: Vec<u32> = match output {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .filter_map(|l| l.trim().parse::<u32>().ok())
                    .collect()
            }
            _ => Vec::new(),
        };

        if !pids.contains(&primary_pid) {
            if std::path::Path::new(&format!("/proc/{}", primary_pid)).exists() {
                pids.push(primary_pid);
            }
        }

        pids
    }

    /// Reads VmRSS (resident memory) in KB from /proc/{pid}/status.
    #[cfg(target_os = "linux")]
    async fn read_rss(pid: u32) -> Option<u64> {
        let path = format!("/proc/{}/status", pid);
        let content = tokio::fs::read_to_string(&path).await.ok()?;

        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                return line.split_whitespace().nth(1)?.parse::<u64>().ok();
            }
        }
        None
    }

    /// Reads utime + stime (CPU ticks) and starttime from /proc/{pid}/stat.
    #[cfg(target_os = "linux")]
    async fn read_cpu_ticks(pid: u32) -> Option<(u64, u64)> {
        let path = format!("/proc/{}/stat", pid);
        let content = tokio::fs::read_to_string(&path).await.ok()?;

        // Skip past "(comm)" which may contain spaces
        let after_comm = content.rfind(')')? + 2;
        let fields: Vec<&str> = content[after_comm..].split_whitespace().collect();

        if fields.len() < 20 {
            return None;
        }

        // utime=field[11], stime=field[12], starttime=field[19]
        let utime: u64 = fields[11].parse().ok()?;
        let stime: u64 = fields[12].parse().ok()?;
        let starttime: u64 = fields[19].parse().ok()?;

        Some((utime + stime, starttime))
    }

    /// Calculates CPU percentage from total ticks and earliest start time.
    #[cfg(target_os = "linux")]
    async fn calculate_cpu_percent(total_ticks: u64, starttime: u64) -> Option<f32> {
        let uptime_content = tokio::fs::read_to_string("/proc/uptime").await.ok()?;
        let uptime_secs: f64 = uptime_content
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;

        let clk_tck: f64 = 100.0; // sysconf(_SC_CLK_TCK)
        let total_time = total_ticks as f64 / clk_tck;
        let process_uptime = uptime_secs - (starttime as f64 / clk_tck);

        if process_uptime > 0.0 {
            Some((total_time / process_uptime * 100.0) as f32)
        } else {
            None
        }
    }
}
