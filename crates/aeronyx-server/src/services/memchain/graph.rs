// ============================================
// File: crates/aeronyx-server/src/services/memchain/graph.rs
// ============================================
//! # Memory Co-occurrence Graph (D8)
//!
//! Maintains a weighted graph of memory co-occurrences: when memories
//! appear together in a recall result, edges are created/strengthened.
//!
//! Used for MVF feature φ₅ (graph centrality) — records that frequently
//! co-occur with other active memories are more "central" and get higher
//! recall scores.
//!
//! ## Storage: `memory_edges` table in SQLite
//! Edges are bidirectional (A→B and B→A both stored).
//! Weight increases by 1.0 on each co-occurrence.
//!
//! ## Performance
//! Only the top 5 recalled memories form edges (capped to avoid O(n²)).
//! All operations use INSERT ... ON CONFLICT DO UPDATE for idempotency.

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use tracing::{debug, error};

/// Update co-occurrence edges for a set of recalled memory IDs.
///
/// Takes the top `max_pairs` (default 5) memory IDs and creates/updates
/// bidirectional edges between all pairs.
///
/// # Arguments
/// * `conn` - SQLite connection (caller holds the Mutex)
/// * `memory_ids` - Record IDs of recalled memories (truncated to 5)
/// * `now` - Current Unix timestamp
pub fn update_cooccurrence(
    conn: &Connection,
    memory_ids: &[[u8; 32]],
    now: i64,
) {
    // Cap to top 5 to avoid O(n²) explosion
    let ids: Vec<&[u8; 32]> = memory_ids.iter().take(5).collect();

    if ids.len() < 2 {
        return; // Need at least 2 memories to form an edge
    }

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let a = ids[i].as_slice();
            let b = ids[j].as_slice();

            // Bidirectional: A→B and B→A
            let _ = conn.execute(
                "INSERT INTO memory_edges (source_id, target_id, edge_type, weight, created_at)
                 VALUES (?1, ?2, 'co_occurred', 1.0, ?3)
                 ON CONFLICT(source_id, target_id)
                 DO UPDATE SET weight = weight + 1.0",
                params![a, b, now],
            );
            let _ = conn.execute(
                "INSERT INTO memory_edges (source_id, target_id, edge_type, weight, created_at)
                 VALUES (?1, ?2, 'co_occurred', 1.0, ?3)
                 ON CONFLICT(source_id, target_id)
                 DO UPDATE SET weight = weight + 1.0",
                params![b, a, now],
            );
        }
    }

    debug!(
        pairs = ids.len() * (ids.len() - 1) / 2,
        "[GRAPH] Co-occurrence edges updated"
    );
}

/// Get neighbors of a memory node, sorted by edge weight descending.
pub fn get_neighbors(
    conn: &Connection,
    memory_id: &[u8; 32],
    limit: usize,
) -> Vec<([u8; 32], f32)> {
    let mut stmt = match conn.prepare(
        "SELECT target_id, weight FROM memory_edges
         WHERE source_id = ?1
         ORDER BY weight DESC
         LIMIT ?2"
    ) {
        Ok(s) => s,
        Err(e) => {
            error!(error = %e, "[GRAPH] get_neighbors prepare failed");
            return Vec::new();
        }
    };

    stmt.query_map(params![memory_id.as_slice(), limit as i64], |row| {
        let blob: Vec<u8> = row.get(0)?;
        let weight: f32 = row.get(1)?;
        let mut id = [0u8; 32];
        if blob.len() == 32 {
            id.copy_from_slice(&blob);
        }
        Ok((id, weight))
    })
    .map(|rows| rows.filter_map(|r| r.ok()).collect())
    .unwrap_or_default()
}

/// Get the degree (number of edges) for a memory node.
pub fn get_degree(conn: &Connection, memory_id: &[u8; 32]) -> u32 {
    conn.query_row(
        "SELECT COUNT(*) FROM memory_edges WHERE source_id = ?1",
        params![memory_id.as_slice()],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0) as u32
}

/// Get the maximum degree among all memories for a given owner.
///
/// Used to normalize φ₅ = degree(m) / max_degree to [0, 1].
pub fn get_max_degree(conn: &Connection, owner: &[u8; 32]) -> u32 {
    conn.query_row(
        "SELECT COALESCE(MAX(cnt), 0) FROM (
            SELECT COUNT(*) as cnt FROM memory_edges e
            JOIN records r ON e.source_id = r.record_id
            WHERE r.owner = ?1
            GROUP BY e.source_id
        )",
        params![owner.as_slice()],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE records (record_id BLOB PRIMARY KEY, owner BLOB);
             CREATE TABLE memory_edges (
                 source_id BLOB NOT NULL, target_id BLOB NOT NULL,
                 edge_type TEXT NOT NULL DEFAULT 'co_occurred',
                 weight REAL NOT NULL DEFAULT 1.0,
                 created_at INTEGER NOT NULL,
                 PRIMARY KEY (source_id, target_id)
             );"
        ).unwrap();
        conn
    }

    #[test]
    fn test_cooccurrence_basic() {
        let conn = setup_db();
        let a = [1u8; 32];
        let b = [2u8; 32];
        let c = [3u8; 32];

        update_cooccurrence(&conn, &[a, b, c], 1000);

        // Should have 6 edges (3 pairs × 2 directions)
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_edges", [], |r| r.get(0)
        ).unwrap();
        assert_eq!(count, 6);
    }

    #[test]
    fn test_weight_accumulation() {
        let conn = setup_db();
        let a = [1u8; 32];
        let b = [2u8; 32];

        update_cooccurrence(&conn, &[a, b], 1000);
        update_cooccurrence(&conn, &[a, b], 2000);

        let w: f32 = conn.query_row(
            "SELECT weight FROM memory_edges WHERE source_id = ?1 AND target_id = ?2",
            params![a.as_slice(), b.as_slice()],
            |r| r.get(0),
        ).unwrap();
        assert!((w - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_degree() {
        let conn = setup_db();
        let a = [1u8; 32];
        let b = [2u8; 32];
        let c = [3u8; 32];

        update_cooccurrence(&conn, &[a, b, c], 1000);

        assert_eq!(get_degree(&conn, &a), 2); // A→B, A→C
        assert_eq!(get_degree(&conn, &b), 2);
    }

    #[test]
    fn test_neighbors() {
        let conn = setup_db();
        let a = [1u8; 32];
        let b = [2u8; 32];
        let c = [3u8; 32];

        update_cooccurrence(&conn, &[a, b, c], 1000);
        // Make A-B co-occur again (higher weight)
        update_cooccurrence(&conn, &[a, b], 2000);

        let neighbors = get_neighbors(&conn, &a, 5);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, b); // B has higher weight (2.0)
        assert!((neighbors[0].1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_cap_at_5() {
        let conn = setup_db();
        let ids: Vec<[u8; 32]> = (0..10u8).map(|i| [i; 32]).collect();

        update_cooccurrence(&conn, &ids, 1000);

        // Only first 5 should form edges: C(5,2) × 2 = 20
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_edges", [], |r| r.get(0)
        ).unwrap();
        assert_eq!(count, 20);
    }
}
