//! Lock-free bucketed transposition table.
//!
//! Layout: `TT_SIZE` buckets × `TT_BUCKET_WIDTH` entries; each entry is TWO
//! AtomicU64 slots — hash and payload. Payload is published BEFORE the hash
//! (Release) and re-verified after read (Acquire) so concurrent writers
//! (Lazy SMP) never yield a torn entry: a reader that sees the hash is
//! guaranteed to see at least that data snapshot.

use super::params;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TTEntry {
    pub score: i32,
    pub depth: i8,
    /// 0=EXACT, 1=LOWERBOUND (≥beta), 2=UPPERBOUND (≤alpha)
    pub flag: u8,
    pub generation: u8,
    pub best_move: u32,
    /// Position is in check (cached to avoid re-checking). Always false in
    /// Taikyoku (no check exists — SPEC §7.3); kept for chess-style engines.
    pub in_check: bool,
}

static TT: OnceLock<Vec<AtomicU64>> = OnceLock::new();

pub(crate) static TT_GEN: AtomicU64 = AtomicU64::new(1);

#[inline]
fn tt() -> &'static Vec<AtomicU64> {
    TT.get_or_init(|| {
        (0..params::TT_SIZE * params::TT_BUCKET_WIDTH * 2)
            .map(|_| AtomicU64::new(0))
            .collect()
    })
}

#[inline]
fn tt_index(hash: u64) -> usize {
    ((hash as usize) & (params::TT_SIZE - 1)) * params::TT_BUCKET_WIDTH * 2
}

const TT_MOVE_MASK: u64 = (1 << 25) - 1;
const TT_CHECK_MASK: u64 = 1 << 25;

#[inline]
pub(crate) fn tt_pack(entry: &TTEntry, gen: u8) -> u64 {
    let sc = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv = entry.best_move & (TT_MOVE_MASK as u32);
    ((sc as u64) << 48)
        | ((entry.depth as u64 & 0x7F) << 41)
        | ((entry.flag as u64 & 0x03) << 39)
        | ((gen as u64) << 31)
        | if entry.in_check { TT_CHECK_MASK } else { 0 }
        | (mv as u64)
}

#[inline]
pub(crate) fn tt_unpack(packed: u64) -> TTEntry {
    TTEntry {
        score: ((packed >> 48) & 0xFFFF) as u16 as i16 as i32,
        depth: ((packed >> 41) & 0x7F) as i8,
        flag: ((packed >> 39) & 0x03) as u8,
        generation: ((packed >> 31) & 0xFF) as u8,
        best_move: (packed & TT_MOVE_MASK) as u32,
        in_check: (packed & TT_CHECK_MASK) != 0,
    }
}

/// Bump on every new `search()` call: entries from older generations lose
/// the replace race, achieving aging without clearing the table.
#[inline]
pub(crate) fn tt_gen() -> u8 {
    (TT_GEN.load(Ordering::Relaxed) & 0xFF) as u8
}

/// Advance the generation counter (called at the start of a new search).
pub(crate) fn tt_new_generation() {
    TT_GEN.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn tt_probe(hash: u64) -> Option<TTEntry> {
    let base = tt_index(hash);
    let t = tt();
    for i in 0..params::TT_BUCKET_WIDTH {
        let idx = base + i * 2;
        // ── Race-safe read ────────────────────────────────────────
        // Read hash with Acquire, snapshot the data, then RE-VERIFY the
        // hash: if it changed, the slot was overwritten mid-read and the
        // entry is treated as a miss.
        let stored = t[idx].load(Ordering::Acquire);
        if stored == hash {
            let entry = tt_unpack(t[idx + 1].load(Ordering::Acquire));
            if t[idx].load(Ordering::Acquire) != stored {
                continue; // slot overwritten mid-read — try next bucket slot
            }
            if entry.depth >= 0 { return Some(entry); }
        }
    }
    None
}

pub(crate) fn tt_store(hash: u64, entry: TTEntry) {
    let base = tt_index(hash);
    let t = tt();
    let gen = tt_gen();
    let mut replace_idx = 0;
    let mut replace_score = i32::MAX;

    for i in 0..params::TT_BUCKET_WIDTH {
        let idx = base + i * 2;
        let old_hash = t[idx].load(Ordering::Relaxed);
        if old_hash == 0 {
            replace_idx = idx;
            break;
        }
        let old = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
        let score = ((old.depth as i32) << 16) - (gen.wrapping_sub(old.generation) as i32);
        if score < replace_score {
            replace_score = score;
            replace_idx = idx;
        }
    }

    // ── Race-safe write ─────────────────────────────────────────
    // Publish data BEFORE the hash, both with Release: a reader that observes
    // the hash (Acquire) is guaranteed to see at least this data snapshot,
    // never a torn mix of old and new entries.
    t[replace_idx + 1].store(tt_pack(&entry, gen), Ordering::Release);
    t[replace_idx].store(hash, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tt_pack_unpack_roundtrip() {
        let entry = TTEntry {
            score: 1234,
            depth: 8,
            flag: 1,
            generation: 0, // ignored by tt_pack — the gen parameter wins
            best_move: 0x1234,
            in_check: true,
        };
        let out = tt_unpack(tt_pack(&entry, 7));
        assert_eq!(out.score, 1234);
        assert_eq!(out.depth, 8);
        assert_eq!(out.flag, 1);
        assert_eq!(out.generation, 7);
        assert_eq!(out.best_move, 0x1234);
        assert!(out.in_check);
        // Score clamping at the ±32000 boundary.
        let mut big = entry;
        big.score = 99_000;
        assert_eq!(tt_unpack(tt_pack(&big, 7)).score, 32_000);
        big.score = -99_000;
        assert_eq!(tt_unpack(tt_pack(&big, 7)).score, -32_000);
    }
}
