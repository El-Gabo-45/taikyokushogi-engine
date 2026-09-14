//! Search heuristics shared across nodes: killer moves, butterfly history
//! and counter moves.
//!
//! All state is global and lock-free (atomics) so Lazy SMP worker threads
//! share statistics without locks. All tables are lazy-initialized and
//! zero-initialized by the OS on first touch.

use super::params;
use super::tt;
use crate::types::NUM_SQUARES;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::OnceLock;
// ── Killer moves ────────────────────────────────────────────────
static KILLERS: OnceLock<Vec<AtomicU64>> = OnceLock::new();

#[inline]
fn killers() -> &'static Vec<AtomicU64> {
    KILLERS.get_or_init(|| (0..256).map(|_| AtomicU64::new(0)).collect())
}

/// Record `mv` as the new first killer at `depth` (old first becomes second).
pub(crate) fn killer_store(depth: u32, mv: u32) {
    let d = depth.min(params::KILLER_MAX_PLY) as usize;
    let slot = &killers()[d];
    // Atomic read-modify-write: a plain load+store is not atomic and, under
    // Lazy SMP, two threads can overwrite each other and drop the old
    // first killer. fetch_update makes the swap race-free.
    let _ = slot.fetch_update(Ordering::Release, Ordering::Relaxed, |cur| {
        let mv0 = cur as u32;
        if mv == mv0 { None } else { Some(mv as u64 | ((mv0 as u64) << 32)) }
    });
}

#[inline]
pub(crate) fn killer_score(depth: u32, mv: u32) -> i32 {
    let d = depth.min(params::KILLER_MAX_PLY) as usize;
    let p = killers()[d].load(Ordering::Relaxed);
    let mv0 = p as u32;
    let mv1 = (p >> 32) as u32;
    if mv == mv0 { params::KILLER1_SCORE } else if mv == mv1 { params::KILLER2_SCORE } else { 0 }
}

// ── Butterfly history ───────────────────────────────────────────
// Indexed by the EXACT (from, to) square pair:
// [2][1296][1296] i32 ≈ 6.7 MB, allocated lazily and zero-initialized
// by the OS on first touch. The previous 256×256 modulo table gave every
// bucket ~25 distinct (from, to) pairs on a 36×36 board (25× signal
// dilution); exact indexing removes the collateral noise entirely.
const HIST_FROM: usize = NUM_SQUARES;
static HIST: OnceLock<Box<[AtomicI32]>> = OnceLock::new();

#[inline]
fn history() -> &'static [AtomicI32] {
    HIST.get_or_init(|| {
        let mut v = Vec::with_capacity(2 * HIST_FROM * HIST_FROM);
        v.resize_with(2 * HIST_FROM * HIST_FROM, || AtomicI32::new(0));
        v.into_boxed_slice()
    })
}

#[inline]
fn history_idx(from: usize, to: usize, side: u8) -> usize {
    side as usize * HIST_FROM * HIST_FROM + from * HIST_FROM + to
}

pub(crate) fn history_store(from: usize, to: usize, depth: u32, side: u8) {
    let idx = history_idx(from, to, side);
    let bonus = (depth * depth).min(params::HIST_BONUS_CAP as u32) as i32;
    history()[idx].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_add(bonus).min(params::HIST_MAX))
    }).ok();
}

/// History gravity: penalize quiets that failed to cause a cutoff, so next
/// time the cutting move (and similar ones) are tried first.
pub(crate) fn history_malus(from: usize, to: usize, depth: u32, side: u8) {
    let idx = history_idx(from, to, side);
    let malus = (depth * depth).min(params::HIST_BONUS_CAP as u32) as i32;
    history()[idx].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(malus).max(0))
    }).ok();
}

#[inline]
pub(crate) fn history_score(from: usize, to: usize, side: u8) -> i32 {
    history()[history_idx(from, to, side)].load(Ordering::Relaxed)
}

// ── Counter moves ───────────────────────────────────────────────
static COUNTER: OnceLock<Vec<AtomicU64>> = OnceLock::new();

#[inline]
fn counter() -> &'static Vec<AtomicU64> {
    COUNTER.get_or_init(|| (0..65536).map(|_| AtomicU64::new(0)).collect())
}

pub(crate) fn counter_store(prev: u32, mv: u32) {
    counter()[(prev as usize) & 0xFFFF].store(mv as u64, Ordering::Relaxed);
}

#[inline]
pub(crate) fn counter_score(prev: u32, mv: u32) -> i32 {
    if prev == 0 { return 0; }
    let stored = counter()[(prev as usize) & 0xFFFF].load(Ordering::Relaxed) as u32;
    if stored == mv { params::COUNTER_SCORE } else { 0 }
}

/// Zero all heuristics and age the TT (used between games / by tests).
#[allow(dead_code)] // used by tests; also the natural reset hook for a future game loop
pub(crate) fn clear() {
    if let Some(h) = HIST.get() { for cell in h.iter() { cell.store(0, Ordering::Relaxed); } }
    if let Some(c) = COUNTER.get() { for cell in c { cell.store(0, Ordering::Relaxed); } }
    if let Some(k) = KILLERS.get() { for cell in k { cell.store(0, Ordering::Relaxed); } }
    tt::tt_new_generation();
}
