//! Tunable search parameters — the single place where every "magic number"
//! of the search lives. Tuning sessions and A/B experiments should only ever
//! touch this file.
//!
//! Grouped by subsystem. Pure helper functions encode margin *formulas* so
//! the gating logic in `pvs`/`qsearch` stays declarative.

use crate::eval::MATE_SCORE;

// ── Move-ordering score ladder ──────────────────────────────────
// Priority: TT move > root hint > tactical (MVV-LVA) > killers > history+counter.
pub const ROOT_HINT_SCORE: i32 = 3_000_000; // previous-iteration best-move bonus
pub const TT_MOVE_SCORE: i32 = 2_000_000;   // hash move always first
pub const TACTICAL_BASE_SCORE: i32 = 1_000_000;
pub const KILLER1_SCORE: i32 = 90_000;
pub const KILLER2_SCORE: i32 = 80_000;
pub const COUNTER_SCORE: i32 = 70_000;
pub const PROMO_ORDER_BONUS: i32 = 5000;
/// Score threshold below which a capture is skipped in quiescence.
pub const QS_PREFILTER: i32 = -300;

// ── Clock polling ───────────────────────────────────────────────
// Nodes are expensive (~150-200us) on a 1296-square board, so the deadline
// is polled every few dozen nodes; the bitwise AND keeps the check free.
pub const NODES_PER_CLOCK_CHECK: u64 = 63;
pub const NODES_PER_CLOCK_CHECK_QS: u64 = 127;

// ── Mate-score safety gate ──────────────────────────────────────
// Pruning heuristics are disabled when a mate score may be in play.
#[inline]
pub fn mate_gate_ok(window_edge: i32) -> bool {
    window_edge > -MATE_SCORE + 100
}

// ── Shallow-eval pruning margins ────────────────────────────────
// Each returns Some(margin) only at the depths where the technique applies.

/// Razoring: `static_eval + margin <= alpha` → prune low.
pub fn razoring_margin(d: u32) -> Option<i32> {
    match d { 0 => Some(400), 1 => Some(600), 2 => Some(900), _ => None }
}

/// Reverse futility (both directions at this node):
/// `static_eval - margin >= beta` → prune high; `static_eval + margin <= alpha` → prune low.
pub fn rfp_margin(d: u32) -> Option<i32> {
    if d <= 3 { Some(150 + 250 * d as i32) } else { None }
}

/// Futility: `static_eval + margin <= alpha` → skip shallow quiet nodes.
pub fn futility_margin(d: u32) -> Option<i32> {
    match d { 0 => Some(80), 1 => Some(160), 2 => Some(240), _ => None }
}

/// ProbCut: `static_eval + margin <= alpha` → statistical prune.
pub fn probcut_margin(d: u32) -> Option<i32> {
    if d >= 4 { Some(500 + 300 * d as i32) } else { None }
}

// ── Null-move pruning ───────────────────────────────────────────
pub const NMP_MIN_DEPTH: u32 = 3;
pub const NMP_NO_PROGRESS_LIMIT: u32 = 100;
pub const NMP_MIN_SIDE_PIECES: usize = 4; // piece_count > 3
#[inline]
pub fn nmp_reduction(d: u32) -> u32 {
    if d >= 6 { 3 } else { 2 }
}

// ── Capture beam (scaled by real branching) ─────────────────────
pub fn capture_beam_base(d: u32) -> u32 {
    if d <= 2 { 24 } else if d <= 4 { 12 } else { 6 }
}
pub const CAPTURE_BEAM_MAX: usize = 96;
pub const BEAM_BRANCH_SCALE: usize = 128;

// ── Late Move Pruning (quiets) ──────────────────────────────────
pub const LMP_BRANCH_SCALE: usize = 128;
pub const LMP_MAX: usize = 64;
#[inline]
pub fn lmp_base(d: u32, improving: bool) -> usize {
    (((3 + d * d) as i32).max(3) as usize) / if improving { 1 } else { 2 }
}

// ── Futility at move level ──────────────────────────────────────
pub const QUIET_FUTILITY_MAX_DEPTH: u32 = 2;
#[inline]
pub fn quiet_futility_margin(d: u32) -> i32 {
    120 * d as i32 + 60
}
pub const CAPTURE_FUTILITY_MARGIN: i32 = 140;

// ── Late Move Reductions ────────────────────────────────────────
pub const LMR_MIN_MOVE_INDEX: usize = 3; // cur_move >= 3
pub const LMR_MIN_DEPTH: u32 = 3;
pub const TT_MOVE_EXT_MIN_DEPTH: u32 = 6;
#[inline]
pub fn lmr_base(cur_move: usize) -> u32 {
    (cur_move as u32 / 3).min(3)
}
#[inline]
pub fn lmr_depth_factor(d: u32) -> u32 {
    (d / 3).min(2)
}
/// A quiet with a strong history is statistically good — soften its reduction.
#[inline]
pub fn lmr_history_soften(history: i32) -> u32 {
    (history / 8_000).min(2) as u32
}

// ── Quiescence ──────────────────────────────────────────────────
pub const MAX_QDEPTH: u32 = 6;
/// Below this piece count, leaves drop into quiescence; above it, the O(1)
/// static eval is used (a full QS at 800 pieces is prohibitive).
pub const QS_LEAF_PIECE_LIMIT: usize = 200;

// ── Internal Iterative Deepening ────────────────────────────────
pub const IID_MIN_DEPTH: u32 = 4;

// ── Root aspiration windows ─────────────────────────────────────
#[inline]
pub fn aspiration_window(depth: u32) -> i32 {
    if depth <= 4 { 64 } else { 200 }
}
/// Window growth factor after the first aspiration failure (×8).
pub const ASPIRATION_WINDOW_GROW: i32 = 8;
/// After this many failed re-searches, keep the bounded result.
pub const ASPIRATION_MAX_FAILS: u32 = 2;
/// Root-node first-move window refinement offset around the best score.
pub const ROOT_WINDOW_REFINE: i32 = 50;
/// Root aspiration is only refined once the best score beats alpha clearly.
pub const ROOT_WINDOW_REFINE_GATE: i32 = 100;

// ── Iterative-deepening time management ─────────────────────────
/// Predicted cost multiplier of the next iteration (branching factor).
pub const NEXT_ITER_COST_FACTOR: u64 = 4;

// ── Transposition table ─────────────────────────────────────────
pub const TT_SIZE: usize = 1 << 22;
pub const TT_BUCKET_WIDTH: usize = 4;

// ── History heuristics ──────────────────────────────────────────
pub const HIST_BONUS_CAP: i32 = 400;   // (depth*depth).min(400)
pub const HIST_MAX: i32 = 32_767;
/// Killer slots are stored per ply; depth is clamped below this.
pub const KILLER_MAX_PLY: u32 = 127;
/// Quiets recorded per node for history-malus on beta cutoff.
pub const QUIETS_TRACKED_MAX: usize = 64;
