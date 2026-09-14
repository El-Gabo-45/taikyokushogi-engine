//! Move scoring and packing for move ordering.
//!
//! Priority ladder: 1) TT/hash move, 2) root hint, 3) tactical MVV-LVA,
//! 4) killers, 5) counter moves, 6) butterfly history.

use super::heuristics;
use super::params;
use crate::board::Board;
use crate::pieces;
use crate::types::*;
use std::sync::OnceLock;

/// Material values indexed by piece type (0 = empty), computed once.
pub(crate) fn piece_vals() -> &'static [i32; 512] {
    static V: OnceLock<[i32; 512]> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = [0i32; 512];
        for pt in 1..=301u16 { v[pt as usize] = pieces::value(pt); }
        v
    })
}

/// Compact key for TT/killer/history tables: from | to<<11 | promo<<24.
#[inline]
pub(crate) fn m_pack(m: &Move) -> u32 {
    (m.from_sq as u32) | ((m.to_sq as u32) << 12) | (if m.promotion { 1 << 24 } else { 0 })
}

/// A capture/igui/mid-capture/range-capture/promotion — searched in stage 1.
#[inline]
pub(crate) fn is_tactical(m: &Move) -> bool {
    m.captured_piece != 0 || m.mid_piece != 0 || m.promotion || m.is_igui
        || (m.range_cap && m.caps_value != 0)
}

/// Priority: 1) Hash move (TT)  2) MVV-LVA captures  3) Killers
/// 4) History  5) Counter
pub(crate) fn score_move(m: &Move, tt_move: u32, hist: i32, cntr: i32, depth: u32) -> i32 {
    let packed = m_pack(m);
    // 1) Hash move (from TT)
    if packed == tt_move { return params::TT_MOVE_SCORE; }
    // 2) Captures: MVV-LVA
    if is_tactical(m) {
        let vals = piece_vals();
        let mut score = params::TACTICAL_BASE_SCORE;
        if m.captured_piece != 0 { score += vals[m.captured_piece as usize] * 100; }
        if m.mid_piece != 0 { score += vals[m.mid_piece as usize] * 100; }
        if m.range_cap && m.caps_value != 0 { score += m.caps_value * 100; }
        if m.promotion { score += params::PROMO_ORDER_BONUS; }
        return score;
    }
    // 3) Killer moves
    let kscore = heuristics::killer_score(depth, packed);
    if kscore > 0 { return kscore; }
    // 4) History heuristic
    // 5) Counter move heuristic
    hist + cntr
}

/// Cheap quiescence ordering score: total captured material (×10) minus the
/// moving piece's value, plus a promotion bonus. Used by QS and by capture
/// futility pruning in pvs.
pub(crate) fn capture_qs_score(board: &Board, m: &Move, values: &[i32; 512]) -> i32 {
    let from_pt = cell_piece(board.cells[m.from_sq as usize]);
    let mut score = 0;
    if m.captured_piece != 0 { score += values[m.captured_piece as usize] * 10; }
    if m.mid_piece != 0 { score += values[m.mid_piece as usize] * 10; }
    if m.range_cap { score += m.caps_value * 10; }
    if m.promotion {
        if let Some(promoted) = pieces::promotes_to(from_pt) {
            score += values[promoted as usize] - values[from_pt as usize];
        } else {
            score += 2500;
        }
    }
    score - values[from_pt as usize]
}
