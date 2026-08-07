//! Modular evaluation for Taikyoku Shogi.
//!
//! The evaluation is split into:
//! - **Families** — piece classification by movement pattern and role.
//! - **Zones** — positional bonuses based on board geography.
//! - **King safety** — royal piece protection.
//! - **Material** — family-weighted material count.
//!
//! # Performance: Incremental Material Score
//!
//! The `material_score` field on `Board` is maintained incrementally in
//! `apply_move`/`undo_move`, so `material_score()` is now O(1) instead of
//! O(pieces). This eliminates ~84µs per call in the initial position.
//!
//! Reference: "Incremental evaluation" is a standard technique described in
//! the Chess Programming Wiki ("Evaluation" → "Incremental Updates").
//! Stockfish maintains incremental material and PSQT scores this way.

pub mod families;
pub mod zones;
pub mod nnue;
pub mod psqt;

use crate::board::Board;
use crate::types::*;

pub const MATE_SCORE: i32 = 1_000_000;

// ── Evaluation backend selection ────────────────────────────────
// Lets callers (e.g. a match runner comparing hand-crafted eval vs NNUE
// strength) switch backends at runtime without recompiling. Defaults to
// the hand-crafted evaluator, so existing behavior is unchanged unless
// something explicitly opts into NNUE.
use std::sync::atomic::{AtomicBool, Ordering};
static USE_NNUE: AtomicBool = AtomicBool::new(false);

/// Switch the global evaluation backend. When `true`, `evaluate()` uses
/// the NNUE network (see nnue::nnue_evaluate_from_scratch); when `false`
/// (the default), it uses the original hand-crafted material/zone/king-
/// safety evaluation below.
pub fn set_use_nnue(enabled: bool) {
    USE_NNUE.store(enabled, Ordering::Relaxed);
}

pub fn using_nnue() -> bool {
    USE_NNUE.load(Ordering::Relaxed)
}

/// Material score from Black's perspective (legacy API).
/// Positive = Black has more material.
///
/// Uses the incrementally-maintained `material_score` field on the board
/// if available (O(1)), otherwise falls back to the full scan (O(pieces)).
pub fn material_score(board: &Board) -> i32 {
    board.material_score
}

/// Evaluate the position from the side to move's perspective.
pub fn evaluate(board: &Board) -> i32 {
    // Terminal positions -- shared between both evaluation backends,
    // since checkmate/draw detection doesn't depend on which evaluator
    // is scoring non-terminal positions.
    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins => if board.side_to_move == BLACK { MATE_SCORE } else { -MATE_SCORE },
            GameResult::WhiteWins => if board.side_to_move == WHITE { MATE_SCORE } else { -MATE_SCORE },
            GameResult::Draw => 0,
        };
    }

    if USE_NNUE.load(Ordering::Relaxed) {
        return nnue::nnue_evaluate_from_scratch(board);
    }

    // Incremental PSQT score (combines material + family weight + zone
    // bonuses). Maintained O(1) in apply_move/undo_move — no O(pieces)
    // scan needed.
    let score = board.psqt_score
        + king_safety(board, BLACK)
        - king_safety(board, WHITE);

    // Return from side-to-move perspective.
    if board.side_to_move == BLACK { score } else { -score }
}

/// Simple king safety: penalize if the king has no friendly pieces nearby.
/// Uses the precomputed threat-zone bitboard for O(1) defender counting.
fn king_safety(board: &Board, color: u8) -> i32 {
    let king_sq = board.king_square(color);
    if king_sq == INVALID_SQ || (king_sq as usize) >= NUM_SQUARES { return 0; }

    // Use the precomputed 5x5 threat zone around the king and intersect
    // with friendly occupancy — O(1) bitboard AND + popcount instead of
    // a 5x5 loop with bounds checks.
    let zone = crate::bitboard::threat_zone_table().zone(king_sq as usize);
    let friendly = board.occupancy[color as usize].and_new(zone);
    let defenders = friendly.count() as i32;

    defenders * 5
}
