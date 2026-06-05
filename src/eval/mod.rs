//! Modular evaluation for Taikyoku Shogi.
//!
//! The evaluation is split into:
//! - **Families** — piece classification by movement pattern and role.
//! - **Zones** — positional bonuses based on board geography.
//! - **King safety** — royal piece protection.
//! - **Material** — family-weighted material count.

pub mod families;
pub mod zones;

use crate::board::Board;
use crate::types::*;
use crate::pieces;
use crate::eval::families::family_value;
use crate::eval::zones::zone_score;

pub const MATE_SCORE: i32 = 1_000_000;

/// Material score from Black's perspective (legacy API).
/// Positive = Black has more material.
pub fn material_score(board: &Board) -> i32 {
    let mut score: i32 = 0;
    for c in 0..2 {
        let sign: i32 = if c == BLACK as usize { 1 } else { -1 };
        for i in 0..board.piece_list_len[c] {
            let sq = board.piece_list[c][i] as usize;
            if sq == INVALID_SQ as usize { continue; }
            let cell = board.cells[sq];
            if cell == EMPTY_CELL { continue; }
            let pt = cell_piece(cell);
            score += sign * pieces::value(pt);
        }
    }
    score
}

/// Evaluate the position from the side to move's perspective.
pub fn evaluate(board: &Board) -> i32 {
    // Terminal positions.
    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins => if board.side_to_move == BLACK { MATE_SCORE } else { -MATE_SCORE },
            GameResult::WhiteWins => if board.side_to_move == WHITE { MATE_SCORE } else { -MATE_SCORE },
            GameResult::Draw => 0,
        };
    }

    // Material + family weight.
    let mut score = 0;
    for c in 0..2 {
        let sign: i32 = if c == BLACK as usize { 1 } else { -1 };
        for i in 0..board.piece_list_len[c] {
            let sq = board.piece_list[c][i] as usize;
            if sq == INVALID_SQ as usize { continue; }
            let cell = board.cells[sq];
            if cell == EMPTY_CELL { continue; }
            let pt = cell_piece(cell);
            score += sign * family_value(pt);
        }
    }

    // Zone bonuses.
    score += zone_score(board, BLACK);
    score -= zone_score(board, WHITE);

    // King safety (simple: penalize exposed king).
    score += king_safety(board, BLACK);
    score -= king_safety(board, WHITE);

    // Return from side-to-move perspective.
    if board.side_to_move == BLACK { score } else { -score }
}

/// Simple king safety: penalize if the king has no friendly pieces nearby.
fn king_safety(board: &Board, color: u8) -> i32 {
    let king_sq = board.king_square(color);
    if king_sq == INVALID_SQ || (king_sq as usize) >= NUM_SQUARES { return 0; }

    let kr = (king_sq as usize) / BOARD_SIZE;
    let kc = (king_sq as usize) % BOARD_SIZE;

    let mut defenders = 0;
    for dr in -2i32..=2 {
        for dc in -2i32..=2 {
            if dr == 0 && dc == 0 { continue; }
            let r = kr as i32 + dr;
            let c = kc as i32 + dc;
            if r < 0 || r >= BOARD_SIZE as i32 || c < 0 || c >= BOARD_SIZE as i32 { continue; }
            let sq = (r as usize) * BOARD_SIZE + (c as usize);
            if sq >= NUM_SQUARES { continue; }
            let cell = board.cells[sq];
            if cell != EMPTY_CELL && cell_color(cell) == color {
                defenders += 1;
            }
        }
    }

    defenders * 5
}
