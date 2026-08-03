//! Precomputed Piece-Square Table (PSQT) evaluation.
//!
//! The PSQT combines material value, family weight, and zone bonus into a
//! single precomputed table indexed by (piece_type, square, color). This
//! makes `evaluate()` a simple O(pieces) sum over the piece list — and with
//! incremental updates on the board, O(1) per call.
//!
//! Reference: "Piece-Square Tables" — Chess Programming Wiki. Stockfish
//! maintains incremental PSQT scores this way.

use crate::types::*;
use crate::pieces;
use crate::eval::families::family_value_fast;
use crate::eval::families::Family;
use crate::eval::zones::Zone;
use std::sync::OnceLock;

/// PSQT value for a piece of type `pt` at square `sq` for color `color`.
/// Positive = good for that color.
pub fn psqt_value(pt: u16, sq: usize, color: u8) -> i32 {
    // Material + family weight (this already includes the family_weight
    // blend from families.rs).
    let mut score = family_value_fast(pt);

    // Zone bonus (same logic as zones.rs::zone_family_bonus).
    let rank = (sq / BOARD_SIZE) as u8;
    let zone = Zone::from_rank(rank, color);
    let fam_enum = crate::eval::families::classify_fast(pt);
    let base = zone.bonus();
    let zone_bonus = match fam_enum {
        Family::Royal     => 0,
        Family::RangeCap  => base * 2,
        Family::Lion      => base * 2,
        Family::Eagle     => base * 2,
        Family::Dragon    => base * 2,
        Family::Demon     => base * 2,
        Family::Standard  => base,
        Family::Slider    => base,
        Family::Stepper   => base / 2,
        Family::Pawn      => base / 2,
        Family::Hook      => base,
        Family::Special   => base * 2,
        Family::Other     => base / 2,
    };
    score += zone_bonus;

    // Small positional bonus: pieces advancing toward the enemy camp get a
    // tiny extra edge; royals penalized for being exposed near center.
    if pieces::is_royal(pt) {
        let r = sq / BOARD_SIZE;
        let c = sq % BOARD_SIZE;
        let center_dist = (r as i32 - 17).abs() + (c as i32 - 17).abs();
        score -= center_dist * 2;
    }

    score
}

/// Precomputed PSQT table: [pt][sq][color]
static PSQT_TABLE: OnceLock<Box<[[[i32; 2]; NUM_SQUARES]; 512]>> = OnceLock::new();

/// Get the precomputed PSQT value for a piece.
#[inline]
pub fn psqt(pt: u16, sq: usize, color: u8) -> i32 {
    PSQT_TABLE.get_or_init(|| {
        let mut table = Box::new([[[0i32; 2]; NUM_SQUARES]; 512]);
        for pt in 1..=301u16 {
            for sq in 0..NUM_SQUARES {
                for color in 0..2u8 {
                    table[pt as usize][sq][color as usize] = psqt_value(pt, sq, color);
                }
            }
        }
        table
    })[(pt as usize).min(511)][sq][color as usize]
}

/// Compute the full PSQT score for a board from Black's perspective.
/// Positive = Black is better.
pub fn psqt_score(board: &crate::board::Board) -> i32 {
    let mut score = 0;
    for c in 0..2 {
        let sign: i32 = if c == 0 { 1 } else { -1 };
        for i in 0..board.piece_list_len[c] {
            let sq = board.piece_list[c][i] as usize;
            if sq >= NUM_SQUARES { continue; }
            let cell = board.cells[sq];
            if cell == EMPTY_CELL { continue; }
            let pt = cell_piece(cell);
            score += sign * psqt(pt, sq, c as u8);
        }
    }
    score
}

/// Get the PSQT delta for a piece moving from `from` to `to`, including
/// captures and promotions. Used for incremental updates.
#[inline]
pub fn psqt_delta(
    from: usize, to: usize,
    pt: u16, color: u8,
    promoted_pt: Option<u16>,
    captured_pt: u16,
    captured_sq: usize,
) -> i32 {
    let sign: i32 = if color == BLACK { 1 } else { -1 };

    // Remove piece from origin
    let mut delta = -sign * psqt(pt, from, color);

    // Add piece at destination (possibly promoted)
    let final_pt = promoted_pt.unwrap_or(pt);
    delta += sign * psqt(final_pt, to, color);

    // Remove captured piece (at destination or elsewhere, e.g. range caps)
    if captured_pt != 0 {
        delta += -sign * psqt(captured_pt, captured_sq, 1 - color);
    }

    delta
}