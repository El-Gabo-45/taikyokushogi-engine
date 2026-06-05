//! Zone-based positional evaluation.
//!
//! The 36×36 board is divided into four concentric zones from each player's
//! perspective.  Pieces gain bonuses for advancing into enemy territory and
//! penalties for retreating into their own camp.

use crate::board::Board;
use crate::types::*;

/// Zone index from the perspective of the side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    /// Ranks 0–8 (own camp).
    OwnCamp,
    /// Ranks 9–17 (inner field).
    InnerField,
    /// Ranks 18–26 (enemy field).
    EnemyField,
    /// Ranks 27–35 (enemy camp).
    EnemyCamp,
}

impl Zone {
    /// Convert a rank (0 = Black's back rank) to a zone for the given color.
    pub fn from_rank(rank: u8, color: u8) -> Zone {
        let r = if color == BLACK { rank } else { 35 - rank };
        match r {
            0..=8   => Zone::OwnCamp,
            9..=17  => Zone::InnerField,
            18..=26 => Zone::EnemyField,
            _       => Zone::EnemyCamp,
        }
    }

    /// Base bonus for a piece in this zone (from the side-to-move perspective).
    pub fn bonus(self) -> i32 {
        match self {
            Zone::OwnCamp     => -10,
            Zone::InnerField  => 0,
            Zone::EnemyField  => 15,
            Zone::EnemyCamp   => 30,
        }
    }
}

/// Zone bonus per family — some families benefit more from advancing.
pub fn zone_family_bonus(zone: Zone, family: crate::eval::families::Family) -> i32 {
    use crate::eval::families::Family;
    let base = zone.bonus();
    match family {
        Family::Royal     => 0,      // King safety is handled elsewhere
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
    }
}

/// Compute the total zone bonus for a side.
pub fn zone_score(board: &Board, color: u8) -> i32 {
    let c = color as usize;
    let mut score = 0;
    for i in 0..board.piece_list_len[c] {
        let sq = board.piece_list[c][i] as usize;
        if sq >= NUM_SQUARES { continue; }
        let cell = board.cells[sq];
        if cell == EMPTY_CELL { continue; }
        let piece = cell_piece(cell);
        let rank = (sq / BOARD_SIZE) as u8;
        let zone = Zone::from_rank(rank, color);
        let fam = crate::eval::families::classify(piece);
        score += zone_family_bonus(zone, fam);
    }
    score
}
