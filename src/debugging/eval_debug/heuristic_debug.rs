use crate::board::Board;
use crate::eval::families::family_value;
use crate::eval::zones::zone_score;
use crate::pieces;
use crate::types::{cell_color, cell_piece, EMPTY_CELL, INVALID_SQ, BLACK, WHITE};

/// Resumen detallado del heuristic para inspección rápida.
#[derive(Debug, Clone)]
pub struct HeuristicDebugReport {
    pub total: i32,
    pub material: i32,
    pub family_weight: i32,
    pub zones: i32,
    pub king_safety: i32,
    pub by_piece: Vec<(String, i32)>,
}

/// Devuelve un reporte legible del valor heurístico actual.
pub fn debug_heuristic(board: &Board) -> HeuristicDebugReport {
    let mut material = 0i32;
    let mut family_weight = 0i32;
    let mut by_piece: Vec<(String, i32)> = Vec::new();

    for color in [BLACK, WHITE] {
        let sign = if color == BLACK { 1 } else { -1 };
        for i in 0..board.piece_list_len[color as usize] {
            let sq = board.piece_list[color as usize][i] as usize;
            if sq == INVALID_SQ as usize {
                continue;
            }
            let cell = board.cells[sq];
            if cell == EMPTY_CELL {
                continue;
            }
            let pt = cell_piece(cell);
            let piece_name = pieces::abbrev(pt).to_string();
            let value = pieces::value(pt);
            let family = family_value(pt);
            material += sign * value;
            family_weight += sign * family;

            if let Some((_, existing)) = by_piece.iter_mut().find(|(name, _)| name == &piece_name) {
                *existing += sign * value;
            } else {
                by_piece.push((piece_name, sign * value));
            }
        }
    }

    by_piece.sort_by(|a, b| b.1.cmp(&a.1));

    let zones = zone_score(board, BLACK) - zone_score(board, WHITE);
    let king_safety = king_safety(board, BLACK) - king_safety(board, WHITE);
    let total = family_weight + zones + king_safety;

    HeuristicDebugReport {
        total,
        material,
        family_weight,
        zones,
        king_safety,
        by_piece,
    }
}

/// Convierte el reporte en una cadena formateada para impresión o logs.
pub fn format_heuristic_debug(board: &Board) -> String {
    let report = debug_heuristic(board);
    let mut lines = vec![
        format!("heuristic_total={}", report.total),
        format!("material={}", report.material),
        format!("family_weight={}", report.family_weight),
        format!("zones={}", report.zones),
        format!("king_safety={}", report.king_safety),
    ];

    if !report.by_piece.is_empty() {
        lines.push("by_piece=".to_string());
        for (name, value) in report.by_piece {
            lines.push(format!("  {name}={value}"));
        }
    }

    lines.join("\n")
}

fn king_safety(board: &Board, color: u8) -> i32 {
    let king_sq = board.king_square(color);
    if king_sq == INVALID_SQ || (king_sq as usize) >= crate::types::NUM_SQUARES {
        return 0;
    }

    let kr = (king_sq as usize) / crate::types::BOARD_SIZE;
    let kc = (king_sq as usize) % crate::types::BOARD_SIZE;

    let mut defenders = 0;
    for dr in -2i32..=2 {
        for dc in -2i32..=2 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let r = kr as i32 + dr;
            let c = kc as i32 + dc;
            if r < 0 || r >= crate::types::BOARD_SIZE as i32 || c < 0 || c >= crate::types::BOARD_SIZE as i32 {
                continue;
            }
            let sq = (r as usize) * crate::types::BOARD_SIZE + (c as usize);
            if sq >= crate::types::NUM_SQUARES {
                continue;
            }
            let cell = board.cells[sq];
            if cell != EMPTY_CELL && cell_color(cell) == color {
                defenders += 1;
            }
        }
    }

    defenders * 5
}
