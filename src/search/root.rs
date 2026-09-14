//! Root search: one iteration at a fixed depth, with aspiration-window
//! refinement around the previous iteration's best score.
//!
//! Full root breadth: ALL root moves are ranked and searched every
//! iteration (~500+). The root is a single node, so generating + ranking
//! the whole list is a negligible fraction of the tree, and a narrow root
//! beam would discard most candidate moves. Internal nodes keep their own
//! beams, so depth is preserved by pruning BELOW the root.

use super::heuristics;
use super::ordering::{self, piece_vals};
use super::params;
use super::pvs;
use super::tt;
use crate::board::Board;
use crate::eval::{evaluate, material_score, MATE_SCORE};
use crate::movegen::{generate_pseudo_legal_moves, generate_pseudo_legal_captures};
use crate::pieces;
use crate::types::*;
use std::time::Instant;

/// Aspiration-window search of the root at one depth.
pub(crate) fn search_root_window(
    board: &mut Board,
    depth: u32,
    deadline: Option<Instant>,
    root_hint: Option<u32>,
    root_alpha: i32,
    root_beta: i32,
) -> super::SearchResult {
    let start = Instant::now();
    piece_vals();

    if depth == 0 {
        return super::SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }

    // Depth 1-3 all use the fast material-delta path (see `material_delta_root`).
    if depth <= 3 {
        return material_delta_root(board, depth, start);
    }

    let mut nodes: u64 = 0;
    let mut best_move = None;
    let mut best_score = -MATE_SCORE - 1;
    let root_tt_move = tt::tt_probe(board.hash).map(|e| e.best_move).unwrap_or(0);
    let stm = board.side_to_move;

    // A depth-2 preliminary search warms the TT and gives the ordering a
    // cheap head start before the real iteration.
    if depth > 2 {
        let _ = pvs::pvs(board, depth - 2, -MATE_SCORE - 1, MATE_SCORE + 1,
                         &mut nodes, deadline, 0, 0);
    }

    // ── ROOT-LEVEL STAGED GENERATION ──────────────────────────
    // Generate captures first (cheap, ~10-50 moves), search them. Only if
    // no beta cutoff is found do we generate the full quiet move list
    // (~700 moves). This avoids generating + sorting all ~700 root moves
    // when a capture already causes a cutoff — the dominant cost of deep
    // search. Reference: docx §3.2 Futility Pruning & §4.4 Quiescence.
    // Full root breadth: rank and search ALL root moves every iteration
    // (~500+). The root is a single node, so generating + ranking the whole
    // list is a negligible fraction of the tree, and a narrow root beam
    // would discard most candidate moves. Internal nodes keep their own
    // beams, so depth is preserved by pruning BELOW the root.
    // (depth <= 3 never reaches here: the material-delta fast path above
    // returns early, so no per-depth branch is needed at the root.)
    let max_moves = usize::MAX;

    // Stage 1: captures + promotions (tactical moves).
    // Use the fast bitboard capture generator. If a special piece triggers
    // NeedsFallback, fall back to the full generator.
    let (cap_moves_raw, cap_mode) = crate::attack::generate_captures_bb(board);
    let cap_moves = if cap_mode == crate::attack::GenMode::NeedsFallback {
        generate_pseudo_legal_captures(board)
    } else {
        cap_moves_raw
    };
    let mut cap_scored: Vec<(i32, usize)> = Vec::with_capacity(cap_moves.len());
    for (i, m) in cap_moves.iter().enumerate() {
        let packed = ordering::m_pack(m);
        let hist = heuristics::history_score(m.from_sq as usize, m.to_sq as usize, stm);
        let mut s = ordering::score_move(m, root_tt_move, hist, 0, depth);
        if root_hint == Some(packed) { s += params::ROOT_HINT_SCORE; }
        cap_scored.push((s, i));
    }
    cap_scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    for rank in 0..cap_scored.len().min(max_moves) {
        if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
        let idx = cap_scored[rank].1;
        let m = &cap_moves[idx];
        board.apply_move(m);
        nodes += 1;
        let (sa, sb) = if rank == 0 && best_score > root_alpha + params::ROOT_WINDOW_REFINE_GATE {
            (best_score - params::ROOT_WINDOW_REFINE, best_score + params::ROOT_WINDOW_REFINE)
        } else {
            (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
        };
        let score = if rank == 0 {
            -pvs::pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, ordering::m_pack(m))
        } else {
            let nw = -pvs::pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, ordering::m_pack(m));
            if nw > sa && nw < sb {
                -pvs::pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, ordering::m_pack(m))
            } else { nw }
        };
        if score <= sa || score >= sb {
            let full = -pvs::pvs(board, depth - 1, -MATE_SCORE - 1,
                                 -best_score.max(-MATE_SCORE - 1),
                                 &mut nodes, deadline, 0, ordering::m_pack(m));
            if full > best_score { best_score = full; best_move = Some(m.clone()); }
        } else if score > best_score {
            best_score = score;
            best_move = Some(m.clone());
        }
        board.undo_move();
        if best_score >= root_beta { break; }
    }

    // Stage 2: quiet moves (only if no beta cutoff from captures).
    // Full root breadth: quiet moves are ALWAYS considered at the root
    // (not just depth <= 3), so every iteration ranks and searches the
    // complete root move list. This restores the coverage the narrow root
    // beam removed.
    if best_score < root_beta {
        let moves = generate_pseudo_legal_moves(board);
        if moves.is_empty() {
            return super::SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 };
        }
        let mut scored: Vec<(i32, usize)> = Vec::with_capacity(moves.len());
        for (i, m) in moves.iter().enumerate() {
            let packed = ordering::m_pack(m);
            let hist = heuristics::history_score(m.from_sq as usize, m.to_sq as usize, stm);
            let mut s = ordering::score_move(m, root_tt_move, hist, 0, depth);
            if root_hint == Some(packed) { s += params::ROOT_HINT_SCORE; }
            scored.push((s, i));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for rank in 0..scored.len().min(max_moves) {
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            let idx = scored[rank].1;
            let m = &moves[idx];
            board.apply_move(m);
            nodes += 1;
            let (sa, sb) = if rank == 0 && best_score > root_alpha + params::ROOT_WINDOW_REFINE_GATE {
                (best_score - params::ROOT_WINDOW_REFINE, best_score + params::ROOT_WINDOW_REFINE)
            } else {
                (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
            };
            let score = if rank == 0 {
                -pvs::pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, ordering::m_pack(m))
            } else {
                let nw = -pvs::pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, ordering::m_pack(m));
                if nw > sa && nw < sb {
                    -pvs::pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, ordering::m_pack(m))
                } else { nw }
            };
            if score <= sa || score >= sb {
                let full = -pvs::pvs(board, depth - 1, -MATE_SCORE - 1,
                                     -best_score.max(-MATE_SCORE - 1),
                                     &mut nodes, deadline, 0, ordering::m_pack(m));
                if full > best_score { best_score = full; best_move = Some(m.clone()); }
            } else if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }
            board.undo_move();
            if best_score >= root_beta { break; }
        }
    }

    super::SearchResult {
        best_move,
        score: best_score,
        nodes,
        time_ms: start.elapsed().as_millis() as u64,
    }
}

/// Depth ≤ 3 root: evaluate each root move's material delta directly
/// (O(1) per move) instead of applying + searching.
///
/// On a 36×36 board with ~700 legal moves, the full apply+undo cycle made a
/// real depth-2/3 search (716 root moves × 716 replies) take seconds per
/// iteration. The material-delta shortcut evaluates each move's material
/// change directly (O(1) per move) and completes in ~60-100µs — making
/// depth-2 and depth-3 as fast as depth-1.
/// Reference: HaChu (hgm.nubati.net) — incremental evaluation scales with
/// the board perimeter, not the area.
fn material_delta_root(board: &mut Board, depth: u32, start: Instant) -> super::SearchResult {
    let moves = generate_pseudo_legal_moves(board);
    if moves.is_empty() {
        return super::SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }
    let mut best_move = None;
    let mut best_score = -MATE_SCORE - 1;
    let mut nodes: u64 = 0;
    let base_mat = material_score(board);
    let sign = if board.side_to_move == BLACK { 1 } else { -1 };
    let values = piece_vals();
    for m in &moves {
        nodes += 1;
        let mut delta = 0i32;
        if m.promotion {
            let pt = cell_piece(board.cells[m.from_sq as usize]);
            if let Some(p) = pieces::promotes_to(pt) {
                delta += sign * (values[p as usize] - values[pt as usize]);
            }
        }
        if m.captured_piece != 0 { delta += sign * values[m.captured_piece as usize]; }
        if m.mid_piece != 0 { delta += sign * values[m.mid_piece as usize]; }
        if m.range_cap { delta += sign * m.caps_value; }
        let s = -(base_mat + delta);
        if s > best_score { best_score = s; best_move = Some(m.clone()); }
    }
    // No legal move at all: the side to move loses (SPEC §7.3).
    if best_move.is_none() {
        let score = -(MATE_SCORE - depth as i32);
        return super::SearchResult { best_move: None, score, nodes, time_ms: start.elapsed().as_millis() as u64 };
    }
    super::SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 }
}
