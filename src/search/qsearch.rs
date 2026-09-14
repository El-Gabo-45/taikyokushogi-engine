//! Quiescence search: capture-only extension at the leaves.
//!
//! Reuses the TT for shallow results (captures transpose constantly on a
//! 36×36 board, so caching here saves whole quiescence subtrees) and is
//! depth-capped (MAX_QDEPTH) because without a cap a single call once
//! consumed 300,000 nodes on this game's open boards.

use super::ordering::{self, piece_vals};
use super::params;
use super::pvs::Ctx;
use super::tt;
use crate::eval::{evaluate, MATE_SCORE};
use crate::movegen::generate_pseudo_legal_captures;
use crate::types::*;

/// Quiescence entry (qd = 0 from the main search).
pub(crate) fn run(ctx: &mut Ctx, alpha: i32, beta: i32) -> i32 {
    ctx.qsearch(alpha, beta, 0)
}

impl<'a> Ctx<'a> {
    fn qsearch(&mut self, mut alpha: i32, beta: i32, qd: u32) -> i32 {
        *self.nodes += 1;
        let init_q_alpha = alpha;
        // Same reasoning as pvs(): nodes here are expensive (movegen ~120-180us),
        // so check the clock much more often than a typical chess engine would.
        if *self.nodes & params::NODES_PER_CLOCK_CHECK_QS == 0 && self.past_deadline() {
            return alpha;
        }

        // Terminal check (same semantics as pvs, with the quiescence ply).
        if let Some(result) = self.board.game_result() {
            let stm = self.board.side_to_move;
            return match result {
                GameResult::BlackWins => {
                    if stm == BLACK { MATE_SCORE - qd as i32 } else { -(MATE_SCORE - qd as i32) }
                }
                GameResult::WhiteWins => {
                    if stm == WHITE { MATE_SCORE - qd as i32 } else { -(MATE_SCORE - qd as i32) }
                }
                GameResult::Draw => 0,
            };
        }

        // ── TT PROBE (quiescence) ─────────────────────────────────────
        // Reuse shallow results at leaf nodes: on a 36×36 board the transposition
        // count in quiescence is enormous (captures transpose constantly), so
        // caching here saves whole quiescence subtrees.
        let q_hash = self.board.hash;
        if qd > 0 {
            if let Some(entry) = tt::tt_probe(q_hash) {
                match entry.flag {
                    0 => return entry.score,
                    1 => if entry.score >= beta { return entry.score; },
                    2 => if entry.score <= alpha { return entry.score; },
                    _ => {}
                }
            }
        }

        // Stand pat
        let stand_pat = evaluate(self.board);
        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }
        if qd >= params::MAX_QDEPTH { return alpha; }

        // Generate only captures and promotions (staged move generation —
        // Reference: docx §3.2 Futility Pruning & §4.4 Quiescence Search).
        // The capture-only generator skips the ~700 quiet moves, so QS now
        // scales with the number of pieces that can actually capture.
        let moves = generate_pseudo_legal_captures(self.board);
        let values = piece_vals();
        let mut scored: Vec<(i32, usize)> = Vec::with_capacity(moves.len());
        for (i, m) in moves.iter().enumerate() {
            let s = ordering::capture_qs_score(self.board, m, values);
            if s < params::QS_PREFILTER {
                continue;
            }
            scored.push((s, i));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        for (i_idx, &(_, i)) in scored.iter().enumerate() {
            if i_idx > 0 && self.past_deadline() { return alpha; }
            let m = &moves[i];
            self.board.apply_move(m);
            let score = -self.qsearch(-beta, -alpha, qd + 1);
            self.board.undo_move();
            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }

        if qd > 0 {
            tt::tt_store(q_hash, tt::TTEntry {
                score: alpha,
                depth: 0,
                flag: if alpha <= init_q_alpha { 2 } else { 0 },
                generation: tt::tt_gen(),
                best_move: 0,
                // Taikyoku has no check (SPEC §7.3) — always false.
                in_check: false,
            });
        }
        alpha
    }
}
