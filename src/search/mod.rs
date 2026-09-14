//! Alpha-beta search for Taikyoku Shogi (36x36 board, ~700 legal moves/node).
//!
//! Module layout (one responsibility per file):
//! * [`params`]     — every tunable constant and margin formula, in one place.
//! * [`tt`]         — lock-free bucketed transposition table (race-safe
//!   publish/reverify protocol for Lazy SMP).
//! * [`heuristics`] — killer moves, butterfly history, counter moves.
//! * [`ordering`]   — move scoring (TT move > MVV-LVA > killers > history).
//! * [`pvs`]        — the PVS core: null-window re-search, late move
//!   reductions (history-softened), null-move pruning, razoring / reverse
//!   futility / futility / ProbCut, internal iterative deepening, staged
//!   move generation with incremental pick-next ordering.
//! * [`qsearch`]    — capture-only quiescence with its own TT traffic.
//! * [`root`]       — one root iteration at a fixed depth (aspiration
//!   windows around the previous score), plus the depth≤3 material-delta
//!   fast path.
//!
//! Iterative deepening with predictive time management lives in [`search`]
//! below.
//!
//! NOTE: this variant has NO check and NO checkmate (SPEC §7.3): the game
//! only ends when a side captures the opponent's LAST royal (SPEC §7.2).

mod heuristics;
mod ordering;
mod params;
mod pvs;
mod qsearch;
mod root;
mod tt;

use crate::board::Board;
use crate::eval::{evaluate, MATE_SCORE};
use crate::types::Move;
use std::time::Instant;

/// Result of a completed search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
}

/// Full search: iterative deepening with aspiration windows and predictive
/// time management.
pub fn search(board: &mut Board, depth: u32, time_limit_ms: u64) -> SearchResult {
    let start = Instant::now();
    // env::var takes a global lock — cache it once per search, not per iteration.
    let debug_log = std::env::var_os("RPS_DEBUG").is_some();
    let deadline = if time_limit_ms > 0 {
        Some(start + std::time::Duration::from_millis(time_limit_ms))
    } else { None };

    tt::tt_new_generation();
    ordering::piece_vals();
    let mut best_result = SearchResult { best_move: None, score: evaluate(board), nodes: 0, time_ms: 0 };
    let mut total_nodes: u64 = 0;
    let mut root_hint: Option<u32> = None;
    let mut score_guess = best_result.score;
    let mut prev_iter_ms: u64 = 0;

    if depth == 0 {
        return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }

    for current_depth in 1..=depth {
        if let Some(dl) = deadline {
            if Instant::now() >= dl { break; }
        }

        // ── PREDICTIVE TIME MANAGEMENT ────────────────────────
        // The next iteration typically costs ~4x the previous one (branching
        // factor). If the last completed iteration's duration times 4 no
        // longer fits in the remaining budget, stop — starting it would burn
        // the rest of the time on an unusable partial result. (Standard
        // technique: predict the next iteration's cost from the previous
        // one, as in Stockfish.)
        if current_depth >= 2 && deadline.is_some() && prev_iter_ms > 0 {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            let remaining = time_limit_ms.saturating_sub(elapsed_ms);
            if prev_iter_ms.saturating_mul(params::NEXT_ITER_COST_FACTOR) > remaining { break; }
        }

        let result = if current_depth <= 1 {
            root::search_root_window(board, current_depth, deadline, root_hint, -MATE_SCORE - 1, MATE_SCORE + 1)
        } else {
            // Aspiration windows at ALL depths >= 2. The previous version
            // disabled them for d >= 5 because a *narrow* (±64) window failed
            // constantly on this game's volatile scores (the eval can jump
            // thousands of centipawns between iterations when a large capture
            // is found), causing long cascades of re-searches. Policy:
            // initial window ±64 (±200 at d >= 5); on the FIRST fail grow ×8;
            // on the SECOND fail fall back to the FULL window. At most two
            // re-searches per iteration, and each failed re-search is much
            // cheaper than a full-window search thanks to TT hits.
            let mut window = params::aspiration_window(current_depth);
            let mut fails = 0u8;
            let mut alpha = score_guess.saturating_sub(window);
            let mut beta = score_guess.saturating_add(window);
            let mut local_result;
            loop {
                local_result = root::search_root_window(board, current_depth, deadline, root_hint, alpha, beta);
                if let Some(dl) = deadline {
                    if Instant::now() >= dl { break; }
                }
                if local_result.score <= alpha || local_result.score >= beta {
                    fails += 1;
                    if fails >= params::ASPIRATION_MAX_FAILS as u8 {
                        break; // keep the (bounded) result — full re-search not worth it
                    }
                    window = (window * params::ASPIRATION_WINDOW_GROW).min(MATE_SCORE);
                    if local_result.score <= alpha {
                        // Fail low: the true score is LOWER than guessed —
                        // recenter the window on the failed score so the next
                        // search is bounded around the right region.
                        score_guess = local_result.score;
                    }
                    alpha = score_guess.saturating_sub(window);
                    beta = score_guess.saturating_add(window);
                    continue;
                }
                break;
            }
            local_result
        };

        if deadline.map(|dl| Instant::now() >= dl).unwrap_or(false) { break; }
        total_nodes = total_nodes.saturating_add(result.nodes);
        if debug_log {
            eprintln!("iter d={} nodes={} score={} t={}ms", current_depth, result.nodes, result.score, result.time_ms);
        }
        root_hint = result.best_move.as_ref().map(|m| ordering::m_pack(m));
        score_guess = result.score;
        prev_iter_ms = result.time_ms;
        best_result = result;
    }

    let elapsed = start.elapsed().as_millis() as u64;
    SearchResult {
        best_move: best_result.best_move,
        score: best_result.score,
        nodes: total_nodes.max(best_result.nodes),
        time_ms: elapsed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Board;

    #[test]
    fn search_initial_reaches_depth_and_finds_a_move() {
        heuristics::clear();
        let mut board = Board::initial();
        let r = board.search(4, 0);
        assert!(r.best_move.is_some(), "depth-4 search must return a move");
        assert!(r.nodes > 0);
        assert!(r.score.abs() < MATE_SCORE, "eval must not look like mate");
    }

    #[test]
    fn history_clear_zeroes_counters() {
        heuristics::history_store(3, 4, 5, 0);
        assert!(heuristics::history_score(3, 4, 0) > 0);
        heuristics::clear();
        assert_eq!(heuristics::history_score(3, 4, 1), 0);
        assert_eq!(heuristics::history_score(0, 0, 0), 0);
    }

    #[test]
    fn killer_store_is_read_back() {
        heuristics::killer_store(10, 0xABCD);
        assert_eq!(heuristics::killer_score(10, 0xABCD), params::KILLER1_SCORE);
    }
}
