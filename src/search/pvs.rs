//! PVS (Principal Variation Search) core.
//!
//! Structure of a node, in strict evaluation order:
//!   1. clock + terminal checks
//!   2. TT probe
//!   3. leaf handling (quiescence / static eval)
//!   4. shallow-eval pruning: razoring → RFP → futility
//!   5. null-move pruning
//!   6. ProbCut
//!   7. internal iterative deepening
//!   8. staged movegen: capture beam (branching-scaled) → quiet LMP stage
//!   9. TT store
//!
//! NOTE: this variant has NO check and NO checkmate (SPEC §7.3): a move may
//! expose royals freely, and the game only ends when a side captures the
//! opponent's LAST royal (SPEC §7.2). There is no legality filtering by
//! king safety, no check extension, and no check-gated pruning. The
//! `in_check` bindings stay in the code (always false) so the gates read
//! naturally and can be wired up if a check rule is ever added.

use super::heuristics;
use super::ordering::{self, piece_vals};
use super::params;
use super::qsearch;
use super::tt;
use crate::board::Board;
use crate::eval::{evaluate, MATE_SCORE};
use crate::types::*;
use std::time::Instant;

/// Per-node search state threaded through the recursive calls, replacing the
/// previous 8-argument `pvs(...)` signature.
pub(crate) struct Ctx<'a> {
    pub board: &'a mut Board,
    pub nodes: &'a mut u64,
    pub deadline: Option<Instant>,
    pub ply: u32,
    /// Packed key of the move played at the parent node (for counter moves).
    pub prev_move: u32,
}

impl<'a> Ctx<'a> {
    #[inline]
    pub(crate) fn past_deadline(&self) -> bool {
        self.deadline.map_or(false, |dl| Instant::now() >= dl)
    }

    /// Returns the score of a terminal position, or None if the game is on.
    fn terminal_score(&self, ply: u32) -> Option<i32> {
        let result = self.board.game_result()?;
        let stm = self.board.side_to_move;
        Some(match result {
            GameResult::BlackWins => {
                if stm == BLACK { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) }
            }
            GameResult::WhiteWins => {
                if stm == WHITE { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) }
            }
            GameResult::Draw => 0,
        })
    }

    /// Shallow-eval pruning that needs no move generation, in the original
    /// evaluation order: razoring → RFP (both directions) → futility.
    /// Returns Some(score) if the node can be pruned outright.
    fn shallow_eval_prune(d: u32, static_eval: i32, alpha: i32, beta: i32, in_check: bool) -> Option<i32> {
        // ── RAZORING (depth ≤ 2) ─────────────────────────────────
        // If static_eval + huge_margin ≤ alpha, prune the node entirely.
        if let Some(margin) = params::razoring_margin(d) {
            if !in_check && params::mate_gate_ok(alpha) && static_eval + margin <= alpha {
                return Some(alpha);
            }
        }
        // ── REVERSE FUTILITY PRUNING (depth ≤ 3) ─────────────────
        // If static_eval - margin ≥ beta, prune (position is too good).
        if let Some(margin) = params::rfp_margin(d) {
            if !in_check && params::mate_gate_ok(alpha) {
                if static_eval - margin >= beta { return Some(beta); }
                if static_eval + margin <= alpha { return Some(alpha); }
            }
        }
        // ── FUTILITY PRUNING (depth ≤ 2) ─────────────────────────
        // Skip shallow quiet nodes when even optimistic gains cannot reach alpha.
        if let Some(margin) = params::futility_margin(d) {
            if !in_check && params::mate_gate_ok(alpha) && static_eval + margin <= alpha {
                return Some(alpha);
            }
        }
        None
    }

    /// Null Move Pruning (depth ≥ 3): give the opponent a free move; if even
    /// then we're still ≥ beta, prune.
    fn null_move_prune(&mut self, d: u32, beta: i32, in_check: bool) -> Option<i32> {
        if d < params::NMP_MIN_DEPTH || in_check { return None; }
        let side = self.board.side_to_move as usize;
        if self.board.no_progress_plies >= params::NMP_NO_PROGRESS_LIMIT
            || self.board.piece_count[side] < params::NMP_MIN_SIDE_PIECES
        {
            return None;
        }
        let r = params::nmp_reduction(d);
        self.board.null_move();
        let null_score = -self.pvs(d.saturating_sub(r), -beta, -(beta - 1));
        self.board.undo_null_move();
        if null_score >= beta { Some(beta) } else { None }
    }

    /// ProbCut (depth ≥ 4): statistical pruning — if the static eval is far
    /// enough below alpha, the probability that any move can raise it above
    /// beta is negligible. Reference: "ProbCut" — Kotani, Computer Shogi.
    fn probcut(d: u32, static_eval: i32, alpha: i32, in_check: bool) -> Option<i32> {
        if let Some(margin) = params::probcut_margin(d) {
            if !in_check && params::mate_gate_ok(alpha) && static_eval + margin <= alpha {
                return Some(alpha);
            }
        }
        None
    }

    /// Internal Iterative Deepening: with no TT move at a non-trivial depth,
    /// run a shallow scout to obtain one from the TT.
    fn iid_move(&mut self, d: u32, hash: u64, tt_move: u32, alpha: i32, beta: i32) -> u32 {
        if tt_move != 0 || d < params::IID_MIN_DEPTH {
            return tt_move;
        }
        let iid_d = d / 2 - 1;
        let _ = self.pvs(iid_d, -beta, -alpha);
        tt::tt_probe(hash).map(|e| e.best_move).unwrap_or(0)
    }

    /// Entry point kept free of state plumbing (see `Ctx`).
    pub fn run(&mut self, depth: u32, alpha: i32, beta: i32) -> i32 {
        // Every caller (full window, null-window probe, aspiration) maintains
        // beta > alpha; the flag logic in the TT store relies on this invariant.
        debug_assert!(beta > alpha, "pvs requires beta > alpha");
        *self.nodes += 1;

        // Time check: a node on this 1296-square board costs ~150-200us, so the
        // deadline is polled every 64 nodes (~10-15ms of real time). The bitwise
        // AND keeps the check itself essentially free.
        if *self.nodes & params::NODES_PER_CLOCK_CHECK == 0 && self.past_deadline() {
            return alpha;
        }

        // Terminal check
        if let Some(score) = self.terminal_score(self.ply) {
            return score;
        }

        // ── TT PROBE ──────────────────────────────────────────────
        let hash = self.board.hash;
        let tt_move = tt::tt_probe(hash).map(|e| e.best_move).unwrap_or(0);

        // Taikyoku has NO check (SPEC §7.3) — see module docs.
        let in_check = false;
        let d = depth; // effective depth (no extension)

        // ── QUIESCENCE AT LEAVES ──────────────────────────────────
        // At d == 0, run quiescence (capture-only) to avoid the horizon effect.
        // For d > 0, we genuinely generate and search moves — no leaf fast-path
        // shortcut that would fake depth. The fast capture generator + small
        // beams keep each node cheap enough to reach depth 6+.
        if d == 0 {
            let total = self.board.piece_count[0] + self.board.piece_count[1];
            if total < params::QS_LEAF_PIECE_LIMIT {
                return qsearch::run(self, alpha, beta);
            }
            return evaluate(self.board);
        }

        // ── STATIC EVAL ───────────────────────────────────────────
        // With the incremental PSQT score, evaluate() is O(1) (just a couple
        // of table lookups + the king-safety term). Use it at every node —
        // no need for the cheaper-but-cruder material-only approximation.
        let static_eval = evaluate(self.board);

        // ── SHALLOW-EVAL PRUNING (razoring / RFP / futility) ──────
        if let Some(score) = Self::shallow_eval_prune(d, static_eval, alpha, beta, in_check) {
            return score;
        }

        // ── NULL MOVE PRUNING ─────────────────────────────────────
        if let Some(score) = self.null_move_prune(d, beta, in_check) {
            return score;
        }

        // ── PROBCUT ───────────────────────────────────────────────
        if let Some(score) = Self::probcut(d, static_eval, alpha, in_check) {
            return score;
        }

        // ── INTERNAL ITERATIVE DEEPENING ──────────────────────────
        // If no TT move, do a shallow search to get one.
        let iid_move = self.iid_move(d, hash, tt_move, alpha, beta);

        self.search_node(d, alpha, beta, static_eval, in_check, iid_move)
    }

    /// Recursive PVS call used inside the node logic.
    #[inline]
    fn pvs(&mut self, depth: u32, alpha: i32, beta: i32) -> i32 {
        self.run(depth, alpha, beta)
    }

    /// The move-generation + move-loop part of a PVS node, shared by `run`.
    fn search_node(&mut self, d: u32, mut alpha: i32, beta: i32,
                   static_eval: i32, in_check: bool, iid_move: u32) -> i32 {
        // Side to move at THIS node, captured before any apply_move flips it:
        // history tables are per-side, and the beta-cutoff blocks below run
        // after board.undo_move() has already restored the parent's side.
        let stm = self.board.side_to_move;

        // ── BEST MOVE AS SCALAR ───────────────────────────────────
        // Track the best move as its packed u32 instead of a cloned Move.
        // The Move struct is a plain POD now (range captures are recomputed in
        // apply_move from from/to, no heap payload), but the TT only needs the
        // packed key anyway — cloning a 16-byte struct per alpha raise is pure
        // waste.
        let mut best_packed: u32 = 0;
        let mut tt_flag: u8 = 2; // UPPERBOUND
        let init_alpha = alpha;
        let mut searched = false;
        // Quiets tried so far at this node (for history malus on beta cutoff).
        // Fixed stack array: at most the beam + 1 quiets are ever tried before
        // the beam break, so 64 slots always suffice — no heap allocation per
        // node.
        let mut quiet_tried: [(u16, u16); params::QUIETS_TRACKED_MAX] = [(0, 0); params::QUIETS_TRACKED_MAX];
        let mut quiet_tried_n = 0usize;

        // ── STAGED MOVE GENERATION ────────────────────────────────
        // Generate captures first (cheap, ~10-50 moves), search them. Only if
        // no beta cutoff is found do we generate the full quiet move list
        // (~700 moves). This avoids generating ~700 quiet moves at every node
        // when a capture already causes a cutoff — the dominant cost of deep
        // search. Reference: docx §3.2 Futility Pruning & §4.4 Quiescence.
        let (cap_moves, cap_mode) = crate::attack::generate_captures_bb(self.board);
        let cap_moves = if cap_mode == crate::attack::GenMode::NeedsFallback {
            crate::movegen::generate_pseudo_legal_captures(self.board)
        } else {
            cap_moves
        };
        // With hundreds of tactical moves per node (median branching 944,
        // max 1254), a fixed 6-24 beam prunes a far larger FRACTION of moves
        // than intended. Scale the threshold with the actual list length so
        // the pruned fraction stays comparable to chess:
        //   beam = base * (1 + tactical_moves / 128)
        let beam_base = params::capture_beam_base(d);
        let rps_beam = (beam_base as usize * (1 + cap_moves.len() / params::BEAM_BRANCH_SCALE))
            .min(params::CAPTURE_BEAM_MAX);
        // ── Incremental move selection (pick-next) ────────────────────
        // Score once into a flat i32 buffer, then repeatedly select the best
        // remaining move: O(k·n) with k ≈ rps_beam (6-24) instead of a full
        // O(n log n) sort of the capture list at every node.
        let mut cap_scores: Vec<i32> = Vec::with_capacity(cap_moves.len());
        for m in cap_moves.iter() {
            let packed = ordering::m_pack(m);
            let hist = heuristics::history_score(m.from_sq as usize, m.to_sq as usize, stm);
            let cntr = heuristics::counter_score(self.prev_move, packed);
            cap_scores.push(ordering::score_move(m, iid_move, hist, cntr, d));
        }

        let mut move_idx = 0usize;
        // ── Stage 1: captures + promotions ────────────────────────────
        loop {
            // Pick the best remaining capture.
            let mut idx = usize::MAX;
            let mut order_score = i32::MIN;
            for (i, &s) in cap_scores.iter().enumerate() {
                if s > order_score { order_score = s; idx = i; }
            }
            if idx == usize::MAX { break; }
            cap_scores[idx] = i32::MIN;
            let packed = ordering::m_pack(&cap_moves[idx]);
            // 0-based index counting LEGAL captures searched so far. Illegal
            // pseudo-legal captures (verified below) never consume a beam slot:
            // move_idx previously advanced before the legality filter, so the
            // first illegal captures could exhaust rps_beam without a single
            // valid move being searched. The increment happens after the filter.
            let cur_move = move_idx;
            if cur_move >= rps_beam && !in_check && searched {
                break;
            }
            if cur_move > 0 && self.past_deadline() { break; }
            let m = &cap_moves[idx];
            // ── CAPTURE FUTILITY PRUNING (depth ≤ 2) ──────────────
            // If even capturing the most valuable pieces on the board plus a
            // safety margin cannot lift the static eval to alpha, this capture
            // cannot possibly raise the score. With ~700 legal moves per node on
            // a 36×36 board, dropping hopeless captures cheaply (O(1) estimate)
            // is a large win — we avoid the full apply + subtree search.
            if d <= 2 && !in_check && cur_move > 0
                && order_score < params::TACTICAL_BASE_SCORE && params::mate_gate_ok(alpha)
            {
                let values = piece_vals();
                let from_pt = cell_piece(self.board.cells[m.from_sq as usize]);
                let gain = ordering::capture_qs_score(self.board, m, values) + values[from_pt as usize];
                if static_eval + gain + params::CAPTURE_FUTILITY_MARGIN <= alpha { continue; }
            }
            self.board.apply_move(m);
            // Every pseudo-legal move is legal in Taikyoku (no check): consume a
            // beam slot now, before the search.
            move_idx += 1;
            searched = true;
            let new_d = d.saturating_sub(1);
            let score = if cur_move == 0 {
                -self.pvs(new_d, -beta, -alpha)
            } else {
                let nw = -self.pvs(new_d, -alpha - 1, -alpha);
                if nw > alpha && nw < beta {
                    -self.pvs(new_d, -beta, -alpha)
                } else { nw }
            };
            self.board.undo_move();
            if score > alpha {
                alpha = score;
                tt_flag = 0;
                best_packed = packed;
            }
            if alpha >= beta {
                tt_flag = 1;
                if order_score < params::TACTICAL_BASE_SCORE {
                    heuristics::killer_store(d, packed);
                    heuristics::history_store(m.from_sq as usize, m.to_sq as usize, d, stm);
                }
                if self.prev_move != 0 { heuristics::counter_store(self.prev_move, packed); }
                // History gravity: penalize the quiets that failed to cause a
                // cutoff, so next time the cutting move (and similar ones) are
                // tried first. (Standard technique: Stockfish's history malus.)
                for &(hf, ht) in &quiet_tried[..quiet_tried_n] {
                    heuristics::history_malus(hf as usize, ht as usize, d, stm);
                }
                break;
            }
        }
        // ── Stage 2: quiet moves (only if no beta cutoff from captures) ──
        // LMP scaled by the REAL branching factor of the node. With ~700 quiets
        // per node on a 36x36 board, a fixed chess-calibrated threshold prunes a
        // far larger FRACTION than intended (TaikyokuShogi-Stockfish measured
        // 84-98% of quiets with chess thresholds on branching ~1000). The fix
        // scales the threshold with the node's branching so the pruned fraction
        // is comparable to chess:
        //   lmp_n = (3 + d^2) / (improving ? 1 : 2) * (1 + quiets / 128)
        if alpha < beta {
            let moves = crate::movegen::generate_pseudo_legal_moves(self.board);
            if moves.is_empty() { return -(MATE_SCORE - self.ply as i32); }
            let mut scored: Vec<(i32, usize, u32)> = Vec::with_capacity(moves.len());
            for (i, m) in moves.iter().enumerate() {
                let packed = ordering::m_pack(m);
                let hist = heuristics::history_score(m.from_sq as usize, m.to_sq as usize, stm);
                let cntr = heuristics::counter_score(self.prev_move, packed);
                let s = ordering::score_move(m, iid_move, hist, cntr, d);
                scored.push((s, i, packed));
            }
            let n_quiets = scored.len();
            let improving = static_eval > alpha;
            let lmp_base = params::lmp_base(d, improving);
            let lmp_n = (lmp_base * (1 + n_quiets / params::LMP_BRANCH_SCALE))
                .min(n_quiets).min(params::LMP_MAX);
            let select_n = lmp_n;
            if select_n > 1 && scored.len() > select_n {
                scored.select_nth_unstable_by(select_n - 1, |a, b| b.0.cmp(&a.0));
            } else {
                scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            }

            for &(order_score, idx, packed) in scored.iter() {
                // cur_move = count of quiets searched so far; only moves that are
                // actually searched consume a slot (the increment is after any
                // pruning continue).
                let cur_move = move_idx;
                if cur_move >= lmp_n && !in_check && searched {
                    break;
                }
                if cur_move > 0 && self.past_deadline() { break; }
                // ── QUIET FUTILITY PRUNING (depth ≤ 2) ────────────
                // A quiet move at low depth cannot change the eval by more than a
                // small margin; if even that margin cannot reach alpha, skip the
                // move entirely (avoids apply + subtree on the
                // hundreds of remaining quiets at each node).
                if d <= params::QUIET_FUTILITY_MAX_DEPTH && !in_check && cur_move > 0
                    && order_score < params::TACTICAL_BASE_SCORE && params::mate_gate_ok(alpha)
                {
                    if static_eval + params::quiet_futility_margin(d) <= alpha { continue; }
                }
                if cur_move > 0 && self.past_deadline() { break; }
                let m = &moves[idx];
                self.board.apply_move(m);
                // Every pseudo-legal move is legal in Taikyoku (no check): consume
                // a beam slot now, before the search.
                move_idx += 1;
                searched = true;
                if order_score < params::TACTICAL_BASE_SCORE {
                    if quiet_tried_n < quiet_tried.len() {
                        quiet_tried[quiet_tried_n] = (m.from_sq, m.to_sq);
                        quiet_tried_n += 1;
                    }
                }
                // Hash-move extension: give the TT's best move one extra ply.
                // NOTE: not a true singular extension (which would re-search at
                // reduced depth with the TT move EXCLUDED to verify it is the
                // only good move) — just a depth bump for the hash move.
                let tt_move_ext = d >= params::TT_MOVE_EXT_MIN_DEPTH && !in_check
                    && iid_move != 0 && packed == iid_move
                    && order_score < params::TACTICAL_BASE_SCORE;
                let reduction = if cur_move >= params::LMR_MIN_MOVE_INDEX && d >= params::LMR_MIN_DEPTH
                    && order_score < params::TACTICAL_BASE_SCORE && !in_check
                {
                    let base = params::lmr_base(cur_move);
                    let depth_factor = params::lmr_depth_factor(d);
                    // A quiet with a strong history is statistically good —
                    // soften its reduction so it is not searched too shallowly
                    // (LMR + history interaction, standard in modern engines).
                    let soften = params::lmr_history_soften(
                        heuristics::history_score(m.from_sq as usize, m.to_sq as usize, stm));
                    (base + depth_factor).saturating_sub(soften)
                } else { 0 };
                let mut new_d = d.saturating_sub(1 + reduction);
                if tt_move_ext { new_d = new_d.saturating_add(1); }
                let score;
                if cur_move == 0 {
                    score = -self.pvs(new_d, -beta, -alpha);
                } else if reduction > 0 {
                    let nw = -self.pvs(new_d, -alpha - 1, -alpha);
                    if nw > alpha && nw < beta {
                        score = -self.pvs(d.saturating_sub(1), -beta, -alpha);
                    } else { score = nw; }
                } else {
                    let nw = -self.pvs(new_d, -alpha - 1, -alpha);
                    if nw > alpha && nw < beta {
                        score = -self.pvs(new_d, -beta, -alpha);
                    } else { score = nw; }
                }
                self.board.undo_move();
                if score > alpha {
                    alpha = score;
                    tt_flag = 0;
                    best_packed = packed;
                }
                if alpha >= beta {
                    tt_flag = 1;
                    if order_score < params::TACTICAL_BASE_SCORE {
                        heuristics::killer_store(d, packed);
                        heuristics::history_store(m.from_sq as usize, m.to_sq as usize, d, stm);
                    }
                    if self.prev_move != 0 { heuristics::counter_store(self.prev_move, packed); }
                    break;
                }
            }
        }

        // ── TT STORE ──────────────────────────────────────────────
        // Store whenever real moves were searched — including all-moves-fail-low
        // nodes (valid UPPERBOUND entries). Skipping those used to waste TT
        // probes on positions the tree reaches repeatedly via transpositions.
        if searched {
            tt::tt_store(self.board.hash, tt::TTEntry {
                score: alpha,
                depth: d.min(120) as i8,
                flag: if alpha <= init_alpha { 2 } else { tt_flag },
                // Coherence: tt_pack overrides this field with the ACTIVE
                // generation, so a hardcoded 0 had no packed effect — but an
                // inspected struct must not lie. Set the real generation.
                generation: tt::tt_gen(),
                best_move: best_packed,
                in_check: false, // no such thing as check in Taikyoku
            });
        }

        alpha
    }
}

/// Free-function entry point with the classic signature, for callers that
/// don't hold a `Ctx` (root search, tests).
pub(crate) fn pvs(board: &mut Board, depth: u32, alpha: i32, beta: i32,
                  nodes: &mut u64, deadline: Option<Instant>, ply: u32,
                  prev_move: u32) -> i32 {
    let mut ctx = Ctx { board, nodes, deadline, ply, prev_move };
    ctx.run(depth, alpha, beta)
}
