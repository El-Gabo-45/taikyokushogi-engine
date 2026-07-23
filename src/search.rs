use crate::types::*;
use crate::pieces;
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::eval::{evaluate, material_score, MATE_SCORE};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// ── Transposition Table (full u64 key, 128-bit entries) ──────────
const TT_SIZE: usize = 1 << 20; // 1M entries, each 16 bytes

#[derive(Clone, Copy)]
struct TTEntry {
    score: i32,
    depth: u8,
    flag: u8,
    best_move: u32,
}

static TT: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn tt() -> &'static Vec<AtomicU64> {
    TT.get_or_init(|| (0..TT_SIZE * 2).map(|_| AtomicU64::new(0)).collect())
}

#[inline]
fn tt_index(hash: u64) -> usize {
    (hash as usize) & (TT_SIZE - 1)
}

#[inline]
fn tt_pack(entry: &TTEntry) -> u64 {
    let score_clamped = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv16 = (entry.best_move & 0xFFFF) as u16;
    ((score_clamped as u64) << 32)
        | ((entry.depth as u64) << 24)
        | ((entry.flag as u64) << 16)
        | (mv16 as u64)
}

#[inline]
fn tt_unpack(packed: u64) -> TTEntry {
    TTEntry {
        score: ((packed >> 32) & 0xFFFF) as u16 as i16 as i32,
        depth: ((packed >> 24) & 0xFF) as u8,
        flag: ((packed >> 16) & 0xFF) as u8,
        best_move: (packed & 0xFFFF) as u32,
    }
}

fn tt_probe(hash: u64) -> Option<(TTEntry, u32)> {
    let idx = tt_index(hash) * 2;
    let t = tt();
    let stored_hash = t[idx].load(Ordering::Relaxed);
    if stored_hash == hash {
        let entry = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
        Some((entry, entry.best_move))
    } else {
        None
    }
}

fn tt_store(hash: u64, entry: TTEntry) {
    let idx = tt_index(hash) * 2;
    let t = tt();
    let old_hash = t[idx].load(Ordering::Relaxed);
    if old_hash == 0 {
        t[idx].store(hash, Ordering::Relaxed);
        t[idx + 1].store(tt_pack(&entry), Ordering::Relaxed);
    } else {
        let old_data = t[idx + 1].load(Ordering::Relaxed);
        let old_entry = tt_unpack(old_data);
        if entry.depth >= old_entry.depth || old_hash != hash {
            t[idx].store(hash, Ordering::Relaxed);
            t[idx + 1].store(tt_pack(&entry), Ordering::Relaxed);
        }
    }
}

// ── Killer Moves ─────────────────────────────────────────────────
static mut KILLER_MOVES: [[u32; 2]; 128] = [[0u32; 2]; 128];

fn killer_store(depth: u32, mv: u32) {
    unsafe {
        let d = depth.min(127) as usize;
        KILLER_MOVES[d][1] = KILLER_MOVES[d][0];
        KILLER_MOVES[d][0] = mv;
    }
}

fn killer_score(depth: u32, mv: u32) -> i32 {
    unsafe {
        let d = depth.min(127) as usize;
        if KILLER_MOVES[d][0] == mv { return 90000; }
        if KILLER_MOVES[d][1] == mv { return 80000; }
    }
    0
}

// ── History Heuristic ────────────────────────────────────────────
const HIST_SIZE: usize = 256;
static HISTORY: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn history() -> &'static Vec<AtomicU64> {
    HISTORY.get_or_init(|| (0..HIST_SIZE * HIST_SIZE).map(|_| AtomicU64::new(0)).collect())
}

fn history_store(from: usize, to: usize, depth: u32) {
    let idx = (from % HIST_SIZE) * HIST_SIZE + (to % HIST_SIZE);
    // Quadratic bonus: depth*depth gives more weight to deeper cutoffs
    let bonus = (depth * depth) as u64;
    history()[idx].fetch_add(bonus, Ordering::Relaxed);
}

fn history_score(from: usize, to: usize) -> i32 {
    let idx = (from % HIST_SIZE) * HIST_SIZE + (to % HIST_SIZE);
    history()[idx].load(Ordering::Relaxed) as i32
}

fn history_clear() {
    if let Some(h) = HISTORY.get() {
        for cell in h.iter() {
            cell.store(0, Ordering::Relaxed);
        }
    }
}

// ── Search Result ────────────────────────────────────────────────
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
}

// ── Move ordering helpers ────────────────────────────────────────
#[inline]
fn m_pack(m: &Move) -> u32 {
    (m.from_sq as u32) | ((m.to_sq as u32) << 12) | (if m.promotion { 1 << 24 } else { 0 })
}

fn move_order_score(m: &Move) -> i32 {
    let mut score = 0i32;
    if m.captured_piece != 0 { score += pieces::value(m.captured_piece) * 10; }
    if m.promotion { score += 5000; }
    if m.mid_piece != 0 { score += pieces::value(m.mid_piece) * 5; }
    if let Some(ref caps) = m.range_caps {
        for &(_, pt, _) in caps { score += pieces::value(pt) * 10; }
    }
    score
}

// ── Public Search ────────────────────────────────────────────────
pub fn search(board: &mut Board, depth: u32, time_limit_ms: u64) -> SearchResult {
    let start = Instant::now();
    let deadline = if time_limit_ms > 0 {
        Some(start + std::time::Duration::from_millis(time_limit_ms))
    } else {
        None
    };

    let mut best_move = None;
    let mut best_score = -MATE_SCORE - 1;
    let mut nodes: u64 = 0;

    let moves = generate_legal_moves(board);
    if moves.is_empty() {
        return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }
    if moves.len() == 1 {
        return SearchResult { best_move: Some(moves[0].clone()), score: 0, nodes: 1, time_ms: 0 };
    }

    // Score moves for ordering
    let mut scored_moves: Vec<(i32, usize)> = moves.iter().enumerate()
        .map(|(i, m)| (move_order_score(m), i))
        .collect();
    scored_moves.sort_by(|a, b| b.0.cmp(&a.0));

    let max_moves = if depth <= 1 { moves.len() } else { moves.len().min(64) };

    if depth <= 1 {
        // Compute baseline material score once, then adjust incrementally per move
        // This avoids apply_move/undo_move (expensive: 800+ piece list updates, hash updates, undo stack)
        // and avoids iterating all 800+ pieces in material_score() for every move
        let base_mat = material_score(board);
        for rank in 0..max_moves {
            let idx = scored_moves[rank].1;
            let m = &moves[idx];
            nodes += 1;

            // Compute material delta from Black's perspective directly from move data
            let mut delta = 0i32;
            let sign = if board.side_to_move == BLACK { 1 } else { -1 };

            // Promotion: piece gains (promoted_value - original_value)
            if m.promotion {
                let pt = cell_piece(board.cells[m.from_sq as usize]);
                if let Some(promo_pt) = pieces::promotes_to(pt) {
                    delta += sign * (pieces::value(promo_pt) - pieces::value(pt));
                }
            }

            // Captures: opponent loses pieces
            if m.captured_piece != 0 {
                delta += sign * pieces::value(m.captured_piece);
            }
            if m.mid_piece != 0 {
                delta += sign * pieces::value(m.mid_piece);
            }
            if let Some(ref caps) = m.range_caps {
                for &(_, pt, _) in caps {
                    delta += sign * pieces::value(pt);
                }
            }

            // Score from opponent's perspective after the move
            let score = -(base_mat + delta);
            if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }
        }
    } else {
        // Iterative deepening: do depth 2 first to get a good TT move for deeper search
        if depth > 2 {
            // Shallow search to seed TT
            let _ = alphabeta(board, depth - 2, -MATE_SCORE - 1, MATE_SCORE + 1,
                             &mut nodes, deadline, 0, false);
        }

        for rank in 0..max_moves {
            let idx = scored_moves[rank].1;
            let m = &moves[idx];
            board.apply_move(m);
            nodes += 1;

            // Aspiration windows for deep searches
            let (search_alpha, search_beta) = if depth > 3 && rank == 0 && best_score > -MATE_SCORE + 100 {
                (best_score - 50, best_score + 50)
            } else {
                (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
            };

            let score = -alphabeta(board, depth - 1, search_alpha, search_beta,
                                    &mut nodes, deadline, 0, true);

            // Aspiration window fail - research with full window
            if score <= search_alpha || score >= search_beta {
                let full_score = -alphabeta(board, depth - 1, -MATE_SCORE - 1,
                                            -best_score.max(-MATE_SCORE - 1),
                                            &mut nodes, deadline, 0, true);
                if full_score > best_score {
                    best_score = full_score;
                    best_move = Some(m.clone());
                }
            } else if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }

            board.undo_move();
            if let Some(dl) = deadline {
                if Instant::now() >= dl { break; }
            }
        }
    }

    let elapsed = start.elapsed().as_millis() as u64;
    SearchResult { best_move, score: best_score, nodes, time_ms: elapsed }
}

// ── Alpha-Beta (optimized) ──────────────────────────────────────
fn alphabeta(board: &mut Board, depth: u32, mut alpha: i32, beta: i32,
             nodes: &mut u64, deadline: Option<Instant>, ply: u32,
             enable_pruning: bool) -> i32 {
    *nodes += 1;

    // Time check
    if *nodes & 4095 == 0 {
        if let Some(dl) = deadline {
            if Instant::now() >= dl { return alpha; }
        }
    }

    // Terminal check
    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins  => if board.side_to_move == BLACK { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) },
            GameResult::WhiteWins  => if board.side_to_move == WHITE { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) },
            GameResult::Draw       => 0,
        };
    }

    // Check extension: extend by 1 ply if in check
    let in_check = is_in_check(board);
    let ext = if in_check && enable_pruning { 1 } else { 0 };
    let effective_depth = depth + ext;

    if effective_depth == 0 {
        let total_pieces = board.piece_count[0] + board.piece_count[1];
        if total_pieces < 200 {
            return quiescence(board, alpha, beta, nodes, deadline);
        }
        return evaluate(board);
    }

    // TT probe (with full u64 hash verification)
    let hash = board.hash;
    let tt_best_move = if let Some((entry, best_mv)) = tt_probe(hash) {
        if entry.depth >= effective_depth as u8 {
            match entry.flag {
                0 => return entry.score,
                1 => { if entry.score >= beta  { return entry.score; } }
                2 => { if entry.score <= alpha { return entry.score; } }
                _ => {}
            }
        }
        best_mv
    } else {
        0
    };

    // Reverse futility pruning (RFP): at shallow depths, if static eval is
    // far below alpha, prune the node entirely (no need to search captures).
    if enable_pruning && depth <= 2 && !in_check {
        let static_eval = evaluate(board);
        let margin = 200 * depth as i32;
        if static_eval + margin <= alpha {
            return alpha;
        }
        if static_eval - margin >= beta {
            return beta;
        }
    }

    // Null move pruning (R=3) — skip if in check
    let side = board.side_to_move as usize;
    if enable_pruning && effective_depth >= 3
        && board.no_progress_plies < 100
        && board.piece_count[side] > 10
        && !in_check
    {
        board.null_move();
        let null_score = -alphabeta(board, effective_depth.saturating_sub(3), -beta, -(beta - 1),
                                    nodes, deadline, ply + 1, enable_pruning);
        board.undo_null_move();
        if null_score >= beta {
            return beta;
        }
    }

    // Generate legal moves
    let moves = generate_legal_moves(board);
    if moves.is_empty() {
        return -(MATE_SCORE - ply as i32);
    }

    // Move ordering
    let mut scored_moves: Vec<(i32, usize)> = moves.iter().enumerate()
        .map(|(i, m)| {
            let packed = m_pack(m);
            let mut score = move_order_score(m);
            if packed == tt_best_move { score += 2_000_000; }
            score += killer_score(effective_depth, packed);
            score += history_score(m.from_sq as usize, m.to_sq as usize);
            (score, i)
        })
        .collect();
    scored_moves.sort_by(|a, b| b.0.cmp(&a.0));

    let max_moves = if effective_depth <= 2 { moves.len() } else { moves.len().min(32) };

    let mut tt_flag: u8 = 2; // UPPERBOUND
    let mut best_local: Option<Move> = None;

    // Store initial alpha for TT flag
    let init_alpha = alpha;

    for (move_idx, &(_, idx)) in scored_moves[..max_moves].iter().enumerate() {
        let m = &moves[idx];

        // Late move pruning: skip moves with low ordering score at depth 0 (after quiescence)
        if enable_pruning && move_idx >= 4 && effective_depth <= 1 && alpha > -MATE_SCORE + 100 && !m.promotion && m.captured_piece == 0 {
            let total_pieces = board.piece_count[0] + board.piece_count[1];
            if total_pieces > 100 {
                continue;
            }
        }

        board.apply_move(m);

        // Late Move Reduction (LMR): reduce depth for late moves
        let new_depth = if enable_pruning && move_idx >= 4 && effective_depth >= 3 && !m.promotion && m.captured_piece == 0 && !in_check {
            effective_depth.saturating_sub(2)
        } else {
            effective_depth.saturating_sub(1)
        };

        // PVS search: search first move with full window, rest with zero window
        let score;
        if move_idx == 0 {
            score = -alphabeta(board, new_depth, -beta, -alpha, nodes, deadline, ply + 1, enable_pruning);
        } else {
            // Search with null window
            let null_score = -alphabeta(board, new_depth, -alpha - 1, -alpha, nodes, deadline, ply + 1, enable_pruning);
            if null_score > alpha && null_score < beta && new_depth < effective_depth.saturating_sub(1) {
                // Re-search with full depth
                score = -alphabeta(board, effective_depth.saturating_sub(1), -beta, -alpha, nodes, deadline, ply + 1, enable_pruning);
            } else {
                score = null_score;
            }
        }

        board.undo_move();

        if score > alpha {
            alpha = score;
            tt_flag = 0; // EXACT
            best_local = Some(m.clone());
        }
        if alpha >= beta {
            tt_flag = 1; // LOWERBOUND
            killer_store(effective_depth, m_pack(m));
            history_store(m.from_sq as usize, m.to_sq as usize, effective_depth);
            break;
        }
    }

    if let Some(bm) = &best_local {
        tt_store(hash, TTEntry {
            score: alpha,
            depth: effective_depth as u8,
            flag: if alpha <= init_alpha { 2 } else { tt_flag },
            best_move: m_pack(bm),
        });
    }

    alpha
}

// ── Quiescence Search ────────────────────────────────────────────
const MAX_QDEPTH: u32 = 4;

fn quiescence(board: &mut Board, alpha: i32, beta: i32,
              nodes: &mut u64, deadline: Option<Instant>) -> i32 {
    quiescence_inner(board, alpha, beta, nodes, deadline, 0)
}

fn quiescence_inner(board: &mut Board, mut alpha: i32, beta: i32,
                    nodes: &mut u64, deadline: Option<Instant>, qdepth: u32) -> i32 {
    *nodes += 1;
    if *nodes & 4095 == 0 {
        if let Some(dl) = deadline {
            if Instant::now() >= dl { return alpha; }
        }
    }

    let stand_pat = evaluate(board);
    if stand_pat >= beta { return beta; }
    if stand_pat > alpha { alpha = stand_pat; }

    if qdepth >= MAX_QDEPTH { return alpha; }

    let moves = generate_legal_moves(board);

    // Sort moves by capture value for better quiescence ordering
    let mut scored_qmoves: Vec<(i32, usize)> = moves.iter().enumerate()
        .filter(|(_, m)| m.captured_piece != 0 || m.mid_piece != 0 || m.is_igui)
        .map(|(i, m)| {
            let mut score = 0;
            if m.captured_piece != 0 { score += pieces::value(m.captured_piece) * 10; }
            if m.mid_piece != 0 { score += pieces::value(m.mid_piece) * 10; }
            (score, i)
        })
        .collect();
    scored_qmoves.sort_by(|a, b| b.0.cmp(&a.0));

    for &(_, i) in &scored_qmoves {
        let m = &moves[i];
        board.apply_move(m);
        let score = -quiescence_inner(board, -beta, -alpha, nodes, deadline, qdepth + 1);
        board.undo_move();
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}

use crate::movegen::is_in_check;