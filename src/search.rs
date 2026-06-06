use crate::types::*;
use crate::pieces;
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::eval::{evaluate, MATE_SCORE};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// ── Transposition Table ──────────────────────────────────────────
const TT_SIZE: usize = 1 << 20; // 1M entries ≈ 8MB

#[derive(Clone, Copy)]
struct TTEntry {
    key: u16,   // lower 16 bits of hash for verification
    score: i32,
    depth: u8,
    flag: u8,   // 0 = Exact, 1 = Lower bound, 2 = Upper bound
    best_move: u32,
}

static TT: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn tt() -> &'static Vec<AtomicU64> {
    TT.get_or_init(|| (0..TT_SIZE).map(|_| AtomicU64::new(0)).collect())
}

#[inline]
fn tt_index(hash: u64) -> usize {
    (hash as usize) & (TT_SIZE - 1)
}

#[inline]
fn tt_pack(entry: &TTEntry) -> u64 {
    let score_clamped = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv16 = (entry.best_move & 0xFFFF) as u16;
    ((entry.key as u64) << 48)
        | ((score_clamped as u64) << 32)
        | ((entry.depth as u64) << 24)
        | ((entry.flag as u64) << 16)
        | (mv16 as u64)
}

#[inline]
fn tt_unpack(packed: u64) -> TTEntry {
    TTEntry {
        key: ((packed >> 48) & 0xFFFF) as u16,
        score: ((packed >> 32) & 0xFFFF) as u16 as i16 as i32,
        depth: ((packed >> 24) & 0xFF) as u8,
        flag: ((packed >> 16) & 0xFF) as u8,
        best_move: (packed & 0xFFFF) as u32,
    }
}

fn tt_probe(hash: u64) -> Option<TTEntry> {
    let idx = tt_index(hash);
    let packed = tt()[idx].load(Ordering::Relaxed);
    if packed == 0 { return None; }
    let entry = tt_unpack(packed);
    if entry.key == (hash as u16) { Some(entry) } else { None }
}

fn tt_store(hash: u64, entry: TTEntry) {
    let idx = tt_index(hash);
    let t = tt();
    let old = t[idx].load(Ordering::Relaxed);
    if old == 0 {
        t[idx].store(tt_pack(&entry), Ordering::Relaxed);
    } else {
        let old_entry = tt_unpack(old);
        if entry.depth >= old_entry.depth {
            t[idx].store(tt_pack(&entry), Ordering::Relaxed);
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
    history()[idx].fetch_add(depth as u64, Ordering::Relaxed);
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

    // Iterative deepening
    history_clear();
    let max_depth = depth.max(1);
    for d in 1..=max_depth {
        let mut local_best: Option<Move> = None;
        let mut local_best_score = -MATE_SCORE - 1;
        let mut local_nodes: u64 = 0;

        let mut scored_moves: Vec<(i32, usize)> = moves.iter().enumerate()
            .map(|(i, m)| (move_order_score(m), i))
            .collect();
        scored_moves.sort_by(|a, b| b.0.cmp(&a.0));

        for &(_, idx) in &scored_moves {
            let m = &moves[idx];
            board.apply_move(m);
            let score = -alphabeta(board, d - 1, -MATE_SCORE - 1,
                                   -local_best_score.max(-MATE_SCORE - 1),
                                   &mut local_nodes, deadline, 0);
            board.undo_move();

            if score > local_best_score {
                local_best_score = score;
                local_best = Some(m.clone());
            }

            if let Some(dl) = deadline {
                if Instant::now() >= dl { break; }
            }
        }

        best_move = local_best;
        best_score = local_best_score;
        nodes = local_nodes;

        if best_score >= MATE_SCORE - 100 { break; }
        if let Some(dl) = deadline {
            if Instant::now() >= dl { break; }
        }
    }

    let elapsed = start.elapsed().as_millis() as u64;
    SearchResult { best_move, score: best_score, nodes, time_ms: elapsed }
}

// ── Alpha-Beta ───────────────────────────────────────────────────
fn alphabeta(board: &mut Board, depth: u32, mut alpha: i32, beta: i32,
             nodes: &mut u64, deadline: Option<Instant>, ply: u32) -> i32 {
    *nodes += 1;

    if *nodes & 4095 == 0 {
        if let Some(dl) = deadline {
            if Instant::now() >= dl { return alpha; }
        }
    }

    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins  => if board.side_to_move == BLACK { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) },
            GameResult::WhiteWins  => if board.side_to_move == WHITE { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) },
            GameResult::Draw       => 0,
        };
    }

    if depth == 0 {
        // Only run quiescence when the board is less crowded (< 200 pieces total).
        // In the opening with 800 pieces, full movegen in every leaf is too expensive.
        let total_pieces = board.piece_count[0] + board.piece_count[1];
        if total_pieces < 200 {
            return quiescence(board, alpha, beta, nodes, deadline);
        }
        return evaluate(board);
    }

    // TT probe — board.hash is O(1), maintained incrementally
    let hash = board.hash;
    if let Some(entry) = tt_probe(hash) {
        if entry.depth >= depth as u8 {
            match entry.flag {
                0 => return entry.score,
                1 => { if entry.score >= beta  { return entry.score; } }
                2 => { if entry.score <= alpha { return entry.score; } }
                _ => {}
            }
        }
    }

    // Null move pruning (R=3):
    // Skip if in check, too few pieces (zugzwang risk), or endgame-ish
    let in_check = is_in_check(board);
    let side = board.side_to_move as usize;
    if depth >= 3
        && !in_check
        && board.no_progress_plies < 100
        && board.piece_count[side] > 10
    {
        board.null_move();
        let null_score = -alphabeta(board, depth.saturating_sub(3), -beta, -(beta - 1),
                                    nodes, deadline, ply + 1);
        board.undo_null_move();
        if null_score >= beta {
            return beta;
        }
    }

    let moves = generate_legal_moves(board);
    if moves.is_empty() {
        return if in_check { -(MATE_SCORE - ply as i32) } else { 0 };
    }

    // Move ordering: TT best move first, then captures + killers + history
    let tt_best_move = tt_probe(board.hash).map(|e| e.best_move);
    let mut scored_moves: Vec<(i32, usize)> = moves.iter().enumerate()
        .map(|(i, m)| {
            let packed = m_pack(m);
            let mut score = move_order_score(m);
            // TT best move gets highest priority
            if Some(packed) == tt_best_move { score += 2_000_000; }
            score += killer_score(depth, packed);
            score += history_score(m.from_sq as usize, m.to_sq as usize);
            (score, i)
        })
        .collect();
    scored_moves.sort_by(|a, b| b.0.cmp(&a.0));

    let mut tt_flag: u8 = 2; // Upper bound
    let mut best_local: Option<Move> = None;

    for (move_idx, &(_, idx)) in scored_moves.iter().enumerate() {
        let m = &moves[idx];
        board.apply_move(m);

        // Late move reductions
        let new_depth = if move_idx >= 4 && depth >= 3 && !in_check
            && !m.promotion && m.captured_piece == 0
        {
            depth - 2
        } else {
            depth - 1
        };

        let mut score = -alphabeta(board, new_depth, -beta, -alpha, nodes, deadline, ply + 1);

        // Re-search if LMR failed high
        if score >= beta && new_depth < depth - 1 {
            score = -alphabeta(board, depth - 1, -beta, -alpha, nodes, deadline, ply + 1);
        }

        board.undo_move();

        if score > alpha {
            alpha = score;
            tt_flag = 0;
            best_local = Some(m.clone());
        }
        if alpha >= beta {
            tt_flag = 1;
            killer_store(depth, m_pack(m));
            history_store(m.from_sq as usize, m.to_sq as usize, depth);
            break;
        }
    }

    if let Some(bm) = &best_local {
        tt_store(hash, TTEntry {
            key: hash as u16,
            score: alpha,
            depth: depth as u8,
            flag: tt_flag,
            best_move: m_pack(bm),
        });
    }

    alpha
}

// ── Quiescence Search ────────────────────────────────────────────
const MAX_QDEPTH: u32 = 4;

fn quiescence(board: &mut Board, mut alpha: i32, beta: i32,
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

    // Depth limit: stop here, return stand_pat
    if qdepth >= MAX_QDEPTH { return alpha; }

    let moves = generate_legal_moves(board);
    for m in &moves {
        // Only consider captures — skip quiet moves entirely
        if m.captured_piece == 0 && m.mid_piece == 0 && !m.is_igui { continue; }
        board.apply_move(m);
        let score = -quiescence_inner(board, -beta, -alpha, nodes, deadline, qdepth + 1);
        board.undo_move();
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}

// ── Check Detection ──────────────────────────────────────────────
fn is_in_check(board: &Board) -> bool {
    let king_sq = board.king_square(board.side_to_move);
    if king_sq == INVALID_SQ { return false; }
    let king_sq_usize = king_sq as usize;

    let opp = 1 - board.side_to_move;
    let rt = crate::types::ray_table();

    for i in 0..board.piece_list_len[opp as usize] {
        let sq = board.piece_list[opp as usize][i] as usize;
        if sq >= NUM_SQUARES { continue; }
        let cell = board.cells[sq];
        if cell == EMPTY_CELL { continue; }
        let pt = cell_piece(cell);
        let mv = pieces::movement(pt);

        // Check jump attacks
        for &(jdr, jdc) in &mv.jumps {
            let r = (sq / BOARD_SIZE) as i32;
            let c = (sq % BOARD_SIZE) as i32;
            let (dr, dc) = if opp == BLACK {
                (jdr as i32, jdc as i32)
            } else {
                (-(jdr as i32), -(jdc as i32))
            };
            let nr = r + dr;
            let nc = c + dc;
            if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 { continue; }
            let nsq = nr as usize * BOARD_SIZE + nc as usize;
            if nsq == king_sq_usize { return true; }
        }

        // Check slide attacks (the part that was missing before)
        for &(dir, max_range) in &mv.slides {
            let ray = rt.ray_for_color(sq, dir as usize, opp);
            let limit = if max_range == 0 { ray.len() } else { (max_range as usize).min(ray.len()) };
            for &rsq in &ray[..limit] {
                let rsq = rsq as usize;
                if rsq == king_sq_usize { return true; }
                if board.cells[rsq] != EMPTY_CELL { break; } // blocked by a piece
            }
        }
    }
    false
}

// ── Helpers ──────────────────────────────────────────────────────
#[inline]
fn m_pack(m: &Move) -> u32 {
    (m.from_sq as u32) | ((m.to_sq as u32) << 12) | (if m.promotion { 1 << 24 } else { 0 })
}

fn move_order_score(m: &Move) -> i32 {
    let mut score = 0i32;
    if m.captured_piece != 0 {
        score += pieces::value(m.captured_piece) * 10;
    }
    if m.promotion { score += 5000; }
    if m.mid_piece != 0 { score += pieces::value(m.mid_piece) * 5; }
    if let Some(ref caps) = m.range_caps {
        for &(_, pt, _) in caps {
            score += pieces::value(pt) * 10;
        }
    }
    score
}