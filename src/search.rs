use crate::types::*;
use crate::pieces;
use crate::board::Board;
use crate::movegen::generate_pseudo_legal_moves;
use crate::movegen::is_in_check;
use crate::eval::{evaluate, material_score, MATE_SCORE};
use std::sync::atomic::{AtomicU64, AtomicI32, Ordering};
use std::sync::{OnceLock, Arc};
use std::time::Instant;
use std::thread;

// ── Transposition Table ─────────────────────────────────────────
const TT_SIZE: usize = 1 << 22;

#[derive(Clone, Copy)]
struct TTEntry {
    score: i32,
    depth: i8,
    flag: u8,       // 0=EXACT, 1=LOWERBOUND (≥beta), 2=UPPERBOUND (≤alpha)
    generation: u8,
    best_move: u32,
}

static TT: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn tt() -> &'static Vec<AtomicU64> {
    TT.get_or_init(|| (0..TT_SIZE * 2).map(|_| AtomicU64::new(0)).collect())
}

#[inline]
fn tt_index(hash: u64) -> usize { (hash as usize) & (TT_SIZE - 1) }

#[inline]
fn tt_pack(entry: &TTEntry, gen: u8) -> u64 {
    let sc = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv16 = (entry.best_move & 0xFFFF) as u16;
    ((sc as u64) << 32) | ((entry.depth as u64 & 0x7F) << 25)
        | ((entry.flag as u64) << 16) | ((gen as u64) << 8) | (mv16 as u64)
}

#[inline]
fn tt_unpack(packed: u64) -> TTEntry {
    TTEntry {
        score: ((packed >> 32) & 0xFFFF) as u16 as i16 as i32,
        depth: ((packed >> 25) & 0x7F) as i8,
        flag: ((packed >> 16) & 0xFF) as u8,
        generation: ((packed >> 8) & 0xFF) as u8,
        best_move: (packed & 0xFFFF) as u32,
    }
}

static TT_GEN: AtomicU64 = AtomicU64::new(1);
fn tt_gen() -> u8 { (TT_GEN.load(Ordering::Relaxed) & 0xFF) as u8 }

fn tt_probe(hash: u64) -> Option<(TTEntry, u32)> {
    let idx = tt_index(hash) * 2;
    let t = tt();
    let stored = t[idx].load(Ordering::Relaxed);
    if stored == hash {
        let entry = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
        if entry.depth >= 0 { return Some((entry, entry.best_move)); }
    }
    None
}

fn tt_store(hash: u64, entry: TTEntry) {
    let idx = tt_index(hash) * 2;
    let t = tt();
    let gen = tt_gen();
    let old_hash = t[idx].load(Ordering::Relaxed);
    let old = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
    let replace = old_hash == 0 || entry.depth > old.depth
        || (entry.depth == old.depth && gen.wrapping_sub(old.generation) > 100);
    if replace {
        t[idx].store(hash, Ordering::Relaxed);
        t[idx + 1].store(tt_pack(&entry, gen), Ordering::Relaxed);
    }
}

// ── Killers ─────────────────────────────────────────────────────
static KILLERS: OnceLock<Vec<AtomicU64>> = OnceLock::new();
fn killers() -> &'static Vec<AtomicU64> {
    KILLERS.get_or_init(|| (0..256).map(|_| AtomicU64::new(0)).collect())
}

fn killer_store(depth: u32, mv: u32) {
    let d = depth.min(127) as usize;
    let slot = &killers()[d];
    let cur = slot.load(Ordering::Relaxed);
    let mv0 = cur as u32;
    if mv != mv0 { slot.store(mv as u64 | ((mv0 as u64) << 32), Ordering::Relaxed); }
}

fn killer_score(depth: u32, mv: u32) -> i32 {
    let d = depth.min(127) as usize;
    let p = killers()[d].load(Ordering::Relaxed);
    let mv0 = p as u32;
    let mv1 = (p >> 32) as u32;
    if mv == mv0 { 90000 } else if mv == mv1 { 80000 } else { 0 }
}

// ── History ─────────────────────────────────────────────────────
const HIST_SZ: usize = 256;
static HIST: OnceLock<Vec<AtomicI32>> = OnceLock::new();
fn history() -> &'static Vec<AtomicI32> {
    HIST.get_or_init(|| (0..HIST_SZ * HIST_SZ).map(|_| AtomicI32::new(0)).collect())
}

fn history_store(from: usize, to: usize, depth: u32) {
    let idx = (from % HIST_SZ) * HIST_SZ + (to % HIST_SZ);
    let bonus = (depth * depth).min(400) as i32;
    history()[idx].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_add(bonus).min(32767))
    }).ok();
}

fn history_score(from: usize, to: usize) -> i32 {
    let idx = (from % HIST_SZ) * HIST_SZ + (to % HIST_SZ);
    history()[idx].load(Ordering::Relaxed)
}

fn history_clear() {
    if let Some(h) = HIST.get() { for cell in h { cell.store(0, Ordering::Relaxed); } }
    counter_clear();
    TT_GEN.fetch_add(1, Ordering::Relaxed);
}

// ── Counter Move ────────────────────────────────────────────────
static COUNTER: OnceLock<Vec<AtomicU64>> = OnceLock::new();
fn counter() -> &'static Vec<AtomicU64> {
    COUNTER.get_or_init(|| (0..65536).map(|_| AtomicU64::new(0)).collect())
}
fn counter_store(prev: u32, mv: u32) {
    counter()[(prev as usize) & 0xFFFF].store(mv as u64, Ordering::Relaxed);
}
fn counter_score(prev: u32, mv: u32) -> i32 {
    if prev == 0 { return 0; }
    let stored = counter()[(prev as usize) & 0xFFFF].load(Ordering::Relaxed) as u32;
    if stored == mv { 70000 } else { 0 }
}
fn counter_clear() {
    if let Some(c) = COUNTER.get() { for cell in c { cell.store(0, Ordering::Relaxed); } }
}

// ── Piece values ────────────────────────────────────────────────
fn piece_vals() -> &'static [i32; 512] {
    static V: OnceLock<[i32; 512]> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = [0i32; 512];
        for pt in 1..=301u16 { v[pt as usize] = pieces::value(pt); }
        v
    })
}

pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
}

#[inline]
fn m_pack(m: &Move) -> u32 {
    (m.from_sq as u32) | ((m.to_sq as u32) << 12) | (if m.promotion { 1 << 24 } else { 0 })
}

fn is_tactical(m: &Move) -> bool {
    m.captured_piece != 0 || m.mid_piece != 0 || m.promotion || m.range_caps.is_some()
}

// ── Move ordering ───────────────────────────────────────────────
// Priority: 1) Hash move (TT)  2) MVV-LVA captures  3) Killers  4) History  5) Counter

// ── Move ordering ───────────────────────────────────────────────
fn score_move(m: &Move, tt_move: u32, killer: u32, hist: i32, cntr: i32) -> i32 {
    let packed = m_pack(m);
    // 1) Hash move (from TT)
    if packed == tt_move { return 2_000_000; }
    // 2) Captures: MVV-LVA
    if is_tactical(m) {
        let vals = piece_vals();
        let mut score = 1_000_000;
        if m.captured_piece != 0 { score += vals[m.captured_piece as usize] * 100; }
        if m.mid_piece != 0 { score += vals[m.mid_piece as usize] * 100; }
        if let Some(ref caps) = m.range_caps {
            for &(_, pt, _) in caps.iter() { score += vals[pt as usize] * 100; }
        }
        if m.promotion { score += 5000; }
        return score;
    }
    // 3) Killer moves
    if packed == killer { return 90000; }
    // 4) History heuristic
    // 5) Counter move heuristic
    hist + cntr
}

// ── Public Search ──────────────────────────────────────────────
pub fn search(board: &mut Board, depth: u32, time_limit_ms: u64) -> SearchResult {
    let start = Instant::now();
    let deadline = if time_limit_ms > 0 {
        Some(start + std::time::Duration::from_millis(time_limit_ms))
    } else { None };

    piece_vals();
    let mut best_move = None;
    let mut best_score = -MATE_SCORE - 1;
    let mut nodes: u64 = 0;

    let moves = generate_pseudo_legal_moves(board);
    if moves.is_empty() {
        return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }

    // Score and sort root moves
    let mut scored: Vec<(i32, usize)> = moves.iter().enumerate()
        .map(|(i, m)| (score_move(m, 0, 0, 0, 0), i))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    let max_moves = if depth <= 1 { moves.len() } else { moves.len().min(64) };

    if depth <= 1 {
        let base_mat = material_score(board);
        for rank in 0..max_moves {
            let idx = scored[rank].1;
            let m = &moves[idx];
            nodes += 1;
            let mut delta = 0i32;
            let sign = if board.side_to_move == BLACK { 1 } else { -1 };
            if m.promotion {
                let pt = cell_piece(board.cells[m.from_sq as usize]);
                if let Some(p) = pieces::promotes_to(pt) {
                    let v = piece_vals();
                    delta += sign * (v[p as usize] - v[pt as usize]);
                }
            }
            let v = piece_vals();
            if m.captured_piece != 0 { delta += sign * v[m.captured_piece as usize]; }
            if m.mid_piece != 0 { delta += sign * v[m.mid_piece as usize]; }
            if let Some(ref caps) = m.range_caps {
                for &(_, pt, _) in caps.iter() { delta += sign * v[pt as usize]; }
            }
            let s = -(base_mat + delta);
            if s > best_score { best_score = s; best_move = Some(m.clone()); }
        }
    } else {
        // Seed TT
        if depth > 2 {
            let _ = pvs(board, depth - 2, -MATE_SCORE - 1, MATE_SCORE + 1,
                        &mut nodes, deadline, 0, true, 0);
        }

        // Lazy SMP helpers
        let num_helpers = if depth >= 4 && time_limit_ms > 100 {
            let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            (cpus.min(4)).saturating_sub(1)
        } else { 0 };

        let shared_nodes = if num_helpers > 0 { Some(Arc::new(AtomicU64::new(0))) } else { None };
        let mut handles = Vec::new();
        if let Some(sn) = &shared_nodes {
            for _ in 0..num_helpers {
                let mut hb = board.clone();
                let sn = Arc::clone(sn);
                let dl = deadline;
                handles.push(thread::spawn(move || {
                    let mut hn = 0u64;
                    let _ = pvs(&mut hb, depth - 1, -MATE_SCORE - 1, MATE_SCORE + 1,
                                &mut hn, dl, 0, true, 0);
                    sn.fetch_add(hn, Ordering::Relaxed);
                }));
            }
        }

        // Root search with aspiration windows
        for rank in 0..max_moves {
            let idx = scored[rank].1;
            let m = &moves[idx];

            board.apply_move(m);
            if is_in_check(board) { board.undo_move(); continue; }
            nodes += 1;

            let (sa, sb) = if depth > 3 && rank == 0 && best_score > -MATE_SCORE + 100 {
                (best_score - 50, best_score + 50)
            } else {
                (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
            };

            // PVS at root: first move full window, rest with null window
            let score = if rank == 0 {
                -pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, true, m_pack(m))
            } else {
                // Null-window search (egaScout)
                let nw = -pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, true, m_pack(m));
                if nw > sa && nw < sb {
                    -pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, true, m_pack(m))
                } else { nw }
            };

            // Research if aspiration failed
            if score <= sa || score >= sb {
                let full = -pvs(board, depth - 1, -MATE_SCORE - 1,
                                -best_score.max(-MATE_SCORE - 1),
                                &mut nodes, deadline, 0, true, m_pack(m));
                if full > best_score { best_score = full; best_move = Some(m.clone()); }
            } else if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }

            board.undo_move();
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
        }

        for h in handles { h.join().ok(); }
        if let Some(sn) = shared_nodes { nodes += sn.load(Ordering::Relaxed); }
    }

    let elapsed = start.elapsed().as_millis() as u64;
    SearchResult { best_move, score: best_score, nodes, time_ms: elapsed }
}

// ── PVS (Principal Variation Search) with egaScout ─────────────
// This is the core search function implementing:
// - PVS/egaScout: full window for first move, null window for rest
// - NMP: Null Move Pruning (R=2 or R=3)
// - RFP: Reverse Futility Pruning
// - Razoring
// - IID: Internal Iterative Deepening
// - LMR: Late Move Reduction
// - LMP: Late Move Pruning
fn pvs(board: &mut Board, depth: u32, mut alpha: i32, beta: i32,
       nodes: &mut u64, deadline: Option<Instant>, ply: u32,
       pruning: bool, prev_move: u32) -> i32 {
    *nodes += 1;

    // Time check
    // NOTE: Taikyoku Shogi nodes are ~150-200us each (1296-square board,
    // 402 pieces/side -> movegen + is_in_check are far costlier than in a
    // normal chess engine). The old 8191-node interval meant up to ~1.2-1.6s
    // could pass between deadline checks, badly overshooting time_limit_ms.
    // 255 keeps the check itself cheap (bitwise AND) while checking the
    // clock roughly every 40-50ms of real time at this engine's node cost.
    if *nodes & 255 == 0 {
        if let Some(dl) = deadline { if Instant::now() >= dl { return alpha; } }
    }

    // Terminal check
    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins => if board.side_to_move == BLACK { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) },
            GameResult::WhiteWins => if board.side_to_move == WHITE { MATE_SCORE - ply as i32 } else { -(MATE_SCORE - ply as i32) },
            GameResult::Draw => 0,
        };
    }

    // Check extension
    let in_check = is_in_check(board);
    let ext = if in_check && pruning { 1 } else { 0 };
    let d = depth + ext; // effective depth

    // ── TT PROBE ──────────────────────────────────────────────
    let hash = board.hash;
    let tt_move = if let Some((entry, best_mv)) = tt_probe(hash) {
        if (entry.depth as u32) >= d {
            match entry.flag {
                0 => return entry.score,
                1 => if entry.score >= beta { return entry.score; },
                2 => if entry.score <= alpha { return entry.score; },
                _ => {}
            }
        }
        best_mv
    } else { 0 };

    // ── QUIESCENCE ────────────────────────────────────────────
    if d == 0 {
        let total = board.piece_count[0] + board.piece_count[1];
        if total < 200 { return quiescence(board, alpha, beta, nodes, deadline); }
        return evaluate(board);
    }

    // ── STATIC EVAL ───────────────────────────────────────────
    let static_eval = evaluate(board);

    // ── RAZORING (depth ≤ 2) ──────────────────────────────────
    // If static_eval + huge_margin ≤ alpha, prune the node entirely
    if pruning && d <= 2 && !in_check && alpha > -MATE_SCORE + 100 {
        let margin = match d { 0 => 400, 1 => 600, _ => 900 };
        if static_eval + margin <= alpha { return alpha; }
    }

    // ── REVERSE FUTILITY PRUNING (depth ≤ 3) ─────────────────
    // If static_eval - margin ≥ beta, prune (position is too good)
    if pruning && d <= 3 && !in_check && alpha > -MATE_SCORE + 100 {
        let margin = 150 + 250 * d as i32;
        if static_eval - margin >= beta { return beta; }
        if static_eval + margin <= alpha { return alpha; }
    }

    // ── NULL MOVE PRUNING (depth ≥ 3) ─────────────────────────
    // Give opponent a free move. If even then we're still ≥ beta, prune.
    let side = board.side_to_move as usize;
    if pruning && d >= 3 && board.no_progress_plies < 100
        && board.piece_count[side] > 3 && !in_check
    {
        let r = if d >= 6 { 3 } else { 2 };
        board.null_move();
        let null_score = -pvs(board, d.saturating_sub(r), -beta, -(beta - 1),
                              nodes, deadline, ply + 1, pruning, 0);
        board.undo_null_move();
        if null_score >= beta { return beta; }
    }

    // ── INTERNAL ITERATIVE DEEPENING ──────────────────────────
    // If no TT move, do a shallow search to get one
    let iid_move = if tt_move == 0 && d >= 4 && pruning {
        let iid_d = d / 2 - 1;
        let _ = pvs(board, iid_d, -beta, -alpha, nodes, deadline, ply, pruning, prev_move);
        tt_probe(hash).map(|(_, mv)| mv).unwrap_or(0)
    } else { tt_move };

    // ── GENERATE MOVES ────────────────────────────────────────
    let moves = generate_pseudo_legal_moves(board);
    if moves.is_empty() { return -(MATE_SCORE - ply as i32); }

    // ── MOVE ORDERING ─────────────────────────────────────────
    // Score each move: hash > MVV-LVA captures > killers > history > counter
    let killer = {
        let d_idx = d.min(127) as usize;
        killers()[d_idx].load(Ordering::Relaxed) as u32
    };

    let mut scored: Vec<(i32, usize, u32)> = moves.iter().enumerate()
        .map(|(i, m)| {
            let packed = m_pack(m);
            let hist = history_score(m.from_sq as usize, m.to_sq as usize);
            let cntr = counter_score(prev_move, packed);
            let s = score_move(m, iid_move, killer, hist, cntr);
            (s, i, packed)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    // Separate into tactical and quiet for beam search
    let beam = if d <= 1 { 8 } else if d <= 2 { 6 } else if d <= 4 { 4 } else { 3 };
    let mut best: Option<Move> = None;
    let mut tt_flag: u8 = 2; // UPPERBOUND
    let init_alpha = alpha;
    let mut searched = false;

    for (move_idx, &(order_score, idx, packed)) in scored.iter().enumerate() {
        // ── LATE MOVE PRUNING (LMP) ───────────────────────────
        // Skip quiet moves beyond the beam at shallow depths
        if pruning && d <= 2 && move_idx >= beam && order_score < 1_000_000
            && alpha > -MATE_SCORE + 100
        {
            continue;
        }

        // Re-check deadline between sibling moves, not just on function entry.
        // Without this, a child pvs() call can return early on deadline but
        // this loop would keep launching more siblings (each re-entering pvs,
        // incrementing nodes, but not necessarily hitting the next 256-node
        // boundary soon) — the overshoot compounds across ply levels.
        // Skip on move_idx==0 since we must search at least one move.
        if move_idx > 0 {
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
        }

        let m = &moves[idx];
        board.apply_move(m);
        if is_in_check(board) { board.undo_move(); continue; }
        searched = true;

        // ── LATE MOVE REDUCTION (LMR) ─────────────────────────
        // Reduce depth for late quiet moves
        let reduction = if pruning && move_idx >= 3 && d >= 3
            && order_score < 1_000_000 && !in_check
        {
            let base = (move_idx / 3).min(3) as u32;
            let depth_factor = (d / 3).min(2);
            base + depth_factor
        } else { 0 };
        let new_d = d.saturating_sub(1 + reduction);

        // ── PVS / egaScout ────────────────────────────────────
        // First move: full window search
        // Subsequent moves: null-window search (egaScout)
        let score;
        if move_idx == 0 {
            score = -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, pruning, packed);
        } else if reduction > 0 {
            // Null-window search with reduced depth
            let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, pruning, packed);
            if nw > alpha && nw < beta {
                // Re-search with full depth (no reduction)
                score = -pvs(board, d.saturating_sub(1), -beta, -alpha,
                            nodes, deadline, ply + 1, pruning, packed);
            } else { score = nw; }
        } else {
            // Null-window search (egaScout)
            let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, pruning, packed);
            if nw > alpha && nw < beta {
                // Re-search with full window
                score = -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, pruning, packed);
            } else { score = nw; }
        }

        board.undo_move();

        if score > alpha {
            alpha = score;
            tt_flag = 0; // EXACT
            best = Some(m.clone());
        }
        if alpha >= beta {
            tt_flag = 1; // LOWERBOUND
            // Store killer and history for quiet moves that cause beta cutoffs
            if order_score < 1_000_000 {
                killer_store(d, packed);
                history_store(m.from_sq as usize, m.to_sq as usize, d);
            }
            if prev_move != 0 { counter_store(prev_move, packed); }
            break;
        }
    }

    // ── DYNAMIC WIDENING ──────────────────────────────────────
    // If no move raised alpha (fail-low), search more moves
    if !searched || (alpha <= init_alpha && !best.is_some()) {
        for &(_score, idx, packed) in &scored[beam..scored.len().min(beam * 2)] {
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            let m = &moves[idx];
            board.apply_move(m);
            if is_in_check(board) { board.undo_move(); continue; }
            let nw = -pvs(board, d.saturating_sub(2), -alpha - 1, -alpha,
                         nodes, deadline, ply + 1, pruning, packed);
            if nw > alpha {
                let score = -pvs(board, d.saturating_sub(1), -beta, -alpha,
                                nodes, deadline, ply + 1, pruning, packed);
                if score > alpha { alpha = score; tt_flag = 0; best = Some(m.clone()); }
            }
            board.undo_move();
            if alpha >= beta { tt_flag = 1; break; }
        }
    }

    // ── TT STORE ──────────────────────────────────────────────
    if let Some(bm) = &best {
        tt_store(hash, TTEntry {
            score: alpha,
            depth: d as i8,
            flag: if alpha <= init_alpha { 2 } else { tt_flag },
            generation: 0,
            best_move: m_pack(bm),
        });
    }

    alpha
}

// ── Quiescence Search ──────────────────────────────────────────
const MAX_QDEPTH: u32 = 6;

fn quiescence(board: &mut Board, alpha: i32, beta: i32,
              nodes: &mut u64, deadline: Option<Instant>) -> i32 {
    quiescence_inner(board, alpha, beta, nodes, deadline, 0)
}

fn quiescence_inner(board: &mut Board, mut alpha: i32, beta: i32,
                    nodes: &mut u64, deadline: Option<Instant>, qd: u32) -> i32 {
    *nodes += 1;
    // Same reasoning as pvs(): nodes here are expensive (movegen ~120-180us),
    // so check the clock much more often than a typical chess engine would.
    if *nodes & 127 == 0 {
        if let Some(dl) = deadline { if Instant::now() >= dl { return alpha; } }
    }

    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins => if board.side_to_move == BLACK { MATE_SCORE - qd as i32 } else { -(MATE_SCORE - qd as i32) },
            GameResult::WhiteWins => if board.side_to_move == WHITE { MATE_SCORE - qd as i32 } else { -(MATE_SCORE - qd as i32) },
            GameResult::Draw => 0,
        };
    }

    // Stand pat
    let stand_pat = evaluate(board);
    if stand_pat >= beta { return beta; }
    if stand_pat > alpha { alpha = stand_pat; }
    if qd >= MAX_QDEPTH { return alpha; }

    // Generate only captures and promotions
    let moves = generate_pseudo_legal_moves(board);
    let mut scored: Vec<(i32, usize)> = moves.iter().enumerate()
        .filter(|(_, m)| m.captured_piece != 0 || m.mid_piece != 0 || m.is_igui || m.promotion)
        .map(|(i, m)| {
            let v = piece_vals();
            let mut s = 0;
            if m.captured_piece != 0 { s += v[m.captured_piece as usize] * 10; }
            if m.mid_piece != 0 { s += v[m.mid_piece as usize] * 10; }
            if m.promotion { s += 5000; }
            (s, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    for (i_idx, &(_, i)) in scored.iter().enumerate() {
        if i_idx > 0 {
            if let Some(dl) = deadline { if Instant::now() >= dl { return alpha; } }
        }
        let m = &moves[i];
        board.apply_move(m);
        if is_in_check(board) { board.undo_move(); continue; }
        let score = -quiescence_inner(board, -beta, -alpha, nodes, deadline, qd + 1);
        board.undo_move();
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}