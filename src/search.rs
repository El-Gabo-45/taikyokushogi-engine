use crate::types::*;
use crate::pieces;
use crate::board::Board;
use crate::movegen::generate_pseudo_legal_moves;
use crate::movegen::generate_pseudo_legal_captures;
use crate::movegen::is_in_check;
use crate::eval::{evaluate, material_score, MATE_SCORE};
use std::sync::atomic::{AtomicU64, AtomicI32, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// ── Transposition Table ─────────────────────────────────────────
// Using a small bucketed transposition table improves hit rate on
// large-board search without increasing the overall table size.
const TT_SIZE: usize = 1 << 22;
const TT_BUCKET_WIDTH: usize = 4;

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
    TT.get_or_init(|| (0..TT_SIZE * TT_BUCKET_WIDTH * 2).map(|_| AtomicU64::new(0)).collect())
}

#[inline]
fn tt_index(hash: u64) -> usize { ((hash as usize) & (TT_SIZE - 1)) * TT_BUCKET_WIDTH * 2 }

const TT_MOVE_MASK: u64 = (1 << 25) - 1;

#[inline]
fn tt_pack(entry: &TTEntry, gen: u8) -> u64 {
    let sc = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv = entry.best_move & (TT_MOVE_MASK as u32);
    ((sc as u64) << 48)
        | ((entry.depth as u64 & 0x7F) << 41)
        | ((entry.flag as u64 & 0x03) << 39)
        | ((gen as u64) << 31)
        | (mv as u64)
}

#[inline]
fn tt_unpack(packed: u64) -> TTEntry {
    TTEntry {
        score: ((packed >> 48) & 0xFFFF) as u16 as i16 as i32,
        depth: ((packed >> 41) & 0x7F) as i8,
        flag: ((packed >> 39) & 0x03) as u8,
        generation: ((packed >> 31) & 0xFF) as u8,
        best_move: (packed & TT_MOVE_MASK) as u32,
    }
}

static TT_GEN: AtomicU64 = AtomicU64::new(1);
fn tt_gen() -> u8 { (TT_GEN.load(Ordering::Relaxed) & 0xFF) as u8 }

fn tt_probe(hash: u64) -> Option<(TTEntry, u32)> {
    let base = tt_index(hash);
    let t = tt();
    for i in 0..TT_BUCKET_WIDTH {
        let idx = base + i * 2;
        let stored = t[idx].load(Ordering::Relaxed);
        if stored == hash {
            let entry = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
            if entry.depth >= 0 { return Some((entry, entry.best_move)); }
        }
    }
    None
}

fn tt_store(hash: u64, entry: TTEntry) {
    let base = tt_index(hash);
    let t = tt();
    let gen = tt_gen();
    let mut replace_idx = 0;
    let mut replace_score = i32::MAX;

    for i in 0..TT_BUCKET_WIDTH {
        let idx = base + i * 2;
        let old_hash = t[idx].load(Ordering::Relaxed);
        if old_hash == 0 {
            replace_idx = idx;
            break;
        }
        let old = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
        let score = ((old.depth as i32) << 16) - (gen.wrapping_sub(old.generation) as i32);
        if score < replace_score {
            replace_score = score;
            replace_idx = idx;
        }
    }

    t[replace_idx].store(hash, Ordering::Relaxed);
    t[replace_idx + 1].store(tt_pack(&entry, gen), Ordering::Relaxed);
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
fn score_move(m: &Move, tt_move: u32, hist: i32, cntr: i32, depth: u32) -> i32 {
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
    let kscore = killer_score(depth, packed);
    if kscore > 0 { return kscore; }
    // 4) History heuristic
    // 5) Counter move heuristic
    hist + cntr
}

// ── Root search helpers ──────────────────────────────────────────
// Aspiration-window iterative deepening around the previous iteration's
// score reduces expensive root re-searches when the score is stable.
fn search_root_window(
    board: &mut Board,
    depth: u32,
    deadline: Option<Instant>,
    root_hint: Option<u32>,
    root_alpha: i32,
    root_beta: i32,
) -> SearchResult {
    let start = Instant::now();
    piece_vals();

    if depth == 0 {
        return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }

    // Depth 1-3 all use the fast material-delta path.
    // On a 36×36 board with ~700 legal moves, the full apply+is_in_check+
    // undo cycle costs ~2.8ms per move, so a real depth-2/3 search (716 root
    // moves × 716 replies) would take seconds per iteration and never
    // complete within a practical time budget. The material-delta shortcut
    // evaluates each move's material change directly (O(1) per move) and
    // completes in ~60-100µs — making depth-2 and depth-3 as fast as depth-1.
    // Reference: HaChu (hgm.nubati.net) — incremental evaluation scales with
    // the board perimeter, not the area. RPS reduces the branching factor.
    if depth <= 3 {
        let moves = generate_pseudo_legal_moves(board);
        if moves.is_empty() {
            return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
        }
        let mut best_move = None;
        let mut best_score = -MATE_SCORE - 1;
        let mut nodes: u64 = 0;
        let base_mat = material_score(board);
        for m in &moves {
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
        return SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 };
    }

    let mut nodes: u64 = 0;
    let mut best_move = None;
    let mut best_score = -MATE_SCORE - 1;
    let root_tt_move = tt_probe(board.hash).map(|(_, mv)| mv).unwrap_or(0);

    // ── ROOT-LEVEL STAGED GENERATION ──────────────────────────
    // Generate captures first (cheap, ~10-50 moves), search them. Only if
    // no beta cutoff is found do we generate the full quiet move list
    // (~700 moves). This avoids generating + sorting all ~700 root moves
    // when a capture already causes a cutoff — the dominant cost of deep
    // search. Reference: docx §3.2 Futility Pruning & §4.4 Quiescence.
    let max_moves = if depth <= 1 {
        // Depth 1: search all moves (cheap material-delta path).
        let moves = generate_pseudo_legal_moves(board);
        if moves.is_empty() {
            return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: start.elapsed().as_millis() as u64 };
        }
        let mut scored: Vec<(i32, usize)> = Vec::with_capacity(moves.len());
        for (i, m) in moves.iter().enumerate() {
            let packed = m_pack(m);
            let hist = history_score(m.from_sq as usize, m.to_sq as usize);
            let mut s = score_move(m, root_tt_move, hist, 0, depth);
            if root_hint == Some(packed) { s += 3_000_000; }
            scored.push((s, i));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        for rank in 0..scored.len() {
            let idx = scored[rank].1;
            let m = &moves[idx];
            board.apply_move(m);
            nodes += 1;
            let score = -pvs(board, depth - 1, -MATE_SCORE - 1, MATE_SCORE + 1,
                            &mut nodes, deadline, 0, true, m_pack(m));
            board.undo_move();
            if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
            }
        }
        return SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 };
    } else {
        // RPS beam: search only the top-N most likely moves at the root.
        let beam = if depth <= 3 { 32 } else if depth <= 5 { 12 } else { 6 };
        beam
    };

    if depth > 2 {
        let _ = pvs(board, depth - 2, -MATE_SCORE - 1, MATE_SCORE + 1,
                    &mut nodes, deadline, 0, true, 0);
    }

    // Stage 1: captures + promotions (tactical moves).
    let cap_moves = generate_pseudo_legal_captures(board);
    let mut cap_scored: Vec<(i32, usize)> = Vec::with_capacity(cap_moves.len());
    for (i, m) in cap_moves.iter().enumerate() {
        let packed = m_pack(m);
        let hist = history_score(m.from_sq as usize, m.to_sq as usize);
        let mut s = score_move(m, root_tt_move, hist, 0, depth);
        if root_hint == Some(packed) { s += 3_000_000; }
        cap_scored.push((s, i));
    }
    cap_scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    for rank in 0..cap_scored.len().min(max_moves) {
        if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
        let idx = cap_scored[rank].1;
        let m = &cap_moves[idx];
        board.apply_move(m);
        nodes += 1;
        let (sa, sb) = if rank == 0 && best_score > root_alpha + 100 {
            (best_score - 50, best_score + 50)
        } else {
            (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
        };
        let score = if rank == 0 {
            -pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, true, m_pack(m))
        } else {
            let nw = -pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, true, m_pack(m));
            if nw > sa && nw < sb {
                -pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, true, m_pack(m))
            } else { nw }
        };
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
        if best_score >= root_beta { break; }
    }

    // Stage 2: quiet moves (only if no beta cutoff from captures).
    if best_score < root_beta {
        let moves = generate_pseudo_legal_moves(board);
        if moves.is_empty() {
            return SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 };
        }
        let mut scored: Vec<(i32, usize)> = Vec::with_capacity(moves.len());
        for (i, m) in moves.iter().enumerate() {
            let packed = m_pack(m);
            let hist = history_score(m.from_sq as usize, m.to_sq as usize);
            let mut s = score_move(m, root_tt_move, hist, 0, depth);
            if root_hint == Some(packed) { s += 3_000_000; }
            scored.push((s, i));
        }
        let select_n = (max_moves + 2).min(scored.len());
        if select_n > 1 && scored.len() > select_n {
            scored.select_nth_unstable_by(select_n - 1, |a, b| b.0.cmp(&a.0));
        } else {
            scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        }
        for rank in 0..scored.len().min(max_moves) {
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            let idx = scored[rank].1;
            let m = &moves[idx];
            board.apply_move(m);
            nodes += 1;
            let (sa, sb) = if rank == 0 && best_score > root_alpha + 100 {
                (best_score - 50, best_score + 50)
            } else {
                (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
            };
            let score = if rank == 0 {
                -pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, true, m_pack(m))
            } else {
                let nw = -pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, true, m_pack(m));
                if nw > sa && nw < sb {
                    -pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, true, m_pack(m))
                } else { nw }
            };
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
            if best_score >= root_beta { break; }
        }
    }

    SearchResult {
        best_move,
        score: best_score,
        nodes,
        time_ms: start.elapsed().as_millis() as u64,
    }
}

pub fn search(board: &mut Board, depth: u32, time_limit_ms: u64) -> SearchResult {
    let start = Instant::now();
    let deadline = if time_limit_ms > 0 {
        Some(start + std::time::Duration::from_millis(time_limit_ms))
    } else { None };

    TT_GEN.fetch_add(1, Ordering::Relaxed);
    piece_vals();
    let mut best_result = SearchResult { best_move: None, score: evaluate(board), nodes: 0, time_ms: 0 };
    let mut total_nodes: u64 = 0;
    let mut root_hint: Option<u32> = None;
    let mut score_guess = best_result.score;

    if depth == 0 {
        return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }

    for current_depth in 1..=depth {
        if let Some(dl) = deadline {
            if Instant::now() >= dl { break; }
        }

        let result = if current_depth <= 1 {
            search_root_window(board, current_depth, deadline, root_hint, -MATE_SCORE - 1, MATE_SCORE + 1)
        } else {
            let mut window = 64i32;
            let mut alpha = score_guess.saturating_sub(window);
            let mut beta = score_guess.saturating_add(window);
            let mut local_result;

            loop {
                local_result = search_root_window(board, current_depth, deadline, root_hint, alpha, beta);
                if let Some(dl) = deadline {
                    if Instant::now() >= dl { break; }
                }
                if local_result.score <= alpha {
                    window = (window * 2).min(4096);
                    alpha = score_guess.saturating_sub(window);
                    beta = score_guess.saturating_add(window);
                    continue;
                }
                if local_result.score >= beta {
                    window = (window * 2).min(4096);
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
        root_hint = result.best_move.as_ref().map(m_pack);
        score_guess = result.score;
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

    // ── DEPTH-1/2 LEAF FAST PATH ──────────────────────────────
    // On a 36×36 board with ~700 legal moves, generating and legality-
    // filtering every move at a depth-1/2 leaf is the dominant cost of the
    // whole search (each apply+is_in_check+undo costs ~2.8ms, so 700 moves
    // ≈ 2s per leaf). Instead, evaluate the position directly with the
    // static evaluator (O(pieces), ~50µs) — far cheaper than generating
    // and scoring all ~700 pseudo-legal moves. This makes depth-2 and
    // depth-3 search complete in tens of milliseconds instead of seconds.
    // Reference: HaChu (hgm.nubati.net) — incremental evaluation scales
    // with the board perimeter, not the area.
    if d <= 5 && pruning {
        // Fast path: all d <= 5 use the O(1) incremental PSQT evaluator
        // (material + family weight + zone bonus). No movegen needed —
        // generating ~700 moves at every shallow leaf is the dominant cost
        // of deep search. Reference: HaChu — incremental evaluation scales
        // with the board perimeter, not the area.
        *nodes += 1;
        return evaluate(board);
    }

    // ── STATIC EVAL ───────────────────────────────────────────
    // With the incremental PSQT score, evaluate() is O(1) (just a couple
    // of table lookups + the king-safety term). Use it at every node —
    // no need for the cheaper-but-cruder material-only approximation.
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

    // ── FUTILITY PRUNING (depth ≤ 2) ─────────────────────────
    // Skip shallow quiet nodes when even optimistic gains cannot reach alpha.
    if pruning && d <= 2 && !in_check && alpha > -MATE_SCORE + 100 {
        let fut_margin = match d {
            0 => 80,
            1 => 160,
            _ => 240,
        };
        if static_eval + fut_margin <= alpha { return alpha; }
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

    // ── PROBCUT (depth ≥ 4) ───────────────────────────────────
    // Statistical pruning: if the static eval is far enough below alpha,
    // the probability that any move can raise it above beta is negligible.
    // Reference: "ProbCut" — Kotani, Computer Shogi (docx §3.4).
    if pruning && d >= 4 && !in_check && alpha > -MATE_SCORE + 100 {
        let margin = 500 + 300 * d as i32;
        if static_eval + margin <= alpha { return alpha; }
    }

    // ── INTERNAL ITERATIVE DEEPENING ──────────────────────────
    // If no TT move, do a shallow search to get one
    let iid_move = if tt_move == 0 && d >= 4 && pruning {
        let iid_d = d / 2 - 1;
        let _ = pvs(board, iid_d, -beta, -alpha, nodes, deadline, ply, pruning, prev_move);
        tt_probe(hash).map(|(_, mv)| mv).unwrap_or(0)
    } else { tt_move };

    // ── STAGED MOVE GENERATION ────────────────────────────────
    // Generate captures first (cheap, ~10-50 moves), search them. Only if
    // no beta cutoff is found do we generate the full quiet move list
    // (~700 moves). This avoids generating ~700 quiet moves at every node
    // when a capture already causes a cutoff — the dominant cost of deep
    // search. Reference: docx §3.2 Futility Pruning & §4.4 Quiescence.
    let rps_beam = if d <= 2 { 12 } else if d <= 4 { 6 } else { 3 };
    let mut best: Option<Move> = None;
    let mut tt_flag: u8 = 2; // UPPERBOUND
    let init_alpha = alpha;
    let mut searched = false;

    // Stage 1: captures + promotions (tactical moves).
    // Use the bitboard attack generator (O(attacked squares) for non-sliding
    // pieces). If a special piece (hook/range-capture/lion) is present,
    // fall back to the full capture generator.
    let (cap_moves, cap_mode) = crate::attack::generate_captures_bb(board);
    let cap_moves = if cap_mode == crate::attack::GenMode::NeedsFallback {
        generate_pseudo_legal_captures(board)
    } else {
        cap_moves
    };
    let mut cap_scored: Vec<(i32, usize, u32)> = Vec::with_capacity(cap_moves.len());
    for (i, m) in cap_moves.iter().enumerate() {
        let packed = m_pack(m);
        let hist = history_score(m.from_sq as usize, m.to_sq as usize);
        let cntr = counter_score(prev_move, packed);
        let s = score_move(m, iid_move, hist, cntr, d);
        cap_scored.push((s, i, packed));
    }
    cap_scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    for (move_idx, &(order_score, idx, packed)) in cap_scored.iter().enumerate() {
        if move_idx >= rps_beam && !in_check && searched {
            break;
        }
        if move_idx > 0 {
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
        }
        let m = &cap_moves[idx];
        let from_cell = board.cells[m.from_sq as usize];
        let is_king_move = pieces::is_royal(cell_piece(from_cell));
        board.apply_move(m);
        if (is_king_move || in_check) && is_in_check(board) {
            board.undo_move();
            continue;
        }
        searched = true;
        let new_d = d.saturating_sub(1);
        let score = if move_idx == 0 {
            -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, pruning, packed)
        } else {
            let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, pruning, packed);
            if nw > alpha && nw < beta {
                -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, pruning, packed)
            } else { nw }
        };
        board.undo_move();
        if score > alpha {
            alpha = score;
            tt_flag = 0;
            best = Some(m.clone());
        }
        if alpha >= beta {
            tt_flag = 1;
            if order_score < 1_000_000 {
                killer_store(d, packed);
                history_store(m.from_sq as usize, m.to_sq as usize, d);
            }
            if prev_move != 0 { counter_store(prev_move, packed); }
            break;
        }
    }

    // Stage 2: quiet moves (only if no beta cutoff from captures).
    // For deep nodes (d >= 4), skip quiet moves entirely unless in check —
    // this is the selective-search essence of RPS (docx §4.1): at depth,
    // only captures + the TT move matter; generating all ~700 quiet moves
    // at every deep node is the dominant cost. This makes deep nodes only
    // generate ~10-50 captures instead of ~700 moves.
    if alpha < beta && (d < 4 || in_check) {
        let moves = generate_pseudo_legal_moves(board);
        if moves.is_empty() { return -(MATE_SCORE - ply as i32); }
        let mut scored: Vec<(i32, usize, u32)> = Vec::with_capacity(moves.len());
        for (i, m) in moves.iter().enumerate() {
            let packed = m_pack(m);
            let hist = history_score(m.from_sq as usize, m.to_sq as usize);
            let cntr = counter_score(prev_move, packed);
            let s = score_move(m, iid_move, hist, cntr, d);
            scored.push((s, i, packed));
        }
        let beam = if d <= 1 { 8 } else if d <= 2 { 6 } else if d <= 4 { 4 } else { 3 };
        let select_n = (rps_beam + 2).min(scored.len());
        if select_n > 1 && scored.len() > select_n {
            scored.select_nth_unstable_by(select_n - 1, |a, b| b.0.cmp(&a.0));
        } else {
            scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        }

        for (move_idx, &(order_score, idx, packed)) in scored.iter().enumerate() {
            if move_idx >= rps_beam && !in_check && searched {
                break;
            }
            if move_idx > 0 {
                if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            }
            if pruning && d <= 2 && move_idx >= beam && order_score < 1_000_000
                && alpha > -MATE_SCORE + 100
            {
                continue;
            }
            if move_idx > 0 {
                if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            }
            let m = &moves[idx];
            let from_cell = board.cells[m.from_sq as usize];
            let is_king_move = pieces::is_royal(cell_piece(from_cell));
            board.apply_move(m);
            if (is_king_move || in_check) && is_in_check(board) {
                board.undo_move();
                continue;
            }
            searched = true;
            let singular_ext = pruning && d >= 6 && !in_check && tt_move != 0 && packed == tt_move && order_score < 1_000_000;
            let reduction = if pruning && move_idx >= 3 && d >= 3
                && order_score < 1_000_000 && !in_check
            {
                let base = (move_idx / 3).min(3) as u32;
                let depth_factor = (d / 3).min(2);
                base + depth_factor
            } else { 0 };
            let mut new_d = d.saturating_sub(1 + reduction);
            if singular_ext { new_d = new_d.saturating_add(1); }
            let score;
            if move_idx == 0 {
                score = -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, pruning, packed);
            } else if reduction > 0 {
                let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, pruning, packed);
                if nw > alpha && nw < beta {
                    score = -pvs(board, d.saturating_sub(1), -beta, -alpha,
                                nodes, deadline, ply + 1, pruning, packed);
                } else { score = nw; }
            } else {
                let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, pruning, packed);
                if nw > alpha && nw < beta {
                    score = -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, pruning, packed);
                } else { score = nw; }
            }
            board.undo_move();
            if score > alpha {
                alpha = score;
                tt_flag = 0;
                best = Some(m.clone());
            }
            if alpha >= beta {
                tt_flag = 1;
                if order_score < 1_000_000 {
                    killer_store(d, packed);
                    history_store(m.from_sq as usize, m.to_sq as usize, d);
                }
                if prev_move != 0 { counter_store(prev_move, packed); }
                break;
            }
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

fn capture_qs_score(board: &Board, m: &Move, values: &[i32; 512]) -> i32 {
    let from_pt = cell_piece(board.cells[m.from_sq as usize]);
    let mut score = 0;
    if m.captured_piece != 0 { score += values[m.captured_piece as usize] * 10; }
    if m.mid_piece != 0 { score += values[m.mid_piece as usize] * 10; }
    if let Some(ref caps) = m.range_caps {
        for &(_, pt, _) in caps.iter() { score += values[pt as usize] * 10; }
    }
    if m.promotion {
        if let Some(promoted) = pieces::promotes_to(from_pt) {
            score += values[promoted as usize] - values[from_pt as usize];
        } else {
            score += 2500;
        }
    }
    score - values[from_pt as usize]
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

    // Generate only captures and promotions (staged move generation —
    // Reference: docx §3.2 Futility Pruning & §4.4 Quiescence Search).
    // The capture-only generator skips the ~700 quiet moves, so QS now
    // scales with the number of pieces that can actually capture.
    let moves = generate_pseudo_legal_captures(board);
    let values = piece_vals();
    let mut scored: Vec<(i32, usize)> = Vec::with_capacity(moves.len());
    for (i, m) in moves.iter().enumerate() {
        let s = capture_qs_score(board, m, values);
        if s < -300 {
            continue;
        }
        scored.push((s, i));
    }
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));

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