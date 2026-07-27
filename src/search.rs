use crate::types::*;
use crate::pieces;
use crate::board::Board;
use crate::movegen::generate_pseudo_legal_moves;
use crate::movegen::is_in_check;
use crate::eval::{evaluate, material_score, MATE_SCORE};
use std::sync::atomic::{AtomicU64, AtomicI32, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// ── Transposition Table (4M entries, depth-preferred + generation aging) ─
const TT_SIZE: usize = 1 << 22;

#[derive(Clone, Copy)]
struct TTEntry {
    score: i32,
    depth: i8,
    flag: u8,
    generation: u8,
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
fn tt_pack_entry(entry: &TTEntry, generation: u8) -> u64 {
    let score_clamped = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv16 = (entry.best_move & 0xFFFF) as u16;
    ((score_clamped as u64) << 32)
        | ((entry.depth as u64 & 0x7F) << 25)
        | ((entry.flag as u64) << 16)
        | ((generation as u64) << 8)
        | (mv16 as u64)
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

static TT_GENERATION: AtomicU64 = AtomicU64::new(1);

fn tt_generation() -> u8 {
    (TT_GENERATION.load(Ordering::Relaxed) & 0xFF) as u8
}

fn tt_probe(hash: u64) -> Option<(TTEntry, u32)> {
    let idx = tt_index(hash) * 2;
    let t = tt();
    let stored_hash = t[idx].load(Ordering::Relaxed);
    if stored_hash == hash {
        let entry = tt_unpack(t[idx + 1].load(Ordering::Relaxed));
        if entry.depth >= 0 {
            return Some((entry, entry.best_move));
        }
    }
    None
}

fn tt_store(hash: u64, entry: TTEntry) {
    let idx = tt_index(hash) * 2;
    let t = tt();
    let gen = tt_generation();
    let old_hash = t[idx].load(Ordering::Relaxed);
    let old_data = t[idx + 1].load(Ordering::Relaxed);
    let old_entry = tt_unpack(old_data);
    let replace = old_hash == 0
        || entry.depth > old_entry.depth
        || (entry.depth == old_entry.depth && gen.wrapping_sub(old_entry.generation) > 100);
    if replace {
        t[idx].store(hash, Ordering::Relaxed);
        t[idx + 1].store(tt_pack_entry(&entry, gen), Ordering::Relaxed);
    }
}

// ── Killers (atomic) ─────────────────────────────────────────────
static KILLER_MOVES: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn killers() -> &'static Vec<AtomicU64> {
    KILLER_MOVES.get_or_init(|| (0..256).map(|_| AtomicU64::new(0)).collect())
}

fn killer_store(depth: u32, mv: u32) {
    let d = depth.min(127) as usize;
    let slot = &killers()[d];
    let current = slot.load(Ordering::Relaxed);
    let mv0 = current as u32;
    if mv != mv0 {
        slot.store(mv as u64 | ((mv0 as u64) << 32), Ordering::Relaxed);
    }
}

fn killer_score(depth: u32, mv: u32) -> i32 {
    let d = depth.min(127) as usize;
    let packed = killers()[d].load(Ordering::Relaxed);
    let mv0 = packed as u32;
    let mv1 = (packed >> 32) as u32;
    if mv == mv0 { 90000 }
    else if mv == mv1 { 80000 }
    else { 0 }
}

// ── History (AtomicI32) ──────────────────────────────────────────
const HIST_SIZE: usize = 256;
static HISTORY: OnceLock<Vec<AtomicI32>> = OnceLock::new();

fn history() -> &'static Vec<AtomicI32> {
    HISTORY.get_or_init(|| (0..HIST_SIZE * HIST_SIZE).map(|_| AtomicI32::new(0)).collect())
}

fn history_store(from: usize, to: usize, depth: u32) {
    let idx = (from % HIST_SIZE) * HIST_SIZE + (to % HIST_SIZE);
    let bonus = (depth * depth).min(400) as i32;
    history()[idx].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_add(bonus).min(32767))
    }).ok();
}

fn history_score(from: usize, to: usize) -> i32 {
    let idx = (from % HIST_SIZE) * HIST_SIZE + (to % HIST_SIZE);
    history()[idx].load(Ordering::Relaxed)
}

fn history_clear() {
    if let Some(h) = HISTORY.get() {
        for cell in h.iter() {
            cell.store(0, Ordering::Relaxed);
        }
    }
    counter_move_clear();
    TT_GENERATION.fetch_add(1, Ordering::Relaxed);
}

// ── Counter Move Heuristic ───────────────────────────────────────
// For (move, depth) store a "counter move" that refuted it
static COUNTER_MOVES: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn counter_moves() -> &'static Vec<AtomicU64> {
    COUNTER_MOVES.get_or_init(|| (0..65536).map(|_| AtomicU64::new(0)).collect())
}

fn counter_move_store(prev_move: u32, counter: u32) {
    let idx = (prev_move as usize) & 0xFFFF;
    counter_moves()[idx].store(counter as u64, Ordering::Relaxed);
}

fn counter_move_score(prev_move: u32, mv: u32) -> i32 {
    let idx = (prev_move as usize) & 0xFFFF;
    let stored = counter_moves()[idx].load(Ordering::Relaxed) as u32;
    if stored == mv { 70000 } else { 0 }
}

fn counter_move_clear() {
    if let Some(h) = COUNTER_MOVES.get() {
        for cell in h.iter() {
            cell.store(0, Ordering::Relaxed);
        }
    }
}

// ── Precomputed piece values ─────────────────────────────────────
fn init_piece_values() -> &'static [i32; 512] {
    static VALS: OnceLock<[i32; 512]> = OnceLock::new();
    VALS.get_or_init(|| {
        let mut vals = [0i32; 512];
        for pt in 1..=301u16 {
            vals[pt as usize] = pieces::value(pt);
        }
        vals
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

fn move_order_score(m: &Move) -> i32 {
    let vals = init_piece_values();
    let mut score = 0i32;
    if m.captured_piece != 0 { score += vals[m.captured_piece as usize] * 10; }
    if m.promotion { score += 5000; }
    if m.mid_piece != 0 { score += vals[m.mid_piece as usize] * 5; }
    if let Some(ref caps) = m.range_caps {
        for &(_, pt, _) in caps { score += vals[pt as usize] * 10; }
    }
    score
}

// ── Move Grouping ────────────────────────────────────────────────
// Split moves into tactical (must-see) and quiet (can be pruned)
struct MoveGroups {
    tactical: Vec<(i32, usize, u32)>, // (score, index, packed)
    quiet: Vec<(i32, usize, u32)>,
}

fn group_moves<'a>(moves: &'a [Move], scored: &[(i32, usize)],
                   depth: u32, iid_move: u32, tt_best: u32,
                   prev_move: u32) -> MoveGroups {
    let mut tactical = Vec::with_capacity(scored.len().min(32));
    let mut quiet = Vec::with_capacity(scored.len().min(32));

    for &(base_score, idx) in scored {
        let m = &moves[idx];
        let packed = m_pack(m);
        let mut score = base_score;
        if packed == iid_move { score += 2_000_000; }
        if packed == tt_best { score += 1_000_000; }
        score += killer_score(depth, packed);
        score += history_score(m.from_sq as usize, m.to_sq as usize);
        score += counter_move_score(prev_move, packed);

        if is_tactical(m) {
            tactical.push((score, idx, packed));
        } else {
            quiet.push((score, idx, packed));
        }
    }
    // Both groups already sorted by the caller's sort
    MoveGroups { tactical, quiet }
}

// ── Beam Width Calculator (Progressive Widening) ────────────────
// Returns beam width for a given depth and move index.
// The beam widens as depth decreases (we're closer to root).
fn beam_width(effective_depth: u32, move_idx: usize, fail_low: bool) -> usize {
    let base = if effective_depth <= 1 { 8 }
        else if effective_depth <= 2 { 6 }
        else if effective_depth <= 4 { 4 }
        else { 3 };
    
    // If we failed low (alpha wasn't raised), widen the beam
    let widened = if fail_low { base * 2 } else { base };
    
    // Progressive widening: for moves beyond the beam, we still search
    // but with reduced depth (handled by LMR)
    widened
}

// ── Public Search ──────────────────────────────────────────────
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

    init_piece_values();

    let moves = generate_pseudo_legal_moves(board);
    if moves.is_empty() {
        return SearchResult { best_move: None, score: evaluate(board), nodes: 1, time_ms: 0 };
    }

    let mut scored_moves: Vec<(i32, usize)> = moves.iter().enumerate()
        .map(|(i, m)| (move_order_score(m), i))
        .collect();
    scored_moves.sort_by(|a, b| b.0.cmp(&a.0));

    let beam = if depth <= 1 { moves.len() } else { 64 };
    let max_moves = beam.min(moves.len());

    if depth <= 1 {
        // Fast material-delta for depth-1
        let base_mat = material_score(board);
        for rank in 0..max_moves {
            let idx = scored_moves[rank].1;
            let m = &moves[idx];
            nodes += 1;
            let mut delta = 0i32;
            let sign = if board.side_to_move == BLACK { 1 } else { -1 };
            if m.promotion {
                let pt = cell_piece(board.cells[m.from_sq as usize]);
                if let Some(promo_pt) = pieces::promotes_to(pt) {
                    let vals = init_piece_values();
                    delta += sign * (vals[promo_pt as usize] - vals[pt as usize]);
                }
            }
            let vals = init_piece_values();
            if m.captured_piece != 0 { delta += sign * vals[m.captured_piece as usize]; }
            if m.mid_piece != 0 { delta += sign * vals[m.mid_piece as usize]; }
            if let Some(ref caps) = m.range_caps {
                for &(_, pt, _) in caps { delta += sign * vals[pt as usize]; }
            }
            let score = -(base_mat + delta);
            if score > best_score { best_score = score; best_move = Some(m.clone()); }
        }
    } else {
        // Seed TT
        if depth > 2 {
            let _ = alphabeta(board, depth - 2, -MATE_SCORE - 1, MATE_SCORE + 1,
                             &mut nodes, deadline, 0, false, 0);
        }

        let mut prev_root_packed = 0u32;
        for rank in 0..max_moves {
            let idx = scored_moves[rank].1;
            let m = &moves[idx];

            board.apply_move(m);
            if is_in_check(board) {
                board.undo_move();
                continue;
            }
            nodes += 1;

            // Aspiration windows
            let (sa, sb) = if depth > 3 && rank == 0 && best_score > -MATE_SCORE + 100 {
                (best_score - 50, best_score + 50)
            } else {
                (-MATE_SCORE - 1, -best_score.max(-MATE_SCORE - 1))
            };

            let score = -alphabeta(board, depth - 1, sa, sb, &mut nodes, deadline, 0, false, m_pack(m));

            // Research if aspiration failed
            if score <= sa || score >= sb {
                let full = -alphabeta(board, depth - 1, -MATE_SCORE - 1,
                                      -best_score.max(-MATE_SCORE - 1),
                                      &mut nodes, deadline, 0, false, m_pack(m));
                if full > best_score { best_score = full; best_move = Some(m.clone()); }
            } else if score > best_score {
                best_score = score;
                best_move = Some(m.clone());
                prev_root_packed = m_pack(m);
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

// ── Alpha-Beta with Advanced Pruning ──────────────────────────
fn alphabeta(board: &mut Board, depth: u32, mut alpha: i32, beta: i32,
             nodes: &mut u64, deadline: Option<Instant>, ply: u32,
             prune_hard: bool, prev_move: u32) -> i32 {
    *nodes += 1;

    if *nodes & 8191 == 0 {
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

    let in_check = is_in_check(board);
    let ext = if in_check && prune_hard { 1 } else { 0 };
    let effective_depth = depth + ext;

    let hash = board.hash;
    let tt_best_move = if let Some((entry, best_mv)) = tt_probe(hash) {
        if (entry.depth as u32) >= effective_depth {
            match entry.flag {
                0 => return entry.score,
                1 => { if entry.score >= beta  { return entry.score; } }
                2 => { if entry.score <= alpha { return entry.score; } }
                _ => {}
            }
        }
        best_mv
    } else { 0 };

    if effective_depth == 0 {
        let total_pieces = board.piece_count[0] + board.piece_count[1];
        if total_pieces < 200 { return quiescence(board, alpha, beta, nodes, deadline); }
        return evaluate(board);
    }

    // ── Razoring / RFP ──────────────────────────────────────
    if prune_hard && effective_depth <= 2 && !in_check {
        let static_eval = evaluate(board);
        if static_eval + 300 + 200 * effective_depth as i32 <= alpha { return alpha; }
        if static_eval - 250 * effective_depth as i32 >= beta { return beta; }
    }
    if prune_hard && depth <= 3 && !in_check && alpha > -MATE_SCORE + 100 {
        let static_eval = evaluate(board);
        if static_eval + 300 + 200 * depth as i32 <= alpha { return alpha; }
    }

    // ── Null Move Pruning ────────────────────────────────────
    let side = board.side_to_move as usize;
    if prune_hard && effective_depth >= 3
        && board.no_progress_plies < 100
        && board.piece_count[side] > 3
        && !in_check
    {
        let r = if effective_depth >= 6 { 3 } else { 2 };
        board.null_move();
        let null_score = -alphabeta(board, effective_depth.saturating_sub(r), -beta, -(beta - 1),
                                    nodes, deadline, ply + 1, prune_hard, 0);
        board.undo_null_move();
        if null_score >= beta { return beta; }
    }

    // ── IID ──────────────────────────────────────────────────
    let iid_move = if tt_best_move == 0 && effective_depth >= 4 && prune_hard {
        let iid_depth = effective_depth / 2 - 1;
        let _ = alphabeta(board, iid_depth, -beta, -alpha, nodes, deadline, ply, prune_hard, prev_move);
        tt_probe(hash).map(|(_, mv)| mv).unwrap_or(0)
    } else { tt_best_move };

    // ── Generate moves ────────────────────────────────────────
    let moves = generate_pseudo_legal_moves(board);
    if moves.is_empty() { return -(MATE_SCORE - ply as i32); }

    // Score and sort
    let mut scored_moves: Vec<(i32, usize)> = moves.iter().enumerate()
        .map(|(i, m)| (move_order_score(m), i))
        .collect();
    scored_moves.sort_by(|a, b| b.0.cmp(&a.0));

    // ── Move GROUPING ─────────────────────────────────────────
    // Separate into tactical (captures/promos) and quiet moves
    let groups = group_moves(&moves, &scored_moves, effective_depth,
                             iid_move, tt_best_move, prev_move);

    // ── BEAM SEARCH + PROGRESSIVE WIDENING ───────────────────
    // Determine beam size. Start narrow, widen if needed.
    let beam = beam_width(effective_depth, 0, false);
    let max_tactical = groups.tactical.len().min(beam);
    let max_quiet = groups.quiet.len().min(beam / 2); // fewer quiet moves

    let mut tt_flag: u8 = 2;
    let mut best_local: Option<Move> = None;
    let init_alpha = alpha;
    let mut fail_low = true; // assume fail-low until we raise alpha

    // ── YOUNG BROTHERS WAIT CONCEPT ─────────────────────────
    // Search tactical moves first. If the first brother (best tactical)
    // doesn't fail high, the quiet brothers are unlikely to fail high.
    // Only search quiet moves if tacticals didn't resolve.
    
    // Phase 1: Search tactical moves within beam
    for (move_idx, &(_score, idx, packed)) in groups.tactical[..max_tactical].iter().enumerate() {
        let m = &moves[idx];

        board.apply_move(m);
        if is_in_check(board) {
            board.undo_move();
            continue;
        }

        // LMR for later tactical moves
        let reduction = if prune_hard && move_idx >= 2 && effective_depth >= 3 && !in_check {
            (move_idx as u32).min(1)
        } else { 0 };
        let new_depth = effective_depth.saturating_sub(1 + reduction);

        let score = if move_idx == 0 {
            -alphabeta(board, new_depth, -beta, -alpha, nodes, deadline, ply + 1, prune_hard, packed)
        } else {
            let nw = -alphabeta(board, new_depth, -alpha - 1, -alpha, nodes, deadline, ply + 1, prune_hard, packed);
            if nw > alpha && nw < beta && reduction > 0 {
                -alphabeta(board, effective_depth.saturating_sub(1), -beta, -alpha,
                          nodes, deadline, ply + 1, prune_hard, packed)
            } else { nw }
        };

        board.undo_move();

        if score > alpha {
            alpha = score;
            tt_flag = 0;
            best_local = Some(m.clone());
            fail_low = false;
            // Store counter move
            if prev_move != 0 { counter_move_store(prev_move, packed); }
        }
        if alpha >= beta {
            tt_flag = 1;
            break;
        }
    }

    // Phase 2: If still not resolved, search quiet moves (YBWC)
    // YBWC: If the tactical search already raised alpha, we only
    // need one quiet move to refute it, so beam is very narrow.
    if alpha < beta && !groups.quiet.is_empty() {
        // If we already found a good move, young brothers wait:
        // we only need to find ONE quiet move that refutes.
        let quiet_beam = if fail_low { max_quiet } else { 1 };
        
        for (move_idx, &(_score, idx, packed)) in groups.quiet[..quiet_beam.min(groups.quiet.len())].iter().enumerate() {
            let m = &moves[idx];

            // Futility pruning for quiet moves beyond the first few
            if move_idx >= 2 && effective_depth <= 2 && alpha > -MATE_SCORE + 100 {
                continue;
            }

            board.apply_move(m);
            if is_in_check(board) {
                board.undo_move();
                continue;
            }

            // LMR for quiet moves (more aggressive)
            let reduction = if prune_hard && effective_depth >= 2 {
                let base = (move_idx / 2 + 1).min(3) as u32;
                let depth_factor = (effective_depth / 3).min(2);
                base + depth_factor
            } else { 0 };
            let new_depth = effective_depth.saturating_sub(1 + reduction);

            let score;
            if move_idx == 0 && fail_low {
                score = -alphabeta(board, new_depth, -beta, -alpha, nodes, deadline, ply + 1, prune_hard, packed);
            } else {
                let nw = -alphabeta(board, new_depth, -alpha - 1, -alpha, nodes, deadline, ply + 1, prune_hard, packed);
                if nw > alpha && nw < beta && reduction > 0 {
                    score = -alphabeta(board, effective_depth.saturating_sub(1), -beta, -alpha,
                                      nodes, deadline, ply + 1, prune_hard, packed);
                } else { score = nw; }
            }

            board.undo_move();

            if score > alpha {
                alpha = score;
                tt_flag = 0;
                best_local = Some(m.clone());
                fail_low = false;
                if prev_move != 0 { counter_move_store(prev_move, packed); }
                // YBWC: we found a refutation, no need to search more quiet moves
                // unless alpha is still not > old alpha significantly
                if move_idx > 0 { break; }
            }
            if alpha >= beta {
                tt_flag = 1;
                killer_store(effective_depth, packed);
                history_store(m.from_sq as usize, m.to_sq as usize, effective_depth);
                if prev_move != 0 { counter_move_store(prev_move, packed); }
                break;
            }
        }
    }

    // ── DYNAMIC TREE SPLITTING ───────────────────────────────
    // If fail_low (no move raised alpha), widen the beam.
    // This implements progressive widening: if the narrow beam
    // failed, search more moves.
    if fail_low && !groups.tactical.is_empty() {
        let wider_beam = (beam * 2).min(groups.tactical.len());
        for &(_score, idx, packed) in &groups.tactical[max_tactical..wider_beam] {
            let m = &moves[idx];
            board.apply_move(m);
            if is_in_check(board) {
                board.undo_move();
                continue;
            }
            let new_depth = effective_depth.saturating_sub(2); // reduced since we're widening
            let nw = -alphabeta(board, new_depth, -alpha - 1, -alpha, nodes, deadline, ply + 1, prune_hard, packed);
            if nw > alpha {
                let score = -alphabeta(board, effective_depth.saturating_sub(1), -beta, -alpha,
                                      nodes, deadline, ply + 1, prune_hard, packed);
                if score > alpha {
                    alpha = score;
                    tt_flag = 0;
                    best_local = Some(m.clone());
                }
            }
            board.undo_move();
            if alpha >= beta { tt_flag = 1; break; }
        }
    }

    // TT store
    if let Some(bm) = &best_local {
        tt_store(hash, TTEntry {
            score: alpha,
            depth: effective_depth as i8,
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
                    nodes: &mut u64, deadline: Option<Instant>, qdepth: u32) -> i32 {
    *nodes += 1;
    if *nodes & 4095 == 0 {
        if let Some(dl) = deadline {
            if Instant::now() >= dl { return alpha; }
        }
    }

    if let Some(result) = board.game_result() {
        return match result {
            GameResult::BlackWins => if board.side_to_move == BLACK { MATE_SCORE - qdepth as i32 } else { -(MATE_SCORE - qdepth as i32) },
            GameResult::WhiteWins => if board.side_to_move == WHITE { MATE_SCORE - qdepth as i32 } else { -(MATE_SCORE - qdepth as i32) },
            GameResult::Draw => 0,
        };
    }

    let stand_pat = evaluate(board);
    if stand_pat >= beta { return beta; }
    if stand_pat > alpha { alpha = stand_pat; }
    if qdepth >= MAX_QDEPTH { return alpha; }

    let moves = generate_pseudo_legal_moves(board);
    let mut scored_qmoves: Vec<(i32, usize)> = moves.iter().enumerate()
        .filter(|(_, m)| m.captured_piece != 0 || m.mid_piece != 0 || m.is_igui || m.promotion)
        .map(|(i, m)| {
            let vals = init_piece_values();
            let mut score = 0;
            if m.captured_piece != 0 { score += vals[m.captured_piece as usize] * 10; }
            if m.mid_piece != 0 { score += vals[m.mid_piece as usize] * 10; }
            if m.promotion { score += 5000; }
            (score, i)
        })
        .collect();
    scored_qmoves.sort_by(|a, b| b.0.cmp(&a.0));

    for &(_, i) in &scored_qmoves {
        let m = &moves[i];
        board.apply_move(m);

        if is_in_check(board) {
            board.undo_move();
            continue;
        }

        let score = -quiescence_inner(board, -beta, -alpha, nodes, deadline, qdepth + 1);
        board.undo_move();
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}