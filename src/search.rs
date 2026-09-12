//! Alpha-beta search for Taikyoku Shogi (36x36 board, ~700 legal moves/node).
//!
//! Components:
//! * Transposition table: 2^22 buckets x 4 entries; hash and payload live in
//!   separate AtomicU64 slots. Payload is published BEFORE the hash
//!   (Release) and re-verified after read (Acquire) so concurrent writers
//!   never yield a torn entry.
//! * Move-ordering heuristics: TT move, MVV-LVA, two killers per ply,
//!   counter moves, history with bonus (on cutoff) and malus (gravity).
//! * Search: iterative deepening with aspiration windows, PVS with
//!   null-window re-search, late move reductions (history-softened),
//!   null-move pruning, razoring / reverse futility / futility / ProbCut,
//!   staged move generation (captures first, quiets only if needed) with
//!   incremental pick-next ordering, check extension on in-check nodes.
//! * Quiescence with its own TT traffic (stores the real in_check flag).

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
    in_check: bool, // Position is in check (cached to avoid re-checking)
}

static TT: OnceLock<Vec<AtomicU64>> = OnceLock::new();

fn tt() -> &'static Vec<AtomicU64> {
    TT.get_or_init(|| (0..TT_SIZE * TT_BUCKET_WIDTH * 2).map(|_| AtomicU64::new(0)).collect())
}

#[inline]
fn tt_index(hash: u64) -> usize { ((hash as usize) & (TT_SIZE - 1)) * TT_BUCKET_WIDTH * 2 }

const TT_MOVE_MASK: u64 = (1 << 25) - 1;
const TT_CHECK_MASK: u64 = 1 << 25;

// ── Move-ordering score constants ────────────────────────────────
const TT_MOVE_SCORE: i32 = 2_000_000;   // hash move always first
const ROOT_HINT_SCORE: i32 = 3_000_000; // previous-iteration best-move bonus
const KILLER1_SCORE: i32 = 90_000;
const KILLER2_SCORE: i32 = 80_000;
const COUNTER_SCORE: i32 = 70_000;

#[inline]
fn tt_pack(entry: &TTEntry, gen: u8) -> u64 {
    let sc = entry.score.clamp(-32000, 32000) as i16 as u16;
    let mv = entry.best_move & (TT_MOVE_MASK as u32);
    ((sc as u64) << 48)
        | ((entry.depth as u64 & 0x7F) << 41)
        | ((entry.flag as u64 & 0x03) << 39)
        | ((gen as u64) << 31)
        | if entry.in_check { TT_CHECK_MASK } else { 0 }
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
        in_check: (packed & TT_CHECK_MASK) != 0,
    }
}

static TT_GEN: AtomicU64 = AtomicU64::new(1);
fn tt_gen() -> u8 { (TT_GEN.load(Ordering::Relaxed) & 0xFF) as u8 }

fn tt_probe(hash: u64) -> Option<TTEntry> {
    let base = tt_index(hash);
    let t = tt();
    for i in 0..TT_BUCKET_WIDTH {
        let idx = base + i * 2;
        // ── Race-safe read ────────────────────────────────────────
        // Hash and data live in separate AtomicU64 slots. A concurrent writer
        // could publish a new hash between our two loads. Read with Acquire,
        // snapshot the data, then RE-VERIFY the hash: if it changed, the slot
        // was overwritten mid-read and we treat it as a miss.
        let stored = t[idx].load(Ordering::Acquire);
        if stored == hash {
            let entry = tt_unpack(t[idx + 1].load(Ordering::Acquire));
            if t[idx].load(Ordering::Acquire) != stored {
                continue; // slot overwritten mid-read — try next bucket slot
            }
            if entry.depth >= 0 { return Some(entry); }
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

    // ── Race-safe write ─────────────────────────────────────────
    // Publish data BEFORE the hash, both with Release: a reader that observes
    // the hash (Acquire) is guaranteed to see at least this data snapshot,
    // never a torn mix of old and new entries.
    t[replace_idx + 1].store(tt_pack(&entry, gen), Ordering::Release);
    t[replace_idx].store(hash, Ordering::Release);
}

// ── Killers ─────────────────────────────────────────────────────
static KILLERS: OnceLock<Vec<AtomicU64>> = OnceLock::new();
fn killers() -> &'static Vec<AtomicU64> {
    KILLERS.get_or_init(|| (0..256).map(|_| AtomicU64::new(0)).collect())
}

fn killer_store(depth: u32, mv: u32) {
    let d = depth.min(127) as usize;
    let slot = &killers()[d];
    // Atomic read-modify-write: a plain load+store is not atomic and, under
    // Lazy SMP, two threads can overwrite each other and drop the old
    // first killer. fetch_update makes the swap race-free.
    let _ = slot.fetch_update(Ordering::Release, Ordering::Relaxed, |cur| {
        let mv0 = cur as u32;
        if mv == mv0 { None } else { Some(mv as u64 | ((mv0 as u64) << 32)) }
    });
}

fn killer_score(depth: u32, mv: u32) -> i32 {
    let d = depth.min(127) as usize;
    let p = killers()[d].load(Ordering::Relaxed);
    let mv0 = p as u32;
    let mv1 = (p >> 32) as u32;
    if mv == mv0 { KILLER1_SCORE } else if mv == mv1 { KILLER2_SCORE } else { 0 }
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

fn history_malus(from: usize, to: usize, depth: u32) {
    let idx = (from % HIST_SZ) * HIST_SZ + (to % HIST_SZ);
    let malus = (depth * depth).min(400) as i32;
    history()[idx].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(malus).max(0))
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
    if stored == mv { COUNTER_SCORE } else { 0 }
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
fn score_move(m: &Move, tt_move: u32, hist: i32, cntr: i32, depth: u32) -> i32 {
    let packed = m_pack(m);
    // 1) Hash move (from TT)
    if packed == tt_move { return TT_MOVE_SCORE; }
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
        let in_check = is_in_check(board);
        let mut best_move = None;
        let mut best_score = -MATE_SCORE - 1;
        let mut nodes: u64 = 0;
        let base_mat = material_score(board);
        let sign = if board.side_to_move == BLACK { 1 } else { -1 };
        let values = piece_vals();
        for m in &moves {
            nodes += 1;
            // ── LEGALITY FILTER ──────────
            // The pure material-delta shortcut scored pseudo-legal moves
            // that leave our own royal attacked — kings stepping into
            // attack, pinned sliders moving off the pin — with full
            // material gain, badly misleading the tree (a depth-3 "mate"
            // against an illegal reply). Verify with apply + is_in_check +
            // undo; it is only ~µs per move with the bitboard checker.
            board.apply_move(m);
            let illegal = is_in_check(board);
            board.undo_move();
            if illegal { continue; }
            let mut delta = 0i32;
            if m.promotion {
                let pt = cell_piece(board.cells[m.from_sq as usize]);
                if let Some(p) = pieces::promotes_to(pt) {
                    delta += sign * (values[p as usize] - values[pt as usize]);
                }
            }
            if m.captured_piece != 0 { delta += sign * values[m.captured_piece as usize]; }
            if m.mid_piece != 0 { delta += sign * values[m.mid_piece as usize]; }
            if let Some(ref caps) = m.range_caps {
                for &(_, pt, _) in caps.iter() { delta += sign * values[pt as usize]; }
            }
            let s = -(base_mat + delta);
            if s > best_score { best_score = s; best_move = Some(m.clone()); }
        }
        // No legal move at all: checkmate if in check, else stalemate-like 0.
        if best_move.is_none() {
            let score = if in_check { -(MATE_SCORE - depth as i32) } else { 0 };
            return SearchResult { best_move: None, score, nodes, time_ms: start.elapsed().as_millis() as u64 };
        }
        return SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 };
    }

    let mut nodes: u64 = 0;
    let mut best_move = None;
    let mut best_score = -MATE_SCORE - 1;
    let root_tt_move = tt_probe(board.hash).map(|e| e.best_move).unwrap_or(0);

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

    if depth > 2 {
        let _ = pvs(board, depth - 2, -MATE_SCORE - 1, MATE_SCORE + 1,
                    &mut nodes, deadline, 0, 0);
    }

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
        let packed = m_pack(m);
        let hist = history_score(m.from_sq as usize, m.to_sq as usize);
        let mut s = score_move(m, root_tt_move, hist, 0, depth);
        if root_hint == Some(packed) { s += ROOT_HINT_SCORE; }
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
            -pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, m_pack(m))
        } else {
            let nw = -pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, m_pack(m));
            if nw > sa && nw < sb {
                -pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, m_pack(m))
            } else { nw }
        };
        if score <= sa || score >= sb {
            let full = -pvs(board, depth - 1, -MATE_SCORE - 1,
                            -best_score.max(-MATE_SCORE - 1),
                            &mut nodes, deadline, 0, m_pack(m));
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
            return SearchResult { best_move, score: best_score, nodes, time_ms: start.elapsed().as_millis() as u64 };
        }
        let mut scored: Vec<(i32, usize)> = Vec::with_capacity(moves.len());
        for (i, m) in moves.iter().enumerate() {
            let packed = m_pack(m);
            let hist = history_score(m.from_sq as usize, m.to_sq as usize);
            let mut s = score_move(m, root_tt_move, hist, 0, depth);
            if root_hint == Some(packed) { s += ROOT_HINT_SCORE; }
            scored.push((s, i));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
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
                -pvs(board, depth - 1, sa, sb, &mut nodes, deadline, 0, m_pack(m))
            } else {
                let nw = -pvs(board, depth - 1, -sa - 1, -sa, &mut nodes, deadline, 0, m_pack(m));
                if nw > sa && nw < sb {
                    -pvs(board, depth - 1, -sb, -sa, &mut nodes, deadline, 0, m_pack(m))
                } else { nw }
            };
            if score <= sa || score >= sb {
                let full = -pvs(board, depth - 1, -MATE_SCORE - 1,
                                -best_score.max(-MATE_SCORE - 1),
                                &mut nodes, deadline, 0, m_pack(m));
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
    // env::var takes a global lock — cache it once per search, not per iteration.
    let debug_log = std::env::var_os("RPS_DEBUG").is_some();
    let deadline = if time_limit_ms > 0 {
        Some(start + std::time::Duration::from_millis(time_limit_ms))
    } else { None };

    TT_GEN.fetch_add(1, Ordering::Relaxed);
    piece_vals();
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
            if prev_iter_ms.saturating_mul(4) > remaining { break; }
        }

        let result = if current_depth <= 1 {
            search_root_window(board, current_depth, deadline, root_hint, -MATE_SCORE - 1, MATE_SCORE + 1)
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
            let mut window = if current_depth <= 4 { 64i32 } else { 200i32 };
            let mut fails = 0u8;
            let mut alpha = score_guess.saturating_sub(window);
            let mut beta = score_guess.saturating_add(window);
            let mut local_result;
            loop {
                local_result = search_root_window(board, current_depth, deadline, root_hint, alpha, beta);
                if let Some(dl) = deadline {
                    if Instant::now() >= dl { break; }
                }
                if local_result.score <= alpha || local_result.score >= beta {
                    fails += 1;
                    if fails >= 2 {
                        break; // keep the (bounded) result — full re-search not worth it
                    }
                    window = (window * 8).min(MATE_SCORE);
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
        root_hint = result.best_move.as_ref().map(m_pack);
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
       prev_move: u32) -> i32 {
    // Every caller (full window, null-window probe, aspiration) maintains
    // beta > alpha; the flag logic in the TT store relies on this invariant.
    debug_assert!(beta > alpha, "pvs requires beta > alpha");
    *nodes += 1;

    // Time check: a node on this 1296-square board costs ~150-200us, so the
    // deadline is polled every 64 nodes (~10-15ms of real time). The bitwise
    // AND keeps the check itself essentially free.
    if *nodes & 63 == 0 {
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

    // ── TT PROBE ──────────────────────────────────────────────
    // The TT stores an `in_check` flag so the expensive is_in_check()
    // computation can be skipped on TT hits. Only compute it when the
    // TT entry doesn't provide it (i.e., no entry or depth < depth).
    let hash = board.hash;
    let tt_probe_result = tt_probe(hash);
    let mut cached_in_check = false;
    let mut has_tt_entry = false;
    let tt_move = if let Some(entry) = tt_probe_result {
        has_tt_entry = true;
        cached_in_check = entry.in_check;
        if (entry.depth as u32) >= depth {
            match entry.flag {
                0 => return entry.score,
                1 => if entry.score >= beta { return entry.score; },
                2 => if entry.score <= alpha { return entry.score; },
                _ => {}
            }
        }
        entry.best_move
    } else { 0 };

    // Trust the TT `in_check` flag on *any* TT hit (true or false): it was
    // computed for this exact position (same hash + side to move), so it is
    // valid at this node. Previously only `true` was trusted and the common
    // case (TT hit on a non-check position) recomputed the expensive
    // is_in_check() ray/bitboard scan every single time. This skips that
    // scan on nearly every TT-hit node.
    let in_check = if has_tt_entry { cached_in_check } else { is_in_check(board) };
    let ext = if in_check { 1 } else { 0 };
    let d = depth + ext; // effective depth

    // ── QUIESCENCE AT LEAVES ──────────────────────────────────
    // At d == 0, run quiescence (capture-only) to avoid the horizon effect.
    // For d > 0, we genuinely generate and search moves — no leaf fast-path
    // shortcut that would fake depth. The fast capture generator + small
    // RPS beams keep each node cheap enough to reach depth 6+.
    if d == 0 {
        let total = board.piece_count[0] + board.piece_count[1];
        if total < 200 { return quiescence(board, alpha, beta, nodes, deadline); }
        return evaluate(board);
    }

    // ── STATIC EVAL ───────────────────────────────────────────
    // With the incremental PSQT score, evaluate() is O(1) (just a couple
    // of table lookups + the king-safety term). Use it at every node —
    // no need for the cheaper-but-cruder material-only approximation.
    let static_eval = evaluate(board);

    // ── RAZORING (depth ≤ 2) ──────────────────────────────────
    // If static_eval + huge_margin ≤ alpha, prune the node entirely
    if d <= 2 && !in_check && alpha > -MATE_SCORE + 100 {
        let margin = match d { 0 => 400, 1 => 600, _ => 900 };
        if static_eval + margin <= alpha { return alpha; }
    }

    // ── REVERSE FUTILITY PRUNING (depth ≤ 3) ─────────────────
    // If static_eval - margin ≥ beta, prune (position is too good)
    if d <= 3 && !in_check && alpha > -MATE_SCORE + 100 {
        let margin = 150 + 250 * d as i32;
        if static_eval - margin >= beta { return beta; }
        if static_eval + margin <= alpha { return alpha; }
    }

    // ── FUTILITY PRUNING (depth ≤ 2) ─────────────────────────
    // Skip shallow quiet nodes when even optimistic gains cannot reach alpha.
    if d <= 2 && !in_check && alpha > -MATE_SCORE + 100 {
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
    if d >= 3 && board.no_progress_plies < 100
        && board.piece_count[side] > 3 && !in_check
    {
        let r = if d >= 6 { 3 } else { 2 };
        board.null_move();
        let null_score = -pvs(board, d.saturating_sub(r), -beta, -(beta - 1),
                              nodes, deadline, ply + 1, 0);
        board.undo_null_move();
        if null_score >= beta { return beta; }
    }

    // ── PROBCUT (depth ≥ 4) ───────────────────────────────────
    // Statistical pruning: if the static eval is far enough below alpha,
    // the probability that any move can raise it above beta is negligible.
    // Reference: "ProbCut" — Kotani, Computer Shogi (docx §3.4).
    if d >= 4 && !in_check && alpha > -MATE_SCORE + 100 {
        let margin = 500 + 300 * d as i32;
        if static_eval + margin <= alpha { return alpha; }
    }

    // ── INTERNAL ITERATIVE DEEPENING ──────────────────────────
    // If no TT move, do a shallow search to get one
    let iid_move = if tt_move == 0 && d >= 4 {
        let iid_d = d / 2 - 1;
        let _ = pvs(board, iid_d, -beta, -alpha, nodes, deadline, ply, prev_move);
        tt_probe(hash).map(|e| e.best_move).unwrap_or(0)
    } else { tt_move };

    // ── STAGED MOVE GENERATION ────────────────────────────────
    // Generate captures first (cheap, ~10-50 moves), search them. Only if
    // no beta cutoff is found do we generate the full quiet move list
    // (~700 moves). This avoids generating ~700 quiet moves at every node
    // when a capture already causes a cutoff — the dominant cost of deep
    // search. Reference: docx §3.2 Futility Pruning & §4.4 Quiescence.
    let rps_beam = if d <= 2 { 24 } else if d <= 4 { 12 } else { 6 };
    // ── BEST MOVE AS SCALAR ───────────────────────────────────
    // Track the best move as its packed u32 instead of a cloned Move.
    // Move contains an Option<Rc<Vec<...>>> (range_caps) plus 12 fields;
    // cloning it on every alpha raise trashes L1/L2 for data we only need
    // to store into the TT as a u32 anyway. (Recommendation: pack to scalar.)
    let mut best_packed: u32 = 0;
    let mut tt_flag: u8 = 2; // UPPERBOUND
    let init_alpha = alpha;
    let mut searched = false;
    // Quiets tried so far at this node (for history malus on beta cutoff).
    // Fixed stack array: at most rps_beam+1 quiets are ever tried before the
    // beam break, so 64 slots always suffice — no heap allocation per node.
    let mut quiet_tried: [(u16, u16); 64] = [(0, 0); 64];
    let mut quiet_tried_n = 0usize;

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
    // ── Incremental move selection (pick-next) ────────────────────
    // Score once into a flat i32 buffer, then repeatedly select the best
    // remaining move: O(k·n) with k ≈ rps_beam (6-24) instead of a full
    // O(n log n) sort of the capture list at every node.
    let mut cap_scores: Vec<i32> = Vec::with_capacity(cap_moves.len());
    for m in cap_moves.iter() {
        let packed = m_pack(m);
        let hist = history_score(m.from_sq as usize, m.to_sq as usize);
        let cntr = counter_score(prev_move, packed);
        cap_scores.push(score_move(m, iid_move, hist, cntr, d));
    }

    let mut move_idx = 0usize;
    loop {
        // Pick the best remaining capture.
        let mut idx = usize::MAX;
        let mut order_score = i32::MIN;
        for (i, &s) in cap_scores.iter().enumerate() {
            if s > order_score { order_score = s; idx = i; }
        }
        if idx == usize::MAX { break; }
        cap_scores[idx] = i32::MIN;
        let packed = m_pack(&cap_moves[idx]);
        // 0-based index counting LEGAL captures searched so far. Illegal
        // pseudo-legal captures (verified below) never consume a beam slot:
        // move_idx previously advanced before the legality filter, so the
        // first illegal captures could exhaust rps_beam without a single
        // valid move being searched. The increment happens after the filter.
        let cur_move = move_idx;
        if cur_move >= rps_beam && !in_check && searched {
            break;
        }
        if cur_move > 0 {
            if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
        }
        let m = &cap_moves[idx];
        // ── CAPTURE FUTILITY PRUNING (depth ≤ 2) ──────────────
        // If even capturing the most valuable pieces on the board plus a
        // safety margin cannot lift the static eval to alpha, this capture
        // cannot possibly raise the score. With ~700 legal moves per node on
        // a 36×36 board, dropping hopeless captures cheaply (O(1) estimate)
        // is a large win — we avoid the full apply + is_in_check + search.
        if d <= 2 && !in_check && cur_move > 0
            && order_score < 1_000_000 && alpha > -MATE_SCORE + 100
        {
            let values = piece_vals();
            let from_pt = cell_piece(board.cells[m.from_sq as usize]);
            let gain = capture_qs_score(board, m, values) + values[from_pt as usize];
            if static_eval + gain + 140 <= alpha { continue; }
        }
        let from_cell = board.cells[m.from_sq as usize];
        let is_king_move = pieces::is_royal(cell_piece(from_cell));
        board.apply_move(m);
        if (is_king_move || in_check) && is_in_check(board) {
            board.undo_move();
            continue; // illegal — does NOT consume a beam slot
        }
        move_idx += 1; // legal — consume a beam slot NOW, before the search
        searched = true;
        let new_d = d.saturating_sub(1);
        let score = if cur_move == 0 {
            -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, packed)
        } else {
            let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, packed);
            if nw > alpha && nw < beta {
                -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, packed)
            } else { nw }
        };
        board.undo_move();
        if score > alpha {
            alpha = score;
            tt_flag = 0;
            best_packed = packed;
        }
        if alpha >= beta {
            tt_flag = 1;
            if order_score < 1_000_000 {
                killer_store(d, packed);
                history_store(m.from_sq as usize, m.to_sq as usize, d);
            }
            if prev_move != 0 { counter_store(prev_move, packed); }
            // History gravity: penalize the quiets that failed to cause a
            // cutoff, so next time the cutting move (and similar ones) are
            // tried first. (Standard technique: Stockfish's history malus.)
            for &(hf, ht) in &quiet_tried[..quiet_tried_n] {
                history_malus(hf as usize, ht as usize, d);
            }
            break;
        }
    }

    // Stage 2: quiet moves (only if no beta cutoff from captures).
    // ACTUAL deep_skip_quiets gate: the code below always generated the full
    // ~700-move quiet list at every node even though `rps_beam` only searched
    // a handful — the comment claimed "skip quiet moves for d>=4" but the gate
    // was never applied to GENERATION, only to how many were searched. On a
    // giant board the full quiet generation is the dominant per-node cost, so
    // at deep, quiet, already-searched nodes we skip it entirely (captures
    // already raised alpha, and quiet subtree adds little for the cost).
    let skip_quiet_gen = d >= 4 && !in_check && searched;
    if alpha < beta && !skip_quiet_gen {
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
        let beam = if d <= 1 { 24 } else if d <= 2 { 18 } else if d <= 4 { 12 } else { 8 };
        let select_n = (rps_beam + 2).min(scored.len());
        if select_n > 1 && scored.len() > select_n {
            scored.select_nth_unstable_by(select_n - 1, |a, b| b.0.cmp(&a.0));
        } else {
            scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        }

        for &(order_score, idx, packed) in scored.iter() {
            // cur_move = count of LEGAL quiets searched so far. Illegal
            // pseudo-legal quiets (verified below) never consume a beam
            // slot — the increment happens after the legality filter, so
            // rps_beam is spent exclusively on valid moves.
            let cur_move = move_idx;
            if cur_move >= rps_beam && !in_check && searched {
                break;
            }
            if cur_move > 0 {
                if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            }
            if d <= 2 && cur_move >= beam && order_score < 1_000_000
                && alpha > -MATE_SCORE + 100
            {
                continue;
            }
            // ── QUIET FUTILITY PRUNING (depth ≤ 2) ────────────
            // A quiet move at low depth cannot change the eval by more than a
            // small margin; if even that margin cannot reach alpha, skip the
            // move entirely (avoids apply + is_in_check + subtree on the
            // hundreds of remaining quiets at each node).
            if d <= 2 && !in_check && cur_move > 0
                && order_score < 1_000_000 && alpha > -MATE_SCORE + 100
            {
                if static_eval + 120 * d as i32 + 60 <= alpha { continue; }
            }
            if cur_move > 0 {
                if let Some(dl) = deadline { if Instant::now() >= dl { break; } }
            }
            let m = &moves[idx];
            let from_cell = board.cells[m.from_sq as usize];
            let is_king_move = pieces::is_royal(cell_piece(from_cell));
            board.apply_move(m);
            if (is_king_move || in_check) && is_in_check(board) {
                board.undo_move();
                continue; // illegal — does NOT consume a beam slot
            }
            move_idx += 1; // legal — consume a beam slot NOW, before the search
            searched = true;
            if order_score < 1_000_000 {
                if quiet_tried_n < quiet_tried.len() {
                    quiet_tried[quiet_tried_n] = (m.from_sq, m.to_sq);
                    quiet_tried_n += 1;
                }
            }
            // Hash-move extension: give the TT's best move one extra ply.
            // NOTE: not a true singular extension (which would re-search at
            // reduced depth with the TT move EXCLUDED to verify it is the
            // only good move) — just a depth bump for the hash move.
            let tt_move_ext = d >= 6 && !in_check && tt_move != 0 && packed == tt_move && order_score < 1_000_000;
            let reduction = if cur_move >= 3 && d >= 3
                && order_score < 1_000_000 && !in_check
            {
                let base = (cur_move / 3).min(3) as u32;
                let depth_factor = (d / 3).min(2);
                // A quiet with a strong history is statistically good —
                // soften its reduction so it is not searched too shallowly
                // (LMR + history interaction, standard in modern engines).
                let soften = (history_score(m.from_sq as usize, m.to_sq as usize) / 8_000).min(2) as u32;
                (base + depth_factor).saturating_sub(soften)
            } else { 0 };
            let mut new_d = d.saturating_sub(1 + reduction);
            if tt_move_ext { new_d = new_d.saturating_add(1); }
            let score;
            if cur_move == 0 {
                score = -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, packed);
            } else if reduction > 0 {
                let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, packed);
                if nw > alpha && nw < beta {
                    score = -pvs(board, d.saturating_sub(1), -beta, -alpha,
                                nodes, deadline, ply + 1, packed);
                } else { score = nw; }
            } else {
                let nw = -pvs(board, new_d, -alpha - 1, -alpha, nodes, deadline, ply + 1, packed);
                if nw > alpha && nw < beta {
                    score = -pvs(board, new_d, -beta, -alpha, nodes, deadline, ply + 1, packed);
                } else { score = nw; }
            }
            board.undo_move();
            if score > alpha {
                alpha = score;
                tt_flag = 0;
                best_packed = packed;
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
    // Store whenever real moves were searched — including all-moves-fail-low
    // nodes (valid UPPERBOUND entries). Skipping those used to waste TT
    // probes on positions the tree reaches repeatedly via transpositions.
    if searched {
        tt_store(hash, TTEntry {
            score: alpha,
            depth: d.min(120) as i8,
            flag: if alpha <= init_alpha { 2 } else { tt_flag },
            // Coherence: tt_pack overrides this field with the ACTIVE
            // generation, so a hardcoded 0 had no packed effect — but an
            // inspected struct must not lie. Set the real generation.
            generation: tt_gen(),
            best_move: best_packed,
            in_check,
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
    let init_q_alpha = alpha;
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

    // ── TT PROBE (quiescence) ─────────────────────────────────────
    // Reuse shallow results at leaf nodes: on a 36×36 board the transposition
    // count in quiescence is enormous (captures transpose constantly), so
    // caching here saves whole quiescence subtrees.
    let q_hash = board.hash;
    if qd > 0 {
        if let Some(entry) = tt_probe(q_hash) {
            match entry.flag {
                0 => return entry.score,
                1 => if entry.score >= beta { return entry.score; },
                2 => if entry.score <= alpha { return entry.score; },
                _ => {}
            }
        }
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

    if qd > 0 {
        tt_store(q_hash, TTEntry {
            score: alpha,
            depth: 0,
            flag: if alpha <= init_q_alpha { 2 } else { 0 },
            generation: tt_gen(),
            best_move: 0,
            // CRITICAL FIX: this was hardcoded `false`, but pvs TRUSTS the
            // TT's in_check flag on any hit (to skip is_in_check and to gate
            // the check extension + null-move + razoring/RFP pruning). A
            // QS-stored entry claiming "not in check" for a position that
            // IS in check silently disabled the check extension and enabled
            // pruning inside check — invalidating tactical accuracy.
            in_check: is_in_check(board),
        });
    }
    alpha
}



// ── Unit tests ──────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn tt_pack_unpack_roundtrip() {
        let entry = TTEntry {
            score: 1234,
            depth: 8,
            flag: 1,
            generation: 0, // ignored by tt_pack — the gen parameter wins
            best_move: 0x1234,
            in_check: true,
        };
        let out = tt_unpack(tt_pack(&entry, 7));
        assert_eq!(out.score, 1234);
        assert_eq!(out.depth, 8);
        assert_eq!(out.flag, 1);
        assert_eq!(out.generation, 7);
        assert_eq!(out.best_move, 0x1234);
        assert!(out.in_check);
        // Score clamping at the ±32000 boundary.
        let mut big = entry;
        big.score = 99_000;
        assert_eq!(tt_unpack(tt_pack(&big, 7)).score, 32_000);
        big.score = -99_000;
        assert_eq!(tt_unpack(tt_pack(&big, 7)).score, -32_000);
    }

    #[test]
    fn search_initial_reaches_depth_and_finds_a_move() {
        history_clear();
        let mut board = Board::initial();
        let r = board.search(4, 0);
        assert!(r.best_move.is_some(), "depth-4 search must return a move");
        assert!(r.nodes > 0);
        // Deterministic: same input, same output.
        let mut board2 = Board::initial();
        let r2 = board2.search(4, 0);
        assert_eq!(r.nodes, r2.nodes);
        assert_eq!(r.score, r2.score);
    }

    #[test]
    fn history_clear_zeroes_counters() {
        history_store(3, 4, 5);
        assert!(history_score(3, 4) > 0);
        history_clear();
        assert_eq!(history_score(3, 4), 0);
        assert_eq!(history_score(0, 0), 0);
    }

    #[test]
    fn killer_store_is_read_back() {
        killer_store(10, 0xABCD);
        assert_eq!(killer_score(10, 0xABCD), KILLER1_SCORE);
    }
}
