// Comprehensive, real-world performance benchmark for the Taikyoku Shogi engine.
//
// Unlike the simpler benchmarks, this one:
//   1. Runs each depth test MULTIPLE times to measure variance (not a single shot).
//   2. Tests depths 1..=10 in different game phases (initial, midgame, endgame).
//   3. Measures nodes, real elapsed time, NPS, time-limit behavior, and best move.
//   4. Reports medians + min/max so the numbers are trustworthy.
//   5. Avoids "no time limit" searches beyond depth 1 — on a 36×36 board with
//      ~500+ moves per position, an unbounded depth-2 search takes many minutes.
//
// Run with: cargo run --release --example bench_thorough
use taikyokushogi::Board;
use std::time::Instant;

const RUNS_PER_DEPTH: usize = 2;
const BUDGET_FAST_MS: u64 = 2_000;
const BUDGET_MID_MS: u64 = 5_000;
const BUDGET_SLOW_MS: u64 = 10_000;
const BUDGET_DEEP_MS: u64 = 15_000;

#[derive(Clone)]
struct RunStats {
    nodes: u64,
    time_ms: u64,
    nps: f64,
    score: i32,
    overshoot_ms: f64,
    best_move: String,
}

fn run_search(board: &mut Board, depth: u32, time_limit_ms: u64) -> RunStats {
    let t = Instant::now();
    let result = board.search(depth, time_limit_ms);
    let elapsed_us = t.elapsed().as_micros() as u64;
    let elapsed_ms = elapsed_us / 1000;
    // Use microseconds for NPS calculation so sub-millisecond searches
    // report a real NPS instead of 0.
    let nps = if elapsed_us > 0 {
        result.nodes as f64 / (elapsed_us as f64 / 1_000_000.0)
    } else {
        0.0
    };
    let overshoot = elapsed_ms as f64 - time_limit_ms as f64;
    RunStats {
        nodes: result.nodes,
        time_ms: elapsed_ms,
        nps,
        score: result.score,
        overshoot_ms: overshoot,
        best_move: result.best_move
            .map(|m| format!("{}", m))
            .unwrap_or_else(|| "none".to_string()),
    }
}

fn median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = vals.len() / 2;
    if vals.len() % 2 == 0 {
        (vals[mid - 1] + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

fn benchmark_depth(
    label: &str,
    board_factory: &dyn Fn() -> Board,
    depth: u32,
    time_limit_ms: u64,
) {
    let mut nodes: Vec<f64> = Vec::with_capacity(RUNS_PER_DEPTH);
    let mut times: Vec<f64> = Vec::with_capacity(RUNS_PER_DEPTH);
    let mut nps_all: Vec<f64> = Vec::with_capacity(RUNS_PER_DEPTH);
    let mut overshoots: Vec<f64> = Vec::with_capacity(RUNS_PER_DEPTH);
    let mut moves: Vec<String> = Vec::with_capacity(RUNS_PER_DEPTH);
    let mut scores: Vec<i32> = Vec::with_capacity(RUNS_PER_DEPTH);

    for _ in 0..RUNS_PER_DEPTH {
        let mut board = board_factory();
        let s = run_search(&mut board, depth, time_limit_ms);
        nodes.push(s.nodes as f64);
        times.push(s.time_ms as f64);
        nps_all.push(s.nps);
        overshoots.push(s.overshoot_ms);
        moves.push(s.best_move);
        scores.push(s.score);
    }

    let med_nodes = median(&mut nodes);
    let med_time = median(&mut times);
    let med_nps = median(&mut nps_all);
    let med_overshoot = median(&mut overshoots);
    let min_time = times.iter().cloned().fold(f64::MAX, |a: f64, b| a.min(b));
    let max_time = times.iter().cloned().fold(0.0f64, |a, b| a.max(b));
    let min_nps = nps_all.iter().cloned().fold(f64::MAX, |a: f64, b| a.min(b));
    let max_nps = nps_all.iter().cloned().fold(0.0f64, |a, b| a.max(b));

    let moves_set: std::collections::BTreeSet<String> = moves.into_iter().collect();
    let move_str = if moves_set.len() <= 3 {
        moves_set.into_iter().collect::<Vec<_>>().join(" | ")
    } else {
        format!("{} distinct moves", moves_set.len())
    };
    let score_str = if scores.iter().all(|&s| s == scores[0]) {
        format!("{}", scores[0])
    } else {
        format!("{}..{}", scores.iter().min().unwrap(), scores.iter().max().unwrap())
    };

    println!(
        "  {} depth={:<2} limit={:>5}ms  med_nodes={:>11}  med_time={:>8.1}ms ({}..{})  med_nps={:>9.0}  nps_range={:>9.0}-{:>9.0}  overshoot={:>+8.1}ms  score={}  best=[{}]  ({})",
        label,
        depth,
        time_limit_ms,
        med_nodes as u64,
        med_time,
        min_time,
        max_time,
        med_nps,
        min_nps,
        max_nps,
        med_overshoot,
        score_str,
        move_str,
        RUNS_PER_DEPTH
    );
}

fn reach_position(plies: usize) -> Board {
    // Use depth-1 selfplay: it's instant even on the full 804-piece board,
    // and produces a natural midgame/endgame position. Depth-2+ without a
    // time limit is impractical here (each ply can take many seconds).
    let mut board = Board::initial();
    for _ in 0..plies {
        let result = board.search(1, 0);
        if let Some(mv) = result.best_move {
            board.apply(&mv);
        } else {
            break;
        }
    }
    board
}

fn main() {
    println!("============================================================");
    println!("  Taikyoku Shogi — Thorough Performance Benchmark");
    println!("  {} runs per depth, median reported", RUNS_PER_DEPTH);
    println!("============================================================\n");

    let now = Instant::now();

    // ── 1. INITIAL POSITION ────────────────────────────────────
    // The starting position is the extreme case: 804 pieces, ~512 legal moves.
    // Depth 1 completes instantly. At depth 2 with only 2s, the engine cannot
    // even finish scanning the root's 512 legal moves.
    println!("── Initial Position (804 pieces, ~512 legal moves) ───────────");
    println!("  Shallow depths only — deeper is impractical on the full board.\n");

    let initial_factory = || Board::initial();

    benchmark_depth("init", &initial_factory, 1, 0);
    benchmark_depth("init", &initial_factory, 2, BUDGET_FAST_MS);
    benchmark_depth("init", &initial_factory, 2, BUDGET_MID_MS);

    // ── 2. MIDGAME POSITION ────────────────────────────────────
    // Reach a natural midgame with 12 plies of depth-2 selfplay.
    println!("\n── Midgame Position (after 12 plies of depth-2 selfplay) ────");

    let midgame = reach_position(12);
    let n_b = midgame.piece_count(taikyokushogi::Color::Black);
    let n_w = midgame.piece_count(taikyokushogi::Color::White);
    println!("  {} pieces (B:{} W:{}) — real depth scaling can be measured here.\n", n_b + n_w, n_b, n_w);

    let midgame_factory = {
        let mg = midgame.clone();
        move || mg.clone()
    };

    // Depth scan: 1-6 with fast budget, 7-10 with mid budget.
    // Shows how far the engine actually gets and how NPS evolves.
    println!("  -- Depth scan (1-6 @ 2s, 7-10 @ 10s) --");
    for depth in 1..=6u32 {
        benchmark_depth("midg", &midgame_factory, depth, BUDGET_FAST_MS);
    }
    for depth in 7..=10u32 {
        benchmark_depth("midg", &midgame_factory, depth, BUDGET_SLOW_MS);
    }

    // Time-limit scaling: same depth, different budgets.
    // Shows how efficiently additional time is converted into nodes.
    println!("\n  -- Time-limit scaling (depth=4 with 250ms→10s) --");
    for limit in [250u64, 500, 1_000, 2_000, 5_000, BUDGET_SLOW_MS] {
        benchmark_depth("midg", &midgame_factory, 4, limit);
    }

    // Deep search: how deep can we get with a long budget?
    println!("\n  -- Deep search (depth 8/10/12 @ 15s) --");
    for depth in [8u32, 10, 12] {
        benchmark_depth("midg", &midgame_factory, depth, BUDGET_DEEP_MS);
    }

    // ── 3. ENDGAME-LIKE POSITION (many pieces traded) ───────────
    println!("\n── Endgame-Like Position (after 40 plies of depth-2 selfplay) ─");
    let endgame = reach_position(40);
    let e_b = endgame.piece_count(taikyokushogi::Color::Black);
    let e_w = endgame.piece_count(taikyokushogi::Color::White);
    println!("  {} pieces (B:{} W:{}) — quiescence search matters here.\n", e_b + e_w, e_b, e_w);

    let endgame_factory = {
        let eg = endgame.clone();
        move || eg.clone()
    };

    for depth in [2u32, 4, 6, 8, 10] {
        benchmark_depth("endg", &endgame_factory, depth, BUDGET_MID_MS);
    }

    let total_s = now.elapsed().as_secs_f64();
    println!("\n============================================================");
    println!("  Total benchmark time: {:.1}s", total_s);
    println!("============================================================");
}