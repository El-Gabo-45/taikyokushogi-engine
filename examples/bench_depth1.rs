// High-precision depth-1 NPS benchmark.
//
// Depth-1 search completes in sub-millisecond time, so a single run
// reports NPS=0 (elapsed rounds to 0ms). To measure the real NPS and its
// variance, this benchmark runs depth-1 search MANY times and aggregates:
//   - total nodes, total time, aggregate NPS
//   - per-run NPS distribution (min / median / mean / max / stddev)
//   - per-run time distribution
//
// Run with: cargo run --release --example bench_depth1
use taikyokushogi::Board;
use std::time::Instant;

const RUNS: usize = 2000;

fn main() {
    println!("============================================================");
    println!("  Taikyoku Shogi — Depth-1 NPS Variance Benchmark");
    println!("  {} runs per position", RUNS);
    println!("============================================================\n");

    // ── Initial position ────────────────────────────────────────
    println!("── Initial Position (804 pieces, ~512 legal moves) ───────────");
    let initial_factory = || Board::initial();
    bench_depth1("init", &initial_factory);

    // ── Midgame position ────────────────────────────────────────
    println!("\n── Midgame Position (after 12 plies of depth-1 selfplay) ────");
    let midgame = reach_position(12);
    let n_b = midgame.piece_count(taikyokushogi::Color::Black);
    let n_w = midgame.piece_count(taikyokushogi::Color::White);
    println!("  {} pieces (B:{} W:{})\n", n_b + n_w, n_b, n_w);
    let midgame_factory = {
        let mg = midgame.clone();
        move || mg.clone()
    };
    bench_depth1("midg", &midgame_factory);

    // ── Endgame position ────────────────────────────────────────
    println!("\n── Endgame-Like Position (after 40 plies of depth-1 selfplay) ─");
    let endgame = reach_position(40);
    let e_b = endgame.piece_count(taikyokushogi::Color::Black);
    let e_w = endgame.piece_count(taikyokushogi::Color::White);
    println!("  {} pieces (B:{} W:{})\n", e_b + e_w, e_b, e_w);
    let endgame_factory = {
        let eg = endgame.clone();
        move || eg.clone()
    };
    bench_depth1("endg", &endgame_factory);
}

fn reach_position(plies: usize) -> Board {
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

fn bench_depth1(label: &str, board_factory: &dyn Fn() -> Board) {
    // Warmup: run a few times to populate caches / TT.
    for _ in 0..10 {
        let mut b = board_factory();
        let _ = b.search(1, 0);
    }

    let mut per_run_nps: Vec<f64> = Vec::with_capacity(RUNS);
    let mut per_run_us: Vec<f64> = Vec::with_capacity(RUNS);
    let mut total_nodes: u64 = 0;
    let mut total_us: u128 = 0;
    let mut nodes_per_run: Vec<u64> = Vec::with_capacity(RUNS);

    for _ in 0..RUNS {
        let mut board = board_factory();
        let t = Instant::now();
        let result = board.search(1, 0);
        let us = t.elapsed().as_micros();
        total_nodes += result.nodes;
        total_us += us;
        nodes_per_run.push(result.nodes);
        per_run_us.push(us as f64);
        if us > 0 {
            per_run_nps.push(result.nodes as f64 / (us as f64 / 1_000_000.0));
        }
    }

    // Aggregate NPS
    let agg_nps = total_nodes as f64 / (total_us as f64 / 1_000_000.0);

    // Per-run stats
    let mean_nps = if per_run_nps.is_empty() { 0.0 } else {
        per_run_nps.iter().sum::<f64>() / per_run_nps.len() as f64
    };
    let min_nps = per_run_nps.iter().cloned().fold(f64::MAX, |a: f64, b| a.min(b));
    let max_nps = per_run_nps.iter().cloned().fold(0.0f64, |a, b| a.max(b));
    let mut sorted_nps = per_run_nps.clone();
    sorted_nps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_nps = sorted_nps[sorted_nps.len() / 2];

    // Stddev
    let variance = if per_run_nps.len() > 1 {
        per_run_nps.iter().map(|v| (v - mean_nps).powi(2)).sum::<f64>() / (per_run_nps.len() - 1) as f64
    } else { 0.0 };
    let stddev_nps = variance.sqrt();

    // Time stats
    let mean_us = per_run_us.iter().sum::<f64>() / per_run_us.len() as f64;
    let min_us = per_run_us.iter().cloned().fold(f64::MAX, |a: f64, b| a.min(b));
    let max_us = per_run_us.iter().cloned().fold(0.0f64, |a, b| a.max(b));
    let mut sorted_us = per_run_us.clone();
    sorted_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = sorted_us[sorted_us.len() / 2];

    // Nodes consistency
    let nodes_set: std::collections::BTreeSet<u64> = nodes_per_run.into_iter().collect();
    let nodes_str = if nodes_set.len() <= 3 {
        nodes_set.into_iter().map(|n| n.to_string()).collect::<Vec<_>>().join(" | ")
    } else {
        format!("{} distinct values", nodes_set.len())
    };

    println!("  {} runs={}  total_nodes={}  total_time={:.1}ms", label, RUNS, total_nodes, total_us as f64 / 1000.0);
    println!("    aggregate NPS: {:.0}", agg_nps);
    println!("    per-run NPS:   min={:.0}  median={:.0}  mean={:.0}  max={:.0}  stddev={:.0}  ({} samples)", min_nps, median_nps, mean_nps, max_nps, stddev_nps, per_run_nps.len());
    println!("    per-run time:  min={:.1}us  median={:.1}us  mean={:.1}us  max={:.1}us", min_us, median_us, mean_us, max_us);
    println!("    nodes/run:     {}", nodes_str);
    println!();
}