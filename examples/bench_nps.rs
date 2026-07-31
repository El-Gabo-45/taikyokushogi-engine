// NPS (Nodes Per Second) benchmark and bottleneck analysis.
// Run with: cargo run --release --example bench_nps
//
// Measures:
//   1. Perft-style NPS (pure movegen + apply/undo, no search)
//   2. Search NPS at various depths with fixed time budgets
//   3. Component-level timing: legal_moves, evaluate, apply+undo
//   4. Pseudo-legal vs legal movegen cost (is_in_check overhead)
//   5. Scaling analysis: how NPS changes with depth
use taikyokushogi::Board;
use std::time::Instant;

/// Perft: count leaf nodes at given depth using only legal move generation.
/// This measures the raw speed of movegen + apply/undo without search overhead.
fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let moves = board.legal_moves();
    if depth == 1 { return moves.len() as u64; }
    let mut nodes = 0u64;
    for m in &moves {
        board.apply(m);
        nodes += perft(board, depth - 1);
        board.undo();
    }
    nodes
}

/// Measure a component by running it N times and returning avg microseconds.
fn bench_component<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    // Warmup
    for _ in 0..(iters / 10).max(1) { f(); }

    let t = Instant::now();
    for _ in 0..iters { f(); }
    let us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  {:<30} {:>8.1} us/call  ({} iters)", label, us, iters);
    us
}

fn main() {
    println!("============================================================");
    println!("  Taikyoku Shogi — NPS Benchmark & Bottleneck Analysis");
    println!("============================================================\n");

    let board = Board::initial();
    let n_black = board.piece_count(taikyokushogi::Color::Black);
    let n_white = board.piece_count(taikyokushogi::Color::White);
    println!("Initial position: {} pieces (B:{} W:{})\n",
             n_black + n_white, n_black, n_white);

    // ── 1. Component-level timing ────────────────────────────────
    println!("── Component Timing (initial position) ──────────────────────");
    {
        let board = Board::initial();

        // legal_moves (includes is_in_check filtering)
        bench_component("legal_moves()", 200, || {
            let b = Board::initial();
            let _ = b.legal_moves();
        });

        // evaluate
        bench_component("evaluate()", 500, || {
            let b = Board::initial();
            let _ = b.evaluate();
        });

        // material_score
        bench_component("material_score()", 500, || {
            let b = Board::initial();
            let _ = b.material_score();
        });

        // apply + undo (single move)
        let moves = board.legal_moves();
        if let Some(first_move) = moves.first() {
            let fm = first_move.clone();
            bench_component("apply() + undo()", 500, || {
                let mut b = Board::initial();
                b.apply(&fm);
                b.undo();
            });
        }

        // Board clone (used in Lazy SMP and filter_legal_moves)
        bench_component("Board::clone()", 1000, || {
            let b = Board::initial();
            let _ = b.clone();
        });
    }

    // ── 2. Perft (pure movegen NPS) ──────────────────────────────
    println!("\n── Perft (pure movegen + apply/undo) ────────────────────────");
    // Only depth 1 and 2 — depth 3 would be ~134M nodes (too slow)
    for depth in 1..=2 {
        let mut board = Board::initial();
        let t = Instant::now();
        let nodes = perft(&mut board, depth);
        let secs = t.elapsed().as_secs_f64();
        let nps = nodes as f64 / secs;
        println!("  perft({}): {:>12} nodes  {:>8.3}s  {:>12.0} nps",
                 depth, nodes, secs, nps);
    }

    // ── 3. Search NPS at fixed depths ────────────────────────────
    println!("\n── Search NPS (fixed time budget) ───────────────────────────");
    println!("  {:<6} {:<8} {:>12} {:>10} {:>12}",
             "depth", "limit", "nodes", "time_ms", "nps");

    for depth in [1u32, 2, 3, 4, 5, 6].iter() {
        let mut board = Board::initial();
        let limit_ms = if *depth <= 2 { 500 } else if *depth <= 4 { 1000 } else { 1500 };
        let result = board.search(*depth, limit_ms);
        let nps = if result.time_ms > 0 {
            result.nodes as f64 / (result.time_ms as f64 / 1000.0)
        } else { 0.0 };
        println!("  {:<6} {:<8} {:>12} {:>10} {:>12.0}",
                 depth, format!("{}ms", limit_ms), result.nodes,
                 result.time_ms, nps);
    }

    // ── 4. Selfplay NPS (realistic game scenario) ────────────────
    println!("\n── Selfplay NPS (10 moves per depth) ────────────────────────");
    for depth in [1u32, 2, 3].iter() {
        let mut board = Board::initial();
        let t = Instant::now();
        let mut total_nodes = 0u64;
        let mut moves_played = 0;
        for _ in 0..10 {
            let result = board.search(*depth, 0);
            total_nodes += result.nodes;
            if let Some(mv) = result.best_move {
                board.apply(&mv);
                moves_played += 1;
            } else {
                break;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let nps = total_nodes as f64 / secs;
        println!("  depth={}  moves={}  nodes={:>10}  time={:.2}s  nps={:.0}",
                 depth, moves_played, total_nodes, secs, nps);
    }

    // ── 5. Midgame NPS (after 12 plies) ──────────────────────────
    println!("\n── Midgame NPS (after 12 plies of selfplay) ────────────────");
    {
        let mut board = Board::initial();
        for _ in 0..12 {
            let result = board.search(2, 0);
            if let Some(mv) = result.best_move {
                board.apply(&mv);
            } else {
                break;
            }
        }

        let n_b = board.piece_count(taikyokushogi::Color::Black);
        let n_w = board.piece_count(taikyokushogi::Color::White);
        println!("  Midgame position: {} pieces (B:{} W:{})", n_b + n_w, n_b, n_w);

        // legal_moves in midgame
        let t = Instant::now();
        let mut total_moves = 0;
        let iters = 50;
        for _ in 0..iters {
            let moves = board.legal_moves();
            total_moves += moves.len();
        }
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        println!("  legal_moves(): {} moves  {:.1}us each", total_moves / iters, us);

        // Search NPS in midgame
        for depth in [2u32, 3, 4].iter() {
            let mut b = board.clone();
            let t = Instant::now();
            let result = b.search(*depth, 1000);
            let secs = t.elapsed().as_secs_f64();
            let nps = result.nodes as f64 / secs;
            println!("  search(depth={}): {} nodes  {:.3}s  {:.0} nps",
                     depth, result.nodes, secs, nps);
        }
    }

    // ── 6. Bottleneck Summary ────────────────────────────────────
    println!("\n── Bottleneck Analysis Summary ──────────────────────────────");
    println!("  • Board size: 36×36 = 1,296 squares, up to 804 pieces");
    println!("  • Move generation iterates all ~402 pieces per side");
    println!("  • is_in_check uses bitboard threat-zone filtering");
    println!("  • Search uses PVS + TT + killers + history + LMR + NMP");
    println!("  • Lazy SMP with up to 3 helper threads at depth ≥ 4");
    println!();
    println!("  Expected bottlenecks (by cost):");
    println!("    1. movegen (generate_pseudo_legal_moves): O(pieces × rays)");
    println!("    2. is_in_check (called per move for legality): O(threat_zone)");
    println!("    3. evaluate (material + zones + king_safety): O(pieces)");
    println!("    4. Board clone (for Lazy SMP + filter_legal_moves)");
    println!("    5. Vec allocations in move ordering (scored.sort_by)");
    println!();
    println!("  Compare the component timings above to identify the actual");
    println!("  hot spot on this machine.");
}