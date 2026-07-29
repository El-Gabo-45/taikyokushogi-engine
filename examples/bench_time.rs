// Verifies the engine respects the assigned time_limit_ms.
// Run with: cargo run --release --example bench_time
use taikyokushogi::Board;
use std::time::Instant;

fn run(depth: u32, limit_ms: u64, n_moves: usize) {
    let mut board = Board::initial();
    let mut max_move_ms = 0f64;
    let t = Instant::now();
    let mut played = 0;

    for _ in 0..n_moves {
        let mt = Instant::now();
        let result = board.search(depth, limit_ms);
        let ms = mt.elapsed().as_secs_f64() * 1000.0;
        if ms > max_move_ms { max_move_ms = ms; }
        if let Some(mv) = result.best_move {
            board.apply(&mv);
            played += 1;
        } else {
            break;
        }
    }

    let overshoot = max_move_ms - limit_ms as f64;
    println!(
        "depth={:<2} limit={:>4}ms  moves={:<3} total={:>7.0}ms  max_single_move={:>7.1}ms  overshoot={:>+7.1}ms",
        depth, limit_ms, played, t.elapsed().as_secs_f64() * 1000.0, max_move_ms, overshoot
    );
}

fn main() {
    println!("Checking that each move respects its assigned time limit.\n");
    println!("('overshoot' should be near 0 or negative; hundreds/thousands of ms");
    println!("would indicate the deadline bug is still present)\n");

    run(2, 500, 10);
    run(3, 500, 10);
    run(4, 500, 8);
    run(5, 500, 6);
}
