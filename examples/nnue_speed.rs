//! Micro-benchmark: NNUE evaluate() from-scratch refresh vs incrementally
//! maintained accumulator, on the same midgame position.
//!
//! Run with: cargo run --release --example nnue_speed
use taikyokushogi::{Board, set_use_nnue};

fn main() {
    set_use_nnue(true);
    let mut board = Board::initial();
    // Build a midgame position (12 plies of depth-2 search, like stress_undo).
    for _ in 0..12 {
        let r = board.search(2, 0);
        match r.best_move { Some(m) => board.apply(&m), None => break }
    }
    let tsfen = board.to_tsfen(); // same position, but fresh boards have no accumulator

    // Warm-up (also initializes the global net).
    let _ = board.evaluate();

    // Incremental path: accumulator is already maintained by the applies.
    let iters = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..iters { let _ = board.evaluate(); }
    let inc = t0.elapsed() / iters as u32;

    // From-scratch path: a board parsed from TSFen has nnue_acc = None, so
    // evaluate() must do the full O(pieces x FT_NEURONS) refresh. Re-parse
    // per iteration (cheap relative to the refresh) to reset the state.
    let mut scr = std::time::Duration::ZERO;
    let mut n_ok = 0;
    for _ in 0..10 {
        if let Ok(mut b) = Board::from_tsfen(&tsfen) {
            let t0 = std::time::Instant::now();
            let _ = b.evaluate();
            scr += t0.elapsed();
            n_ok += 1;
        }
    }
    if n_ok > 0 { scr /= n_ok as u32; }

    println!("NNUE evaluate()  incremental : {:>10.3?}", inc);
    println!("NNUE evaluate()  from-scratch: {:>10.3?}", scr);
    if inc.as_micros() > 0 {
        println!("speedup: {:>8.1}x", scr.as_nanos() as f64 / inc.as_nanos() as f64);
    }
    set_use_nnue(false);
}
