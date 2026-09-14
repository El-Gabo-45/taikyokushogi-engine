// Confirms the hand-crafted vs NNUE evaluation toggle switches which code
// path runs. The NNUE accumulator is maintained incrementally by
// apply_move/undo_move now (see training/README.md), but this example calls
// evaluate() on a freshly-built board whose accumulator is still empty, so
// the single call includes one from-scratch refresh plus the one-time
// random-weight initialization. For incremental-vs-scratch timings see the
// nnue_speed example.
//
// Usage: cargo run --release --example toggle_nnue
//    or: TAIKYOKU_NNUE_PATH=/path/to/trained.nnue cargo run --release --example toggle_nnue

use taikyokushogi::Board;
use std::time::Instant;

fn main() {
    taikyokushogi::set_use_nnue(false);
    println!("using_nnue() = {}", taikyokushogi::using_nnue());
    let board = Board::initial();
    let t = Instant::now();
    let score_handcrafted = board.evaluate();
    println!("hand-crafted eval: score = {}, took {:?}", score_handcrafted, t.elapsed());

    taikyokushogi::set_use_nnue(true);
    println!("\nusing_nnue() = {}", taikyokushogi::using_nnue());
    let t = Instant::now();
    let score_nnue = board.evaluate();
    println!("NNUE eval:          score = {}, took {:?}", score_nnue, t.elapsed());

    if score_handcrafted != score_nnue {
        println!("\nOK: the two backends produced different scores, confirming the toggle switches evaluation logic.");
    } else {
        println!("\nNOTE: scores matched -- unlikely but not impossible by coincidence with an untrained/random NNUE.");
    }
    println!("\nNOTE: the NNUE timing above is for a SINGLE evaluate() call on a");
    println!("freshly-built board, so it includes one from-scratch accumulator");
    println!("refresh plus one-time random-weight initialization. After moves are");
    println!("applied, evaluate() uses the incrementally-maintained accumulator");
    println!("(see the nnue_speed example for the incremental vs scratch timing).");
}
