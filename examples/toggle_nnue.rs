// Confirms the hand-crafted vs NNUE evaluation toggle switches which code
// path runs, using a SINGLE evaluate() call (not a full search) since
// nnue_evaluate_from_scratch rebuilds the Accumulator from scratch on
// every call (O(pieces * FT_NEURONS), not incremental yet -- see the
// comment on nnue_evaluate_from_scratch in nnue.rs and training/README.md's
// "Known limitation: NNUE eval is not yet wired for incremental updates"
// section). A full multi-thousand-node search with this is currently WAY
// too slow to be practical -- that's a real, known limitation, not
// something this example works around.
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
    println!("\nNOTE: the NNUE timing above is for a SINGLE evaluate() call. Using");
    println!("this inside search (thousands of calls per move) is currently too");
    println!("slow to be practical without incremental accumulator updates --");
    println!("see training/README.md for details and next steps.");
}
