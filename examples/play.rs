//! Play a real game of Taikyoku Shogi with actual search.
//!
//! ```sh
//! cargo run --release --example play
//! ```

use taikyokushogi::Board;
use std::time::Instant;

fn main() {
    println!("=== Taikyoku Shogi Self-Play ===");
    println!("Board: 36x36 = {} squares", 36*36);
    println!();

    // Play a game with depth-1 search (full search)
    let mut board = Board::initial();
    println!("Initial: Black={} White={}",
             board.piece_count(taikyokushogi::Color::Black),
             board.piece_count(taikyokushogi::Color::White));

    let start = Instant::now();
    let mut total_nodes: u64 = 0;
    let mut total_moves: u32 = 0;

    for move_no in 0..50 {
        if let Some(result) = board.game_result() {
            println!("Game over at move {}: {:?}", move_no, result);
            break;
        }

        // Use search with depth=1, no time limit, at every other move
        // (depth=0 is just eval, which is fast)
        let result = if move_no % 2 == 0 {
            board.search(1, 0)  // depth 1
        } else {
            board.search(0, 0)  // just eval
        };

        total_nodes += result.nodes;

        if let Some(mv) = result.best_move {
            if move_no < 3 || move_no % 10 == 9 {
                println!("Move {:3}: {}->{} score={:6} nodes={:7} time={}ms",
                         move_no + 1, mv.from(), mv.to(), result.score, result.nodes, result.time_ms);
            }
            board.apply(&mv);
            total_moves += 1;
        } else {
            println!("No legal moves at move {}", move_no + 1);
            break;
        }
    }

    let elapsed = start.elapsed();
    let ms = elapsed.as_millis() as u64;
    println!();
    println!("=== Results ===");
    println!("Moves played: {}", total_moves);
    println!("Total nodes: {}", total_nodes);
    println!("Total time: {}ms", ms);
    if ms > 0 {
        println!("Search speed: {:.0} nps", total_nodes as f64 / (ms as f64 / 1000.0));
    }
    println!("Final: Black={} White={}",
             board.piece_count(taikyokushogi::Color::Black),
             board.piece_count(taikyokushogi::Color::White));
    println!("Material score (Black): {}", board.material_score());
}