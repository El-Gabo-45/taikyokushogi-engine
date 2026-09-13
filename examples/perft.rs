//! Deterministic perft (node count) from the initial position.
//!
//! Reference values from the TaikyokuShogi-Stockfish audit (README.md):
//!   perft(1) = 488
//!   perft(2) = 237,684
//!   perft(3) = 124,729,180
//! Run with: cargo run --release --example perft -- [depth]
use taikyokushogi::Board;

fn perft(mut board: &mut Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let moves = board.legal_moves();
    if depth == 1 { return moves.len() as u64; }
    let mut n = 0u64;
    for m in moves {
        board.apply(&m);
        n += perft(board, depth - 1);
        board.undo();
    }
    n
}

fn main() {
    let depth = std::env::args().nth(1).map(|a| a.parse::<u32>().unwrap()).unwrap_or(2);
    let mut board = Board::initial();
    let t = std::time::Instant::now();
    let n = perft(&mut board, depth);
    let ms = t.elapsed().as_millis() as u64;
    println!("perft({}) = {}   (reference perft(1)=488 perft(2)=237684)", depth, n);
    println!("elapsed: {}ms", ms);
}
