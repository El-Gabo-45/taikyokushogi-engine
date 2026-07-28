use std::env;
use taikyokushogi::Board;

fn main() {
    let args: Vec<String> = env::args().collect();
    let depth = args.get(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(2);
    let time_limit = args.get(2).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1000);
    let debug = args.iter().any(|arg| arg == "--debug" || arg == "-d");

    if debug {
        std::env::set_var("TAIKYOKUSHOGI_DEBUG_CLI", "1");
    }

    let mut board = Board::initial();
    let result = board.search(depth, time_limit);

    println!("best_move={:?}", result.best_move.as_ref().map(|mv| mv.to_string()));
    println!("score={}", result.score);
    println!("nodes={}", result.nodes);
    println!("time_ms={}", result.time_ms);
}
