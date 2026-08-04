use taikyokushogi::Board;
use std::time::Instant;
fn main() {
    let mut board = Board::initial();
    for _ in 0..12 {
        let r = board.search(2, 0);
        if let Some(mv) = r.best_move { board.apply(&mv); } else { break; }
    }
    println!("Midgame: B={} W={}", board.piece_count(taikyokushogi::Color::Black), board.piece_count(taikyokushogi::Color::White));
    for depth in [2u32, 4, 5, 6, 8, 10] {
        let t = Instant::now();
        let r = board.search(depth, 2000);
        let ms = t.elapsed().as_millis();
        println!("depth={:<2} nodes={:>8} time={:>6}ms nps={:>9}", depth, r.nodes, ms,
            if ms > 0 { r.nodes as f64 / (ms as f64 / 1000.0) } else { 0.0 });
    }
}
