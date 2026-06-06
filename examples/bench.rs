use taikyokushogi::Board;
use std::time::Instant;

fn main() {
    let b = Board::initial();

    // 1. Medir movegen
    let t = Instant::now();
    let mut total = 0;
    for _ in 0..100 {
        let moves = b.legal_moves();
        total += moves.len();
    }
    let us = t.elapsed().as_micros() as f64 / 100.0;
    println!("movegen: {} moves  {:.1}us each", total / 100, us);

    // 2. Medir eval
    let t = Instant::now();
    for _ in 0..1000 {
        let _s = b.evaluate();
    }
    let us = t.elapsed().as_micros() as f64 / 1000.0;
    println!("eval: {:.1}us each", us);

    // 3. Medir depth-1 search completo
    let mut board = Board::initial();
    let t = Instant::now();
    let result = board.search(1, 0);
    let ms = t.elapsed().as_micros() as f64 / 1000.0;
    println!("depth-1 search: {} nodes  {:.2}ms  {:.0} nps",
             result.nodes, ms, result.nodes as f64 / (ms / 1000.0));

    // 4. Medir solo movegen + apply/undo depth-1 (sin eval)
    let moves = b.legal_moves();
    let mut board2 = Board::initial();
    let t = Instant::now();
    let mut nodes = 0u64;
    for m in &moves {
        board2.apply(m);
        nodes += 1;
        board2.undo();
    }
    let ms2 = t.elapsed().as_micros() as f64 / 1000.0;
    println!("movegen+apply+undo ({:.0} moves): {:.2}ms  {:.0} nodes/s",
             moves.len() as f64, ms2, nodes as f64 / (ms2 / 1000.0));
}
