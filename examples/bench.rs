use taikyokushogi::Board;
use std::time::Instant;
use std::io::Write;

fn flush() {
    std::io::stdout().flush().unwrap();
}

fn main() {
    println!("start"); flush();

    let mut b = Board::initial();
    println!("board ready"); flush();

    let t = Instant::now();
    let r = b.search(1, 30000);
    println!("D1: {}ms nodes={} score={}", t.elapsed().as_millis(), r.nodes, r.score); flush();
}