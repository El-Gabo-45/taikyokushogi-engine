use taikyokushogi::Board;

fn main() {
    let board = Board::initial();
    let legal = board.legal_moves();
    println!("legal_moves = {}", legal.len());
    // Estimate how many are captures (any move that removes an opponent piece).
    let cap = legal
        .iter()
        .filter(|m| {
            let s = m.to_string();
            s.contains('x') || s.contains('+') // crude capture/promo marker
        })
        .count();
    println!("approx captures/promos (by notation) = {}", cap);
}

