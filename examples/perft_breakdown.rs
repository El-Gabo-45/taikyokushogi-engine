//! Breaks down perft(1) from the initial position by piece type and checks
use taikyokushogi::{Board, piece_info};
use std::collections::HashMap;

fn main() {
    let board = Board::initial();
    let moves = board.legal_moves();
    println!("total = {}", moves.len());

    let mut by_piece: HashMap<String, usize> = HashMap::new();
    for m in &moves {
        let sq = m.from();
        let p = board.get(sq.row, sq.col).unwrap();
        *by_piece.entry(p.name().to_string()).or_insert(0) += 1;
    }
    let mut v: Vec<(String, usize)> = by_piece.into_iter().collect();
    v.sort();
    for (name, n) in v { println!("{:>24}  {}", name, n); }

    // duplicate detection: same effect signature
    let mut seen = std::collections::HashSet::new();
    let mut dups = 0;
    for m in &moves {
        let r = m.raw();
        let key = (r.from_sq, r.to_sq, r.promotion, r.mid_sq,
                   r.range_cap, r.caps_value, r.captured_piece);
        if !seen.insert(key) {
            dups += 1;
            let sq = m.from();
            let p = board.get(sq.row, sq.col).unwrap();
            println!("DUP: {} ({},{}) -> {} promo={} mid={}", p.name(), sq.row, sq.col, r.to_sq, r.promotion, r.mid_sq);
        }
    }
    if dups == 0 { println!("no duplicates by (from,to,promo,mid,ncaps)"); }
    let _ = piece_info("");
}

