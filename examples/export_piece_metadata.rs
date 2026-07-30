// Exports piece metadata needed by the Python NNUE training pipeline
// (training/features.py) to a small JSON file, so that piece-type
// properties (specifically: which piece_type IDs are royal, i.e. count as
// "the king" for HalfKP feature indexing) never have to be hand-copied or
// re-derived in Python. If pieces.rs ever changes the piece set, just
// re-run this and re-export -- training/features.py always reads from
// this file rather than hardcoding anything.
//
// Usage: cargo run --release --example export_piece_metadata > training/piece_metadata.json

fn main() {
    let n = taikyokushogi::num_piece_types();
    let mut royal_types: Vec<u16> = Vec::new();
    for pt in 1..=(n as u16) {
        if taikyokushogi::is_royal_piece_type(pt) {
            royal_types.push(pt);
        }
    }

    println!("{{");
    println!("  \"num_piece_types\": {},", n);
    println!("  \"royal_piece_types\": [{}]", 
              royal_types.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
    println!("}}");

    eprintln!("num_piece_types = {}", n);
    eprintln!("royal_piece_types = {:?}", royal_types);
}
