use taikyokushogi::Board;
use std::time::Instant;

fn main() {
    let b = Board::initial();
    let moves = b.legal_moves();
    
    // Clasificar movidas
    let mut normal = 0;
    let mut with_capture = 0;
    let mut igui = 0;
    let mut range = 0;
    let mut mid = 0;
    
    for m in &moves {
        // Acceder a los campos internos via Display o debug
        let s = format!("{:?}", m);
        if s.contains("is_igui: true") { igui += 1; }
        else if s.contains("range_caps: Some") { range += 1; }
        else if s.contains("mid_sq:") && !s.contains("mid_sq: 65535") { mid += 1; }
        else if s.contains("captured_piece: 0") { normal += 1; }
        else { with_capture += 1; }
    }
    eprintln!("normal={} capture={} igui={} range={} mid={}", normal, with_capture, igui, range, mid);

    // Medir solo apply+undo de movidas sin capturas
    let mut b2 = Board::initial();
    let quiet: Vec<_> = moves.iter().filter(|m| {
        let s = format!("{:?}", m);
        s.contains("captured_piece: 0") && !s.contains("is_igui: true")
    }).collect();
    eprintln!("quiet moves: {}", quiet.len());
    
    let t = Instant::now();
    for _ in 0..100000 {
        for m in &quiet {
            b2.apply(m);
            b2.undo();
        }
    }
    let ms = t.elapsed().as_millis();
    let ops = 100000u64 * quiet.len() as u64;
    eprintln!("quiet apply+undo: {}M ops en {}ms => {:.1}M ops/seg",
        ops/1_000_000, ms, ops as f64 / (ms as f64 / 1000.0) / 1_000_000.0);
}