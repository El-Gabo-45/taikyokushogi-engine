//! Random-walk invariant oracle: after every apply/undo pair, verifies that
//! the board layout (TSFen), incremental material score and piece counts all
//! return exactly to the pre-move state. Exercises hooks, range captures,
//! lions and promotions across a long walk from the initial position.
use taikyokushogi::{Board, Color};

struct Snapshot {
    tsfen: String,
    mat: i32,
    black: usize,
    white: usize,
}

fn snap(b: &Board) -> Snapshot {
    Snapshot {
        tsfen: b.to_tsfen(),
        mat: b.material_score(),
        black: b.piece_count(Color::Black),
        white: b.piece_count(Color::White),
    }
}

fn main() {
    let mut board = Board::initial();
    let mut walk: Vec<taikyokushogi::Move> = Vec::new();
    let mut mismatches = 0usize;
    let mut checks = 0usize;
    // Snapshot of the position on which the current walk stack operates:
    // after a bulk unwind the board must return exactly here.
    let mut base_snap = snap(&board);

    for ply in 0..3000 {
        let before = snap(&board);
        let moves = board.legal_moves();
        if moves.is_empty() { break; }
        let m = moves[ply % moves.len()].clone();
        board.apply(&m);
        board.undo();
        let after = snap(&board);
        checks += 1;
        if before.tsfen != after.tsfen || before.mat != after.mat
            || before.black != after.black || before.white != after.white {
            mismatches += 1;
            println!("MISMATCH at ply {} move {}->{}", ply, m.raw().from_sq, m.raw().to_sq);
            if mismatches > 5 { break; }
        }
        // advance the walk (with occasional deep unwind)
        let m = moves[(ply * 7919 + 13) % moves.len()].clone();
        board.apply(&m);
        walk.push(m);
        if walk.len() >= 40 {
            while let Some(m) = walk.pop() {
                board.undo();
                let _ = m;
            }
            let back = snap(&board);
            if back.tsfen != base_snap.tsfen || back.mat != base_snap.mat
                || back.black != base_snap.black || back.white != base_snap.white {
                mismatches += 1;
                println!("BULK UNWIND MISMATCH at ply {}", ply);
                if mismatches > 5 { break; }
            }
            base_snap = snap(&board);
        }
    }
    println!("checks={} mismatches={}", checks, mismatches);
    if mismatches == 0 { println!("ORACLE OK"); }
}
