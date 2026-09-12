use taikyokushogi::Board;
use taikyokushogi::Color;

fn counts(b: &Board) -> (usize, usize) {
    (b.piece_count(Color::Black), b.piece_count(Color::White))
}

fn main() {
    // 1) Build a midgame position like depth_quick does: 12 plies of depth-2.
    let mut board = Board::initial();
    for _ in 0..12 {
        let r = board.search(2, 0);
        if let Some(mv) = r.best_move {
            board.apply(&mv);
        } else {
            break;
        }
    }
    let baseline = counts(&board);
    println!("midgame counts = {:?}", baseline);

    // 2) LIFO walk: push random (first legal) moves, then unwind all, check
    //    counts return exactly to baseline after each full unwind.
    let mut stack: Vec<taikyokushogi::Move> = Vec::new();
    let mut mismatches = 0;
    for trial in 0..400 {
        let moves = board.legal_moves();
        if moves.is_empty() {
            break;
        }
        // Pick a deterministic rotating move so many distinct move types run.
        let m = moves[trial % moves.len()].clone();
        board.apply(&m);
        stack.push(m);

        if stack.len() >= 6 {
            while let Some(_m) = stack.pop() {
                board.undo();
            }
            let back = counts(&board);
            if back != baseline {
                println!(
                    "MISMATCH after unwinding trial={} midgame-baseline={:?} got={:?}",
                    trial, baseline, back
                );
                mismatches += 1;
                if mismatches > 5 {
                    break;
                }
            }
        }
    }
    println!(
        "done. mismatches={} final={:?} baseline={:?}",
        mismatches,
        counts(&board),
        baseline
    );
}
