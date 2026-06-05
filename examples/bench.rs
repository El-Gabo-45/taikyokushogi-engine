use taikyokushogi::Board;
use std::time::Instant;

fn main() {
    println!("=== Self-Play Benchmark ===");
    
    let mut total_moves = 0u64;
    let mut total_nodes = 0u64;
    let start = Instant::now();
    
    for game in 0..10 {
        let mut b = Board::initial();
        let mut moves = 0u32;
        let game_start = Instant::now();
        
        loop {
            if b.game_result().is_some() {
                break;
            }
            
            let result = b.search(2, 100);
            if let Some(mv) = result.best_move {
                b.apply(&mv);
                moves += 1;
                total_nodes += result.nodes;
                total_moves += 1;
            } else {
                break;
            }
            
            if moves > 200 {
                break;
            }
        }
        
        let elapsed = game_start.elapsed().as_millis();
        println!("Game {}: {} moves in {}ms ({:.1} moves/s)", 
                 game + 1, moves, elapsed, moves as f64 / (elapsed as f64 / 1000.0));
    }
    
    let total_elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Results ===");
    println!("Total moves: {}", total_moves);
    println!("Total nodes: {}", total_nodes);
    println!("Total time: {:.2}s", total_elapsed);
    println!("Moves/sec: {:.1}", total_moves as f64 / total_elapsed);
    println!("Nodes/sec: {:.0}", total_nodes as f64 / total_elapsed);
}
