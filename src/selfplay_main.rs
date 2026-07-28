use taikyokushogi::selfplay::{SelfPlayConfig, run_selfplay};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // args: <num_games> <depth> <max_moves> <num_workers> <time_limit_ms>
    let num_games = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let max_moves = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let num_workers = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let time_limit_ms = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(500);

    println!("=== Taikyoku Shogi Self-Play ===");
    println!("Games:         {}", num_games);
    println!("Depth:         {}", depth);
    println!("Max moves:     {}", if max_moves == 0 { "unlimited".to_string() } else { max_moves.to_string() });
    println!("Workers:       {}", num_workers);
    println!("Time limit/ms: {}", time_limit_ms);
    println!("Data:          training_data/");
    println!("SQLite:        training_data/games.db");
    println!();

    let config = SelfPlayConfig {
        num_games,
        num_workers,
        depth,
        time_limit_ms,
        max_moves,
        output_dir: "training_data".to_string(),
        save_interval: 100,
        sqlite_path: Some("training_data/games.db".to_string()),
    };

    run_selfplay(Some(config));
}
