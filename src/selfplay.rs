//! GPU-accelerated self-play for Taikyoku Shogi.
use crate::types::*;
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::search;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam::channel::{bounded, Sender, Receiver};

#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub num_games: usize,
    pub num_workers: usize,
    pub depth: u32,
    pub time_limit_ms: u64,
    pub max_moves: u32,
    pub output_dir: String,
    pub save_interval: usize,
    pub sqlite_path: Option<String>,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_games: 4,
            num_workers: 2,
            depth: 2,
            time_limit_ms: 500,
            max_moves: 0,
            output_dir: "training_data".to_string(),
            save_interval: 100,
            sqlite_path: Some("training_data/games.db".to_string()),
        }
    }
}

fn should_stop_game(move_count: u32, max_moves: u32, terminal: Option<GameResult>) -> bool {
    if terminal.is_some() {
        return true;
    }
    max_moves > 0 && move_count >= max_moves
}

/// A single training sample: board position + move played + game outcome.
///
/// IMPORTANT: `#[repr(C)]` is required here. This struct is serialized to
/// disk as raw bytes (see `writer_loop` below, which does an `unsafe`
/// byte-for-byte memory dump). Without `repr(C)`, Rust is free to reorder
/// fields however it likes to minimize padding, and that order is not
/// guaranteed to stay the same across compiler versions or even separate
/// builds. That would make any external reader (e.g. the Python training
/// script) unable to reliably parse the file format. `repr(C)` fixes the
/// field order to exactly the declaration order below, matching a plain C
/// struct, so this layout can be documented once and relied on forever.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TrainingSample {
    pub board: [u16; 1296],
    pub side_to_move: u8,
    pub move_from: u16,
    pub move_to: u16,
    pub move_promo: u8,
    pub result: i8,
    pub policy_target: u16,
    pub value_target: f32,
    pub move_number: u16,
}

#[derive(Clone, Debug)]
pub struct GameMove {
    pub move_number: u16,
    pub from: u16,
    pub to: u16,
    pub promo: u8,
    pub captured_piece: u16,
    pub side: u8,
    pub black_pieces: usize,
    pub white_pieces: usize,
}

#[derive(Clone, Debug)]
pub struct GameRecord {
    pub game_id: u64,
    pub result: i8,
    pub moves: Vec<GameMove>,
    pub total_moves: u32,
    pub depth: u32,
    pub duration_ms: u64,
}

#[derive(Debug, Default)]
pub struct SelfPlayStats {
    pub games_completed: AtomicUsize,
    pub total_moves: AtomicU64,
    pub total_nodes: AtomicU64,
    pub total_time_ms: AtomicU64,
}

impl SelfPlayStats {
    pub fn print(&self) {
        let games = self.games_completed.load(Ordering::Relaxed);
        let moves = self.total_moves.load(Ordering::Relaxed);
        let nodes = self.total_nodes.load(Ordering::Relaxed);
        let time = self.total_time_ms.load(Ordering::Relaxed);
        println!("=== Self-Play Stats ===");
        println!("Games: {}", games);
        if games > 0 {
            println!("Moves: {} ({:.1} moves/game)", moves, moves as f64 / games as f64);
        }
        if time > 0 {
            println!("Nodes: {} ({:.0} nps)", nodes, nodes as f64 / (time as f64 / 1000.0));
        }
        println!("Time: {:.1}s", time as f64 / 1000.0);
    }
}

struct SelfPlayWorker {
    id: usize,
    config: SelfPlayConfig,
    stats: Arc<SelfPlayStats>,
    sample_tx: Sender<TrainingSample>,
    game_tx: Sender<GameRecord>,
}

impl SelfPlayWorker {
    fn play_one_game(&mut self, game_id: u64) -> (Vec<TrainingSample>, GameRecord) {
        let mut board = Board::new();
        board.setup_initial();
        let start = std::time::Instant::now();

        let mut move_count = 0u32;
        let mut game_samples: Vec<TrainingSample> = Vec::new();
        let mut game_moves: Vec<GameMove> = Vec::new();
        let mut result_val: i8 = 0;

        loop {
            let terminal = board.game_result();
            if should_stop_game(move_count, self.config.max_moves, terminal) {
                result_val = match terminal {
                    Some(GameResult::BlackWins) => 1i8,
                    Some(GameResult::WhiteWins) => -1i8,
                    Some(GameResult::Draw) => 0i8,
                    None => 0i8,
                };
                break;
            }

            let search_result = search(&mut board, self.config.depth, self.config.time_limit_ms);
            let best_move = match search_result.best_move {
                Some(m) => m,
                None => {
                    result_val = if board.side_to_move == BLACK { -1i8 } else { 1i8 };
                    break;
                }
            };

            let side = board.side_to_move;
            let black_pieces = board.piece_count[0];
            let white_pieces = board.piece_count[1];

            game_moves.push(GameMove {
                move_number: (move_count + 1) as u16,
                from: best_move.from_sq,
                to: best_move.to_sq,
                promo: best_move.promotion as u8,
                captured_piece: best_move.captured_piece,
                side,
                black_pieces,
                white_pieces,
            });

            let mut sample = TrainingSample {
                board: [0; 1296],
                side_to_move: side,
                move_from: best_move.from_sq,
                move_to: best_move.to_sq,
                move_promo: best_move.promotion as u8,
                result: 0,
                policy_target: 0,
                value_target: 0.0,
                move_number: board.move_number as u16,
            };
            for sq in 0..1296 {
                let cell = board.cells[sq];
                if cell != EMPTY_CELL {
                    // NOTE: piece_type can be up to 301 (9 bits), so it must
                    // NOT be OR'd directly with (color << 8) -- that would
                    // clobber bit 8 of piece_type for any piece_type >= 256
                    // (there are ~45 such piece types in the full Taikyoku
                    // Shogi piece set). Color now goes in bit 9 instead, and
                    // piece_type is masked to 9 bits defensively even though
                    // it should never exceed that range.
                    sample.board[sq] = (cell_piece(cell) & 0x1FF) | ((cell_color(cell) as u16) << 9);
                }
            }
            // Use pseudo-legal moves for policy target (much faster than generate_legal_moves)
            let pseudo_moves = crate::movegen::generate_pseudo_legal_moves(&board);
            for (idx, m) in pseudo_moves.iter().enumerate() {
                if m.from_sq == best_move.from_sq && m.to_sq == best_move.to_sq && m.promotion == best_move.promotion {
                    sample.policy_target = idx as u16;
                    break;
                }
            }
            game_samples.push(sample);

            board.apply_move(&best_move);
            move_count += 1;
            self.stats.total_moves.fetch_add(1, Ordering::Relaxed);
            self.stats.total_nodes.fetch_add(search_result.nodes, Ordering::Relaxed);
            self.stats.total_time_ms.fetch_add(search_result.time_ms, Ordering::Relaxed);
        }

        for s in &mut game_samples {
            let from_perspective = if s.side_to_move == BLACK { result_val as f32 } else { -(result_val as f32) };
            s.value_target = from_perspective;
            s.result = from_perspective as i8;
        }

        let record = GameRecord {
            game_id,
            result: result_val,
            moves: game_moves,
            total_moves: move_count,
            depth: self.config.depth,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        (game_samples, record)
    }

    fn run(&mut self) {
        let total_games = self.config.num_games;
        let num_workers = self.config.num_workers.max(1);
        let games_per_worker = (total_games + num_workers - 1) / num_workers;
        let my_start = self.id * games_per_worker;
        let my_end = std::cmp::min(my_start + games_per_worker, total_games);
        if my_start >= total_games {
            eprintln!("[worker-{}] no games assigned", self.id);
            return;
        }
        let my_count = my_end - my_start;
        eprintln!("[worker-{}] will play {} games ({}..{})", self.id, my_count, my_start, my_end);

        for (i, game_id) in (my_start..my_end).enumerate() {
            let (samples, record) = self.play_one_game(game_id as u64);
            let n = samples.len();
            let result_str = match record.result { 1 => "BlackWins", -1 => "WhiteWins", _ => "Draw" };
            for s in samples {
                let _ = self.sample_tx.send(s);
            }
            let _ = self.game_tx.send(record);
            self.stats.games_completed.fetch_add(1, Ordering::Relaxed);
            eprintln!("[worker-{}] game {}/{} (id={}) finished: {} moves, result={}",
                self.id, i + 1, my_count, game_id, n, result_str);
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_full_games_and_fast_search() {
        let cfg = SelfPlayConfig::default();
        assert_eq!(cfg.max_moves, 0);
        assert!(cfg.depth >= 2);
        assert!(cfg.time_limit_ms > 0);
    }

    #[test]
    fn stop_condition_respects_unbounded_mode() {
        assert!(!should_stop_game(1000, 0, None));
        assert!(should_stop_game(1000, 10, None));
        assert!(should_stop_game(10, 10, None));
        assert!(should_stop_game(5, 10, Some(GameResult::Draw)));
    }
}

pub struct SelfPlayCoordinator {
    config: SelfPlayConfig,
    stats: Arc<SelfPlayStats>,
    workers: Vec<JoinHandle<()>>,
    writer_handle: Option<JoinHandle<()>>,
    db_handle: Option<JoinHandle<()>>,
    sample_tx: Sender<TrainingSample>,
    game_tx: Sender<GameRecord>,
}

impl SelfPlayCoordinator {
    pub fn new(config: SelfPlayConfig) -> Self {
        let stats = Arc::new(SelfPlayStats::default());
        let (sample_tx, sample_rx) = bounded(10000);
        let (game_tx, game_rx) = bounded(1000);

        let output_dir = config.output_dir.clone();
        let save_interval = config.save_interval;
        let writer_handle = std::thread::spawn(move || {
            Self::writer_loop(sample_rx, output_dir, save_interval);
        });

        let db_handle = if let Some(db_path) = &config.sqlite_path {
            let db_path = db_path.clone();
            let output_dir = config.output_dir.clone();
            Some(std::thread::spawn(move || {
                Self::db_loop(game_rx, db_path, output_dir);
            }))
        } else {
            None
        };

        Self {
            config, stats, workers: Vec::new(),
            writer_handle: Some(writer_handle), db_handle,
            sample_tx, game_tx,
        }
    }

    fn writer_loop(rx: Receiver<TrainingSample>, output_dir: String, save_interval: usize) {
        std::fs::create_dir_all(&output_dir).ok();
        let mut buffer: Vec<TrainingSample> = Vec::new();
        let mut file_count = 0;
        let mut last_print = std::time::Instant::now();
        let mut total_saved = 0u64;
        while let Ok(sample) = rx.recv() {
            buffer.push(sample);
            if buffer.len() >= save_interval {
                let filename = format!("{}/samples_{:06}.bin", output_dir, file_count);
                let mut data = Vec::with_capacity(buffer.len() * std::mem::size_of::<TrainingSample>());
                for s in &buffer {
                    let bytes = unsafe { std::slice::from_raw_parts(s as *const TrainingSample as *const u8, std::mem::size_of::<TrainingSample>()) };
                    data.extend_from_slice(bytes);
                }
                let _ = std::fs::write(&filename, &data);
                total_saved += buffer.len() as u64;
                println!("[writer] Saved {} samples (total {}) to {}", buffer.len(), total_saved, filename);
                buffer.clear();
                file_count += 1;
            }
            if last_print.elapsed().as_secs() >= 2 {
                last_print = std::time::Instant::now();
                println!("[writer] buffer={} samples pending", buffer.len());
            }
        }
        if !buffer.is_empty() {
            let filename = format!("{}/samples_{:06}.bin", output_dir, file_count);
            let mut data = Vec::with_capacity(buffer.len() * std::mem::size_of::<TrainingSample>());
            for s in &buffer {
                let bytes = unsafe { std::slice::from_raw_parts(s as *const TrainingSample as *const u8, std::mem::size_of::<TrainingSample>()) };
                data.extend_from_slice(bytes);
            }
            let _ = std::fs::write(&filename, &data);
            total_saved += buffer.len() as u64;
            println!("[writer] Flushed {} samples (total {}) to {}", buffer.len(), total_saved, filename);
        }
    }

    fn db_loop(rx: Receiver<GameRecord>, db_path: String, output_dir: String) {
        std::fs::create_dir_all(&output_dir).ok();
        let conn = match rusqlite::Connection::open(&db_path) {
            Ok(c) => c,
            Err(e) => { eprintln!("[db] open failed: {}", e); return; }
        };
        if let Err(e) = conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                result INTEGER NOT NULL,
                total_moves INTEGER NOT NULL,
                depth INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                created_at REAL NOT NULL DEFAULT (julianday('now'))
            );
            CREATE TABLE IF NOT EXISTS moves (
                game_id INTEGER NOT NULL,
                move_number INTEGER NOT NULL,
                from_sq INTEGER NOT NULL,
                to_sq INTEGER NOT NULL,
                promo INTEGER NOT NULL,
                captured INTEGER NOT NULL,
                side INTEGER NOT NULL,
                black_pieces INTEGER NOT NULL,
                white_pieces INTEGER NOT NULL,
                PRIMARY KEY (game_id, move_number)
            );"
        ) {
            eprintln!("[db] schema failed");
            return;
        }
        let mut total = 0u64;
        let mut last_print = std::time::Instant::now();
        while let Ok(record) = rx.recv() {
            if let Err(e) = conn.execute(
                "INSERT INTO games (id, result, total_moves, depth, duration_ms) VALUES (?, ?, ?, ?, ?)",
                rusqlite::params![record.game_id as i64, record.result as i64, record.total_moves as i64, record.depth as i64, record.duration_ms as i64]
            ) {
                eprintln!("[db] insert game failed: {}", e);
                continue;
            }
            let tx = match conn.unchecked_transaction() { Ok(t) => t, Err(_) => continue };
            let mut failed = false;
            for m in &record.moves {
                if tx.execute(
                    "INSERT INTO moves (game_id, move_number, from_sq, to_sq, promo, captured, side, black_pieces, white_pieces) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rusqlite::params![
                        record.game_id as i64, m.move_number as i64, m.from as i64, m.to as i64,
                        m.promo as i64, m.captured_piece as i64, m.side as i64,
                        m.black_pieces as i64, m.white_pieces as i64,
                    ]
                ).is_err() { failed = true; break; }
            }
            if !failed { let _ = tx.commit(); }
            total += 1;
            if last_print.elapsed().as_secs() >= 2 {
                last_print = std::time::Instant::now();
                println!("[db] games={} written to {}", total, db_path);
            }
        }
        println!("[db] Done: {} games written to {}", total, db_path);
    }

    pub fn start(&mut self) {
        let num_workers = self.config.num_workers.max(1);
        for i in 0..num_workers {
            let tx = self.sample_tx.clone();
            let gx = self.game_tx.clone();
            let config = self.config.clone();
            let stats = self.stats.clone();
            let handle = std::thread::spawn(move || {
                let mut worker = SelfPlayWorker { id: i, config, stats, sample_tx: tx, game_tx: gx };
                worker.run();
            });
            self.workers.push(handle);
        }
    }

    pub fn wait(mut self) {
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
        drop(self.sample_tx);
        drop(self.game_tx);
        if let Some(writer) = self.writer_handle.take() {
            let _ = writer.join();
        }
        if let Some(db) = self.db_handle.take() {
            let _ = db.join();
        }
        println!("\nSelf-play completed!");
        self.stats.print();
    }
}

pub fn run_selfplay(config: Option<SelfPlayConfig>) {
    let config = config.unwrap_or_default();
    let max_moves = if config.max_moves == 0 { "unlimited".to_string() } else { config.max_moves.to_string() };
    println!("Starting self-play: {} games, {} workers, depth={}, max_moves={}",
        config.num_games, config.num_workers, config.depth, max_moves);
    let mut coord = SelfPlayCoordinator::new(config);
    coord.start();
    coord.wait();
}
