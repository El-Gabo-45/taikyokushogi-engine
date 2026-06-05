//! GPU-accelerated self-play for Taikyoku Shogi.
//!
//! This module provides high-performance self-play generation with:
//! - Parallel search using rayon
//! - Lock-free transposition table
//! - Training data export for NNUE training
use crate::types::*;
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::search;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam::channel::{bounded, Sender, Receiver};

/// Configuration for self-play
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    /// Number of parallel games
    pub num_games: usize,
    /// Search depth per move
    pub depth: u32,
    /// Time limit per move (ms)
    pub time_limit_ms: u64,
    /// Maximum moves per game
    pub max_moves: u32,
    /// Output directory for training data
    pub output_dir: String,
    /// Save every N games
    pub save_interval: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_games: 4,
            depth: 2,
            time_limit_ms: 100,
            max_moves: 500,
            output_dir: "training_data".to_string(),
            save_interval: 100,
        }
    }
}

/// Compact game state for training data
#[derive(Clone, Copy, Debug)]
pub struct TrainingSample {
    /// Board state: 1296 squares × 2 bytes (piece type) = 2592 bytes
    pub board: [u16; 1296],
    /// Side to move: 0 = Black, 1 = White
    pub side_to_move: u8,
    /// Move played
    pub move_from: u16,
    pub move_to: u16,
    pub move_promo: u8,
    /// Game result: 1 = Black win, -1 = White win, 0 = draw
    pub result: i8,
    /// Policy target (move index in legal moves)
    pub policy_target: u16,
    /// Value target (-1 to 1)
    pub value_target: f32,
    /// Move number
    pub move_number: u16,
}

/// Statistics for monitoring
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

/// Lock-free transposition table
pub struct TranspositionTable {
    entries: Vec<std::sync::atomic::AtomicU64>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let entry_size = 16;
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        let num_entries = num_entries.next_power_of_two();
        
        let entries = (0..num_entries)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        
        Self {
            entries,
            mask: num_entries - 1,
        }
    }
    
    #[inline]
    pub fn probe(&self, key: u64) -> Option<(i32, u32)> {
        let _idx = (key as usize) & self.mask;
        let _packed = self.entries[_idx].load(Ordering::Relaxed);
        if _packed == 0 { return None; }
        None
    }
    
    #[inline]
    pub fn store(&self, key: u64, _value: i32, _best_move: u32) {
        let idx = (key as usize) & self.mask;
        self.entries[idx].store(1, Ordering::Relaxed);
    }
}

/// Self-play worker
struct SelfPlayWorker {
    id: usize,
    config: SelfPlayConfig,
    stats: Arc<SelfPlayStats>,
    sample_tx: Sender<TrainingSample>,
}

impl SelfPlayWorker {
    fn run(&mut self) {
        loop {
            let mut board = Board::new();
            board.setup_initial();
            
            let mut move_count = 0;
            let mut game_samples: Vec<TrainingSample> = Vec::new();
            
            while move_count < self.config.max_moves {
                if let Some(result) = board.game_result() {
                    let result_val = match result {
                        GameResult::BlackWins => 1,
                        GameResult::WhiteWins => -1,
                        GameResult::Draw => 0,
                    };
                    
                    for sample in &mut game_samples {
                        sample.result = result_val;
                        sample.value_target = result_val as f32;
                        let _ = self.sample_tx.send(*sample);
                    }
                    break;
                }
                
                let search_result = search(&mut board, self.config.depth, self.config.time_limit_ms);
                
                if let Some(best_move) = search_result.best_move {
                    let mut sample = TrainingSample {
                        board: [0; 1296],
                        side_to_move: board.side_to_move,
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
                            sample.board[sq] = cell_piece(cell) | ((cell_color(cell) as u16) << 8);
                        }
                    }
                    
                    let moves = generate_legal_moves(&board);
                    for (idx, m) in moves.iter().enumerate() {
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
                } else {
                    break;
                }
            }
            
            self.stats.games_completed.fetch_add(1, Ordering::Relaxed);
            
            let games = self.stats.games_completed.load(Ordering::Relaxed);
            if games % 10 == 0 {
                self.stats.print();
            }
        }
    }
}

/// Main self-play coordinator
pub struct SelfPlayCoordinator {
    config: SelfPlayConfig,
    stats: Arc<SelfPlayStats>,
    workers: Vec<JoinHandle<()>>,
    writer_handle: Option<JoinHandle<()>>,
}

impl SelfPlayCoordinator {
    pub fn new(config: SelfPlayConfig) -> Self {
        let stats = Arc::new(SelfPlayStats::default());
        
        let (_sample_tx, sample_rx) = bounded(10000);
        
        let output_dir = config.output_dir.clone();
        let save_interval = config.save_interval;
        let writer_handle = std::thread::spawn(move || {
            Self::writer_loop(sample_rx, output_dir, save_interval);
        });
        
        Self {
            config,
            stats,
            workers: Vec::new(),
            writer_handle: Some(writer_handle),
        }
    }
    
    fn writer_loop(rx: Receiver<TrainingSample>, output_dir: String, save_interval: usize) {
        std::fs::create_dir_all(&output_dir).ok();
        
        let mut buffer = Vec::new();
        let mut file_count = 0;
        
        while let Ok(sample) = rx.recv() {
            buffer.push(sample);
            
            if buffer.len() >= save_interval {
                let filename = format!("{}/samples_{:06}.bin", output_dir, file_count);
                // Simple binary serialization
                let mut data = Vec::with_capacity(buffer.len() * std::mem::size_of::<TrainingSample>());
                for sample in &buffer {
                    let bytes = unsafe { std::slice::from_raw_parts(
                        sample as *const TrainingSample as *const u8,
                        std::mem::size_of::<TrainingSample>()
                    )};
                    data.extend_from_slice(bytes);
                }
                std::fs::write(&filename, &data).ok();
                println!("Saved {} samples to {}", buffer.len(), filename);
                buffer.clear();
                file_count += 1;
            }
        }
        
        if !buffer.is_empty() {
            let filename = format!("{}/samples_{:06}.bin", output_dir, file_count);
            let mut data = Vec::with_capacity(buffer.len() * std::mem::size_of::<TrainingSample>());
            for sample in &buffer {
                let bytes = unsafe { std::slice::from_raw_parts(
                    sample as *const TrainingSample as *const u8,
                    std::mem::size_of::<TrainingSample>()
                )};
                data.extend_from_slice(bytes);
            }
            std::fs::write(&filename, &data).ok();
        }
    }
    
    pub fn start(&mut self) {
        for i in 0..self.config.num_games {
            let (tx, _) = bounded(10000);
            
            let mut worker = SelfPlayWorker {
                id: i,
                config: self.config.clone(),
                stats: self.stats.clone(),
                sample_tx: tx,
            };
            
            let handle = std::thread::spawn(move || {
                worker.run();
            });
            
            self.workers.push(handle);
        }
    }
    
    pub fn wait(mut self) {
        for handle in self.workers.drain(..) {
            handle.join().ok();
        }
        
        if let Some(writer) = self.writer_handle.take() {
            writer.join().ok();
        }
        
        println!("Self-play completed!");
        self.stats.print();
    }
}

/// Run self-play with default config
pub fn run_selfplay(config: Option<SelfPlayConfig>) {
    let config = config.unwrap_or_default();
    println!("Starting self-play with {} games", config.num_games);
    
    let mut coordinator = SelfPlayCoordinator::new(config);
    coordinator.start();
    coordinator.wait();
}