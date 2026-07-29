//! NNUE (Efficiently Updatable Neural Network) for Taikyoku Shogi.
//!
//! Architecture (from the document):
//! - Virtual features: 1,296 × 209 × 2 = 541,632
//! - Feature Transformer: 512 neurons with 8M hash buckets
//! - Factorized layers: 1024→1024→256 → 1024→512→128 → 128→1
//!
//! The NNUE replaces the hand-crafted evaluation with a learned one.
//! It uses HalfKP (King + Piece) features that are incrementally updated.

use crate::board::Board;
use crate::types::*;
use crate::bitboard::Bitboard1296;
use std::sync::OnceLock;

// ── Constants ───────────────────────────────────────────────────
pub const NUM_PIECE_TYPES: usize = 209;
pub const NUM_SQUARES: usize = 1296;
pub const NUM_COLORS: usize = 2;
pub const FT_FEATURES: usize = NUM_SQUARES * NUM_PIECE_TYPES * NUM_COLORS; // 541,632
pub const FT_NEURONS: usize = 512;
pub const FT_BUCKETS: usize = 8_000_000;

// L1: 1024 neurons (512 × 2 perspectives)
// L2: 512 neurons
// L3: 128 neurons
// Output: 1 neuron (score)
const L1_SIZE: usize = 1024;
const L2_SIZE: usize = 512;
const L3_SIZE: usize = 128;

// ── Quantization ────────────────────────────────────────────────
// We use 16-bit integers for the network (i16) with SCALE factor
const SCALE: i32 = 256; // Q8.8 fixed point

// ── Feature Index Calculation ──────────────────────────────────
// HalfKP feature: (king_sq * NUM_PIECE_TYPES * NUM_SQUARES) + (piece_sq * NUM_PIECE_TYPES) + piece_type
// But we use the "half" approach: only features for the side to move's king.
// This gives us ~541K features instead of 1M+.

#[inline]
pub fn feature_index(king_sq: usize, piece_sq: usize, piece_type: u16, color: u8, perspective: u8) -> usize {
    let pt = piece_type as usize;
    if color == perspective {
        // Our piece: index based on king square
        (king_sq * NUM_PIECE_TYPES * NUM_SQUARES) + (piece_sq * NUM_PIECE_TYPES) + pt
    } else {
        // Enemy piece: offset by half the feature space
        FT_FEATURES / 2 + (king_sq * NUM_PIECE_TYPES * NUM_SQUARES) + (piece_sq * NUM_PIECE_TYPES) + pt
    }
}

// ── Feature Transformer (Accumulator) ──────────────────────────
// Stores the accumulated activations for each perspective.
// Updated incrementally when pieces move.

#[derive(Clone, Debug)]
pub struct Accumulator {
    /// Accumulated values for the current perspective (FT_NEURONS)
    pub white: Vec<i16>,
    pub black: Vec<i16>,
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            white: vec![0i16; FT_NEURONS],
            black: vec![0i16; FT_NEURONS],
        }
    }

    /// Refresh accumulator from scratch (slow, used for initialization)
    pub fn refresh(&mut self, board: &Board, ft: &FeatureTransformer) {
        self.white.fill(0);
        self.black.fill(0);
        
        // Add features for all pieces from both perspectives
        for color in 0..2 {
            let c = color as usize;
            let king_sq = board.king_square(color as u8) as usize;
            if king_sq >= NUM_SQUARES { continue; }
            
            for i in 0..board.piece_list_len[c] {
                let sq = board.piece_list[c][i] as usize;
                if sq >= NUM_SQUARES { continue; }
                let cell = board.cells[sq];
                if cell == EMPTY_CELL { continue; }
                let pt = cell_piece(cell);
                
                let idx_white = feature_index(king_sq, sq, pt, color as u8, WHITE);
                let idx_black = feature_index(king_sq, sq, pt, color as u8, BLACK);
                
                // Add feature weights to accumulators
                for n in 0..FT_NEURONS {
                    self.white[n] = self.white[n].saturating_add(ft.weights[idx_white][n]);
                    self.black[n] = self.black[n].saturating_add(ft.weights[idx_black][n]);
                }
            }
        }
    }

    /// Update accumulator when a piece moves (incremental)
    pub fn update_move(&mut self, board: &Board, from: usize, to: usize, pt: u16, color: u8,
                       captured_pt: u16, captured_color: u8, ft: &FeatureTransformer) {
        let king_sq_white = board.king_square(WHITE) as usize;
        let king_sq_black = board.king_square(BLACK) as usize;
        
        // Remove piece from old position
        if king_sq_white < NUM_SQUARES {
            let old_idx = feature_index(king_sq_white, from, pt, color, WHITE);
            for n in 0..FT_NEURONS {
                self.white[n] = self.white[n].saturating_sub(ft.weights[old_idx][n]);
            }
        }
        if king_sq_black < NUM_SQUARES {
            let old_idx = feature_index(king_sq_black, from, pt, color, BLACK);
            for n in 0..FT_NEURONS {
                self.black[n] = self.black[n].saturating_sub(ft.weights[old_idx][n]);
            }
        }
        
        // Add piece to new position
        if king_sq_white < NUM_SQUARES {
            let new_idx = feature_index(king_sq_white, to, pt, color, WHITE);
            for n in 0..FT_NEURONS {
                self.white[n] = self.white[n].saturating_add(ft.weights[new_idx][n]);
            }
        }
        if king_sq_black < NUM_SQUARES {
            let new_idx = feature_index(king_sq_black, to, pt, color, BLACK);
            for n in 0..FT_NEURONS {
                self.black[n] = self.black[n].saturating_add(ft.weights[new_idx][n]);
            }
        }
        
        // Remove captured piece
        if captured_pt != 0 {
            if king_sq_white < NUM_SQUARES {
                let cap_idx = feature_index(king_sq_white, to, captured_pt, captured_color, WHITE);
                for n in 0..FT_NEURONS {
                    self.white[n] = self.white[n].saturating_sub(ft.weights[cap_idx][n]);
                }
            }
            if king_sq_black < NUM_SQUARES {
                let cap_idx = feature_index(king_sq_black, to, captured_pt, captured_color, BLACK);
                for n in 0..FT_NEURONS {
                    self.black[n] = self.black[n].saturating_sub(ft.weights[cap_idx][n]);
                }
            }
        }
    }
}

// ── Feature Transformer Weights ────────────────────────────────
// Stored as i16 for quantization. In practice, these would be loaded from a file.

pub struct FeatureTransformer {
    /// weights[feature_idx][neuron] = i16 weight
    pub weights: Vec<Vec<i16>>,
    /// biases[neuron] = i16 bias
    pub biases: Vec<i16>,
}

impl FeatureTransformer {
    pub fn new_random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(FT_FEATURES);
        for _ in 0..FT_FEATURES {
            let mut row = Vec::with_capacity(FT_NEURONS);
            for _ in 0..FT_NEURONS {
                row.push(rng.gen_range(-128..128) as i16);
            }
            weights.push(row);
        }
        let biases = vec![0i16; FT_NEURONS];
        FeatureTransformer { weights, biases }
    }

    /// Forward pass: compute clipped ReLU of (weights · features + bias)
    pub fn forward(&self, acc: &Accumulator, perspective: u8) -> Vec<i16> {
        let acc_ref = if perspective == WHITE { &acc.white } else { &acc.black };
        let mut output = Vec::with_capacity(FT_NEURONS);
        for n in 0..FT_NEURONS {
            let val = (acc_ref[n] as i32) + (self.biases[n] as i32);
            // Clipped ReLU: clamp to [0, 255] then quantize to i16
            let clamped = val.clamp(0, 255) as i16;
            output.push(clamped);
        }
        output
    }
}

// ── Factorized Layer ───────────────────────────────────────────
// A factorized layer: input → hidden → output with ReLU

#[derive(Clone, Debug)]
pub struct FactorizedLayer {
    /// weights[input][hidden]
    pub w1: Vec<Vec<i16>>,
    /// biases[hidden]
    pub b1: Vec<i16>,
    /// weights[hidden][output]
    pub w2: Vec<Vec<i16>>,
    /// biases[output]
    pub b2: Vec<i16>,
}

impl FactorizedLayer {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut w1 = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            let mut row = Vec::with_capacity(hidden_size);
            for _ in 0..hidden_size {
                row.push(rng.gen_range(-64..64) as i16);
            }
            w1.push(row);
        }
        let b1 = vec![0i16; hidden_size];
        
        let mut w2 = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            let mut row = Vec::with_capacity(output_size);
            for _ in 0..output_size {
                row.push(rng.gen_range(-64..64) as i16);
            }
            w2.push(row);
        }
        let b2 = vec![0i16; output_size];
        
        FactorizedLayer { w1, b1, w2, b2 }
    }

    pub fn forward(&self, input: &[i16]) -> Vec<i16> {
        let hidden_size = self.b1.len();
        let output_size = self.b2.len();
        
        // Hidden layer: ReLU(W1 · input + b1)
        let mut hidden = Vec::with_capacity(hidden_size);
        for h in 0..hidden_size {
            let mut sum = self.b1[h] as i32;
            for (i, &val) in input.iter().enumerate() {
                sum += (val as i32) * (self.w1[i][h] as i32);
            }
            // Clipped ReLU
            hidden.push((sum / SCALE).clamp(0, 255) as i16);
        }
        
        // Output layer: W2 · hidden + b2
        let mut output = Vec::with_capacity(output_size);
        for o in 0..output_size {
            let mut sum = self.b2[o] as i32;
            for (h, &val) in hidden.iter().enumerate() {
                sum += (val as i32) * (self.w2[h][o] as i32);
            }
            output.push((sum / SCALE).clamp(-32000, 32000) as i16);
        }
        
        output
    }
}

// ── NNUE Evaluator ─────────────────────────────────────────────
// The complete NNUE evaluation function.

pub struct NnueEvaluator {
    pub ft: FeatureTransformer,
    pub l1: FactorizedLayer,  // 1024 → 1024 → 256
    pub l2: FactorizedLayer,  // 1024 → 512 → 128
    pub l3: FactorizedLayer,  // 512 → 128 → 64
    pub output: LinearLayer,  // 128 → 1
}

impl NnueEvaluator {
    pub fn new_random() -> Self {
        NnueEvaluator {
            ft: FeatureTransformer::new_random(),
            l1: FactorizedLayer::new(FT_NEURONS * 2, 1024, 256),
            l2: FactorizedLayer::new(1024, 512, 128),
            l3: FactorizedLayer::new(512, 128, 64),
            output: LinearLayer::new(128, 1),
        }
    }

    /// Evaluate the position from the side to move's perspective.
    /// Returns a score in centipawns (like the hand-crafted eval).
    pub fn evaluate(&self, board: &Board, acc: &Accumulator) -> i32 {
        // Forward pass through the network
        let ft_white = self.ft.forward(acc, WHITE);
        let ft_black = self.ft.forward(acc, BLACK);
        
        // Concatenate both perspectives
        let mut ft_concat = Vec::with_capacity(FT_NEURONS * 2);
        ft_concat.extend_from_slice(&ft_white);
        ft_concat.extend_from_slice(&ft_black);
        
        // L1: 1024 → 256
        let l1_out = self.l1.forward(&ft_concat);
        
        // L2: 1024 → 128
        let l2_out = self.l2.forward(&l1_out);
        
        // L3: 512 → 64
        let l3_out = self.l3.forward(&l2_out);
        
        // Output: 128 → 1
        let out = self.output.forward(&l3_out);
        
        // Convert to centipawns
        let score = out[0] as i32;
        
        // Return from side to move's perspective
        if board.side_to_move == BLACK { score } else { -score }
    }
}

// ── Linear Layer ────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct LinearLayer {
    pub weights: Vec<Vec<i16>>,
    pub biases: Vec<i16>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut weights = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            let mut row = Vec::with_capacity(output_size);
            for _ in 0..output_size {
                row.push(rng.gen_range(-32..32) as i16);
            }
            weights.push(row);
        }
        let biases = vec![0i16; output_size];
        
        LinearLayer { weights, biases }
    }

    pub fn forward(&self, input: &[i16]) -> Vec<i16> {
        let output_size = self.biases.len();
        let mut output = Vec::with_capacity(output_size);
        for o in 0..output_size {
            let mut sum = self.biases[o] as i32;
            for (i, &val) in input.iter().enumerate() {
                sum += (val as i32) * (self.weights[i][o] as i32);
            }
            output.push((sum / SCALE).clamp(-32000, 32000) as i16);
        }
        output
    }
}

// ── Global NNUE instance ───────────────────────────────────────
static NNUE: OnceLock<NnueEvaluator> = OnceLock::new();

pub fn nnue() -> &'static NnueEvaluator {
    NNUE.get_or_init(|| {
        // In production, this would load from a file
        // For now, create a random network for testing
        NnueEvaluator::new_random()
    })
}

/// NNUE evaluation: returns score from side to move's perspective.
/// Uses the global NNUE instance.
pub fn nnue_evaluate(board: &Board, acc: &Accumulator) -> i32 {
    nnue().evaluate(board, acc)
}

// ── Serialization ──────────────────────────────────────────────
// For saving/loading trained networks

impl NnueEvaluator {
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        
        // Save feature transformer
        let ft_bytes = unsafe {
            std::slice::from_raw_parts(
                self.ft.weights.as_ptr() as *const u8,
                self.ft.weights.len() * std::mem::size_of::<Vec<i16>>(),
            )
        };
        file.write_all(ft_bytes)?;
        
        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        // TODO: implement loading
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "Loading not yet implemented"))
    }
}