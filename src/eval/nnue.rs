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
// NUM_PIECE_TYPES must match `taikyokushogi::num_piece_types()` (i.e.
// `pieces::PIECE_DEFS.len()`) exactly -- verified at 301 for the current
// piece set (checked via a small runtime probe against this crate, not
// assumed from the original design document, which had said 209 before
// the full Taikyoku Shogi piece set was implemented). If pieces.rs ever
// adds/removes piece types, this constant and any already-trained NNUE
// weights must be regenerated together.
pub const NUM_PIECE_TYPES: usize = 301;
pub const NUM_SQUARES: usize = 1296;
pub const NUM_COLORS: usize = 2;
// FT_FEATURES is now driven by FT_BUCKETS (see feature_index below), not
// by the raw HalfKP feature count. The raw count -- king_sq * piece_sq *
// piece_type, doubled for both perspectives -- would be
// NUM_SQUARES * NUM_SQUARES * NUM_PIECE_TYPES * NUM_COLORS =~ 1.01 billion,
// which is infeasible to represent as a dense embedding table (would need
// ~1TB just for the first layer at FT_NEURONS=512, i16). FT_BUCKETS below
// applies feature hashing (index % (FT_BUCKETS/2) per perspective half) to
// compress that space to a fixed, trainable size, at the cost of a small,
// well-distributed collision rate (~1011M / 8M =~ 126 raw indices sharing
// each bucket on average) -- this is the standard technique used whenever
// a HalfKP-style feature space is too large to represent densely (this is
// what the FT_BUCKETS constant's name already implied, but it was
// previously declared and never actually used in feature_index).
pub const FT_FEATURES: usize = FT_BUCKETS;
pub const FT_NEURONS: usize = 512;
pub const FT_BUCKETS: usize = 200_000;

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
// The raw index above ranges over NUM_SQUARES * NUM_SQUARES * NUM_PIECE_TYPES
// (~505M) per perspective. We hash it down to FT_BUCKETS/2 via modulo
// before adding the perspective offset, so the final result always lands
// in [0, FT_BUCKETS). See training/features.py's feature_index() for the
// exact Python mirror this must match bit-for-bit -- the modulo must use
// the SAME divisor (FT_BUCKETS/2) on both sides or exported weights will
// not line up with what the Rust inference code looks up.

#[inline]
pub fn feature_index(king_sq: usize, piece_sq: usize, piece_type: u16, color: u8, perspective: u8) -> usize {
    let pt = piece_type as usize;
    let half_buckets = FT_BUCKETS / 2;
    let raw = (king_sq * NUM_PIECE_TYPES * NUM_SQUARES) + (piece_sq * NUM_PIECE_TYPES) + pt;
    let hashed = raw % half_buckets;
    if color == perspective {
        // Our piece: index based on king square
        hashed
    } else {
        // Enemy piece: offset by half the feature space
        half_buckets + hashed
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
    pub l1: FactorizedLayer,  // 1024 → 1024(hidden) → 256
    pub l2: FactorizedLayer,  // 256 → 512(hidden) → 128
    pub l3: FactorizedLayer,  // 128 → 128(hidden) → 64
    pub output: LinearLayer,  // 64 → 1
}

impl NnueEvaluator {
    pub fn new_random() -> Self {
        NnueEvaluator {
            ft: FeatureTransformer::new_random(),
            // Chain: ft_concat (2*FT_NEURONS=1024) -> 256 -> 128 -> 64 -> 1.
            // Each layer's input_size must equal the previous layer's
            // output_size -- previously l2/l3/output declared input sizes
            // (1024/512/128) that didn't match what they actually received
            // (256/128/64), silently wasting most of each layer's weights
            // without crashing (FactorizedLayer::forward only reads
            // input.len() rows of its weight matrix, so a too-small input
            // just leaves the rest of the matrix unused rather than panicking).
            l1: FactorizedLayer::new(FT_NEURONS * 2, 1024, 256),
            l2: FactorizedLayer::new(256, 512, 128),
            l3: FactorizedLayer::new(128, 128, 64),
            output: LinearLayer::new(64, 1),
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

/// Returns the global NNUE instance, loading it from the path in the
/// TAIKYOKU_NNUE_PATH environment variable if set. Falls back to a
/// randomly-initialized network (with a loud stderr warning) if the env
/// var is unset or the file fails to load -- this keeps the engine
/// functional without a trained network (e.g. for tests, or before
/// training has produced a usable checkpoint), but makes it very visible
/// when that's happening rather than silently playing on untrained
/// weights with no explanation.
pub fn nnue() -> &'static NnueEvaluator {
    NNUE.get_or_init(|| {
        match std::env::var("TAIKYOKU_NNUE_PATH") {
            Ok(path) => match NnueEvaluator::load_from_file(&path) {
                Ok(net) => {
                    eprintln!("[nnue] loaded weights from {}", path);
                    net
                }
                Err(e) => {
                    eprintln!(
                        "[nnue] WARNING: failed to load TAIKYOKU_NNUE_PATH={} ({}); \
                         falling back to RANDOM (untrained) weights -- evaluation \
                         quality will be effectively random until this is fixed",
                        path, e
                    );
                    NnueEvaluator::new_random()
                }
            },
            Err(_) => {
                eprintln!(
                    "[nnue] WARNING: TAIKYOKU_NNUE_PATH not set; using RANDOM \
                     (untrained) weights -- evaluation quality will be effectively \
                     random. Set TAIKYOKU_NNUE_PATH to a .nnue file exported by \
                     training/export.py to use a trained network."
                );
                NnueEvaluator::new_random()
            }
        }
    })
}

/// NNUE evaluation: returns score from side to move's perspective.
/// Uses the global NNUE instance.
pub fn nnue_evaluate(board: &Board, acc: &Accumulator) -> i32 {
    nnue().evaluate(board, acc)
}

/// Stateless NNUE evaluation: builds the Accumulator from scratch (via
/// refresh()) on every call, then evaluates. Convenient for callers (like
/// search.rs) that don't yet maintain an incremental Accumulator across
/// moves -- at the cost of O(pieces * FT_NEURONS) work per call instead
/// of the O(1) amortized cost incremental updates would give. Swapping
/// this out for incremental accumulator maintenance (update on
/// apply_move/undo_move, mirroring how the hand-crafted eval doesn't need
/// this because it's not incremental either) is a real performance
/// optimization opportunity for later, not required for correctness.
pub fn nnue_evaluate_from_scratch(board: &Board) -> i32 {
    let mut acc = Accumulator::new();
    acc.refresh(board, &nnue().ft);
    nnue().evaluate(board, &acc)
}

// ── Serialization ──────────────────────────────────────────────
// For saving/loading trained networks.
//
// FILE FORMAT (all values little-endian; see training/export.py for the
// PyTorch-side writer that must produce byte-identical output):
//
//   magic:        4 bytes  = b"NNU1"  (format version tag)
//   ft_weights:   FT_FEATURES * FT_NEURONS * i16   (row-major: [feature][neuron])
//   ft_biases:    FT_NEURONS * i16
//   l1_w1:        (FT_NEURONS*2) * 1024 * i16
//   l1_b1:        1024 * i16
//   l1_w2:        1024 * 256 * i16
//   l1_b2:        256 * i16
//   l2_w1:        256 * 512 * i16
//   l2_b1:        512 * i16
//   l2_w2:        512 * 128 * i16
//   l2_b2:        128 * i16
//   l3_w1:        128 * 128 * i16
//   l3_b1:        128 * i16
//   l3_w2:        128 * 64 * i16
//   l3_b2:        64 * i16
//   output_w:     64 * 1 * i16
//   output_b:     1 * i16
//
// This is a flat, fully-documented layout (as opposed to raw struct memory
// dumps) specifically so it can be written by an external tool (the PyTorch
// training/export pipeline) without needing to replicate Rust's internal
// memory representation of Vec<Vec<i16>>, which is NOT a contiguous array
// (each inner Vec is a separate heap allocation) and cannot be safely cast
// to bytes directly.

const NNUE_FILE_MAGIC: &[u8; 4] = b"NNU1";

fn write_i16_matrix<W: std::io::Write>(w: &mut W, m: &[Vec<i16>]) -> std::io::Result<()> {
    for row in m {
        for &v in row {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_i16_vec<W: std::io::Write>(w: &mut W, v: &[i16]) -> std::io::Result<()> {
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_i16_matrix<R: std::io::Read>(r: &mut R, rows: usize, cols: usize) -> std::io::Result<Vec<Vec<i16>>> {
    let mut out = Vec::with_capacity(rows);
    let mut buf = [0u8; 2];
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            r.read_exact(&mut buf)?;
            row.push(i16::from_le_bytes(buf));
        }
        out.push(row);
    }
    Ok(out)
}

fn read_i16_vec<R: std::io::Read>(r: &mut R, len: usize) -> std::io::Result<Vec<i16>> {
    let mut out = Vec::with_capacity(len);
    let mut buf = [0u8; 2];
    for _ in 0..len {
        r.read_exact(&mut buf)?;
        out.push(i16::from_le_bytes(buf));
    }
    Ok(out)
}

impl NnueEvaluator {
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(NNUE_FILE_MAGIC)?;

        write_i16_matrix(&mut w, &self.ft.weights)?;
        write_i16_vec(&mut w, &self.ft.biases)?;

        write_i16_matrix(&mut w, &self.l1.w1)?;
        write_i16_vec(&mut w, &self.l1.b1)?;
        write_i16_matrix(&mut w, &self.l1.w2)?;
        write_i16_vec(&mut w, &self.l1.b2)?;

        write_i16_matrix(&mut w, &self.l2.w1)?;
        write_i16_vec(&mut w, &self.l2.b1)?;
        write_i16_matrix(&mut w, &self.l2.w2)?;
        write_i16_vec(&mut w, &self.l2.b2)?;

        write_i16_matrix(&mut w, &self.l3.w1)?;
        write_i16_vec(&mut w, &self.l3.b1)?;
        write_i16_matrix(&mut w, &self.l3.w2)?;
        write_i16_vec(&mut w, &self.l3.b2)?;

        write_i16_matrix(&mut w, &self.output.weights)?;
        write_i16_vec(&mut w, &self.output.biases)?;

        w.flush()?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        use std::io::{BufReader, Read};
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != NNUE_FILE_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad NNUE file magic: expected {:?}, got {:?} -- wrong file or format version",
                    NNUE_FILE_MAGIC, magic
                ),
            ));
        }

        let ft_weights = read_i16_matrix(&mut r, FT_FEATURES, FT_NEURONS)?;
        let ft_biases = read_i16_vec(&mut r, FT_NEURONS)?;

        let l1_w1 = read_i16_matrix(&mut r, FT_NEURONS * 2, 1024)?;
        let l1_b1 = read_i16_vec(&mut r, 1024)?;
        let l1_w2 = read_i16_matrix(&mut r, 1024, 256)?;
        let l1_b2 = read_i16_vec(&mut r, 256)?;

        let l2_w1 = read_i16_matrix(&mut r, 256, 512)?;
        let l2_b1 = read_i16_vec(&mut r, 512)?;
        let l2_w2 = read_i16_matrix(&mut r, 512, 128)?;
        let l2_b2 = read_i16_vec(&mut r, 128)?;

        let l3_w1 = read_i16_matrix(&mut r, 128, 128)?;
        let l3_b1 = read_i16_vec(&mut r, 128)?;
        let l3_w2 = read_i16_matrix(&mut r, 128, 64)?;
        let l3_b2 = read_i16_vec(&mut r, 64)?;

        let output_weights = read_i16_matrix(&mut r, 64, 1)?;
        let output_biases = read_i16_vec(&mut r, 1)?;

        Ok(NnueEvaluator {
            ft: FeatureTransformer { weights: ft_weights, biases: ft_biases },
            l1: FactorizedLayer { w1: l1_w1, b1: l1_b1, w2: l1_w2, b2: l1_b2 },
            l2: FactorizedLayer { w1: l2_w1, b1: l2_b1, w2: l2_w2, b2: l2_b2 },
            l3: FactorizedLayer { w1: l3_w1, b1: l3_b1, w2: l3_w2, b2: l3_b2 },
            output: LinearLayer { weights: output_weights, biases: output_biases },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end check that a .nnue file exported by training/export.py
    /// loads correctly and produces a usable forward pass. Only runs when
    /// the TAIKYOKU_TEST_NNUE_PATH env var points at an actual exported
    /// file (set by training/README.md's verification step) -- it's a
    /// skip, not a failure, when unset, since most `cargo test` runs
    /// won't have a trained network sitting around.
    ///
    /// This is the test that actually proves the PyTorch side (model.py's
    /// dimensions, export.py's transposes and quantization) agrees with
    /// the Rust side (NnueEvaluator's dimensions and load_from_file's
    /// exact-size reads) -- a mismatch anywhere in that chain surfaces
    /// here as either an `io::Error` from load_from_file (wrong file
    /// size) or a panic during evaluate() (out-of-bounds index from a
    /// shape mismatch that happened to produce a same-sized file, e.g. a
    /// transpose bug on a square-ish matrix).
    #[test]
    fn load_and_evaluate_exported_nnue() {
        let path = match std::env::var("TAIKYOKU_TEST_NNUE_PATH") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skipping: TAIKYOKU_TEST_NNUE_PATH not set");
                return;
            }
        };

        let net = NnueEvaluator::load_from_file(&path)
            .unwrap_or_else(|e| panic!("failed to load {}: {}", path, e));

        assert_eq!(net.ft.weights.len(), FT_BUCKETS);
        assert_eq!(net.ft.weights[0].len(), FT_NEURONS);
        assert_eq!(net.ft.biases.len(), FT_NEURONS);
        assert_eq!(net.l1.w1.len(), FT_NEURONS * 2);
        assert_eq!(net.output.weights.len(), 64);
        assert_eq!(net.output.biases.len(), 1);

        // Forward pass on the initial position -- just needs to run
        // without panicking (out-of-bounds indexing would panic here if
        // any dimension were wrong) and return a finite score.
        let mut board = crate::board::Board::new();
        board.setup_initial();
        let mut acc = Accumulator::new();
        acc.refresh(&board, &net.ft);
        let score = net.evaluate(&board, &acc);
        eprintln!("loaded {} OK, initial position score = {}", path, score);
        assert!(score.abs() < 1_000_000, "score {} looks unreasonable", score);
    }
}