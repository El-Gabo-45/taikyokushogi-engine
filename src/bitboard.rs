//! Bitboard1296 — Wide bitboard for 36×36 Taikyoku Shogi.
//!
//! 36×36 = 1,296 squares. Each bitboard is 21 u64s = 168 bytes.
//! This fits in 3 L1 cache lines and supports fast SIMD operations.
//!
//! Design inspired by the document's proposal for Taikyoku Shogi.

use crate::types::*;

/// Number of u64 words needed to represent 1,296 squares.
pub const U64_COUNT: usize = (NUM_SQUARES + 63) / 64; // 21

/// A bitboard for the 36×36 board, stored as 21 consecutive u64 words.
/// Total size: 168 bytes (3 cache lines).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Bitboard1296 {
    pub words: [u64; U64_COUNT],
}

impl Bitboard1296 {
    /// Create an empty bitboard.
    #[inline(always)]
    pub const fn new() -> Self {
        Bitboard1296 { words: [0u64; U64_COUNT] }
    }

    /// Create a bitboard with all bits set.
    #[inline(always)]
    pub fn all() -> Self {
        let mut bb = Bitboard1296::new();
        // Set all bits in all words
        for w in &mut bb.words {
            *w = !0u64;
        }
        // Clear the extra bits beyond 1,296
        let extra = U64_COUNT * 64 - NUM_SQUARES;
        if extra > 0 {
            let last = U64_COUNT - 1;
            bb.words[last] &= !0u64 >> extra;
        }
        bb
    }

    /// Test if a square is set.
    #[inline(always)]
    pub fn get(&self, sq: u16) -> bool {
        let word = (sq as usize) >> 6;
        let bit = 1u64 << (sq & 63);
        (self.words[word] & bit) != 0
    }

    /// Test if a square is set (usize version).
    #[inline(always)]
    pub fn get_usize(&self, sq: usize) -> bool {
        let word = sq >> 6;
        let bit = 1u64 << (sq & 63);
        (self.words[word] & bit) != 0
    }

    /// Set a square.
    #[inline(always)]
    pub fn set(&mut self, sq: u16) {
        let word = (sq as usize) >> 6;
        self.words[word] |= 1u64 << (sq & 63);
    }

    /// Set a square (usize version).
    #[inline(always)]
    pub fn set_usize(&mut self, sq: usize) {
        let word = sq >> 6;
        self.words[word] |= 1u64 << (sq & 63);
    }

    /// Clear a square.
    #[inline(always)]
    pub fn clear(&mut self, sq: u16) {
        let word = (sq as usize) >> 6;
        self.words[word] &= !(1u64 << (sq & 63));
    }

    /// Clear a square (usize version).
    #[inline(always)]
    pub fn clear_usize(&mut self, sq: usize) {
        let word = sq >> 6;
        self.words[word] &= !(1u64 << (sq & 63));
    }

    /// Bitwise AND with another bitboard, store in self.
    #[inline(always)]
    pub fn and(&mut self, other: &Bitboard1296) {
        for i in 0..U64_COUNT {
            self.words[i] &= other.words[i];
        }
    }

    /// Bitwise OR with another bitboard, store in self.
    #[inline(always)]
    pub fn or(&mut self, other: &Bitboard1296) {
        for i in 0..U64_COUNT {
            self.words[i] |= other.words[i];
        }
    }

    /// Bitwise XOR with another bitboard, store in self.
    #[inline(always)]
    pub fn xor(&mut self, other: &Bitboard1296) {
        for i in 0..U64_COUNT {
            self.words[i] ^= other.words[i];
        }
    }

    /// Bitwise NOT (in-place).
    #[inline(always)]
    pub fn not(&mut self) {
        for w in &mut self.words {
            *w = !*w;
        }
        // Clear extra bits
        let extra = U64_COUNT * 64 - NUM_SQUARES;
        if extra > 0 {
            let last = U64_COUNT - 1;
            self.words[last] &= !0u64 >> extra;
        }
    }

    /// Return a new bitboard that is the AND of self and other.
    #[inline(always)]
    pub fn and_new(&self, other: &Bitboard1296) -> Bitboard1296 {
        let mut result = *self;
        result.and(other);
        result
    }

    /// Return a new bitboard that is the OR of self and other.
    #[inline(always)]
    pub fn or_new(&self, other: &Bitboard1296) -> Bitboard1296 {
        let mut result = *self;
        result.or(other);
        result
    }

    /// Return a new bitboard that is the XOR of self and other.
    #[inline(always)]
    pub fn xor_new(&self, other: &Bitboard1296) -> Bitboard1296 {
        let mut result = *self;
        result.xor(other);
        result
    }

    /// Return a new bitboard that is the NOT of self.
    #[inline(always)]
    pub fn not_new(&self) -> Bitboard1296 {
        let mut result = *self;
        result.not();
        result
    }

    /// Check if the bitboard is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Count the number of set bits.
    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.words.iter().map(|&w| w.count_ones()).sum()
    }

    /// Iterate over all set squares, calling `f` for each.
    #[inline(always)]
    pub fn iter<F: FnMut(u16)>(&self, mut f: F) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros();
                f((word_idx as u16) << 6 | bit as u16);
                w &= w - 1;
            }
        }
    }

    /// Iterate over all set squares, calling `f` for each (usize version).
    #[inline(always)]
    pub fn iter_usize<F: FnMut(usize)>(&self, mut f: F) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros();
                f((word_idx << 6) | bit as usize);
                w &= w - 1;
            }
        }
    }

    /// Get the first set square, or None if empty.
    #[inline(always)]
    pub fn first(&self) -> Option<u16> {
        for (word_idx, &word) in self.words.iter().enumerate() {
            if word != 0 {
                let bit = word.trailing_zeros();
                return Some((word_idx as u16) << 6 | bit as u16);
            }
        }
        None
    }

    /// Get the first set square as usize, or None if empty.
    #[inline(always)]
    pub fn first_usize(&self) -> Option<usize> {
        for (word_idx, &word) in self.words.iter().enumerate() {
            if word != 0 {
                let bit = word.trailing_zeros();
                return Some((word_idx << 6) | bit as usize);
            }
        }
        None
    }

    /// Get the last (highest-index) set square as usize, or None if empty.
    /// Used together with first_usize to find the "first blocker" along a
    /// ray depending on whether the ray's square indices increase or
    /// decrease in the direction of travel (see is_in_check_bitboard).
    #[inline(always)]
    pub fn last_usize(&self) -> Option<usize> {
        for (word_idx, &word) in self.words.iter().enumerate().rev() {
            if word != 0 {
                let bit = 63 - word.leading_zeros();
                return Some((word_idx << 6) | bit as usize);
            }
        }
        None
    }

    /// Remove and return the first set square.
    #[inline(always)]
    pub fn pop_first(&mut self) -> Option<u16> {
        for (word_idx, word) in self.words.iter_mut().enumerate() {
            if *word != 0 {
                let bit = word.trailing_zeros();
                *word &= *word - 1;
                return Some((word_idx as u16) << 6 | bit as u16);
            }
        }
        None
    }

    /// Shift all bits by `delta` squares (positive = forward, negative = backward).
    /// This is useful for generating pawn attacks, etc.
    #[inline(always)]
    pub fn shift(&self, delta: i32) -> Bitboard1296 {
        if delta == 0 { return *self; }
        let mut result = Bitboard1296::new();
        self.iter_usize(|sq| {
            let new_sq = sq as i32 + delta;
            if new_sq >= 0 && new_sq < NUM_SQUARES as i32 {
                result.set_usize(new_sq as usize);
            }
        });
        result
    }

    /// Shift by row delta and column delta.
    #[inline(always)]
    pub fn shift_rc(&self, dr: i32, dc: i32) -> Bitboard1296 {
        if dr == 0 && dc == 0 { return *self; }
        let mut result = Bitboard1296::new();
        self.iter_usize(|sq| {
            let r = (sq / BOARD_SIZE) as i32 + dr;
            let c = (sq % BOARD_SIZE) as i32 + dc;
            if r >= 0 && r < BOARD_SIZE as i32 && c >= 0 && c < BOARD_SIZE as i32 {
                result.set_usize((r as usize) * BOARD_SIZE + (c as usize));
            }
        });
        result
    }

    /// Create a bitboard with a single square set.
    #[inline(always)]
    pub fn single(sq: u16) -> Self {
        let mut bb = Bitboard1296::new();
        bb.set(sq);
        bb
    }

    /// Create a bitboard with a single square set (usize version).
    #[inline(always)]
    pub fn single_usize(sq: usize) -> Self {
        let mut bb = Bitboard1296::new();
        bb.set_usize(sq);
        bb
    }

    /// Create a bitboard from a list of squares.
    #[inline(always)]
    pub fn from_squares(squares: &[u16]) -> Self {
        let mut bb = Bitboard1296::new();
        for &sq in squares {
            bb.set(sq);
        }
        bb
    }

    /// Create a bitboard from a list of usize squares.
    #[inline(always)]
    pub fn from_squares_usize(squares: &[usize]) -> Self {
        let mut bb = Bitboard1296::new();
        for &sq in squares {
            bb.set_usize(sq);
        }
        bb
    }

    /// Fill a rectangle on the board.
    #[inline(always)]
    pub fn fill_rect(&mut self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) {
        for r in row_start..=row_end {
            for c in col_start..=col_end {
                self.set_usize(r * BOARD_SIZE + c);
            }
        }
    }
}

impl std::fmt::Debug for Bitboard1296 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bitboard1296(count={})", self.count())
    }
}

impl std::ops::BitAnd for &Bitboard1296 {
    type Output = Bitboard1296;
    fn bitand(self, rhs: &Bitboard1296) -> Bitboard1296 {
        self.and_new(rhs)
    }
}

impl std::ops::BitOr for &Bitboard1296 {
    type Output = Bitboard1296;
    fn bitor(self, rhs: &Bitboard1296) -> Bitboard1296 {
        self.or_new(rhs)
    }
}

impl std::ops::BitXor for &Bitboard1296 {
    type Output = Bitboard1296;
    fn bitxor(self, rhs: &Bitboard1296) -> Bitboard1296 {
        self.xor_new(rhs)
    }
}

impl std::ops::Not for &Bitboard1296 {
    type Output = Bitboard1296;
    fn not(self) -> Bitboard1296 {
        self.not_new()
    }
}

// ── Precomputed attack tables ──────────────────────────────────

/// For each square and each direction, the set of squares along that ray.
/// Stored as bitboards for fast attack generation.
pub struct AttackTable {
    /// rays[sq * NUM_DIRS + dir] = bitboard of squares along that ray
    pub rays: Vec<Bitboard1296>,
    /// first_blocker[sq * NUM_DIRS + dir] = first blocker square (or INVALID_SQ)
    pub first_blocker: Vec<u16>,
}

impl AttackTable {
    pub fn new() -> Self {
        let total = NUM_SQUARES * NUM_DIRS;
        let mut rays = Vec::with_capacity(total);
        let mut first_blocker = vec![INVALID_SQ; total];

        for sq in 0..NUM_SQUARES {
            let r = sq_row(sq) as i32;
            let c = sq_col(sq) as i32;
            for dir in 0..NUM_DIRS {
                let dr = DIR_DR[dir];
                let dc = DIR_DC[dir];
                let mut bb = Bitboard1296::new();
                let mut cr = r + dr;
                let mut cc = c + dc;
                let mut first = INVALID_SQ;
                let mut is_first = true;
                while cr >= 0 && cr < BOARD_SIZE as i32 && cc >= 0 && cc < BOARD_SIZE as i32 {
                    let target = (cr as usize) * BOARD_SIZE + (cc as usize);
                    bb.set_usize(target);
                    if is_first {
                        first = target as u16;
                        is_first = false;
                    }
                    cr += dr;
                    cc += dc;
                }
                rays.push(bb);
                first_blocker[sq * NUM_DIRS + dir] = first;
            }
        }

        AttackTable { rays, first_blocker }
    }

    /// Get the ray from `sq` in direction `dir` as a bitboard.
    #[inline(always)]
    pub fn ray_bb(&self, sq: usize, dir: usize) -> &Bitboard1296 {
        &self.rays[sq * NUM_DIRS + dir]
    }

    /// Get the first square along the ray from `sq` in direction `dir`.
    #[inline(always)]
    pub fn first_along(&self, sq: usize, dir: usize) -> u16 {
        self.first_blocker[sq * NUM_DIRS + dir]
    }

    /// Get ray adjusted for color.
    #[inline(always)]
    pub fn ray_for_color(&self, sq: usize, dir: usize, color: u8) -> &Bitboard1296 {
        if color == BLACK {
            &self.rays[sq * NUM_DIRS + dir]
        } else {
            &self.rays[sq * NUM_DIRS + ((dir + 4) % 8)]
        }
    }
}

use std::sync::OnceLock;
static ATTACK_TABLE: OnceLock<AttackTable> = OnceLock::new();

pub fn attack_table() -> &'static AttackTable {
    ATTACK_TABLE.get_or_init(AttackTable::new)
}

// ── Precomputed king area (1-step) table ───────────────────────
pub struct KingAreaTable {
    /// king_area[sq] = bitboard of all squares within 1 step
    pub areas: Vec<Bitboard1296>,
}

impl KingAreaTable {
    pub fn new() -> Self {
        let mut areas = Vec::with_capacity(NUM_SQUARES);
        for sq in 0..NUM_SQUARES {
            let r = sq_row(sq) as i32;
            let c = sq_col(sq) as i32;
            let mut bb = Bitboard1296::new();
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let nr = r + dr;
                    let nc = c + dc;
                    if nr >= 0 && nr < BOARD_SIZE as i32 && nc >= 0 && nc < BOARD_SIZE as i32 {
                        bb.set_usize((nr as usize) * BOARD_SIZE + (nc as usize));
                    }
                }
            }
            areas.push(bb);
        }
        KingAreaTable { areas }
    }

    #[inline(always)]
    pub fn area(&self, sq: usize) -> &Bitboard1296 {
        &self.areas[sq]
    }
}

static KING_AREA: OnceLock<KingAreaTable> = OnceLock::new();

pub fn king_area_table() -> &'static KingAreaTable {
    KING_AREA.get_or_init(KingAreaTable::new)
}

// ── Precomputed limited range tables ───────────────────────────
// For pieces that move 1-3 squares in a direction
pub struct LimitedRangeTable {
    /// For each (sq, dir, max_dist), the bitboard of reachable squares
    pub attacks: Vec<Bitboard1296>,
}

impl LimitedRangeTable {
    pub fn new() -> Self {
        let total = NUM_SQUARES * NUM_DIRS * 4; // max_dist 1..=3
        let mut attacks = Vec::with_capacity(total);
        for sq in 0..NUM_SQUARES {
            let r = sq_row(sq) as i32;
            let c = sq_col(sq) as i32;
            for dir in 0..NUM_DIRS {
                let dr = DIR_DR[dir];
                let dc = DIR_DC[dir];
                for max_dist in 1..=3 {
                    let mut bb = Bitboard1296::new();
                    for step in 1..=max_dist {
                        let nr = r + dr * step as i32;
                        let nc = c + dc * step as i32;
                        if nr >= 0 && nr < BOARD_SIZE as i32 && nc >= 0 && nc < BOARD_SIZE as i32 {
                            bb.set_usize((nr as usize) * BOARD_SIZE + (nc as usize));
                        }
                    }
                    attacks.push(bb);
                }
            }
        }
        LimitedRangeTable { attacks }
    }

    #[inline(always)]
    pub fn attacks(&self, sq: usize, dir: usize, max_dist: u8) -> &Bitboard1296 {
        let idx = (sq * NUM_DIRS + dir) * 4 + (max_dist as usize).saturating_sub(1).min(3);
        &self.attacks[idx]
    }
}

static LIMITED_RANGE: OnceLock<LimitedRangeTable> = OnceLock::new();

pub fn limited_range_table() -> &'static LimitedRangeTable {
    LIMITED_RANGE.get_or_init(LimitedRangeTable::new)
}