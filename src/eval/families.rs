//! Piece family classification.
//!
//! Each piece is assigned to one or more families based on its movement
//! pattern and game role.  The evaluation function then works in family
//! space rather than over individual piece IDs.
//!
//! Adding a new piece only requires updating its movement definition; the
//! classification is derived automatically from the movement primitives.

use crate::pieces;
use crate::types::*;
use crate::types::{N, S, E, W, NE, SE, SW, NW};

/// Symbolic family of a piece type.  Pieces can belong to multiple families
/// (e.g. a Lion is both a `Lion` and a `Promoted` family).  The classifier
/// returns the *primary* family — the most strategically relevant one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// K, CP — capture = win.
    Royal,
    /// GG, VG, BG, RO, VD, +FCR — fly over lower-rank pieces.
    RangeCap,
    /// LN, LI, +FFI, +BSP — multi-step area movers with igui.
    Lion,
    /// FE, EL, HF, +GEA, +GHK — Queen + long-range jump combinations.
    Eagle,
    /// DK, DH, RD, GD, RD, BL — half-rook + half-bishop style pieces.
    Dragon,
    /// FR, FT, FO, DM, FI, WA, WD, F — long-range attacker.
    Demon,
    /// Standard pieces: CN, SD, RN, GE, GD, WO.
    Standard,
    /// Long-range sliders: R, B, DH, DK, FH, etc.
    Slider,
    /// Stepping pieces: G, S, C, M, K, CP, etc.
    Stepper,
    /// Pawn-row pieces: P, GB, D, N, L, OC, TG.
    Pawn,
    /// Hook movers: HM, LO, CA, PC.
    Hook,
    /// Range-cap or Lion-iguib only.
    Special,
    /// Fallback when the classifier can't decide.
    Other,
}

impl Family {
    /// Human-readable label (used in logs and analysis).
    pub fn label(self) -> &'static str {
        match self {
            Family::Royal     => "Royal",
            Family::RangeCap  => "Range-capturer",
            Family::Lion      => "Lion (area)",
            Family::Eagle     => "Eagle (queen+jump)",
            Family::Dragon    => "Dragon (W+B)",
            Family::Demon     => "Demon (long-range)",
            Family::Standard  => "Standard (B3R)",
            Family::Slider    => "Slider",
            Family::Stepper   => "Stepper",
            Family::Pawn      => "Pawn-rank",
            Family::Hook      => "Hook mover",
            Family::Special   => "Special",
            Family::Other     => "Other",
        }
    }
}

/// Per-family value weight.  The engine's raw `pieces::value` is a baseline;
/// the family weight acts as a multiplier / strategic bias.
pub const FAMILY_WEIGHT: &[i32] = &[
    0, // unused index 0
    100,  // Royal
    90,   // RangeCap
    85,   // Lion
    70,   // Eagle
    60,   // Dragon
    50,   // Demon
    45,   // Standard
    30,   // Slider
    25,   // Stepper
    10,   // Pawn
    20,   // Hook
    60,   // Special
    15,   // Other
];

/// Compute the family of a piece type.
pub fn classify(piece_type: u16) -> Family {
    if piece_type == 0 {
        return Family::Other;
    }
    let m = pieces::movement(piece_type);
    let rank = pieces::rank(piece_type);

    // Royal overrides everything.
    if rank == RANK_ROYAL {
        return Family::Royal;
    }

    // Range capturers.
    if !m.range_capture.is_empty() {
        return Family::RangeCap;
    }

    // Lions (area movers with igui).
    if m.area > 0 && m.igui {
        return Family::Lion;
    }

    // Hook movers.
    if m.hook.is_some() {
        return Family::Hook;
    }

    // Eagles: queen + significant jump capability.
    let queen = m.slides.iter().all(|&(d, r)| r == 0) && m.slides.len() == 8;
    if queen && m.jumps.len() >= 1 {
        return Family::Eagle;
    }

    // Dragons: half-queen (W + B or B + R).
    if m.slides.len() == 4 && m.jumps.iter().all(|&(r, c)| r.abs() == c.abs())
        && !m.jumps.is_empty()
    {
        return Family::Dragon;
    }

    // Demons: queen + long-range attacks (more than 6 squares in a dir).
    let has_long_slide = m.slides.iter().any(|&(_, r)| r == 0 || r >= 5);
    let wide = m.slides.len() >= 4;
    if has_long_slide && wide {
        return Family::Demon;
    }

    // Standards: B + R combination (a rook + bishop base).
    let has_b = m.slides.iter().any(|&(d, _)| matches!(d as usize, NE | SE | SW | NW));
    let has_r = m.slides.iter().any(|&(d, _)| matches!(d as usize, N | S | E | W));
    if has_b && has_r {
        return Family::Standard;
    }

    // Sliders vs steppers.
    if m.slides.iter().any(|&(_, r)| r == 0) {
        return Family::Slider;
    }

    // Pawn row.
    // Hard-coded pawn-like pieces: tiny value and forward-only step.
    let max_range = m.slides.iter().map(|&(_, r)| r).max().unwrap_or(0);
    if max_range <= 2 {
        // Check piece name for "Pawn" by value — pawns are < 1500.
        if pieces::value(piece_type) <= 1000 {
            return Family::Pawn;
        }
        return Family::Stepper;
    }

    Family::Other
}

/// Promotion multiplier — promoted forms are worth considerably more.
pub const PROMOTION_MULT: i32 = 2;

/// Compute the value contribution of a single piece.
pub fn family_value(piece_type: u16) -> i32 {
    let fam = classify(piece_type);
    let weight = FAMILY_WEIGHT[fam as usize];
    let base = pieces::value(piece_type);

    // Promoted pieces get a strong bonus.
    let promo_mult = if pieces::is_royal(piece_type) {
        1
    } else {
        // Heuristic: pieces whose name starts with "+" are promoted. The
        // engine stores them with the same `value()` as their base, so we
        // add a flat bonus here.
        let abbrev = pieces::abbrev(piece_type);
        if abbrev.starts_with('+') { PROMOTION_MULT } else { 1 }
    };

    // Blend the per-piece value and the family weight (50/50).
    (base * promo_mult + weight * 50) / 2
}