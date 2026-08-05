//! Bitboard attack generation for Taikyoku Shogi.
//!
//! The core cost in `movegen::generate_pseudo_legal_moves` is walking every
//! piece and, for each, iterating its `Movement` Vecs (heap reads) plus
//! per-jump bounds checks. On a 36×36 board with ~400 pieces that's
//! O(pieces × movement_vectors) pointer chases and divisions.
//!
//! This module precomputes, per (piece_type, color), a **compact relative
//! attack template** in flat arrays (no heap Vecs):
//! - `jumps[(dr, dc); N]` — non-sliding deltas (jumps, steps, area, igui)
//! - `slides[(dir, max_range); N]` — sliding move specifiers
//!
//! The generation then avoids `pieces::movement()` heap reads for the ~90%
//! of piece types that are simple jumps/steps/slides. Special pieces
//! (hooks, range-capturers, lion mid-captures) still fall back to
//! `pieces::movement` for correctness.
//!
//! Reference: docx §5.3 "Bitboards y Estructuras de Datos Incrementales" —
//! bitboard-based generation scales with the board perimeter, not the area.

use crate::types::*;
use crate::pieces;
use crate::board::Board;
use std::sync::OnceLock;

const MAX_JUMP_DELTAS: usize = 32;
const MAX_SLIDES: usize = 10;

const HOOK_ORTHO_NS: [usize; 2] = [N, S];
const HOOK_ORTHO_EW: [usize; 2] = [E, W];
const HOOK_TURN_NE: [usize; 2] = [NW, SE];
const HOOK_TURN_SE: [usize; 2] = [NE, SW];
const HOOK_TURN_SW: [usize; 2] = [SE, NW];
const HOOK_TURN_NW: [usize; 2] = [NE, SW];

/// Flat movement template per (piece_type, color). Only valid for pieces
/// whose moves are pure jumps/steps/slides/area/igui (no hook, no
/// range_capture, no lion mid-capture). `valid=false` → must use fallback.
#[derive(Clone, Copy)]
pub struct Template {
    pub jumps: [(i8, i8); MAX_JUMP_DELTAS],
    pub n_jumps: u8,
    pub slides: [(u8, u8); MAX_SLIDES],
    pub n_slides: u8,
    pub has_igui: bool,
    pub area: u8,
    pub valid: bool,
}

const fn empty_template() -> Template {
    Template {
        jumps: [(0, 0); MAX_JUMP_DELTAS],
        n_jumps: 0,
        slides: [(0, 0); MAX_SLIDES],
        n_slides: 0,
        has_igui: false,
        area: 0,
        valid: false,
    }
}

static TEMPLATES: OnceLock<Box<[[Template; 2]; 512]>> = OnceLock::new();

pub fn templates() -> &'static [[Template; 2]; 512] {
    TEMPLATES.get_or_init(|| {
        let mut table = Box::new([[empty_template(); 2]; 512]);
        for pt in 1..=511u16 {
            let mv = pieces::movement(pt);
            // Skip special pieces that need the fallback path.
            let special = mv.hook.is_some()
                || !mv.range_capture.is_empty()
                || (mv.area >= 2 && !mv.jumps.is_empty()); // lion mid-capture
            if special { continue; }

            for color in 0..2u8 {
                let mut t = empty_template();
                let flip: i8 = if color == BLACK { 1 } else { -1 };
                let mut ti = 0usize;

                // Jumps.
                for &(jdr, jdc) in &mv.jumps {
                    if ti >= MAX_JUMP_DELTAS { break; }
                    t.jumps[ti] = (jdr * flip, jdc * flip);
                    ti += 1;
                }

                // Steps (range == 1 slides).
                for &(dir, max_range) in &mv.slides {
                    if max_range == 1 && ti < MAX_JUMP_DELTAS {
                        let (dr, dc) = get_deltas(dir as usize, color);
                        t.jumps[ti] = (dr as i8, dc as i8);
                        ti += 1;
                    }
                }

                // Area: radius-1 or radius-2 squares (no mid-capture since
                // those are excluded above; pure distance-1/2 jumps).
                if mv.area > 0 {
                    for dr in -2i8..=2 {
                        for dc in -2i8..=2 {
                            if dr == 0 && dc == 0 { continue; }
                            if dr.abs() > 1 && mv.area < 2 { continue; }
                            if dc.abs() > 1 && mv.area < 2 { continue; }
                            if ti >= MAX_JUMP_DELTAS { break; }
                            t.jumps[ti] = (dr, dc);
                            ti += 1;
                        }
                    }
                }

                t.n_jumps = ti as u8;

                // Slides.
                let mut si = 0usize;
                for &(dir, max_range) in &mv.slides {
                    if max_range == 1 { continue; }
                    if si >= MAX_SLIDES { break; }
                    t.slides[si] = (dir, max_range);
                    si += 1;
                }
                t.n_slides = si as u8;

                t.has_igui = mv.igui;
                t.area = mv.area;
                t.valid = true;

                table[pt as usize][color as usize] = t;
            }
        }
        table
    })
}

/// Generate pseudo-legal captures using the flat templates (memory-safe,
/// no huge precomputed bitboard table). For each piece, walks its jumps
/// (checking opponent occupancy) and slides (ray walk with blocking).
/// Handles hooks and range-capturers directly so it does NOT fall back to
/// the slow `movegen::generate_pseudo_legal_captures` for the common
/// special pieces. Only lion mid-captures (area>=2 with jumps) trigger
/// `NeedsFallback`.
pub fn generate_captures_bb(board: &Board) -> (Vec<crate::types::Move>, GenMode) {
    let color = board.side_to_move;
    let c = color as usize;
    let t = templates();
    let rt = ray_table();
    let mut moves = Vec::with_capacity(64);
    let mut mode = GenMode::AllFast;

    for i in 0..board.piece_list_len[c] {
        let sq = board.piece_list[c][i] as usize;
        if sq == INVALID_SQ as usize { continue; }
        let cell = board.cells[sq];
        if cell == EMPTY_CELL { continue; }
        let pt = cell_piece(cell);
        let tmpl = &t[(pt as usize).min(511)][color as usize];

        if !tmpl.valid {
            // Special piece: handle hooks, range-capturers, AND lion
            // mid-captures directly so we NEVER fall back to the slow
            // movegen path.
            let mv = pieces::movement(pt);
            if mv.hook.is_some() {
                gen_hook_captures(board, sq, pt, color, mv, rt, &mut moves);
            }
            if !mv.range_capture.is_empty() {
                gen_range_capture_captures(board, sq, pt, color, mv, rt, &mut moves);
            }
            if mv.area >= 2 {
                gen_lion_captures(board, sq, pt, color, mv, &mut moves);
            }
            continue;
        }

        let sq_r = sq_row(sq) as i32;
        let sq_c = sq_col(sq) as i32;

        // Jumps/steps/area: only capture if target is an enemy piece.
        for j in 0..tmpl.n_jumps as usize {
            let (dr, dc) = tmpl.jumps[j];
            let nr = sq_r + dr as i32;
            let nc = sq_c + dc as i32;
            if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 { continue; }
            let nsq = (nr as usize) * BOARD_SIZE + (nc as usize);
            let target = board.cells[nsq];
            if target != EMPTY_CELL && cell_color(target) != color {
                push_move(&mut moves, sq as u16, nsq as u16, pt, color, target);
            }
        }

        // Sliders: walk rays (occupancy-dependent blocking).
        for j in 0..tmpl.n_slides as usize {
            let (dir, max_range) = tmpl.slides[j];
            walk_ray_captures(board, rt, sq, pt, color, dir as usize, max_range, &mut moves);
        }

        // Igui (capture in place).
        if tmpl.has_igui {
            for d in 0..NUM_DIRS {
                if let Some(nsq) = step_sq(sq, d, color) {
                    let target = board.cells[nsq];
                    if target != EMPTY_CELL && cell_color(target) != color {
                        push_move_igui(&mut moves, sq as u16, pt, color, target);
                    }
                }
            }
        }
    }

    (moves, mode)
}

/// Generate hook-move captures (orthogonal or diagonal hook movers).
fn gen_hook_captures(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
                     rt: &RayTable, moves: &mut Vec<crate::types::Move>) {
    let dirs: &[usize] = match mv.hook {
        Some(HookType::Orthogonal) => &[N, E, S, W],
        Some(HookType::Diagonal) => &[NE, SE, SW, NW],
        None => return,
    };
    for &d in dirs {
        let ray = rt.ray_for_color(sq, d, color);
        for &mid_sq in ray.iter() {
            let mid = mid_sq as usize;
            let target = board.cells[mid];
            if target != EMPTY_CELL {
                if cell_color(target) != color {
                    push_move(moves, sq as u16, mid_sq, pt, color, target);
                }
                break;
            }
            let turn_dirs: &[usize] = match mv.hook {
                Some(HookType::Orthogonal) => {
                    if d == N || d == S { &HOOK_ORTHO_NS } else { &HOOK_ORTHO_EW }
                }
                Some(HookType::Diagonal) => match d {
                    NE => &HOOK_TURN_NE,
                    SE => &HOOK_TURN_SE,
                    SW => &HOOK_TURN_SW,
                    NW => &HOOK_TURN_NW,
                    _ => &[],
                },
                None => &[],
            };
            for &td in turn_dirs {
                let turn_ray = rt.ray_for_color(mid, td, color);
                for &tsq in turn_ray {
                    let t = board.cells[tsq as usize];
                    if t == EMPTY_CELL { continue; }
                    if cell_color(t) != color {
                        push_move(moves, sq as u16, tsq, pt, color, t);
                        break;
                    } else { break; }
                }
            }
        }
    }
}

/// Generate range-capture moves (captures pieces of lower rank along rays).
fn gen_range_capture_captures(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
                              rt: &RayTable, moves: &mut Vec<crate::types::Move>) {
    let piece_rank = pieces::rank(pt);
    for &dir in &mv.range_capture {
        let ray = rt.ray_for_color(sq, dir as usize, color);
        for &rsq in ray {
            let target = board.cells[rsq as usize];
            if target == EMPTY_CELL { continue; }
            let t_pt = cell_piece(target);
            let t_rank = pieces::rank(t_pt);
            if t_rank > piece_rank {
                push_move(moves, sq as u16, rsq, pt, color, target);
            } else { break; }
        }
    }
}

/// Generate lion area captures — the radius-1 and radius-2 moves, including
/// mid-captures (capture a piece on the way to a second target). This
/// eliminates the slow fallback to movegen for lions.
fn gen_lion_captures(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
                     moves: &mut Vec<crate::types::Move>) {
    let r = sq_row(sq) as i32;
    let c = sq_col(sq) as i32;

    for d1 in 0..NUM_DIRS {
        let (dr1, dc1) = get_deltas(d1, color);
        let r1 = r + dr1;
        let c1 = c + dc1;
        if r1 < 0 || r1 >= BOARD_SIZE as i32 || c1 < 0 || c1 >= BOARD_SIZE as i32 { continue; }
        let sq1 = r1 as usize * BOARD_SIZE + c1 as usize;
        let t1 = board.cells[sq1];
        if t1 != EMPTY_CELL && cell_color(t1) == color { continue; }

        // Direct capture at first step (if enemy).
        if t1 != EMPTY_CELL && cell_color(t1) != color {
            push_move(moves, sq as u16, sq1 as u16, pt, color, t1);
        }

        if mv.area >= 2 {
            for d2 in 0..NUM_DIRS {
                let (dr2, dc2) = get_deltas(d2, color);
                let r2 = r1 + dr2;
                let c2 = c1 + dc2;
                if r2 < 0 || r2 >= BOARD_SIZE as i32 || c2 < 0 || c2 >= BOARD_SIZE as i32 { continue; }
                let sq2 = r2 as usize * BOARD_SIZE + c2 as usize;
                if sq2 == sq { continue; }
                let t2 = board.cells[sq2];
                if t2 != EMPTY_CELL && cell_color(t2) == color { continue; }

                // Mid-capture: capture t1 on the way to sq2.
                if t1 != EMPTY_CELL && cell_color(t1) != color {
                    let mut m = crate::types::Move::simple(sq as u16, sq2 as u16);
                    m.mid_sq = sq1 as u16;
                    m.mid_piece = cell_piece(t1);
                    m.mid_color = cell_color(t1);
                    if t2 != EMPTY_CELL {
                        m.captured_piece = cell_piece(t2);
                        m.captured_color = cell_color(t2);
                    }
                    moves.push(m);
                } else if t2 != EMPTY_CELL && cell_color(t2) != color {
                    // Direct capture at second step (path empty).
                    push_move(moves, sq as u16, sq2 as u16, pt, color, t2);
                }
            }
        }
    }
}

#[inline]
fn can_promote(pt: u16) -> bool {
    pieces::promotes_to(pt).is_some()
}

#[inline]
fn walk_ray_captures(board: &Board, rt: &RayTable,
                     sq: usize, pt: u16, color: u8, dir: usize, max_range: u8,
                     moves: &mut Vec<crate::types::Move>) {
    let ray = rt.ray_for_color(sq, dir, color);
    let limit = if max_range == 0 { ray.len() } else { (max_range as usize).min(ray.len()) };
    for j in 0..limit {
        let rsq = ray[j] as usize;
        let target = board.cells[rsq];
        if target == EMPTY_CELL {
            // Non-capturing promotion: only include if entering promo zone.
            if in_promo_zone(rsq, color) && can_promote(pt) {
                push_move(moves, sq as u16, rsq as u16, pt, color, EMPTY_CELL);
            }
        } else if cell_color(target) != color {
            push_move(moves, sq as u16, rsq as u16, pt, color, target);
            break;
        } else {
            break;
        }
    }
}

/// Fast path for a single non-special piece. Generates jump/step/slide/area/
/// igui moves into `moves` using the flat template `tmpl`.
pub fn fast_piece(
    board: &Board, sq: usize, pt: u16, color: u8,
    tmpl: &Template, rt: &RayTable, moves: &mut Vec<crate::types::Move>,
) {
    let sq_r = sq_row(sq) as i32;
    let sq_c = sq_col(sq) as i32;

    for j in 0..tmpl.n_jumps as usize {
        let (dr, dc) = tmpl.jumps[j];
        let nr = sq_r + dr as i32;
        let nc = sq_c + dc as i32;
        if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 { continue; }
        let nsq = (nr as usize) * BOARD_SIZE + (nc as usize);
        let target = board.cells[nsq];
        if target == EMPTY_CELL {
            push_move(moves, sq as u16, nsq as u16, pt, color, EMPTY_CELL);
        } else if cell_color(target) != color {
            push_move(moves, sq as u16, nsq as u16, pt, color, target);
        }
    }

    for j in 0..tmpl.n_slides as usize {
        let (dir, max_range) = tmpl.slides[j];
        walk_ray(board, rt, sq, pt, color, dir as usize, max_range, moves);
    }

    if tmpl.has_igui {
        for d in 0..NUM_DIRS {
            if let Some(nsq) = step_sq(sq, d, color) {
                let target = board.cells[nsq];
                if target != EMPTY_CELL && cell_color(target) != color {
                    push_move_igui(moves, sq as u16, pt, color, target);
                }
            }
        }
    }
}

/// Indicates whether the fast-path generator covered all pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenMode {
    AllFast,
    NeedsFallback,
}

/// Fast path: generate pseudo-legal moves using flat templates for pieces
/// without hooks/range-captures/lion-mid-captures. When `NeedsFallback` is
/// returned, `moves` is incomplete (special pieces omitted) — the caller
/// must regenerate with `movegen::generate_pseudo_legal_moves`.
pub fn generate_simple_moves(board: &Board) -> (Vec<crate::types::Move>, GenMode) {
    let color = board.side_to_move;
    let c = color as usize;
    let t = templates();
    let rt = ray_table();
    let mut moves = Vec::with_capacity(384);
    let mut mode = GenMode::AllFast;

    for i in 0..board.piece_list_len[c] {
        let sq = board.piece_list[c][i] as usize;
        if sq == INVALID_SQ as usize { continue; }
        let cell = board.cells[sq];
        if cell == EMPTY_CELL { continue; }
        let pt = cell_piece(cell);
        let tmpl = &t[(pt as usize).min(511)][color as usize];

        if !tmpl.valid {
            mode = GenMode::NeedsFallback;
            continue;
        }

        let sq_r = sq_row(sq) as i32;
        let sq_c = sq_col(sq) as i32;

        for j in 0..tmpl.n_jumps as usize {
            let (dr, dc) = tmpl.jumps[j];
            let nr = sq_r + dr as i32;
            let nc = sq_c + dc as i32;
            if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 { continue; }
            let nsq = (nr as usize) * BOARD_SIZE + (nc as usize);
            let target = board.cells[nsq];
            if target == EMPTY_CELL {
                push_move(&mut moves, sq as u16, nsq as u16, pt, color, EMPTY_CELL);
            } else if cell_color(target) != color {
                push_move(&mut moves, sq as u16, nsq as u16, pt, color, target);
            }
        }

        for j in 0..tmpl.n_slides as usize {
            let (dir, max_range) = tmpl.slides[j];
            walk_ray(board, rt, sq, pt, color, dir as usize, max_range, &mut moves);
        }

        if tmpl.has_igui {
            for d in 0..NUM_DIRS {
                if let Some(nsq) = step_sq(sq, d, color) {
                    let target = board.cells[nsq];
                    if target != EMPTY_CELL && cell_color(target) != color {
                        push_move_igui(&mut moves, sq as u16, pt, color, target);
                    }
                }
            }
        }
    }

    (moves, mode)
}

#[inline]
fn push_move(moves: &mut Vec<crate::types::Move>, from: u16, to: u16, pt: u16,
             color: u8, target: Cell) {
    let captured = if target != EMPTY_CELL { cell_piece(target) } else { 0 };
    let cap_color = if target != EMPTY_CELL { cell_color(target) } else { 0 };
    moves.push(crate::types::Move {
        from_sq: from, to_sq: to, promotion: false,
        captured_piece: captured, captured_color: cap_color,
        is_igui: false, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
        range_caps: None,
    });
}

#[inline]
fn push_move_igui(moves: &mut Vec<crate::types::Move>, from: u16, pt: u16,
                  color: u8, target: Cell) {
    let captured = cell_piece(target);
    let cap_color = cell_color(target);
    moves.push(crate::types::Move {
        from_sq: from, to_sq: from,
        promotion: false,
        captured_piece: captured, captured_color: cap_color,
        is_igui: true, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
        range_caps: None,
    });
}

#[inline]
fn walk_ray(board: &Board, rt: &RayTable,
            sq: usize, pt: u16, color: u8, dir: usize, max_range: u8,
            moves: &mut Vec<crate::types::Move>) {
    let ray = rt.ray_for_color(sq, dir, color);
    let limit = if max_range == 0 { ray.len() } else { (max_range as usize).min(ray.len()) };
    for j in 0..limit {
        let rsq = ray[j] as usize;
        let target = board.cells[rsq];
        if target == EMPTY_CELL {
            push_move(moves, sq as u16, rsq as u16, pt, color, EMPTY_CELL);
        } else if cell_color(target) != color {
            push_move(moves, sq as u16, rsq as u16, pt, color, target);
            break;
        } else {
            break;
        }
    }
}