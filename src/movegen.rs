use crate::types::*;
use crate::pieces;
use crate::board::Board;
use std::sync::OnceLock;

const HOOK_ORTHO_NS: [usize; 2] = [N, S];
const HOOK_ORTHO_EW: [usize; 2] = [E, W];
const HOOK_TURN_NE: [usize; 2] = [NW, SE];
const HOOK_TURN_SE: [usize; 2] = [NE, SW];
const HOOK_TURN_SW: [usize; 2] = [SE, NW];
const HOOK_TURN_NW: [usize; 2] = [NE, SW];

// ── PRECOMPUTED JUMP DESTINATIONS ──────────────────────────────
// HaChu technique: precompute all jump destination squares for each
// (piece_type, square, color) combination. This eliminates the per-jump
// bounds checking and arithmetic in gen_jumps, turning it into a simple
// array lookup. Reference: docx §5.1 — "incremental move generation
// scales with the board perimeter, not the area".
//
// Format: JUMP_TABLE[pt][sq][color] -> &[(dest_sq, is_forward)]
// We store dest_sq as u16 (INVALID_SQ if off-board).
static JUMP_TABLE: OnceLock<Box<[[[[u16; 8]; 2]; NUM_SQUARES]; 512]>> = OnceLock::new();

fn jump_table() -> &'static [[[[u16; 8]; 2]; NUM_SQUARES]; 512] {
    JUMP_TABLE.get_or_init(|| {
        let mut table = Box::new([[[[INVALID_SQ; 8]; 2]; NUM_SQUARES]; 512]);
        for pt in 1..=301u16 {
            let mv = pieces::movement(pt);
            if mv.jumps.is_empty() { continue; }
            for sq in 0..NUM_SQUARES {
                let r = sq_row(sq) as i32;
                let c = sq_col(sq) as i32;
                for color in 0..2u8 {
                    let mut dests = [INVALID_SQ; 8];
                    for (j, &(jdr, jdc)) in mv.jumps.iter().enumerate() {
                        if j >= 8 { break; }
                        let (dr, dc) = if color == BLACK {
                            (jdr as i32, jdc as i32)
                        } else {
                            (-(jdr as i32), -(jdc as i32))
                        };
                        let nr = r + dr;
                        let nc = c + dc;
                        if nr >= 0 && nr < BOARD_SIZE as i32 && nc >= 0 && nc < BOARD_SIZE as i32 {
                            dests[j] = (nr as usize * BOARD_SIZE + nc as usize) as u16;
                        }
                    }
                    table[pt as usize][sq][color as usize] = dests;
                }
            }
        }
        table
    })
}


/// Generate pseudo-legal moves (fast, no legality filtering).
/// Does NOT filter out moves that leave king in check.
pub fn generate_pseudo_legal_moves(board: &Board) -> Vec<Move> {
    let color = board.side_to_move;
    let c = color as usize;
    let rt = ray_table();
    let jt = jump_table();
    let mut moves = Vec::with_capacity(512);

    for i in 0..board.piece_list_len[c] {
        let sq = board.piece_list[c][i] as usize;
        if sq == INVALID_SQ as usize { continue; }
        let cell = board.cells[sq];
        if cell == EMPTY_CELL { continue; }
        let pt = cell_piece(cell);
        let mv = pieces::movement(pt);

        gen_slides(board, sq, pt, color, mv, rt, &mut moves);
        gen_jumps_fast(board, sq, pt, color, mv, jt, &mut moves);

        if mv.hook.is_some() {
            gen_hooks(board, sq, pt, color, mv, rt, &mut moves);
        }
        if mv.area > 0 {
            gen_area(board, sq, pt, color, mv, &mut moves);
        }
        if !mv.range_capture.is_empty() {
            gen_range_capture(board, sq, pt, color, mv, rt, &mut moves);
        }
        if mv.igui {
            gen_igui(board, sq, pt, color, &mut moves);
        }
    }
    
    moves
}

/// Generate legal moves (slower: filters pseudo-legal moves for legality).
pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    let pseudo = generate_pseudo_legal_moves(board);
    filter_legal_moves(board, pseudo)
}

fn filter_legal_moves(board: &Board, mut moves: Vec<Move>) -> Vec<Move> {
    let mut legal_moves = Vec::with_capacity(moves.len());
    let mut board_copy = board.clone_without_history();

    for m in moves.drain(..) {
        board_copy.apply_move(&m);
        if !is_in_check(&board_copy) {
            legal_moves.push(m);
        }
        board_copy.undo_move();
    }

    legal_moves
}

/// Public check-detection entry point. Uses the bitboard-filtered version,
/// which only walks opposing pieces that are geometrically capable of
/// threatening the king (per ThreatZoneTable) instead of every piece in
/// piece_list. Falls back to the scalar full-scan automatically whenever
/// debug_assertions are on, to catch any future desync between
/// board.occupancy and board.cells immediately rather than silently
/// missing a check.
pub fn is_in_check(board: &Board) -> bool {
    let result = is_in_check_bitboard(board);
    #[cfg(debug_assertions)]
    {
        let reference = is_in_check_scalar(board);
        debug_assert_eq!(
            result, reference,
            "is_in_check_bitboard disagreed with is_in_check_scalar -- bitboard occupancy is out of sync with cells"
        );
    }
    result
}

/// Bitboard-accelerated check detection. Intersects the opponent's
/// occupancy bitboard with a precomputed "threat zone" for the king's
/// square (see ThreatZoneTable in bitboard.rs) to get the small set of
/// opposing pieces that could *possibly* threaten the king under any
/// movement rule in this game, then runs the exact same rule logic as
/// is_in_check_scalar (via piece_gives_check) only on that filtered set.
/// This turns an O(pieces_per_side) scan (up to ~402) into a scan over
/// however many opposing pieces are actually near the king -- typically
/// far fewer, especially in the opening/midgame.
fn is_in_check_bitboard(board: &Board) -> bool {
    let king_sq = board.king_square(board.side_to_move);
    if king_sq == INVALID_SQ { return false; }
    let king_sq_usize = king_sq as usize;
    let opp = 1 - board.side_to_move;
    let rt = crate::types::ray_table();
    let king_row = king_sq_usize / BOARD_SIZE;
    let king_col = king_sq_usize % BOARD_SIZE;

    let zone = crate::bitboard::threat_zone_table().zone(king_sq_usize);
    let candidates = board.occupancy[opp as usize].and_new(zone);

    let mut found = false;
    candidates.iter_usize(|sq| {
        if found { return; }
        if piece_gives_check(board, sq, opp, rt, king_sq_usize, king_row, king_col) {
            found = true;
        }
    });
    found
}

/// Reference (scalar) check detection. Iterates opposing pieces and, for
/// each, walks rays/deltas using precomputed RayTable lookups. This is the
/// original, verified-correct implementation — kept as the ground truth
/// that the bitboard-accelerated version (is_in_check_bitboard) is tested
/// against, and as an automatic fallback if bitboard occupancy is ever
/// found to be out of sync with `cells`.
fn is_in_check_scalar(board: &Board) -> bool {
    let king_sq = board.king_square(board.side_to_move);
    if king_sq == INVALID_SQ { return false; }
    let king_sq_usize = king_sq as usize;
    let opp = 1 - board.side_to_move;
    let rt = crate::types::ray_table();
    let king_row = king_sq_usize / BOARD_SIZE;
    let king_col = king_sq_usize % BOARD_SIZE;
    
    for i in 0..board.piece_list_len[opp as usize] {
        let sq = board.piece_list[opp as usize][i] as usize;
        if sq >= NUM_SQUARES { continue; }
        if piece_gives_check(board, sq, opp, rt, king_sq_usize, king_row, king_col) {
            return true;
        }
    }
    false
}

/// Checks whether the opposing piece at `sq` gives check to the king at
/// `king_sq_usize`. Shared by both is_in_check_scalar (called for every
/// piece in piece_list) and is_in_check_bitboard (called only for pieces
/// surviving the ThreatZoneTable filter) so the two implementations can
/// never drift apart in rule logic -- only in which squares they check.
#[inline]
fn piece_gives_check(
    board: &Board, sq: usize, opp: u8, rt: &RayTable,
    king_sq_usize: usize, king_row: usize, king_col: usize,
) -> bool {
        let cell = board.cells[sq];
        if cell == EMPTY_CELL { return false; }
        
        let pt = cell_piece(cell);
        let mv = pieces::movement(pt);
        let psq_row = sq / BOARD_SIZE;
        let psq_col = sq % BOARD_SIZE;
        
        let dr = if psq_row > king_row { psq_row - king_row } else { king_row - psq_row };
        let dc = if psq_col > king_col { psq_col - king_col } else { king_col - psq_col };
        let max_dist = dr.max(dc);
        
        // Jumps — use precomputed JUMP_TABLE for O(1) destination lookup
        // instead of per-jump arithmetic + bounds checking.
        if !mv.jumps.is_empty() {
            let jt = jump_table();
            let dests = &jt[pt as usize][sq][opp as usize];
            for j in 0..mv.jumps.len().min(8) {
                if dests[j] as usize == king_sq_usize { return true; }
            }
        }

        // Slides: use bitboard ray to check if king is reachable
        if !mv.slides.is_empty() {
            for &(dir, max_range) in &mv.slides {
                let ray = rt.ray_for_color(sq, dir as usize, opp);
                let limit = if max_range == 0 { ray.len() } else { (max_range as usize).min(ray.len()) };
                let check_count = limit.min(8);
                for j in 0..check_count {
                    let rsq = ray[j] as usize;
                    if rsq == king_sq_usize { return true; }
                    if board.cells[rsq] != EMPTY_CELL { break; }
                }
            }
        }
        
        // Hook moves
        if mv.hook.is_some() && max_dist < 8 {
            let dirs: &[usize] = match mv.hook {
                Some(HookType::Orthogonal) => &[N, E, S, W],
                Some(HookType::Diagonal) => &[NE, SE, SW, NW],
                None => &[],
            };
            for &d in dirs {
                let ray = rt.ray_for_color(sq, d, opp);
                for &mid_sq in ray.iter() {
                    let mid = mid_sq as usize;
                    let target = board.cells[mid];
                    if target != EMPTY_CELL { break; }
                    let turn_dirs: &[usize] = match mv.hook {
                        Some(HookType::Orthogonal) => {
                            if d == N || d == S { &HOOK_ORTHO_EW } else { &HOOK_ORTHO_NS }
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
                        let turn_ray = rt.ray_for_color(mid, td, opp);
                        for &tsq in turn_ray {
                            let t = board.cells[tsq as usize];
                            if t == EMPTY_CELL { continue; }
                            if cell_color(t) != opp {
                                if tsq as usize == king_sq_usize { return true; }
                                break;
                            } else { break; }
                        }
                    }
                }
            }
        }
        
        // Area moves
        if mv.area > 0 && max_dist <= 3 {
            let r = psq_row as i32;
            let c = psq_col as i32;
            for d1 in 0..NUM_DIRS {
                let (dr1, dc1) = get_deltas(d1, opp);
                let r1 = r + dr1; let c1 = c + dc1;
                if r1 < 0 || r1 >= BOARD_SIZE as i32 || c1 < 0 || c1 >= BOARD_SIZE as i32 { continue; }
                let sq1 = r1 as usize * BOARD_SIZE + c1 as usize;
                let t1 = board.cells[sq1];
                if t1 != EMPTY_CELL && cell_color(t1) == opp { continue; }
                if sq1 == king_sq_usize { return true; }
                if mv.area >= 2 {
                    for d2 in 0..NUM_DIRS {
                        let (dr2, dc2) = get_deltas(d2, opp);
                        let r2 = r1 + dr2; let c2 = c1 + dc2;
                        if r2 < 0 || r2 >= BOARD_SIZE as i32 || c2 < 0 || c2 >= BOARD_SIZE as i32 { continue; }
                        let sq2 = r2 as usize * BOARD_SIZE + c2 as usize;
                        if sq2 == sq { continue; }
                        let t2 = board.cells[sq2];
                        if t2 != EMPTY_CELL && cell_color(t2) == opp { continue; }
                        if sq2 == king_sq_usize { return true; }
                    }
                }
            }
        }
        
        // Range capture
        if !mv.range_capture.is_empty() {
            let piece_rank = pieces::rank(pt);
            for &dir in &mv.range_capture {
                let ray = rt.ray_for_color(sq, dir as usize, opp);
                for &rsq in ray {
                    let target = board.cells[rsq as usize];
                    if target == EMPTY_CELL {
                        if rsq as usize == king_sq_usize { return true; }
                    } else {
                        let t_pt = cell_piece(target);
                        let t_rank = pieces::rank(t_pt);
                        if t_rank > piece_rank {
                            if rsq as usize == king_sq_usize { return true; }
                        } else { break; }
                    }
                }
            }
        }
        
        // Igui
        if mv.igui && max_dist <= 1 {
            for d in 0..NUM_DIRS {
                if let Some(nsq) = step_sq(sq, d, opp) {
                    let target = board.cells[nsq];
                    if target != EMPTY_CELL && cell_color(target) != opp {
                        if nsq == king_sq_usize { return true; }
                    }
                }
            }
        }

        false
}

#[inline]
fn can_promote(pt: u16) -> bool {
    pieces::promotes_to(pt).is_some()
}

fn add_move(moves: &mut Vec<Move>, from: u16, to: u16, pt: u16, color: u8, target: Cell) {
    let captured = if target != EMPTY_CELL { cell_piece(target) } else { 0 };
    let cap_color = if target != EMPTY_CELL { cell_color(target) } else { 0 };

    if !can_promote(pt) {
        moves.push(Move {
            from_sq: from, to_sq: to, promotion: false,
            captured_piece: captured, captured_color: cap_color,
            is_igui: false, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
            range_caps: None,
        });
        return;
    }

    let from_in_zone = in_promo_zone(from as usize, color);
    let to_in_zone = in_promo_zone(to as usize, color);
    let is_capture = captured != 0;

    let may_promote =
        (!from_in_zone && to_in_zone) ||
        (from_in_zone && to_in_zone && is_capture);

    let must_promote = to_in_zone
        && is_farthest_rank(to as usize, color)
        && pieces::must_promote_at_far_rank(pt);

    if must_promote {
        moves.push(Move {
            from_sq: from, to_sq: to, promotion: true,
            captured_piece: captured, captured_color: cap_color,
            is_igui: false, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
            range_caps: None,
        });
    } else if may_promote {
        moves.push(Move {
            from_sq: from, to_sq: to, promotion: false,
            captured_piece: captured, captured_color: cap_color,
            is_igui: false, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
            range_caps: None,
        });
        moves.push(Move {
            from_sq: from, to_sq: to, promotion: true,
            captured_piece: captured, captured_color: cap_color,
            is_igui: false, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
            range_caps: None,
        });
    } else {
        moves.push(Move {
            from_sq: from, to_sq: to, promotion: false,
            captured_piece: captured, captured_color: cap_color,
            is_igui: false, mid_sq: INVALID_SQ, mid_piece: 0, mid_color: 0,
            range_caps: None,
        });
    }
}

fn gen_slides(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
              rt: &RayTable, moves: &mut Vec<Move>) {
    for &(dir, max_range) in &mv.slides {
        let ray = rt.ray_for_color(sq, dir as usize, color);
        let limit = if max_range == 0 { ray.len() } else { (max_range as usize).min(ray.len()) };

        for j in 0..limit {
            let target_sq = ray[j] as usize;
            let target = board.cells[target_sq];
            if target == EMPTY_CELL {
                add_move(moves, sq as u16, target_sq as u16, pt, color, EMPTY_CELL);
            } else if cell_color(target) != color {
                add_move(moves, sq as u16, target_sq as u16, pt, color, target);
                break;
            } else {
                break;
            }
        }
    }
}

/// Fast jump generation using precomputed destination table.
/// Eliminates per-jump bounds checking and arithmetic — just a lookup
/// into JUMP_TABLE[pt][sq][color] followed by an occupancy check.
fn gen_jumps_fast(
    board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
    jt: &[[[[u16; 8]; 2]; NUM_SQUARES]; 512],
    moves: &mut Vec<Move>,
) {
    if mv.jumps.is_empty() { return; }
    let dests = &jt[pt as usize][sq][color as usize];
    for j in 0..mv.jumps.len().min(8) {
        let nsq = dests[j];
        if nsq == INVALID_SQ { continue; }
        let nsq_u = nsq as usize;
        let target = board.cells[nsq_u];
        if target == EMPTY_CELL {
            add_move(moves, sq as u16, nsq, pt, color, EMPTY_CELL);
        } else if cell_color(target) != color {
            add_move(moves, sq as u16, nsq, pt, color, target);
        }
    }
}

/// Original gen_jumps — kept for reference/debugging.
#[allow(dead_code)]
fn gen_jumps(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
             moves: &mut Vec<Move>) {
    let r = sq_row(sq) as i32;
    let c = sq_col(sq) as i32;
    for &(jdr, jdc) in &mv.jumps {
        let (dr, dc) = if color == BLACK {
            (jdr as i32, jdc as i32)
        } else {
            (-(jdr as i32), -(jdc as i32))
        };
        let nr = r + dr;
        let nc = c + dc;
        if nr < 0 || nr >= BOARD_SIZE as i32 || nc < 0 || nc >= BOARD_SIZE as i32 { continue; }
        let nsq = nr as usize * BOARD_SIZE + nc as usize;
        let target = board.cells[nsq];
        if target == EMPTY_CELL {
            add_move(moves, sq as u16, nsq as u16, pt, color, EMPTY_CELL);
        } else if cell_color(target) != color {
            add_move(moves, sq as u16, nsq as u16, pt, color, target);
        }
    }
}

fn gen_hooks(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
             rt: &RayTable, moves: &mut Vec<Move>) {
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
                    add_move(moves, sq as u16, mid_sq, pt, color, target);
                }
                break;
            }
            let turn_dirs: &[usize] = match mv.hook {
                Some(HookType::Orthogonal) => {
                    if d == N || d == S { &HOOK_ORTHO_EW } else { &HOOK_ORTHO_NS }
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
                    if t == EMPTY_CELL {
                        add_move(moves, sq as u16, tsq, pt, color, EMPTY_CELL);
                    } else if cell_color(t) != color {
                        add_move(moves, sq as u16, tsq, pt, color, t);
                        break;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

fn gen_area(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
            moves: &mut Vec<Move>) {
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

        add_move(moves, sq as u16, sq1 as u16, pt, color, t1);

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

                if t1 != EMPTY_CELL && cell_color(t1) != color {
                    let mut m = Move::simple(sq as u16, sq2 as u16);
                    m.mid_sq = sq1 as u16;
                    m.mid_piece = cell_piece(t1);
                    m.mid_color = cell_color(t1);
                    if t2 != EMPTY_CELL {
                        m.captured_piece = cell_piece(t2);
                        m.captured_color = cell_color(t2);
                    }
                    moves.push(m);
                } else {
                    add_move(moves, sq as u16, sq2 as u16, pt, color, t2);
                }
            }
        }
    }
}

fn gen_range_capture(board: &Board, sq: usize, pt: u16, color: u8, mv: &Movement,
                     rt: &RayTable, moves: &mut Vec<Move>) {
    use std::rc::Rc;
    let piece_rank = pieces::rank(pt);

    for &dir in &mv.range_capture {
        let ray = rt.ray_for_color(sq, dir as usize, color);
        // Grows as captures accumulate along the ray. Wrapped in Rc once
        // captures exist so every Move sharing this prefix just bumps a
        // refcount (O(1)) instead of deep-copying the Vec (O(n) per move,
        // O(n^2) over the whole ray) — this was the movegen hot spot.
        let mut captured_list: Vec<(u16, u16, u8)> = Vec::new();
        let mut shared: Option<Rc<Vec<(u16, u16, u8)>>> = None;

        for &rsq in ray {
            let target = board.cells[rsq as usize];
            if target == EMPTY_CELL {
                let mut m = Move::simple(sq as u16, rsq);
                if let Some(ref rc) = shared {
                    m.range_caps = Some(Rc::clone(rc));
                }
                moves.push(m);
            } else {
                let t_pt = cell_piece(target);
                let t_rank = pieces::rank(t_pt);
                if t_rank > piece_rank {
                    captured_list.push((rsq, t_pt, cell_color(target)));
                    // Re-share the updated prefix once per new capture
                    // (not once per generated move).
                    shared = Some(Rc::new(captured_list.clone()));
                    let from_in = in_promo_zone(sq, color);
                    let to_in = in_promo_zone(rsq as usize, color);
                    let may_promo = can_promote(pt) && (
                        (!from_in && to_in) || (from_in && to_in)
                    );
                    let must_promo = may_promo && is_farthest_rank(rsq as usize, color)
                        && pieces::must_promote_at_far_rank(pt);
                    if must_promo {
                        let mut m = Move::simple(sq as u16, rsq);
                        m.captured_piece = t_pt;
                        m.captured_color = cell_color(target);
                        m.range_caps = shared.clone();
                        m.promotion = true;
                        moves.push(m);
                    } else if may_promo {
                        let mut m1 = Move::simple(sq as u16, rsq);
                        m1.captured_piece = t_pt;
                        m1.captured_color = cell_color(target);
                        m1.range_caps = shared.clone();
                        moves.push(m1);
                        let mut m2 = Move::simple(sq as u16, rsq);
                        m2.captured_piece = t_pt;
                        m2.captured_color = cell_color(target);
                        m2.range_caps = shared.clone();
                        m2.promotion = true;
                        moves.push(m2);
                    } else {
                        let mut m = Move::simple(sq as u16, rsq);
                        m.captured_piece = t_pt;
                        m.captured_color = cell_color(target);
                        m.range_caps = shared.clone();
                        moves.push(m);
                    }
                } else {
                    break;
                }
            }
        }
    }
}

fn gen_igui(board: &Board, sq: usize, pt: u16, color: u8, moves: &mut Vec<Move>) {
    for d in 0..NUM_DIRS {
        if let Some(nsq) = step_sq(sq, d, color) {
            let target = board.cells[nsq];
            if target != EMPTY_CELL && cell_color(target) != color {
                let in_zone = in_promo_zone(sq, color);
                let may_promo = can_promote(pt) && in_zone;
                if may_promo {
                    let mut m1 = Move::simple(sq as u16, sq as u16);
                    m1.captured_piece = cell_piece(target);
                    m1.captured_color = cell_color(target);
                    m1.is_igui = true;
                    moves.push(m1);
                    let mut m2 = Move::simple(sq as u16, sq as u16);
                    m2.captured_piece = cell_piece(target);
                    m2.captured_color = cell_color(target);
                    m2.is_igui = true;
                    m2.promotion = true;
                    moves.push(m2);
                } else {
                    let mut m = Move::simple(sq as u16, sq as u16);
                    m.captured_piece = cell_piece(target);
                    m.captured_color = cell_color(target);
                    m.is_igui = true;
                    moves.push(m);
                }
            }
        }
    }
}