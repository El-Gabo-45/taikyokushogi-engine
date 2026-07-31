use crate::types::*;
use crate::pieces;
use crate::bitboard::Bitboard1296;
use crate::bitboard::U64_COUNT;

pub struct Board {
    pub cells: [Cell; NUM_SQUARES],
    pub side_to_move: u8,
    pub move_number: u32,
    /// Plies since last capture or promotion (for 500-move draw rule).
    pub no_progress_plies: u32,
    // Piece lists per color
    pub piece_list: [[u16; MAX_PIECES_PER_SIDE]; 2],
    pub piece_list_len: [usize; 2],
    pub piece_count: [usize; 2],
    // Royal piece tracking
    pub royal_list: [[u16; MAX_ROYALS]; 2],
    pub royal_count: [usize; 2],
    // O(1) reverse index: piece_index[sq] = position in piece_list for that square
    pub piece_index: [u16; NUM_SQUARES],
    // Incremental Zobrist hash — updated in apply_move/undo_move
    pub hash: u64,
    // ── INCREMENTAL MATERIAL SCORE ────────────────────────────────
    // Material score from Black's perspective, maintained incrementally
    // in apply_move/undo_move. Positive = Black has more material.
    // This makes material_score() O(1) instead of O(pieces).
    // Reference: Chess Programming Wiki, "Incremental Updates".
    pub material_score: i32,
    // Undo stack
    history: Vec<UndoInfo>,
    // ── BITBOARD OCCUPANCY ────────────────────────────────────────
    // Occupancy per color: fast attack generation and check detection
    pub occupancy: [Bitboard1296; 2],  // [0]=BLACK, [1]=WHITE
    pub all_occupancy: Bitboard1296,
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Board {
            cells: self.cells,
            side_to_move: self.side_to_move,
            move_number: self.move_number,
            no_progress_plies: self.no_progress_plies,
            piece_list: self.piece_list,
            piece_list_len: self.piece_list_len,
            piece_count: self.piece_count,
            royal_list: self.royal_list,
            royal_count: self.royal_count,
            piece_index: self.piece_index,
            hash: self.hash,
            material_score: self.material_score,
            history: self.history.clone(),
            occupancy: self.occupancy,
            all_occupancy: self.all_occupancy,
        }
    }
}

impl Board {
    pub fn new() -> Self {
        Board {
            cells: [EMPTY_CELL; NUM_SQUARES],
            side_to_move: BLACK,
            move_number: 1,
            no_progress_plies: 0,
            piece_list: [[INVALID_SQ; MAX_PIECES_PER_SIDE]; 2],
            piece_list_len: [0; 2],
            piece_count: [0; 2],
            royal_list: [[INVALID_SQ; MAX_ROYALS]; 2],
            royal_count: [0; 2],
            piece_index: [INVALID_SQ; NUM_SQUARES],
            hash: 0,
            material_score: 0,
            history: Vec::new(),
            occupancy: [Bitboard1296::new(), Bitboard1296::new()],
            all_occupancy: Bitboard1296::new(),
        }
    }

    pub fn setup_initial(&mut self) {
        self.cells = [EMPTY_CELL; NUM_SQUARES];
        self.side_to_move = BLACK;
        self.move_number = 1;
        self.no_progress_plies = 0;
        self.piece_list = [[INVALID_SQ; MAX_PIECES_PER_SIDE]; 2];
        self.piece_list_len = [0; 2];
        self.piece_count = [0; 2];
        self.royal_list = [[INVALID_SQ; MAX_ROYALS]; 2];
        self.royal_count = [0; 2];
        self.piece_index = [INVALID_SQ; NUM_SQUARES];
        self.hash = 0;
        self.material_score = 0;
        self.history.clear();
        self.occupancy = [Bitboard1296::new(), Bitboard1296::new()];
        self.all_occupancy = Bitboard1296::new();

        // Place Black's pieces (rows 24-35)
        for (rank_idx, rank_str) in pieces::SETUP_RANKS.iter().enumerate() {
            let rank_pieces = pieces::parse_setup_rank(rank_str);
            let row = 35 - rank_idx;
            for (col, piece_opt) in rank_pieces.iter().enumerate() {
                if let Some(pt) = piece_opt {
                    self.place_piece(row, col, *pt, BLACK);
                }
            }
        }

        // Place White's pieces (rows 0-11) — 180-degree rotation
        for (rank_idx, rank_str) in pieces::SETUP_RANKS.iter().enumerate() {
            let rank_pieces = pieces::parse_setup_rank(rank_str);
            let row = rank_idx;
            let len = rank_pieces.len();
            for (col, piece_opt) in rank_pieces.iter().enumerate() {
                if let Some(pt) = piece_opt {
                    self.place_piece(row, len - 1 - col, *pt, WHITE);
                }
            }
        }
    }

    fn place_piece(&mut self, row: usize, col: usize, pt: u16, color: u8) {
        let sq = sq_index(row, col);
        self.cells[sq] = make_cell(pt, color);
        let c = color as usize;

        let idx = self.piece_list_len[c];
        self.piece_index[sq] = idx as u16;
        self.piece_list[c][idx] = sq as u16;
        self.piece_list_len[c] = idx + 1;
        self.piece_count[c] += 1;

        // Update bitboards
        self.occupancy[c].set_usize(sq);
        self.all_occupancy.set_usize(sq);

        // Update incremental material score
        let val = pieces::value(pt) as i32;
        if color == BLACK { self.material_score += val; }
        else { self.material_score -= val; }

        if pieces::is_royal(pt) {
            let ri = self.royal_count[c];
            self.royal_list[c][ri] = sq as u16;
            self.royal_count[c] = ri + 1;
        }

        self.hash ^= zobrist_piece_key(pt, sq, color);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn at(&self, sq: usize) -> Cell {
        self.cells[sq]
    }

    pub fn apply_move(&mut self, m: &Move) {
        let from = m.from_sq as usize;
        let to = m.to_sq as usize;
        let from_cell = self.cells[from];
        let to_cell = self.cells[to];
        let pt = cell_piece(from_cell);
        let color = cell_color(from_cell);
        let c = color as usize;
        let _opp = 1 - c;

        let mut undo = UndoInfo {
            from_sq: m.from_sq, to_sq: m.to_sq,
            from_cell, to_cell,
            side: self.side_to_move,
            move_number: self.move_number,
            mid_sq: m.mid_sq, mid_cell: EMPTY_CELL,
            range_caps: None,
            no_progress_plies: self.no_progress_plies,
            hash: self.hash,
            material_score: self.material_score,
        };

        let is_capture = to_cell != EMPTY_CELL
            || m.mid_sq != INVALID_SQ
            || m.range_caps.is_some()
            || m.is_igui;
        if is_capture || m.promotion {
            self.no_progress_plies = 0;
        } else {
            self.no_progress_plies += 1;
        }

        // Handle range captures
        if let Some(ref caps) = m.range_caps {
            let mut saved = Vec::new();
            for &(sq, _cap_pt, _cap_color) in caps.iter() {
                let cap_cell = self.cells[sq as usize];
                saved.push((sq, cap_cell));
                if cap_cell != EMPTY_CELL {
                    self.hash ^= zobrist_piece_key(
                        cell_piece(cap_cell), sq as usize, cell_color(cap_cell));
                    // Update incremental material score for captured piece
                    let cap_val = pieces::value(cell_piece(cap_cell)) as i32;
                    let cap_c = cell_color(cap_cell);
                    if cap_c == BLACK { self.material_score -= cap_val; }
                    else { self.material_score += cap_val; }
                }
                self.remove_from_lists(sq as usize);
                self.cells[sq as usize] = EMPTY_CELL;
            }
            undo.range_caps = Some(saved);
        }

        // Handle lion mid-capture
        if m.mid_sq != INVALID_SQ {
            let msq = m.mid_sq as usize;
            undo.mid_cell = self.cells[msq];
            if undo.mid_cell != EMPTY_CELL {
                self.hash ^= zobrist_piece_key(
                    cell_piece(undo.mid_cell), msq, cell_color(undo.mid_cell));
                // Update incremental material score for mid-captured piece
                let mid_val = pieces::value(cell_piece(undo.mid_cell)) as i32;
                let mid_c = cell_color(undo.mid_cell);
                if mid_c == BLACK { self.material_score -= mid_val; }
                else { self.material_score += mid_val; }
            }
            self.remove_from_lists(msq);
            self.cells[msq] = EMPTY_CELL;
        }

        // Handle igui
        if m.is_igui {
            if to_cell != EMPTY_CELL {
                self.hash ^= zobrist_piece_key(
                    cell_piece(to_cell), to, cell_color(to_cell));
                // Update incremental material score for igui-captured piece
                let cap_val = pieces::value(cell_piece(to_cell)) as i32;
                let cap_c = cell_color(to_cell);
                if cap_c == BLACK { self.material_score -= cap_val; }
                else { self.material_score += cap_val; }
                self.remove_from_lists(to);
                self.cells[to] = EMPTY_CELL;
            }
            if m.promotion {
                if let Some(promo_pt) = pieces::promotes_to(pt) {
                    self.hash ^= zobrist_piece_key(pt, from, color);
                    // Update material score for promotion
                    let old_val = pieces::value(pt) as i32;
                    let new_val = pieces::value(promo_pt) as i32;
                    let delta = new_val - old_val;
                    if color == BLACK { self.material_score += delta; }
                    else { self.material_score -= delta; }
                    let new_cell = make_cell(promo_pt, color);
                    self.cells[from] = new_cell;
                    self.hash ^= zobrist_piece_key(promo_pt, from, color);
                    self.update_royal_status(from, pt, promo_pt, c);
                }
            }
            self.side_to_move = 1 - self.side_to_move;
            self.hash ^= zobrist_side_key();
            if self.side_to_move == BLACK { self.move_number += 1; }
            self.history.push(undo);
            return;
        }

        // Remove piece from origin
        self.hash ^= zobrist_piece_key(pt, from, color);
        self.cells[from] = EMPTY_CELL;
        self.remove_sq_from_piece_list(from, c);
        // Update bitboards for from-square
        self.occupancy[c].clear_usize(from);
        self.all_occupancy.clear_usize(from);
        if pieces::is_royal(pt) {
            self.remove_sq_from_royal_list(from, c);
        }

        // Capture at destination
        if to_cell != EMPTY_CELL {
            self.hash ^= zobrist_piece_key(
                cell_piece(to_cell), to, cell_color(to_cell));
            // Update incremental material score for captured piece
            let cap_val = pieces::value(cell_piece(to_cell)) as i32;
            let cap_c = cell_color(to_cell);
            if cap_c == BLACK { self.material_score -= cap_val; }
            else { self.material_score += cap_val; }
            self.remove_from_lists(to);
            // Bitboard cleanup for captured piece is handled in remove_from_lists
        }

        let final_pt = if m.promotion {
            pieces::promotes_to(pt).unwrap_or(pt)
        } else {
            pt
        };

        // Update material score for promotion (if applicable)
        if m.promotion && final_pt != pt {
            let old_val = pieces::value(pt) as i32;
            let new_val = pieces::value(final_pt) as i32;
            let delta = new_val - old_val;
            if color == BLACK { self.material_score += delta; }
            else { self.material_score -= delta; }
        }

        // Place at destination
        self.hash ^= zobrist_piece_key(final_pt, to, color);
        self.cells[to] = make_cell(final_pt, color);
        self.add_sq_to_piece_list(to, c);
        // Update bitboards for to-square
        self.occupancy[c].set_usize(to);
        self.all_occupancy.set_usize(to);
        if pieces::is_royal(final_pt) {
            let ri = self.royal_count[c];
            self.royal_list[c][ri] = to as u16;
            self.royal_count[c] = ri + 1;
        }

        self.side_to_move = 1 - self.side_to_move;
        self.hash ^= zobrist_side_key();
        if self.side_to_move == BLACK { self.move_number += 1; }
        self.history.push(undo);
    }

    pub fn undo_move(&mut self) -> bool {
        let undo = match self.history.pop() {
            Some(u) => u,
            None => return false,
        };

        self.hash = undo.hash;
        self.side_to_move = undo.side;
        self.move_number = undo.move_number;
        self.no_progress_plies = undo.no_progress_plies;
        self.material_score = undo.material_score;

        let from = undo.from_sq as usize;
        let to = undo.to_sq as usize;

        // Igui: fall back to rebuild (rare)
        if undo.from_sq == undo.to_sq {
            self.cells[from] = undo.from_cell;
            if undo.mid_sq != INVALID_SQ {
                self.cells[undo.mid_sq as usize] = undo.mid_cell;
            }
            self.rebuild_lists();
            return true;
        }

        // ── Incremental undo ─────────────────────────────────────────────

        // 1. Remove piece from destination
        let dest_cell = self.cells[to];
        if dest_cell != EMPTY_CELL {
            let dest_pt = cell_piece(dest_cell);
            let dest_c = cell_color(dest_cell) as usize;
            self.remove_sq_from_piece_list(to, dest_c);
            // Update bitboards
            self.occupancy[dest_c].clear_usize(to);
            self.all_occupancy.clear_usize(to);
            if pieces::is_royal(dest_pt) { self.remove_sq_from_royal_list(to, dest_c); }
            self.cells[to] = EMPTY_CELL;
        }

        // 2. Restore original piece at origin
        let orig_pt = cell_piece(undo.from_cell);
        let orig_c = cell_color(undo.from_cell) as usize;
        self.cells[from] = undo.from_cell;
        self.add_sq_to_piece_list(from, orig_c);
        // Update bitboards
        self.occupancy[orig_c].set_usize(from);
        self.all_occupancy.set_usize(from);
        if pieces::is_royal(orig_pt) {
            self.royal_list[orig_c][self.royal_count[orig_c]] = from as u16;
            self.royal_count[orig_c] += 1;
        }

        // 3. Restore captured piece at destination
        if undo.to_cell != EMPTY_CELL {
            let cap_pt = cell_piece(undo.to_cell);
            let cap_c = cell_color(undo.to_cell) as usize;
            self.cells[to] = undo.to_cell;
            self.add_sq_to_piece_list(to, cap_c);
            // Update bitboards
            self.occupancy[cap_c].set_usize(to);
            self.all_occupancy.set_usize(to);
            if pieces::is_royal(cap_pt) {
                self.royal_list[cap_c][self.royal_count[cap_c]] = to as u16;
                self.royal_count[cap_c] += 1;
            }
        }

        // 4. Restore mid-capture (lion)
        if undo.mid_sq != INVALID_SQ && undo.mid_cell != EMPTY_CELL {
            let msq = undo.mid_sq as usize;
            let mid_pt = cell_piece(undo.mid_cell);
            let mid_c = cell_color(undo.mid_cell) as usize;
            self.cells[msq] = undo.mid_cell;
            self.add_sq_to_piece_list(msq, mid_c);
            self.occupancy[mid_c].set_usize(msq);
            self.all_occupancy.set_usize(msq);
            if pieces::is_royal(mid_pt) {
                self.royal_list[mid_c][self.royal_count[mid_c]] = msq as u16;
                self.royal_count[mid_c] += 1;
            }
        }

        // 5. Restore range captures
        if let Some(ref caps) = undo.range_caps {
            for &(sq, cell) in caps {
                if cell != EMPTY_CELL {
                    let squ = sq as usize;
                    let cap_pt = cell_piece(cell);
                    let cap_c = cell_color(cell) as usize;
                    self.cells[squ] = cell;
                    self.add_sq_to_piece_list(squ, cap_c);
                    self.occupancy[cap_c].set_usize(squ);
                    self.all_occupancy.set_usize(squ);
                    if pieces::is_royal(cap_pt) {
                        self.royal_list[cap_c][self.royal_count[cap_c]] = sq;
                        self.royal_count[cap_c] += 1;
                    }
                }
            }
        }

        true
    }

    pub fn rebuild_lists_pub(&mut self) {
        self.rebuild_lists();
    }

    fn rebuild_lists(&mut self) {
        self.piece_list_len = [0; 2];
        self.piece_count = [0; 2];
        self.royal_count = [0; 2];
        self.occupancy = [Bitboard1296::new(), Bitboard1296::new()];
        self.all_occupancy = Bitboard1296::new();
        self.material_score = 0;

        for sq in 0..NUM_SQUARES {
            let cell = self.cells[sq];
            if cell != EMPTY_CELL {
                let pt = cell_piece(cell);
                let color = cell_color(cell);
                let c = color as usize;
                let idx = self.piece_list_len[c];
                self.piece_index[sq] = idx as u16;
                self.piece_list[c][idx] = sq as u16;
                self.piece_list_len[c] = idx + 1;
                self.piece_count[c] += 1;
                self.occupancy[c].set_usize(sq);
                self.all_occupancy.set_usize(sq);
                // Rebuild material score
                let val = pieces::value(pt) as i32;
                if color == BLACK { self.material_score += val; }
                else { self.material_score -= val; }
                if pieces::is_royal(pt) {
                    let ri = self.royal_count[c];
                    self.royal_list[c][ri] = sq as u16;
                    self.royal_count[c] = ri + 1;
                }
            } else {
                self.piece_index[sq] = INVALID_SQ;
            }
        }
    }

    // remove_from_lists: used by apply_move for captured pieces
    fn remove_from_lists(&mut self, sq: usize) {
        let cell = self.cells[sq];
        if cell == EMPTY_CELL { return; }
        let pt = cell_piece(cell);
        let color = cell_color(cell) as usize;
        // Update bitboards BEFORE removing from list (we need cell info)
        self.occupancy[color].clear_usize(sq);
        self.all_occupancy.clear_usize(sq);
        self.remove_sq_from_piece_list(sq, color);
        if pieces::is_royal(pt) {
            self.remove_sq_from_royal_list(sq, color);
        }
    }

    // O(1) removal using piece_index
    fn remove_sq_from_piece_list(&mut self, sq: usize, color: usize) {
        let idx = self.piece_index[sq] as usize;
        let len = self.piece_list_len[color];
        if idx >= len { return; }
        let last_sq = self.piece_list[color][len - 1] as usize;
        self.piece_list[color][idx] = last_sq as u16;
        self.piece_index[last_sq] = idx as u16;
        self.piece_list[color][len - 1] = INVALID_SQ;
        self.piece_index[sq] = INVALID_SQ;
        self.piece_list_len[color] = len - 1;
        self.piece_count[color] -= 1;
    }

    fn remove_sq_from_royal_list(&mut self, sq: usize, color: usize) {
        let sq16 = sq as u16;
        let len = self.royal_count[color];
        for i in 0..len {
            if self.royal_list[color][i] == sq16 {
                self.royal_list[color][i] = self.royal_list[color][len - 1];
                self.royal_list[color][len - 1] = INVALID_SQ;
                self.royal_count[color] = len - 1;
                return;
            }
        }
    }

    fn add_sq_to_piece_list(&mut self, sq: usize, color: usize) {
        let idx = self.piece_list_len[color];
        if idx >= MAX_PIECES_PER_SIDE { return; }
        self.piece_index[sq] = idx as u16;
        self.piece_list[color][idx] = sq as u16;
        self.piece_list_len[color] = idx + 1;
        self.piece_count[color] += 1;
    }

    fn update_royal_status(&mut self, sq: usize, old_pt: u16, new_pt: u16, color: usize) {
        if pieces::is_royal(old_pt) && !pieces::is_royal(new_pt) {
            self.remove_sq_from_royal_list(sq, color);
        } else if !pieces::is_royal(old_pt) && pieces::is_royal(new_pt) {
            let ri = self.royal_count[color];
            self.royal_list[color][ri] = sq as u16;
            self.royal_count[color] = ri + 1;
        }
    }

    pub fn king_square(&self, color: u8) -> u16 {
        let c = color as usize;
        if self.royal_count[c] > 0 {
            self.royal_list[c][0]
        } else {
            INVALID_SQ
        }
    }

    pub fn game_result(&self) -> Option<GameResult> {
        let b = self.royal_count[BLACK as usize] > 0;
        let w = self.royal_count[WHITE as usize] > 0;
        match (b, w) {
            (true, true) => {
                if self.no_progress_plies >= DRAW_PLIES {
                    Some(GameResult::Draw)
                } else {
                    None
                }
            }
            (true, false) => Some(GameResult::BlackWins),
            (false, true) => Some(GameResult::WhiteWins),
            (false, false) => Some(GameResult::Draw),
        }
    }

    pub fn display(&self) -> String {
        let mut s = String::new();
        for r in 0..BOARD_SIZE {
            s.push_str(&format!("{:2} ", BOARD_SIZE - r));
            for c in 0..BOARD_SIZE {
                let cell = self.cells[sq_index(r, c)];
                if cell == EMPTY_CELL {
                    s.push_str(".. ");
                } else {
                    let pt = cell_piece(cell);
                    let color = cell_color(cell);
                    let prefix = if color == WHITE { 'v' } else { '^' };
                    let ab = pieces::abbrev(pt);
                    s.push(prefix);
                    s.push_str(&format!("{:<2}", &ab[..ab.len().min(2)]));
                }
            }
            s.push('\n');
        }
        s
    }

    pub fn null_move(&mut self) {
        let undo = UndoInfo {
            from_sq: INVALID_SQ,
            to_sq: INVALID_SQ,
            from_cell: EMPTY_CELL,
            to_cell: EMPTY_CELL,
            side: self.side_to_move,
            move_number: self.move_number,
            mid_sq: INVALID_SQ,
            mid_cell: EMPTY_CELL,
            range_caps: None,
            no_progress_plies: self.no_progress_plies,
            hash: self.hash,
            material_score: self.material_score,
        };
        self.no_progress_plies += 1;
        self.side_to_move = 1 - self.side_to_move;
        self.hash ^= zobrist_side_key();
        if self.side_to_move == BLACK { self.move_number += 1; }
        self.history.push(undo);
    }

    pub fn undo_null_move(&mut self) {
        if let Some(undo) = self.history.pop() {
            self.hash = undo.hash;
            self.side_to_move = undo.side;
            self.move_number = undo.move_number;
            self.no_progress_plies = undo.no_progress_plies;
            self.material_score = undo.material_score;
        }
    }
}