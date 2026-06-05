/** Cell piece data returned by the API */
export interface CellData {
  piece: string;
  color: number;   // 0 = Black, 1 = White
  name: string;
  value: number;
  is_royal: boolean;
}

/** Full board state from /api/state */
export interface BoardState {
  board: (CellData | null)[][];
  side_to_move: number;
  move_number: number;
  black_pieces: number;
  white_pieces: number;
  game_result: string | null;
  mode: string;
  move_log: string[];
  score_history: number[];
  game_over: boolean;
}

/** A legal destination for a selected piece */
export interface MoveData {
  to: [number, number];
  promotion: boolean;
  is_igui: boolean;
  captured: string | null;
}

export interface MoveResponse {
  ok: boolean;
  error?: string;
  board?: BoardState;
  last_move?: { from: [number, number]; to: [number, number] };
}

export type Square = [number, number];
export type PieceColor = 0 | 1;