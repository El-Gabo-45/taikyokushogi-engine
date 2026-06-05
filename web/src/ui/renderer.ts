/**
 * Canvas-based board renderer.
 *
 * Draws the 36×36 board, pieces, highlights and overlay indicators.
 * Not intended to be fast — clarity and readability are prioritised.
 */

import type { BoardState, CellData, MoveData } from '../types/board';
import { pieceKanji } from '../data/kanji';

// ── Layout constants ──────────────────────────────────────────────
export const CELL = 28;           // px per square
export const COORD = 18;          // px for coordinate margins
export const BOARD = 36;          // squares per side
export const BOARD_PX = CELL * BOARD; // inner board width / height

export const CANVAS_W = COORD + BOARD_PX + COORD;
export const CANVAS_H = CANVAS_W;

// ── Colours (tuned for dark theme) ────────────────────────────────
const LIGHT = '#c8b07a';
const DARK  = '#a68a5b';

const BG  = '#0a0a1a';
const SEL = '#ffd700';
const LEGAL_EMPTY = 'rgba(92,184,92,0.55)';
const LEGAL_CAPT  = 'rgba(217,83,79,0.55)';
const LAST_MOVE   = 'rgba(74,144,217,0.45)';
const HOVER       = 'rgba(255,215,0,0.20)';

const BLACK_PIECE  = '#1a0505';
const WHITE_PIECE  = '#f5f5ff';
const ROYAL_BLACK  = '#c8a030';
const ROYAL_WHITE  = '#3050c8';
const UNDERLINE    = '#ccccff';
const COORD_COLOR  = '#666';

// ── Drawing primitives ────────────────────────────────────────────

function cellX(col: number): number { return COORD + col * CELL; }
function cellY(row: number): number { return COORD + row * CELL; }

// ── Public render function ────────────────────────────────────────

export interface RenderOptions {
  state:       BoardState;
  selectedSq:  [number, number] | null;
  legalMoves:  MoveData[];
  hoverSq:     [number, number] | null;
  lastFrom:    [number, number] | null;
  lastTo:      [number, number] | null;
}

export function render(ctx: CanvasRenderingContext2D, opts: RenderOptions): void {
  const { state, selectedSq, legalMoves, hoverSq, lastFrom, lastTo } = opts;

  const W = ctx.canvas.width;
  const H = ctx.canvas.height;

  // ── Background ──────────────────────────────────────────────
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  // ── Guard: no board state ──────────────────────────────────
  if (!state.board) {
    ctx.fillStyle = '#999';
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Loading...', W / 2, H / 2);
    return;
  }

  // ── Board cells ────────────────────────────────────────────
  for (let r = 0; r < BOARD; r++) {
    for (let c = 0; c < BOARD; c++) {
      const x = cellX(c);
      const y = cellY(r);
      ctx.fillStyle = (r + c) % 2 === 0 ? LIGHT : DARK;
      ctx.fillRect(x, y, CELL, CELL);

      // Last-move highlight
      if (lastFrom?.[0] === r && lastFrom?.[1] === c) {
        ctx.fillStyle = LAST_MOVE;
        ctx.fillRect(x, y, CELL, CELL);
      }
      if (lastTo?.[0] === r && lastTo?.[1] === c) {
        ctx.fillStyle = LAST_MOVE;
        ctx.fillRect(x, y, CELL, CELL);
      }
    }
  }

  // ── Legal-move indicators ──────────────────────────────────
  for (const m of legalMoves) {
    const [tr, tc] = m.to;
    const x = cellX(tc);
    const y = cellY(tr);
    const isCapture = state.board[tr][tc] !== null;

    if (isCapture) {
      ctx.fillStyle = LEGAL_CAPT;
      ctx.fillRect(x, y, CELL, CELL);
    } else {
      ctx.fillStyle = LEGAL_EMPTY;
      ctx.beginPath();
      ctx.arc(x + CELL / 2, y + CELL / 2, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── Pieces ─────────────────────────────────────────────────
  for (let r = 0; r < BOARD; r++) {
    for (let c = 0; c < BOARD; c++) {
      const piece: CellData | null = state.board[r][c];
      if (!piece) continue;

      const x = cellX(c);
      const y = cellY(r);

      // Selection highlight
      if (selectedSq?.[0] === r && selectedSq?.[1] === c) {
        ctx.fillStyle = SEL;
        ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);
      }

      // Hover
      if (hoverSq?.[0] === r && hoverSq?.[1] === c && selectedSq === null) {
        ctx.fillStyle = HOVER;
        ctx.fillRect(x, y, CELL, CELL);
      }

      // Royal glow
      if (piece.is_royal) {
        ctx.fillStyle = piece.color === 0 ? ROYAL_BLACK : ROYAL_WHITE;
        ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);
      }

      // Kanji symbol (from the abbreviation returned by the API)
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = piece.color === 0 ? BLACK_PIECE : WHITE_PIECE;
      ctx.fillText(pieceKanji(piece.piece), x + CELL / 2, y + CELL / 2);

      // White underline
      if (piece.color === 1) {
        ctx.strokeStyle = UNDERLINE;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + 5, y + CELL - 4);
        ctx.lineTo(x + CELL - 5, y + CELL - 4);
        ctx.stroke();
      }
    }
  }

  // ── Coordinates ────────────────────────────────────────────
  ctx.fillStyle = COORD_COLOR;
  ctx.font = '9px monospace';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < BOARD; i++) {
    const label = String(BOARD - i);
    const x = cellX(i) + CELL / 2;
    const y = cellY(i) + CELL / 2;

    // Row left
    ctx.textAlign = 'right';
    ctx.fillText(label, COORD - 4, y);
    // Row right
    ctx.textAlign = 'left';
    ctx.fillText(label, COORD + BOARD_PX + 4, y);

    // Col top
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(label, x, COORD - 4);
    // Col bottom
    ctx.textBaseline = 'top';
    ctx.fillText(label, x, COORD + BOARD_PX + 4);
  }
}