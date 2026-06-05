/**
 * Input controller.
 *
 * Translates mouse events on the canvas into game actions and
 * triggers re-renders when the game state changes.
 */

import type { GameState } from '../game/state';
import { render, CANVAS_W, CANVAS_H, CELL, COORD, BOARD } from './renderer';

export class Controller {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly state: GameState;
  private hoverSq: [number, number] | null = null;
  private rafId = 0;

  constructor(
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    state: GameState,
  ) {
    this.canvas = canvas;
    this.ctx = ctx;
    this.state = state;
    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;

    this.bindEvents();
    state.subscribe(() => this.scheduleDraw());
  }

  // ---------------------------------------------------------------
  // Mouse helpers
  // ---------------------------------------------------------------

  /** Convert a canvas-relative pixel coordinate to (row, col). */
  private pixelToSquare(px: number, py: number): [number, number] | null {
    const c = Math.floor((px - COORD) / CELL);
    const r = Math.floor((py - COORD) / CELL);
    if (r < 0 || r >= BOARD || c < 0 || c >= BOARD) return null;
    return [r, c];
  }

  /** Get mouse position relative to the canvas, accounting for CSS scaling. */
  private mousePos(e: MouseEvent): { px: number; py: number } {
    const rect = this.canvas.getBoundingClientRect();
    const sx = this.canvas.width / rect.width;
    const sy = this.canvas.height / rect.height;
    return {
      px: (e.clientX - rect.left) * sx,
      py: (e.clientY - rect.top) * sy,
    };
  }

  // ---------------------------------------------------------------
  // Event binding
  // ---------------------------------------------------------------

  private bindEvents(): void {
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => this.onMouseLeave());
    this.canvas.addEventListener('click', (e) => this.onClick(e));
  }

  private onMouseMove(e: MouseEvent): void {
    const { px, py } = this.mousePos(e);
    const sq = this.pixelToSquare(px, py);

    if (sq) {
      if (!this.hoverSq || this.hoverSq[0] !== sq[0] || this.hoverSq[1] !== sq[1]) {
        this.hoverSq = sq;
        this.scheduleDraw();
      }
    } else if (this.hoverSq) {
      this.hoverSq = null;
      this.scheduleDraw();
    }
  }

  private onMouseLeave(): void {
    if (this.hoverSq) {
      this.hoverSq = null;
      this.scheduleDraw();
    }
  }

  private onClick(e: MouseEvent): void {
    const { px, py } = this.mousePos(e);
    const sq = this.pixelToSquare(px, py);
    if (!sq) return;

    const [r, c] = sq;
    const board = this.state.board;
    if (!board) return;

    // If a piece is selected and we click a legal target → play
    if (this.state.selectedSq) {
      const target = this.state.legalMoves.find(m => m.to[0] === r && m.to[1] === c);
      if (target && !board.game_over) {
        const [fr, fc] = this.state.selectedSq;
        this.state.play(fr, fc, r, c, target.promotion);
        return;
      }
    }

    // Otherwise try selecting a piece
    const cell = board.board[r][c];
    if (cell && cell.color === board.side_to_move) {
      this.state.select(r, c);
    } else {
      this.state.clearSelection();
    }
  }

  // ---------------------------------------------------------------
  // Rendering loop
  // ---------------------------------------------------------------

  private scheduleDraw(): void {
    if (this.rafId) return;
    this.rafId = requestAnimationFrame(() => {
      this.rafId = 0;
      this.draw();
    });
  }

  private draw(): void {
    const board = this.state.board;
    if (!board) return;

    render(this.ctx, {
      state: board,
      selectedSq: this.state.selectedSq,
      legalMoves: this.state.legalMoves,
      hoverSq: this.hoverSq,
      lastFrom: this.state.lastMoveFrom,
      lastTo: this.state.lastMoveTo,
    });
  }
}