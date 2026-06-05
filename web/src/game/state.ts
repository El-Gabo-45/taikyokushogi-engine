/**
 * Game state machine.
 *
 * Owns the authoritative board state and exposes actions that synchronise
 * with the backend and update the UI subscription.
 */

import type { BoardState, Square, MoveData } from '../types/board';
import * as api from '../api/client';

export type Mode = 'human_vs_random' | 'human_vs_ai' | 'random_vs_random' | 'ai_vs_ai';

export type Listener = () => void;

/** Runtime settings that affect AI auto-play behaviour. */
export interface Settings {
  mode: Mode;
  humanColor: 0 | 1;
  aiDepth: number;
}

export class GameState {
  board: BoardState | null = null;
  selectedSq: Square | null = null;
  legalMoves: MoveData[] = [];
  lastMoveFrom: Square | null = null;
  lastMoveTo: Square | null = null;
  settings: Settings = { mode: 'human_vs_random', humanColor: 0, aiDepth: 2 };

  private readonly listeners = new Set<Listener>();
  private aiPending = false;

  // ----------------------------------------------------------------
  // Subscriptions
  // ----------------------------------------------------------------

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify() {
    for (const fn of this.listeners) fn();
  }

  // ----------------------------------------------------------------
  // API-driven actions
  // ----------------------------------------------------------------

  /** Initialise or reload the board from the backend. */
  async init(): Promise<void> {
    this.board = await api.fetchState();
    this.selectedSq = null;
    this.legalMoves = [];
    this.lastMoveFrom = null;
    this.lastMoveTo = null;
    this.notify();
    await this.tryAutoPlay();
  }

  /** Start a new game. */
  async reset(): Promise<void> {
    await api.newGame();
    await this.init();
  }

  /** Undo the last move. */
  async undo(): Promise<void> {
    if (!this.board || this.board.game_over) return;
    const res = await api.undo();
    if (res.ok) await this.init();
  }

  /** Select a square and fetch its legal moves. */
  async select(r: number, c: number): Promise<void> {
    if (!this.board) return;
    this.selectedSq = [r, c];
    const resp = await api.fetchMoves(r, c);
    this.legalMoves = resp.moves;
    this.notify();
  }

  clearSelection(): void {
    this.selectedSq = null;
    this.legalMoves = [];
    this.notify();
  }

  /** Execute a human move. */
  async play(fr: number, fc: number, tr: number, tc: number, promotion: boolean): Promise<void> {
    if (!this.board || this.board.game_over) return;
    this.selectedSq = null;
    this.legalMoves = [];
    const resp = await api.sendMove([fr, fc], [tr, tc], promotion);
    if (resp.ok && resp.board) {
      this.board = resp.board;
      this.lastMoveFrom = resp.last_move?.from ?? null;
      this.lastMoveTo = resp.last_move?.to ?? null;
      this.notify();
      await this.tryAutoPlay();
    }
  }

  /** Ask the AI to move. */
  async aiMove(depth?: number, timeLimit?: number): Promise<void> {
    if (!this.board || this.board.game_over || this.aiPending) return;
    this.aiPending = true;
    try {
      const d = depth ?? this.settings.aiDepth;
      const resp = await api.sendAiMove(d, timeLimit ?? 30_000);
      if (resp.ok && resp.board) {
        this.board = resp.board;
        this.lastMoveFrom = resp.last_move?.from ?? null;
        this.lastMoveTo = resp.last_move?.to ?? null;
        this.notify();
        await this.tryAutoPlay();
      }
    } finally {
      this.aiPending = false;
    }
  }

  // ----------------------------------------------------------------
  // Auto-play
  // ----------------------------------------------------------------

  private shouldAutoPlay(): boolean {
    if (!this.board || this.board.game_over || this.aiPending) return false;
    const { mode, humanColor } = this.settings;
    const side = this.board.side_to_move;
    if (mode === 'ai_vs_ai') return true;
    if (mode === 'random_vs_random') return true;
    if (mode === 'human_vs_random' && side !== humanColor) return true;
    if (mode === 'human_vs_ai' && side !== humanColor) return true;
    return false;
  }

  private async tryAutoPlay(): Promise<void> {
    if (!this.shouldAutoPlay()) return;
    const delay = this.settings.mode === 'ai_vs_ai' || this.settings.mode === 'random_vs_random'
      ? 200
      : 100;
    await new Promise((r) => setTimeout(r, delay));
    await this.aiMove();
  }
}