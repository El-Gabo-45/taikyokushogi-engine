/**
 * HTTP client for the Taikyoku Shogi engine API.
 *
 * All methods return fresh data from the Rust backend.
 */

import type { BoardState, MoveResponse } from '../types/board';

const BASE = '';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path}: ${res.status}`);
  return res.json();
}

/** Fetch the full board state */
export function fetchState(): Promise<BoardState> {
  return get<BoardState>('/api/state');
}

/** Fetch legal moves from a specific square */
export function fetchMoves(r: number, c: number) {
  return get<{ moves: import('../types/board').MoveData[]; from: [number, number] }>(
    `/api/moves?r=${r}&c=${c}`,
  );
}

/** Apply a human move */
export function sendMove(
  from: [number, number],
  to: [number, number],
  promotion = false,
): Promise<MoveResponse> {
  return post<MoveResponse>('/api/move', { from, to, promotion });
}

/** Ask the AI to generate and apply a move */
export function sendAiMove(depth = 0, timeLimit = 30_000): Promise<MoveResponse> {
  return post<MoveResponse>('/api/ai-move', { depth, time_limit: timeLimit });
}

/** Start a new game */
export function newGame(): Promise<{ ok: boolean }> {
  return post('/api/new-game');
}

/** Undo the last move */
export function undo(): Promise<{ ok: boolean; error?: string }> {
  return post('/api/undo');
}