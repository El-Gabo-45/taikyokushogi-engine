/**
 * Updates the sidebar DOM panels from board state.
 */

import type { BoardState } from '../types/board';

const $ = document.getElementById.bind(document);

export function updatePanels(state: BoardState): void {
  // ── Info section ──────────────────────────────────────────
  $('info-turn')!.textContent = state.side_to_move === 0 ? 'Black' : 'White';
  $('info-move')!.textContent = String(state.move_number);
  $('info-black')!.textContent = String(state.black_pieces);
  $('info-white')!.textContent = String(state.white_pieces);

  const scoreHistory = state.score_history ?? [];
  const score = scoreHistory.length > 0 ? scoreHistory[scoreHistory.length - 1] : 0;
  $('info-score')!.textContent = score > 0 ? `+${score}` : String(score);

  $('status-msg')!.textContent = state.game_result
    ? state.game_result.replace('_', ' ').toUpperCase()
    : state.game_over
      ? 'GAME OVER'
      : '-';

  $('status')!.textContent =
    `${state.black_pieces} vs ${state.white_pieces} pieces | Move ${state.move_number}`;

  // ── Move log ──────────────────────────────────────────────
  const logEl = $('move-log')!;
  logEl.innerHTML = state.move_log.map(e => `<div class="entry">${e}</div>`).join('');
  logEl.scrollTop = logEl.scrollHeight;
}