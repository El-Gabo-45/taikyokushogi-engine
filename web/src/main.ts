/**
 * Application entry point.
 *
 * Wires the game state, canvas renderer, input controller and
 * DOM panels together, then boots the first game.
 */

import { GameState } from './game/state';
import { Controller } from './ui/controller';
import { updatePanels } from './ui/panel';

// ── Bootstrap ──────────────────────────────────────────────────────

const canvas = document.getElementById('board-canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const state = new GameState();
// Constructed for side effects: binds events & subscribes to state
new Controller(canvas, ctx, state);

// Update DOM panels whenever the game state changes
state.subscribe(() => {
  if (state.board) updatePanels(state.board);
});

// Wire toolbar buttons (declared as global functions for onclick=)
const sel = (id: string) => document.getElementById(id) as HTMLSelectElement;

(window as any).newGame = () => {
  state.settings.mode = sel('mode-select').value as any;
  state.settings.humanColor = Number(sel('color-select').value) as 0 | 1;
  state.settings.aiDepth = Number(sel('ai-depth').value);
  state.reset();
};

(window as any).undoMove = () => {
  state.undo();
};

let autoOn = false;
(window as any).toggleAuto = () => {
  autoOn = !autoOn;
  const btn = document.getElementById('auto-btn')!;
  btn.textContent = autoOn ? 'Stop' : 'Auto';
  btn.style.background = autoOn ? '#e94560' : '#0f3460';
  if (autoOn) {
    state.settings.mode = sel('mode-select').value as any;
    state.settings.humanColor = Number(sel('color-select').value) as 0 | 1;
    state.settings.aiDepth = Number(sel('ai-depth').value);
    // Kick-start auto loop — the state machine will recurse
    state.aiMove();
  }
};

// ── Go ────────────────────────────────────────────────────────────
// Initialize game with error handling
state.init().catch((err) => {
  const status = document.getElementById('status');
  if (status) status.textContent = `⚠️ Error: ${err.message}. Check if backend is running on port 5000.`;
  console.error('Failed to initialize game:', err);
});
