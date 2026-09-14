# Taikyoku Shogi Engine

A complete engine for **Taikyoku Shogi** (大局将棋, "ultimate chess"), the largest known variant of shogi.

- **36 x 36** board (1,296 squares)
- **804 pieces** (402 per side)
- **209 piece types** with distinct movement patterns
- **92 additional promoted forms**

## Rules Overview

### Objective

Capture all of your opponent's **royal pieces** (King and Crown Prince). The game ends immediately when a player's last royal piece is captured. Unlike standard shogi, there is no check or checkmate — you may move your King into danger.

### Turns

Black (先手) moves first. Players alternate moving exactly one piece per turn. Passing is not allowed. Captured pieces are permanently removed (no drops).

### Movement

Pieces move orthogonally or diagonally. Most pieces have unique movement patterns combining:

| Type | Description |
|------|-------------|
| **Step** | Move exactly 1 square |
| **Slide** | Move any number of empty squares in a line |
| **Limited** | Move up to N squares in a line (2-7) |
| **Jump** | Leap to a square, bypassing intervening pieces |
| **Hook** | Slide along a line, optionally turn 90 degrees once, then continue sliding |
| **Area** | Take multiple steps per turn (Lion-type moves) |
| **Range capture** | Fly over pieces along a line, capturing all of lower rank |

See [PIECES.md](PIECES.md) for the complete movement reference for every piece.

### Promotion

The **promotion zone** is the 11 rows on the opponent's side of the board.

- A piece **may promote** when it moves into the promotion zone from outside, or when it captures an opponent's piece while inside the zone.
- A piece that entered the zone without promoting may only promote later by capturing inside the zone, or by leaving and re-entering.
- Promotion is **optional**, except for forward-only pieces (Pawn, Stone General, Iron General, Dog, Wood General, Incense Chariot, Ox Chariot, Fierce Tiger) which **must promote** upon reaching the farthest rank.
- Promotion is permanent — a piece cannot revert to its unpromoted form.

### Range Capture Ranking

Range-capturing pieces (Great General, Vice General, etc.) can fly over and capture all pieces along a line, but only pieces of **lower rank**:

1. King, Crown Prince — cannot be range-captured
2. Great General
3. Vice General
4. Flying General, Angle General, Fierce Dragon, Flying Crocodile

### Draw by No Progress (500-Move Rule)

The game is automatically drawn if **500 consecutive full moves** (1,000 plies) pass with neither player making a capture nor a promotion. This prevents games from continuing indefinitely when neither side can make progress. The counter resets whenever a piece is captured or a promotion occurs.

### Initial Setup

Each side's 402 pieces occupy 12 ranks. Black occupies the bottom of the board (rows 25-36); White mirrors from the top (rows 1-12). The King sits at the center of the back rank, flanked by the Crown Prince.

## Getting Started

### Requirements

- A recent stable **Rust toolchain** (the crate targets Rust edition 2021).
- **Node.js + npm** only if you want to rebuild the `web/` frontend
  (TypeScript + Vite; a prebuilt `dist/` is committed).
- **Python 3.9+** only for the NNUE training pipeline in `training/` (see
  [training/README.md](training/README.md)).

### Build

```bash
git clone https://github.com/jh85/taikyokushogi.git
cd taikyokushogi

# Build the engine, server, CLI tools and examples in release mode
cargo build --release
```

### Web GUI

```bash
# Rebuild the frontend (only needed after editing web/src)
cd web && npm install && npm run build && cd ..

# Start the HTTP server: REST API + static web frontend
cargo run --release --bin taikyokushogi-server
# Then open http://localhost:8000 in your browser
```

### Web GUI Features

- **Game modes**: Human vs Random, Human vs AI, Random vs Random, AI vs AI
- Play as **Black or White**; pick the AI strength (Random, D1, D2, D3)
- Click a piece to see its legal moves highlighted on the board
  (green = move, red = capture)
- Hover and last-move highlights, board coordinates, kanji piece glyphs
- Sidebar shows the side to move, move number, piece counts, material
  score, status and the full move log
- **New Game** / **Undo** buttons and an **Auto** play toggle

### REST API

The server exposes a small JSON REST API — `/api/state`, `/api/moves`,
`/api/piece-info/{abbrev}`, `/api/new-game`, `/api/move`, `/api/ai-move`,
`/api/undo` — consumed by the frontend in `web/src/api/client.ts`.
See `src/main.rs` for the route handlers.

### Command-Line Tools

| Binary | Purpose |
|---|---|
| `taikyokushogi-server` | HTTP server + web GUI (REST API in `src/main.rs`) |
| `debug-cli` | CLI search harness: best move, score, nodes, time |
| `selfplay` | Self-play training-data generator for the NNUE pipeline |

```bash
# Search the initial position at depth 5 with a 10 s time limit
cargo run --release --bin debug-cli 5 10000

# Generate 100 self-play games (depth 3, 4 workers, 200 ms per move)
cargo run --release --bin selfplay 100 3 0 4 200
# -> writes training_data/samples_*.bin and training_data/games.db
```

### Examples and Benchmarks

All examples run with `cargo run --release --example <name>`:

| Example | Purpose |
|---|---|
| `bench_fixed` | Deterministic fixed-depth search benchmark (depths as args) |
| `bench_nps` | NPS + component timing (movegen, eval, apply/undo) + bottleneck analysis |
| `check_movecounts` | Legal move count from the initial position |
| `stress_undo` | Apply/undo stress test (checks the board state round-trips) |
| `bench`, `bench_deep`, `bench_depth1`, `bench_midgame`, `bench_thorough`, `bench_time` | Additional benchmarks |
| `export_piece_metadata` | Writes `training/piece_metadata.json` for the NNUE pipeline |
| `toggle_nnue` | Runtime NNUE on/off smoke test |
| `play` | Minimal command-line game loop |
| `check_db` | Self-play SQLite database sanity checks |

```bash
cargo run --release --example bench_fixed 4 5 6   # fixed-depth search at depths 4, 5, 6
cargo run --release --example check_movecounts    # -> 512 legal moves
cargo run --release --example bench_nps           # full benchmark report
```

## Project Structure

```
taikyokushogi-engine/
  src/
    lib.rs              # Public crate API (Board, Move, PieceInfo, search, TSFEN, ...)
    main.rs             # HTTP server + REST API + static web GUI   (bin: taikyokushogi-server)
    debug_cli.rs        # CLI search harness                        (bin: debug-cli)
    selfplay_main.rs    # Self-play entry point                     (bin: selfplay)
    selfplay.rs         # Self-play logic + TrainingSample binary format
    types.rs            # Core types, constants, ray tables
    pieces.rs           # 301 piece types, Betza notation parser
    board.rs            # Board representation, apply/undo, game rules
    movegen.rs          # Legal move generation
    attack.rs           # Attack / ray bitboards
    bitboard.rs         # Bitboard helpers
    search.rs           # PVS + TT + move ordering + pruning + quiescence + Lazy SMP
    tsfen.rs            # TSFEN position notation (encode / parse)
    eval/
      mod.rs            # Evaluator dispatcher (hand-crafted <-> NNUE)
      families.rs       # Piece-family material values
      psqt.rs           # Piece-square tables
      zones.rs          # Threat zones and king-safety heuristics
      nnue.rs           # NNUE (HalfKP-style) neural evaluator + .nnue loader
    debugging/          # Search/eval introspection utilities
  examples/             # Benchmarks, smoke tests and tooling (see above)
  web/                  # TypeScript + Vite frontend (served by the Rust server)
    src/
      main.ts           #   Entry point
      game/state.ts     #   Game state machine + modes
      api/client.ts     #   REST client
      ui/               #   Canvas renderer, input controller, sidebar panels
      data/kanji.ts     #   Piece kanji glyphs
  training/             # NNUE PyTorch pipeline — see training/README.md
  training_data/        # Generated self-play .bin samples + games.db
  PIECES.md             # Complete piece movement reference
  Cargo.toml            # Rust project config (bins, features, examples)
```

## Performance

Measured on the development machine with the release-mode benchmarks
(`cargo run --release --example bench_nps` and `--example bench_fixed`).

| Metric | Value |
|---|---|
| Legal moves from the initial position | 512 |
| Move generation (`legal_moves`) | ~0.73 ms |
| Static evaluation (`evaluate`) | ~51 µs |
| `apply` + `undo` round trip | ~49 µs |
| Perft(2) | 260,975 nodes @ ~1.09 M nodes/s |
| Fixed-depth search — depth 4 | ~0.57 s (~175 k nodes/s) |
| Fixed-depth search — depth 5 | ~1.4 s (~195 k nodes/s) |
| Fixed-depth search — depth 6 | ~2.4 s (~205 k nodes/s) |

The search (`src/search.rs`) uses iterative deepening with aspiration
windows, principal-variation search (PVS), a transposition table with
lock-free concurrent access, killer / counter / history move ordering,
null-move pruning, razoring / reverse-futility / futility / ProbCut
pruning, staged move generation and quiescence search. At depth ≥ 4,
Lazy SMP parallelism spawns up to 3 helper threads.

## Neural Evaluation (NNUE)

An experimental neural-network evaluator (`src/eval/nnue.rs`,
HalfKP-style features) can be enabled at runtime with
`taikyokushogi::set_use_nnue(true)`; the weights are loaded from the path
in the `TAIKYOKU_NNUE_PATH` environment variable. Run the `toggle_nnue`
example for a smoke test.

The accumulator is now maintained **incrementally** by `Board::apply_move`
(O(FT_NEURONS) deltas per moved/captured piece, full refresh only when a
royal anchor changes) and restored in O(1) on undo, so per-evaluate cost no
longer includes the O(pieces × FT_NEURONS) rebuild. The remaining per-call
cost is the network forward pass itself (~3 ms with the current dense
layers — fine for validation and training pipelines; faster inference needs
the flattened-weight redesign planned for the next architecture).

The full PyTorch training pipeline — generating data with `selfplay`,
feature extraction, training, and exporting `.nnue` files — lives in
[training/](training/README.md).

## Using as a Rust Crate

Add to your `Cargo.toml`:

```toml
[dependencies]
taikyokushogi = "0.1"
```

```rust
use taikyokushogi::{Board, Color};

let mut board = Board::initial();
let moves = board.legal_moves();
println!("{} legal moves", moves.len()); // 512

board.apply(&moves[0]);
println!("Score: {}", board.material_score());
board.undo();

// Search: depth + time limit in ms (0 = no limit)
let result = board.search(2, 5000);
if let Some(mv) = result.best_move {
    println!("Best: {}, score: {}", mv, result.score);
}

// TSFEN notation (the "FEN" of Taikyoku Shogi)
let fen = board.to_tsfen();
let restored = Board::from_tsfen(fen).unwrap();

// Piece metadata
let info = taikyokushogi::piece_info("LN").unwrap();
println!("{}: value={} area={} igui={}",
         info.name, info.value, info.area_steps, info.has_igui);

// Switch between the hand-crafted and the NNUE evaluator
// (requires a trained .nnue file — see training/README.md)
// taikyokushogi::set_use_nnue(true);
```

## Cargo Features

- `cpu` (default) — pure-CPU engine
- `gpu-cuda`, `gpu-metal`, `gpu-wgpu`, `gpu-vulkan` — GPU evaluation via `burn`
- `nnue` — enables the NNUE evaluator (`burn` + `ndarray`)

## Credits and copyright

This is a *fork* of **[taikyokushogi](https://github.com/jh85/taikyokushogi)**, a complete engine for Taikyoku Shogi originally developed by **[jh85](https://github.com/jh85)**.

The original base code is released under the [MIT License](LICENSE). We thank the original author for his excellent work on the optimization and logic of the 36×36 board in Rust.

## References

- [Taikyoku shogi — Wikipedia](https://en.wikipedia.org/wiki/Taikyoku_shogi)
