# NNUE Training Pipeline

This directory trains a neural network evaluator for the Taikyoku Shogi
engine (`src/eval/nnue.rs`) using PyTorch, then exports the trained
weights to a binary format the Rust engine can load.

## Status: what works, what doesn't yet

**Works, tested end-to-end:**
- Parsing `.bin` training data from `selfplay` (`dataset.py`)
- Converting positions to HalfKP-style features (`features.py`)
- The PyTorch model architecture, forward + backward pass (`model.py`)
- The training loop, including checkpointing and resume (`train.py`)
- Exporting a checkpoint to a `.nnue` file the Rust engine loads
  (`export.py`)
- Loading that `.nnue` file in Rust and running a forward pass
  (`NnueEvaluator::load_from_file` + the `load_and_evaluate_exported_nnue`
  test in `src/eval/nnue.rs`)
- Toggling between the hand-crafted evaluator and NNUE at runtime
  (`taikyokushogi::set_use_nnue(true/false)`)

**Known limitation -- NNUE is not yet wired for incremental accumulator
updates:**
`nnue_evaluate_from_scratch` (used when `set_use_nnue(true)` is active)
rebuilds the Accumulator from scratch on every call -- O(pieces *
FT_NEURONS), roughly 400,000 operations per call, since Taikyoku Shogi has
up to ~400 pieces per side. Measured: **~1.8 seconds for a single
`evaluate()` call**, vs ~58 microseconds for the hand-crafted evaluator --
about 31,000x slower. This is fine for testing that the pipeline works
end-to-end, but is NOT usable inside a real search (which calls evaluate()
thousands of times per move). Real NNUE engines avoid this by updating the
accumulator incrementally in `apply_move`/`undo_move` (add/remove only the
features that changed, rather than recomputing all of them) -- that's real
follow-up work, not done in this session, and worth doing before trying to
actually play games with the trained network at any reasonable depth.

## Pipeline overview

```
selfplay (Rust binary)          ->  training_data/samples_*.bin
        |
        v
dataset.py (parse .bin)         ->  Sample objects
        |
        v
features.py (board -> features) ->  HalfKP feature indices per perspective
        |
        v
model.py (PyTorch nn.Module)    ->  NnueModel
        |
        v
train.py (training loop)        ->  checkpoints/*.pt
        |
        v
export.py (quantize + write)    ->  trained.nnue
        |
        v
src/eval/nnue.rs::load_from_file -> usable in the Rust engine
```

## Step-by-step

### 1. Generate training data

From the repo root:

```bash
cargo build --release --bin selfplay
./target/release/selfplay <num_games> <depth> <time_limit_ms>
# e.g.: ./target/release/selfplay 1000 3 200
```

This writes `training_data/samples_*.bin`. **If you have .bin files from
before this session's fixes to `selfplay.rs` (the `#[repr(C)]` fix and the
piece-encoding bit-overlap fix), delete them and regenerate** -- they have
corrupted piece_type data for ~45 of the 301 piece types (any with
piece_type >= 256).

### 2. Generate piece metadata (only needed once, or after pieces.rs changes)

```bash
cargo run --release --example export_piece_metadata > training/piece_metadata.json
```

This tells `features.py` which piece_type IDs count as "royal" (needed to
determine king_sq for HalfKP feature indexing). Already generated and
committed as of this session (`{"num_piece_types": 301,
"royal_piece_types": [29, 88]}`), but re-run this if the piece set in
`pieces.rs` ever changes.

### 3. Set up PyTorch with your RX 470

Your card doesn't have ROCm support, so you're using a Docker-based
workaround project. The general shape (adjust to whatever specific image
your ROCm-compat project provides):

```bash
docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v $(pwd):/workspace \
    -w /workspace/training \
    <your-rocm-compat-image> \
    bash
```

Once inside the container, install the Python dependencies (if the image
doesn't already have them):

```bash
pip install torch numpy
```

Then verify PyTorch actually sees your GPU before starting a real
training run:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU found')"
```

If this prints `False`, something's off with the Docker device passthrough
or the ROCm-compat layer -- fix that before training (training will still
run on CPU, just extremely slowly given the model's ~100M+ parameters).

### 4. Train

```bash
python3 train.py --data-dir ../training_data --epochs 20 --batch-size 4096
```

Useful flags:
- `--batch-size`: 4096 is a reasonable starting point for 8GB VRAM given
  this model's size (~104M params, dominated by the feature transformer --
  see "Memory budget" below). Reduce if you hit out-of-memory errors.
- `--num-workers`: CPU processes for data loading/feature extraction in
  parallel with GPU training. Start around 4; tune based on your CPU core
  count.
- `--resume checkpoints/latest.pt`: continue an interrupted run.
- `--device cuda` / `--device cpu`: force a device if auto-detection picks
  the wrong one.

Checkpoints are saved to `checkpoints/` every `--checkpoint-every-steps`
steps and at the end of every epoch. `checkpoints/latest.pt` always points
at the most recent one, for convenient `--resume`.

### 5. Export to a `.nnue` file

```bash
python3 export.py --checkpoint checkpoints/latest.pt --output ../trained.nnue
```

### 6. Use the trained network in the engine

```bash
export TAIKYOKU_NNUE_PATH=/absolute/path/to/trained.nnue
```

Then, from Rust code (or the Python bindings, or `debug-cli`), call
`taikyokushogi::set_use_nnue(true)` to switch the evaluator. Without
`TAIKYOKU_NNUE_PATH` set, the engine falls back to randomly-initialized
(untrained) NNUE weights with a loud warning on stderr -- it will NOT
silently use the hand-crafted evaluator instead, so don't forget to set
the env var.

### Verifying the exported `.nnue` file loads correctly

There's an internal Rust test that loads a `.nnue` file and runs a real
forward pass on the initial position, to catch any architecture mismatch
between the PyTorch and Rust sides (wrong dimensions, wrong weight
transpose, etc. -- these would show up as either a load error or a panic,
not a silent wrong answer, precisely because of the shape assertions in
both `export.py` and `load_from_file`):

```bash
TAIKYOKU_TEST_NNUE_PATH=/path/to/trained.nnue cargo test --release --lib eval::nnue::tests::load_and_evaluate_exported_nnue -- --nocapture
```

## Architecture notes / design decisions made this session

### Piece type count: 301, not 209

`nnue.rs` previously hardcoded `NUM_PIECE_TYPES = 209`, left over from an
earlier design doc, before the full Taikyoku Shogi piece set was
implemented. The actual count (verified via
`taikyokushogi::num_piece_types()` at runtime) is **301**. Fixed in
`nnue.rs`; `model.py` and `features.py` match.

### Feature hashing: FT_BUCKETS = 200,000

The raw HalfKP feature space (`king_sq * piece_sq * piece_type`, doubled
for both perspectives) is ~1.01 **billion** possible indices --
`FeatureTransformer::new_random()` originally declared `FT_FEATURES =
541,632` (or, after the piece-count fix, `780,192`), wildly mismatched
with what `feature_index()` could actually produce. This was caught
immediately when mirroring the architecture in PyTorch: `nn.Embedding`
raised an `IndexError` the moment a real index was looked up.

There was an unused `FT_BUCKETS = 8_000_000` constant already in the code,
suggesting feature hashing (`index % FT_BUCKETS`) was the intended fix but
never implemented. It's implemented now (in both `nnue.rs` and
`features.py` / `model.py`, using the identical formula) -- but
`8,000,000` buckets at `FT_NEURONS=512` needs **~65GB of VRAM to train**
(weights + gradients + Adam's two moment buffers, all fp32), completely
infeasible on an 8GB card. **Reduced to `FT_BUCKETS = 200,000`** (~1.6GB
for the feature transformer during training), leaving comfortable headroom
for batch size and the rest of the model on an 8GB GPU. If you have more
VRAM available later and want fewer hash collisions (currently ~5,000 raw
indices sharing each bucket on average), this is the constant to raise --
just change it in `nnue.rs`, `model.py`, and `features.py` together, and
retrain from scratch (existing checkpoints/.nnue files won't be
compatible with a different bucket count).

### Dimension-chaining bug in the original NnueEvaluator

The original `NnueEvaluator::new_random()` declared layer dimensions that
didn't chain correctly (`l2` expected 1024 inputs but `l1` only produced
256, etc.). This never crashed -- `FactorizedLayer::forward` silently only
reads as many weight-matrix rows as the actual input length, so a
too-small input just left most of each layer's weights unused rather than
panicking -- but it meant 3 of 4 downstream layers were wasting the vast
majority of their parameters. Fixed to chain correctly: 1024 -> 256 -> 128
-> 64 -> 1.

### `TrainingSample` needed `#[repr(C)]`

Without it, Rust is free to reorder struct fields for padding efficiency,
which is exactly what it was doing (verified empirically: field order
differed from declaration order). Since `TrainingSample` is serialized to
disk via a raw byte dump, this made the `.bin` file format
compiler-version-dependent and impossible for an external tool (this
Python pipeline) to parse reliably. Fixed by adding `#[repr(C)]`, which
fixes the layout to declaration order. See the docstring at the top of
`dataset.py` for the exact byte layout this produces.

### Board cell encoding bit-overlap bug

`selfplay.rs` encoded `board[sq] = piece_type | (color << 8)`. With 301
piece types (needing 9 bits, since 301 > 255), any piece_type >= 256 had
its high bit silently clobbered by the color bit. Fixed to
`(piece_type & 0x1FF) | (color << 9)`. **Any `.bin` files generated before
this fix have corrupted data for the affected piece types and should be
regenerated**, not just re-parsed with the new decoder -- the original
bit was actually lost, not just misinterpreted.

### `save_to_file`/`load_from_file` were non-functional

The original `save_to_file` cast `Vec<Vec<i16>>` (a vector of separately
heap-allocated inner vectors) directly to bytes via `slice::from_raw_parts`
on the outer vector's pointer -- this serializes the inner `Vec`s'
internal pointer/capacity/length metadata (24 bytes of garbage per row),
not the actual i16 weight values, and only for the feature transformer
(l1/l2/l3/output weren't saved at all). `load_from_file` just returned
`Err("not yet implemented")`. Both are rewritten in this session with a
flat, fully-documented binary format (see the "Serialization" section at
the top of the relevant part of `nnue.rs`) that `export.py` writes and
`load_from_file` reads -- verified working end-to-end via the Rust test
described above.

### King square reconstruction is an approximation

`board.king_square(color)` in the live engine returns
`royal_list[color][0]` -- the first royal piece added, by game history.
That order isn't recoverable from a single board snapshot (all
`TrainingSample` stores). `features.py`'s `find_king_squares` instead uses
the royal piece at the lowest square index as a deterministic
approximation. This only affects positions where a color has 2+ royal
pieces alive simultaneously (a subset of positions), and only changes
which of several valid king-relative feature encodings gets used -- it's
a small, bounded source of label noise, not a correctness bug. See the
docstring in `features.py` for more detail, and for the real fix if it
turns out to matter (have `selfplay.rs` record king_sq directly into
`TrainingSample` instead of reconstructing it in Python).

## Memory budget reference (FT_BUCKETS = 200,000)

| Component | Params | Size (fp32, training) |
|---|---|---|
| Feature transformer | 200,000 × 512 ≈ 102.4M | ~1.6 GB (weights only) |
| Adam optimizer state (2×) | | ~3.3 GB |
| L1/L2/L3/output | ~1.5M | ~24 MB |
| **Total (weights+grad+Adam, FT only)** | | **~6.5 GB** |

Leaves headroom on an 8GB card for activations and batch size, but not a
huge amount -- if you hit OOM, first try reducing `--batch-size`, then
consider whether `FT_BUCKETS` needs to come down further.
