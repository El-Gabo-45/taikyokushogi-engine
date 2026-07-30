"""
Parser for the .bin training sample files written by src/selfplay.rs.

WHY THIS FILE'S BYTE LAYOUT MATTERS
------------------------------------
`TrainingSample` in selfplay.rs is written to disk via a raw, unsafe memory
dump (`std::slice::from_raw_parts(s as *const TrainingSample as *const u8, ...)`).
That means the exact byte layout depends on how Rust lays the struct out in
memory. As of this training pipeline, `TrainingSample` was updated to use
`#[repr(C)]`, which fixes the field order to the declaration order and pads
only as needed for alignment (measured empirically against rustc 1.91,
release profile):

    offset   size   field
    ------   ----   -----------------
    0        2592   board: [u16; 1296]   (little-endian, one u16 per square)
    2592     1      side_to_move: u8
    2593     1      (padding)
    2594     2      move_from: u16
    2596     2      move_to: u16
    2598     1      move_promo: u8
    2599     1      result: i8
    2600     2      policy_target: u16
    2602     2      (padding)
    2604     4      value_target: f32
    2608     2      move_number: u16
    2610     2      (padding, to align struct size to 4 bytes)
    ------
    2612 bytes total per sample

IMPORTANT: if you generated .bin files with an OLDER version of selfplay.rs
(before `#[repr(C)]` was added), this layout will NOT match -- the compiler
was free to reorder fields, and empirically it did (a different, larger
padding layout was measured: side_to_move at offset 2604 instead of 2592,
etc). If you hit assertion failures in `sanity_check_sample` below, or
garbage-looking boards, delete the old training_data/*.bin files and
regenerate them with the fixed selfplay.rs.

BOARD CELL ENCODING (different from the engine's internal Cell encoding!)
---------------------------------------------------------------------------
Each `board[sq]` value is NOT the same encoding as the engine's internal
`Cell` type (which is `(piece_type << 1) | color`). Instead, per
selfplay.rs's sample-filling code:

    board[sq] = (piece_type & 0x1FF) | (color << 9)   # 0 means empty square

So: piece_type = board[sq] & 0x1FF, color = (board[sq] >> 9) & 1,
and board[sq] == 0 means the square is empty.

NOTE on piece_type bit width: this game has 301 piece types (verified via
`taikyokushogi::num_piece_types()`), which needs 9 bits (up to 511), not 8.
An earlier version of selfplay.rs packed color into bit 8 directly
(`piece_type | (color << 8)`), which silently corrupted piece_type for any
piece_type >= 256 (about 45 of the 301 piece types) by clobbering their
high bit with the color bit. If you have .bin files generated before this
was fixed, regenerate them -- there is no reliable way to recover the
correct piece_type for the affected pieces after the fact.
"""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass

BOARD_SQUARES = 1296
NUM_PIECE_TYPES = 301  # must match taikyokushogi::num_piece_types() exactly
                       # (verified at runtime against the Rust crate; see
                       # training/README.md for how to re-check this if
                       # pieces.rs ever changes)

# Struct format string for Python's `struct` module, matching the repr(C)
# layout above exactly (little-endian, explicit padding via 'x').
#   1296H = board (1296 x u16)
#   B     = side_to_move
#   x     = 1 byte padding
#   HH    = move_from, move_to
#   B     = move_promo
#   b     = result (signed)
#   H     = policy_target
#   xx    = 2 bytes padding
#   f     = value_target
#   H     = move_number
#   xx    = 2 bytes padding (struct size rounds up to align of 4)
SAMPLE_FORMAT = "<1296H B x HH B b H xx f H xx"
SAMPLE_SIZE = struct.calcsize(SAMPLE_FORMAT)


@dataclass
class Sample:
    board: np.ndarray       # shape (1296,), uint16, raw engine encoding
    side_to_move: int
    move_from: int
    move_to: int
    move_promo: int
    result: int
    policy_target: int
    value_target: float
    move_number: int


def sanity_check_sample(s: Sample, strict: bool = True) -> list[str]:
    """Returns a list of problems found (empty list = looks OK).

    This exists because the byte layout of TrainingSample depends on Rust's
    repr, and it's easy to get subtly wrong (wrong padding, wrong field
    order from an old file, etc). Call this on at least the first few
    samples of any new dataset before trusting it.
    """
    problems = []
    if s.side_to_move not in (0, 1):
        problems.append(f"side_to_move={s.side_to_move} not in {{0,1}}")
    if not (0 <= s.move_from < BOARD_SQUARES):
        problems.append(f"move_from={s.move_from} out of range")
    if not (0 <= s.move_to < BOARD_SQUARES):
        problems.append(f"move_to={s.move_to} out of range")
    if s.move_promo not in (0, 1):
        problems.append(f"move_promo={s.move_promo} not in {{0,1}}")
    if s.result not in (-1, 0, 1):
        problems.append(f"result={s.result} not in {{-1,0,1}}")
    if not (-1.0 <= s.value_target <= 1.0):
        problems.append(f"value_target={s.value_target} out of [-1,1]")
    occupied = s.board[s.board != 0]
    if len(occupied) > 0:
        piece_types = occupied & 0x1FF
        colors = (occupied >> 9) & 1
        if piece_types.max() > NUM_PIECE_TYPES:
            problems.append(
                f"max piece_type in board={piece_types.max()} exceeds "
                f"NUM_PIECE_TYPES={NUM_PIECE_TYPES} -- wrong layout, stale "
                f"NUM_PIECE_TYPES constant, or a pre-fix .bin file with the "
                f"old (color << 8) encoding that corrupts piece_type >= 256"
            )
        if not np.all((colors == 0) | (colors == 1)):
            problems.append("board contains color values other than 0/1 -- likely wrong byte layout")
    if strict and problems:
        raise ValueError(f"Sample failed sanity check: {problems}")
    return problems


def iter_samples_from_file(path: Path):
    """Yields Sample objects from a single .bin file, one at a time."""
    data = path.read_bytes()
    if len(data) % SAMPLE_SIZE != 0:
        raise ValueError(
            f"{path}: file size {len(data)} is not a multiple of "
            f"SAMPLE_SIZE={SAMPLE_SIZE} -- wrong struct layout assumed, or "
            f"truncated/corrupt file"
        )
    n = len(data) // SAMPLE_SIZE
    for i in range(n):
        chunk = data[i * SAMPLE_SIZE:(i + 1) * SAMPLE_SIZE]
        unpacked = struct.unpack(SAMPLE_FORMAT, chunk)
        board = np.array(unpacked[0:BOARD_SQUARES], dtype=np.uint16)
        rest = unpacked[BOARD_SQUARES:]
        (side_to_move, move_from, move_to, move_promo, result,
         policy_target, value_target, move_number) = rest
        yield Sample(
            board=board,
            side_to_move=side_to_move,
            move_from=move_from,
            move_to=move_to,
            move_promo=move_promo,
            result=result,
            policy_target=policy_target,
            value_target=value_target,
            move_number=move_number,
        )


class TrainingDataset:
    """A simple in-memory-index dataset: indexes (file, offset) pairs for
    every sample across all samples_*.bin files without loading all sample
    contents into RAM up front, then reads+unpacks a single sample's bytes
    on demand in __getitem__. This keeps RAM usage bounded even for
    datasets with many millions of samples (selfplay.rs can generate a lot
    of these), at the cost of a small amount of disk I/O per access --
    acceptable since training reads are already randomized/shuffled by the
    DataLoader, which does many small reads regardless of dataset
    implementation.

    Used by train.py via torch.utils.data.DataLoader(dataset, ...,
    collate_fn=collate_samples).
    """

    def __init__(self, training_data_dir: Path, validate_first_n: int = 50):
        self.training_data_dir = Path(training_data_dir)
        files = sorted(self.training_data_dir.glob("samples_*.bin"))
        if not files:
            raise FileNotFoundError(
                f"No samples_*.bin files found in {training_data_dir}. "
                f"Run the `selfplay` binary first to generate training data."
            )
        self.index = []  # list of (file_path, byte_offset)
        for f in files:
            size = f.stat().st_size
            if size % SAMPLE_SIZE != 0:
                raise ValueError(
                    f"{f}: file size {size} is not a multiple of "
                    f"SAMPLE_SIZE={SAMPLE_SIZE} -- truncated/corrupt file, "
                    f"or generated with a different (older) selfplay.rs "
                    f"struct layout. Regenerate this file."
                )
            n = size // SAMPLE_SIZE
            for i in range(n):
                self.index.append((f, i * SAMPLE_SIZE))

        # Validate a sample of entries up front so a bad byte-layout
        # assumption fails loudly before a multi-hour training run, not
        # silently in the middle of it.
        checked = 0
        for f, offset in self.index:
            if checked >= validate_first_n:
                break
            sanity_check_sample(self._read_at(f, offset), strict=True)
            checked += 1

    def _read_at(self, path: Path, offset: int) -> Sample:
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(SAMPLE_SIZE)
        unpacked = struct.unpack(SAMPLE_FORMAT, chunk)
        board = np.array(unpacked[0:BOARD_SQUARES], dtype=np.uint16)
        rest = unpacked[BOARD_SQUARES:]
        (side_to_move, move_from, move_to, move_promo, result,
         policy_target, value_target, move_number) = rest
        return Sample(
            board=board, side_to_move=side_to_move, move_from=move_from,
            move_to=move_to, move_promo=move_promo, result=result,
            policy_target=policy_target, value_target=value_target,
            move_number=move_number,
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Sample:
        f, offset = self.index[idx]
        return self._read_at(f, offset)


def load_dataset_dir(training_data_dir: Path, max_samples: int | None = None,
                      validate_first_n: int = 20):
    """Loads all samples_*.bin files in a directory into a list of Sample.

    Validates the first `validate_first_n` samples with sanity_check_sample
    (raises immediately if the layout looks wrong) so a bad byte-layout
    assumption fails loudly and early instead of silently training garbage.
    """
    files = sorted(training_data_dir.glob("samples_*.bin"))
    if not files:
        raise FileNotFoundError(
            f"No samples_*.bin files found in {training_data_dir}. "
            f"Run the `selfplay` binary first to generate training data."
        )

    samples = []
    checked = 0
    for f in files:
        for s in iter_samples_from_file(f):
            if checked < validate_first_n:
                sanity_check_sample(s, strict=True)
                checked += 1
            samples.append(s)
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../training_data")
    print(f"SAMPLE_SIZE = {SAMPLE_SIZE} bytes")
    samples = load_dataset_dir(data_dir, max_samples=1000)
    print(f"Loaded {len(samples)} samples (capped at 1000 for this check) from {data_dir}")
    if samples:
        s = samples[0]
        print(f"First sample: side_to_move={s.side_to_move} move={s.move_from}->{s.move_to} "
              f"value_target={s.value_target} result={s.result} move_number={s.move_number}")
        occupied = int(np.count_nonzero(s.board))
        print(f"Occupied squares in first sample: {occupied}")
