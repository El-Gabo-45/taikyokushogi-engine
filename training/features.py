"""
Converts a raw Sample (from dataset.py) into the active HalfKP feature
indices the model (model.py) expects, mirroring `feature_index()` and
`Accumulator::refresh()` in src/eval/nnue.rs exactly.

KNOWN APPROXIMATION -- read this before trusting results on positions with
multiple royal pieces of the same color alive at once:
---------------------------------------------------------------------------
nnue.rs determines "the king square" for a color via
`board.king_square(color)`, which returns `royal_list[color][0]` -- the
FIRST royal piece in insertion order (see board.rs). That order depends on
the game's move history (which royal piece was placed/promoted first),
which is NOT recoverable from a single board snapshot alone (which is all
TrainingSample stores).

This module instead picks the royal piece at the LOWEST square index when
a color has more than one royal piece alive (Taikyoku Shogi supports up to
MAX_ROYALS=8 per side, though in practice this usually means King + Crown
Prince, piece_types 29 and 88 per piece_metadata.json). This is
deterministic and reproducible, but will occasionally disagree with what
the live engine would have used as `king_sq` for `Accumulator::refresh` in
that specific game.

Impact: this only affects positions where a color has 2+ royal pieces
alive simultaneously (a strict subset of all positions), and even then
only changes which of several valid king-relative feature encodings is
used -- it does not create invalid features or corrupt the piece_type/color
data. In practice this is expected to be a small, bounded source of label
noise, not a correctness bug. If you want exact fidelity here, the real
fix is to have selfplay.rs record king_sq directly into TrainingSample
instead of reconstructing it in Python -- flagged as a possible follow-up,
not done here to avoid re-changing the on-disk sample format twice in one
session.
"""

import json
from pathlib import Path

import numpy as np
import torch

from dataset import Sample, BOARD_SQUARES

NUM_PIECE_TYPES = 301
NUM_SQUARES = 1296
FT_BUCKETS = 200_000  # must match FT_BUCKETS in nnue.rs exactly
FT_FEATURES = FT_BUCKETS

BLACK = 0
WHITE = 1

_METADATA_PATH = Path(__file__).parent / "piece_metadata.json"


def _load_royal_piece_types() -> frozenset[int]:
    if not _METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{_METADATA_PATH} not found. Generate it first with:\n"
            f"  cargo run --release --example export_piece_metadata > training/piece_metadata.json\n"
            f"(run from the repo root, with the Rust project built)."
        )
    with open(_METADATA_PATH) as f:
        meta = json.load(f)
    if meta["num_piece_types"] != NUM_PIECE_TYPES:
        raise ValueError(
            f"piece_metadata.json says num_piece_types={meta['num_piece_types']}, "
            f"but this module hardcodes NUM_PIECE_TYPES={NUM_PIECE_TYPES}. "
            f"The piece set changed -- update NUM_PIECE_TYPES here (and in "
            f"model.py, dataset.py, and nnue.rs) to match, then regenerate "
            f"training data and metadata."
        )
    return frozenset(meta["royal_piece_types"])


ROYAL_PIECE_TYPES = _load_royal_piece_types()


def feature_index(king_sq: int, piece_sq: int, piece_type: int, color: int, perspective: int) -> int:
    """Exact mirror of feature_index() in nnue.rs, INCLUDING the feature
    hashing step (index % (FT_BUCKETS/2)). The raw HalfKP index space
    (king_sq * piece_sq * piece_type) is ~505M per perspective -- far too
    large for a dense embedding table -- so it's hashed down to
    FT_BUCKETS/2 buckets per perspective before the perspective offset is
    added. This MUST use the same divisor as nnue.rs or exported weights
    will not correspond to the same features the Rust inference code
    looks up."""
    half_buckets = FT_BUCKETS // 2
    raw = (king_sq * NUM_PIECE_TYPES * NUM_SQUARES) + (piece_sq * NUM_PIECE_TYPES) + piece_type
    hashed = raw % half_buckets
    if color == perspective:
        return hashed
    else:
        return half_buckets + hashed


def decode_board(board: np.ndarray):
    """Yields (sq, piece_type, color) for every occupied square. Matches
    the (piece_type & 0x1FF) | (color << 9) encoding fixed in selfplay.rs
    this session -- see dataset.py's module docstring for the full byte
    layout and why this encoding (not the engine's internal Cell type) is
    what's on disk."""
    occupied_sqs = np.nonzero(board)[0]
    for sq in occupied_sqs:
        v = int(board[sq])
        piece_type = v & 0x1FF
        color = (v >> 9) & 1
        yield int(sq), piece_type, color


def find_king_squares(board: np.ndarray) -> tuple[int | None, int | None]:
    """Returns (black_king_sq, white_king_sq), or None for a color with no
    royal piece on the board (shouldn't normally happen mid-game, but
    defensive: a checkmate-ending position might have the losing side's
    last royal just captured, depending on exactly when the sample was
    recorded relative to game-end detection).

    See the module docstring for the "lowest square index" tie-breaking
    approximation used when a color has multiple royal pieces alive.
    """
    black_king = None
    white_king = None
    for sq, piece_type, color in decode_board(board):
        if piece_type in ROYAL_PIECE_TYPES:
            if color == BLACK and (black_king is None or sq < black_king):
                black_king = sq
            elif color == WHITE and (white_king is None or sq < white_king):
                white_king = sq
    return black_king, white_king


# Max pieces we'll ever need to pad to. Taikyoku Shogi has up to ~402
# pieces per side (804 total on a full board), but MAX_PIECES_PER_SIDE in
# types.rs is 410 -- pad generously above the theoretical max per side so
# truncation never silently drops a piece.
MAX_ACTIVE_FEATURES = 420


def sample_to_features(s: Sample):
    """Returns (white_idx, white_mask, black_idx, black_mask) as numpy
    arrays of shape (MAX_ACTIVE_FEATURES,), suitable for batching. Mirrors
    Accumulator::refresh() in nnue.rs: for each occupied square, for each
    perspective, one feature index is added (relative to that
    perspective's king square).

    Note: unlike nnue.rs, which maintains SEPARATE accumulators per
    perspective (white/black) each keyed by that perspective's OWN king
    square, this function computes indices for both perspectives from the
    same board snapshot in one pass, which is mathematically the same
    thing nnue.rs's Accumulator::refresh does (it also loops over all
    pieces once and computes both idx_white and idx_black per piece, see
    the `for n in 0..FT_NEURONS` loop in refresh()) -- just restructured
    for batch-friendly numpy/torch output instead of an incremental i16
    accumulator update loop.
    """
    black_king, white_king = find_king_squares(s.board)

    white_indices = np.zeros(MAX_ACTIVE_FEATURES, dtype=np.int64)
    white_mask = np.zeros(MAX_ACTIVE_FEATURES, dtype=np.float32)
    black_indices = np.zeros(MAX_ACTIVE_FEATURES, dtype=np.int64)
    black_mask = np.zeros(MAX_ACTIVE_FEATURES, dtype=np.float32)

    i = 0
    for sq, piece_type, color in decode_board(s.board):
        if i >= MAX_ACTIVE_FEATURES:
            raise ValueError(
                f"Position has more than MAX_ACTIVE_FEATURES={MAX_ACTIVE_FEATURES} "
                f"pieces -- increase MAX_ACTIVE_FEATURES in features.py."
            )
        if white_king is not None:
            white_indices[i] = feature_index(white_king, sq, piece_type, color, WHITE)
            white_mask[i] = 1.0
        if black_king is not None:
            black_indices[i] = feature_index(black_king, sq, piece_type, color, BLACK)
            black_mask[i] = 1.0
        i += 1

    return white_indices, white_mask, black_indices, black_mask


def batch_to_tensors(samples: list[Sample], device: str = "cpu"):
    """Converts a list of Sample into the batched tensors NnueModel.forward
    expects. This is the main entry point used by train.py's DataLoader
    collate function."""
    n = len(samples)
    white_idx = np.zeros((n, MAX_ACTIVE_FEATURES), dtype=np.int64)
    white_mask = np.zeros((n, MAX_ACTIVE_FEATURES), dtype=np.float32)
    black_idx = np.zeros((n, MAX_ACTIVE_FEATURES), dtype=np.int64)
    black_mask = np.zeros((n, MAX_ACTIVE_FEATURES), dtype=np.float32)
    side_to_move = np.zeros(n, dtype=np.int64)
    value_target = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(samples):
        wi, wm, bi, bm = sample_to_features(s)
        white_idx[i] = wi
        white_mask[i] = wm
        black_idx[i] = bi
        black_mask[i] = bm
        side_to_move[i] = s.side_to_move
        value_target[i] = s.value_target

    return {
        "white_idx": torch.from_numpy(white_idx).to(device),
        "white_mask": torch.from_numpy(white_mask).to(device),
        "black_idx": torch.from_numpy(black_idx).to(device),
        "black_mask": torch.from_numpy(black_mask).to(device),
        "side_to_move": torch.from_numpy(side_to_move).to(device),
        "value_target": torch.from_numpy(value_target).to(device),
    }
