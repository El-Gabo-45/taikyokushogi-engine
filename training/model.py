"""
PyTorch mirror of the NNUE architecture defined in src/eval/nnue.rs.

Every dimension here must match nnue.rs exactly, or the exported weights
won't be loadable by the Rust side (see export.py, which writes the exact
byte format documented in nnue.rs's "Serialization" section).

ARCHITECTURE (from nnue.rs, after the dimension-chaining fix applied in
this session -- see git history / changes.diff for what was wrong before):

    Feature Transformer (HalfKP-style, feature-hashed):
        Raw HalfKP index space (king_sq * piece_sq * piece_type) is
        ~505M per perspective -- too large for a dense embedding table --
        so it's hashed down via modulo to FT_BUCKETS/2 buckets per
        perspective (FT_BUCKETS = 200,000 total, matching FT_BUCKETS in
        nnue.rs). See features.py's feature_index() for the exact hash.
        FT_NEURONS  = 512
        Two independent forward passes (white perspective, black
        perspective), each: clipped_relu(accumulated_features + bias),
        clamped to [0, 255].

    Concatenation: [ft_white (512), ft_black (512)] -> 1024

    l1: FactorizedLayer(1024 -> hidden 1024 -> 256), ReLU between the two
        internal linears (see FactorizedLayer.forward in nnue.rs)
    l2: FactorizedLayer(256 -> hidden 512 -> 128)
    l3: FactorizedLayer(128 -> hidden 128 -> 64)
    output: LinearLayer(64 -> 1)

    Final score = output[0], negated if side_to_move == WHITE (see
    NnueEvaluator::evaluate in nnue.rs: `if board.side_to_move == BLACK
    { score } else { -score }` -- BLACK=0, WHITE=1 per types.rs).

QUANTIZATION NOTE: nnue.rs runs inference in i16 fixed-point (Q8.8, SCALE
= 256) for speed. We train in float32 here (standard practice -- this is
exactly what Stockfish's NNUE trainer does too) and only quantize to i16
at export time (see export.py). The float32 model's forward pass mirrors
the *mathematical* operation of the i16 version (clipped ReLU to a [0,1]
range scaled appropriately) closely enough that post-training
quantization works well in practice; we are not trying to exactly
replicate integer rounding behavior during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SQUARES = 1296
NUM_PIECE_TYPES = 301  # must match taikyokushogi::num_piece_types()
NUM_COLORS = 2
FT_BUCKETS = 200_000  # must match FT_BUCKETS in nnue.rs exactly
FT_FEATURES = FT_BUCKETS  # feature-hashed embedding table size (see features.py)
FT_NEURONS = 512

BLACK = 0
WHITE = 1


def feature_index(king_sq: int, piece_sq: int, piece_type: int, color: int, perspective: int) -> int:
    """Python mirror of feature_index() in nnue.rs (with feature hashing
    applied, see features.py for the canonical version used during
    training). Kept here for reference / unit testing against the Rust
    side -- the actual batched version used during training is
    `batch_feature_indices` in features.py (vectorized, not a per-sample
    Python loop, since a training batch touches this millions of times)."""
    half_buckets = FT_BUCKETS // 2
    raw = (king_sq * NUM_PIECE_TYPES * NUM_SQUARES) + (piece_sq * NUM_PIECE_TYPES) + piece_type
    hashed = raw % half_buckets
    if color == perspective:
        return hashed
    else:
        return half_buckets + hashed


class FeatureTransformer(nn.Module):
    """HalfKP feature transformer. Implemented as an nn.EmbeddingBag-style
    sum over active feature indices, exactly like Stockfish's NNUE trainer
    does it -- the alternative (a dense 780,192 x 512 matmul against a
    mostly-zero one-hot vector) would waste enormous amounts of compute
    and memory for no benefit, since a Taikyoku Shogi position only has a
    few hundred pieces on the board (i.e. a few hundred active features
    out of 780,192, per perspective).
    """

    def __init__(self):
        super().__init__()
        # weight[feature_idx, neuron] -- same layout as nnue.rs's
        # `weights: Vec<Vec<i16>>` (row-major: [feature][neuron]).
        self.weight = nn.Embedding(FT_FEATURES, FT_NEURONS)
        self.bias = nn.Parameter(torch.zeros(FT_NEURONS))
        # Small init (matches nnue.rs's `rng.gen_range(-128..128)` roughly
        # in relative scale once you account for float vs the i16/SCALE
        # fixed-point range used at inference time).
        nn.init.uniform_(self.weight.weight, -0.05, 0.05)

    def forward(self, active_indices: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """
        active_indices: (batch, max_pieces) int64, feature indices for
            this perspective (padded with 0 where active_mask is False).
        active_mask: (batch, max_pieces) bool/float, 1.0 where the entry
            in active_indices is a real active feature, 0.0 for padding.
        Returns: (batch, FT_NEURONS) clipped-ReLU activations, mirroring
            FeatureTransformer::forward in nnue.rs (clamp to [0, 255] in
            the i16 version; here we clamp to [0, 1] since we work in a
            normalized float domain and rescale at export/quantization
            time instead of baking the 255 clamp into training).
        """
        # (batch, max_pieces, FT_NEURONS)
        looked_up = self.weight(active_indices)
        masked = looked_up * active_mask.unsqueeze(-1)
        summed = masked.sum(dim=1)  # (batch, FT_NEURONS)
        activated = summed + self.bias
        return torch.clamp(activated, 0.0, 1.0)


class FactorizedLayer(nn.Module):
    """Mirrors FactorizedLayer in nnue.rs: input -> hidden (ReLU) -> output.
    Note nnue.rs's forward() only has ONE ReLU, applied to the hidden
    layer's output before the second linear -- there is no activation
    after the second linear (w2/b2) inside this module; the outer
    NnueEvaluator chains these back-to-back without an extra activation
    between FactorizedLayer instances either. We mirror that exactly:
    ReLU only between w1 and w2 within a single FactorizedLayer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class NnueModel(nn.Module):
    """Full network, mirroring NnueEvaluator in nnue.rs (post dimension fix):
        ft -> concat(white, black) -> l1 -> l2 -> l3 -> output -> scalar score
    """

    def __init__(self):
        super().__init__()
        self.ft = FeatureTransformer()
        self.l1 = FactorizedLayer(FT_NEURONS * 2, 1024, 256)
        self.l2 = FactorizedLayer(256, 512, 128)
        self.l3 = FactorizedLayer(128, 128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, white_idx, white_mask, black_idx, black_mask, side_to_move):
        """
        white_idx/white_mask, black_idx/black_mask: as in
            FeatureTransformer.forward, one pair per perspective.
        side_to_move: (batch,) int, 0=BLACK, 1=WHITE (matches types.rs).
        Returns: (batch, 1) predicted value in [-1, 1] (tanh-squashed --
            see note below), from the SIDE TO MOVE's perspective, matching
            value_target's convention in the training data (dataset.py /
            selfplay.rs: +1 if the side to move in that position went on
            to win, -1 if it lost, 0 for a draw).
        """
        ft_white = self.ft(white_idx, white_mask)
        ft_black = self.ft(black_idx, black_mask)
        concat = torch.cat([ft_white, ft_black], dim=1)  # (batch, 1024)

        x = self.l1(concat)
        x = self.l2(x)
        x = self.l3(x)
        raw_score = self.output(x).squeeze(-1)  # (batch,)

        # nnue.rs negates the raw score when side_to_move == WHITE:
        #   if board.side_to_move == BLACK { score } else { -score }
        # We do the same, then squash with tanh to land in [-1, 1] to
        # match value_target's range. tanh is applied only during
        # training for the value-loss target; the exported integer
        # network at inference time in Rust does NOT apply tanh (it
        # returns a raw centipawn-like score, per nnue.rs) -- the sign
        # flip by side_to_move is the only piece of this that needs to
        # carry over to the exported weights' *usage* in Rust, and it
        # already exists there (see NnueEvaluator::evaluate). Training
        # in a bounded [-1,1] space with tanh simply makes the value
        # loss well-behaved; it does not change what gets exported.
        sign = torch.where(side_to_move == BLACK, 1.0, -1.0)
        signed_score = raw_score * sign
        return torch.tanh(signed_score).unsqueeze(-1)
