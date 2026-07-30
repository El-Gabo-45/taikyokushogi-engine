"""
Exports a trained PyTorch checkpoint to the flat binary .nnue format that
NnueEvaluator::load_from_file in src/eval/nnue.rs reads (see the
"Serialization" section in nnue.rs for the exact byte layout this must
produce).

CRITICAL DETAIL -- weight matrix orientation:
-----------------------------------------------
PyTorch's nn.Linear stores weight as shape [output_size, input_size]
(because it computes y = x @ W.T + b). nnue.rs's FactorizedLayer/LinearLayer
store weights as `weights[input][hidden_or_output]` (row-major, indexed by
INPUT first -- see the `/// weights[input][hidden]` doc comment on
FactorizedLayer in nnue.rs, and LinearLayer's `weights: Vec<Vec<i16>>` used
as `self.weights[i][o]` in its forward()). These are TRANSPOSED relative
to each other. Every weight matrix below is explicitly `.T`'d before
writing to account for this -- if you modify this script, preserve that
transpose or the exported network will silently compute garbage (no crash,
just wrong numbers, since both sides are simple 2D arrays of the right
total size either way).

QUANTIZATION:
-------------
nnue.rs runs inference in i16 fixed-point, Q8.8 (SCALE=256, i.e. real_value
= i16_value / 256). We quantize every float32 weight as
round(clamp(weight * SCALE, -32768, 32767)). This matches the scale used
in nnue.rs's own random-init ranges (e.g. FeatureTransformer::new_random
draws from -128..128, LinearLayer from -32..32) closely enough to produce
sensible fixed-point weights after training in float32 -- this is standard
post-training quantization, the same approach Stockfish's NNUE trainer
uses (train in float, quantize once at export time).

Usage:
    python3 export.py --checkpoint checkpoints/latest.pt --output ../trained.nnue
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import NnueModel, FT_BUCKETS, FT_NEURONS

SCALE = 256
NNUE_FILE_MAGIC = b"NNU1"


def quantize(arr: np.ndarray) -> np.ndarray:
    """float32 array -> i16 array, Q8.8 fixed point, matching nnue.rs's SCALE."""
    scaled = np.round(arr.astype(np.float64) * SCALE)
    clamped = np.clip(scaled, -32768, 32767)
    return clamped.astype(np.int16)


def write_i16_matrix(f, arr: np.ndarray):
    """Writes a 2D array row-major, little-endian i16, matching
    write_i16_matrix in nnue.rs exactly (same iteration order: for row in
    rows, for value in row)."""
    assert arr.ndim == 2
    f.write(arr.astype("<i2").tobytes())


def write_i16_vec(f, arr: np.ndarray):
    assert arr.ndim == 1
    f.write(arr.astype("<i2").tobytes())


def export(model: NnueModel, output_path: Path):
    model.eval()
    with torch.no_grad():
        ft_weights = quantize(model.ft.weight.weight.cpu().numpy())
        ft_biases = quantize(model.ft.bias.cpu().numpy())
        assert ft_weights.shape == (FT_BUCKETS, FT_NEURONS), ft_weights.shape

        l1_w1 = quantize(model.l1.fc1.weight.cpu().numpy().T)
        l1_b1 = quantize(model.l1.fc1.bias.cpu().numpy())
        l1_w2 = quantize(model.l1.fc2.weight.cpu().numpy().T)
        l1_b2 = quantize(model.l1.fc2.bias.cpu().numpy())

        l2_w1 = quantize(model.l2.fc1.weight.cpu().numpy().T)
        l2_b1 = quantize(model.l2.fc1.bias.cpu().numpy())
        l2_w2 = quantize(model.l2.fc2.weight.cpu().numpy().T)
        l2_b2 = quantize(model.l2.fc2.bias.cpu().numpy())

        l3_w1 = quantize(model.l3.fc1.weight.cpu().numpy().T)
        l3_b1 = quantize(model.l3.fc1.bias.cpu().numpy())
        l3_w2 = quantize(model.l3.fc2.weight.cpu().numpy().T)
        l3_b2 = quantize(model.l3.fc2.bias.cpu().numpy())

        output_w = quantize(model.output.weight.cpu().numpy().T)
        output_b = quantize(model.output.bias.cpu().numpy())

    expected_shapes = {
        "ft_weights": (FT_BUCKETS, FT_NEURONS), "ft_biases": (FT_NEURONS,),
        "l1_w1": (FT_NEURONS * 2, 1024), "l1_b1": (1024,),
        "l1_w2": (1024, 256), "l1_b2": (256,),
        "l2_w1": (256, 512), "l2_b1": (512,),
        "l2_w2": (512, 128), "l2_b2": (128,),
        "l3_w1": (128, 128), "l3_b1": (128,),
        "l3_w2": (128, 64), "l3_b2": (64,),
        "output_w": (64, 1), "output_b": (1,),
    }
    actual = {
        "ft_weights": ft_weights, "ft_biases": ft_biases,
        "l1_w1": l1_w1, "l1_b1": l1_b1, "l1_w2": l1_w2, "l1_b2": l1_b2,
        "l2_w1": l2_w1, "l2_b1": l2_b1, "l2_w2": l2_w2, "l2_b2": l2_b2,
        "l3_w1": l3_w1, "l3_b1": l3_b1, "l3_w2": l3_w2, "l3_b2": l3_b2,
        "output_w": output_w, "output_b": output_b,
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = actual[name].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {expected_shape}, "
                f"got {actual_shape}. This means model.py's architecture "
                f"no longer matches nnue.rs's NnueEvaluator -- check both "
                f"for recent changes."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(NNUE_FILE_MAGIC)
        write_i16_matrix(f, ft_weights)
        write_i16_vec(f, ft_biases)
        write_i16_matrix(f, l1_w1)
        write_i16_vec(f, l1_b1)
        write_i16_matrix(f, l1_w2)
        write_i16_vec(f, l1_b2)
        write_i16_matrix(f, l2_w1)
        write_i16_vec(f, l2_b1)
        write_i16_matrix(f, l2_w2)
        write_i16_vec(f, l2_b2)
        write_i16_matrix(f, l3_w1)
        write_i16_vec(f, l3_b1)
        write_i16_matrix(f, l3_w2)
        write_i16_vec(f, l3_b2)
        write_i16_matrix(f, output_w)
        write_i16_vec(f, output_b)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True,
                     help="Path to a .pt checkpoint saved by train.py")
    ap.add_argument("--output", type=Path, default=Path("../trained.nnue"),
                     help="Output path for the .nnue file")
    args = ap.parse_args()

    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = NnueModel()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  (from epoch {ckpt.get('epoch')}, global_step {ckpt.get('global_step')})")

    export(model, args.output)
    print("\nDone. Load this in Rust with:")
    print(f'  NnueEvaluator::load_from_file("{args.output}")')


if __name__ == "__main__":
    main()
