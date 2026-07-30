"""
Training loop for the Taikyoku Shogi NNUE.

Usage:
    python3 train.py --data-dir ../training_data --epochs 20 --batch-size 4096

For the RX 470 (no ROCm support) + Docker-based PyTorch workaround setup:
see README.md in this directory for the exact Docker invocation. Once
inside that container, this script doesn't need any changes -- it just
needs `torch.cuda.is_available()` (or whatever device the container
exposes) to return True and will use it automatically.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TrainingDataset, Sample
from features import batch_to_tensors
from model import NnueModel


def collate_samples(samples: list[Sample]):
    """DataLoader collate_fn: turns a list of raw Sample into the batched
    tensors the model expects. Runs on CPU (in DataLoader worker
    processes, if num_workers > 0) -- the actual .to(device) transfer
    happens in the training loop after this, so multiple CPU workers can
    prepare batches in parallel while the GPU trains on a previous batch.
    """
    return batch_to_tensors(samples, device="cpu")


def evaluate(model, loader, device, max_batches: int | None = None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(batch["white_idx"], batch["white_mask"],
                         batch["black_idx"], batch["black_mask"],
                         batch["side_to_move"])
            target = batch["value_target"].unsqueeze(-1)
            loss = F.mse_loss(pred, target, reduction="sum")
            total_loss += loss.item()
            total_samples += target.size(0)
    model.train()
    return total_loss / max(total_samples, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("../training_data"),
                     help="Directory containing samples_*.bin files from the selfplay binary")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.02,
                     help="Fraction of samples held out for validation")
    ap.add_argument("--num-workers", type=int, default=4,
                     help="DataLoader worker processes for CPU-side batch prep")
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--checkpoint-every-steps", type=int, default=2000,
                     help="Save a checkpoint every N training steps, in addition to once per epoch")
    ap.add_argument("--resume", type=Path, default=None,
                     help="Path to a checkpoint (.pt) to resume training from")
    ap.add_argument("--device", type=str, default=None,
                     help="Force a device (e.g. 'cuda', 'cpu'). Default: auto-detect.")
    ap.add_argument("--log-every-steps", type=int, default=50)
    args = ap.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    print(f"Loading dataset index from {args.data_dir} ...")
    full_dataset = TrainingDataset(args.data_dir)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val
    print(f"Total samples: {n_total:,} ({n_train:,} train / {n_val:,} val)")

    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_samples,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_samples,
        pin_memory=(device.type == "cuda"),
    )

    model = NnueModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}, global_step {global_step}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int, tag: str):
        path = args.checkpoint_dir / f"nnue_epoch{epoch}_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        # Also keep a stable "latest" pointer for convenience (e.g. for
        # export.py, or resuming without hunting for the newest filename).
        latest_path = args.checkpoint_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, latest_path)
        print(f"  Saved checkpoint: {path}")

    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            pred = model(batch["white_idx"], batch["white_mask"],
                         batch["black_idx"], batch["black_mask"],
                         batch["side_to_move"])
            target = batch["value_target"].unsqueeze(-1)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * target.size(0)
            running_count += target.size(0)
            global_step += 1

            if global_step % args.log_every_steps == 0:
                avg_loss = running_loss / max(running_count, 1)
                elapsed = time.time() - epoch_start
                samples_per_sec = running_count / max(elapsed, 1e-6)
                print(f"  epoch {epoch} step {step} (global {global_step}) "
                      f"loss={avg_loss:.6f} samples/s={samples_per_sec:.0f}")

            if global_step % args.checkpoint_every_steps == 0:
                save_checkpoint(epoch, f"step{global_step}")

        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} done in {epoch_time:.1f}s -- val_loss={val_loss:.6f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        save_checkpoint(epoch, "end")

    print("\nTraining complete. Run export.py against the latest checkpoint")
    print("to produce a .nnue file the Rust engine can load.")


if __name__ == "__main__":
    main()
