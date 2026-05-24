"""Train the V1 dynamic-gesture (temporal) classifier.

Mirrors apps/trainer/train_static.py: a standalone script configured by
environment variables, runnable directly (CPU-friendly). Reads the
train/val parquet caches produced by `sigil dataset-v1 build`, trains
the temporal Transformer, and writes:

    <output_dir>/best.pt          best-val-accuracy checkpoint
    <output_dir>/history.json     per-epoch train/val loss + accuracy
    <output_dir>/label_map.json   index → class name (canonical order)

Environment variables (all optional except where noted):
    SIGIL_V1_DATA_DIR     dir containing train.parquet + val.parquet
                          (default: datasets/processed/v1)
    SIGIL_V1_OUTPUT_DIR   where to write outputs
                          (default: models/temporal-classifier/v1)
    SIGIL_EPOCHS          default 30
    SIGIL_BATCH_SIZE      default 32
    SIGIL_LR              default 5e-4
    SIGIL_LABEL_SMOOTHING default 0.05
    SIGIL_AUG_NOISE       Gaussian aug sigma, default 0.01 (0 disables)
    SIGIL_SEED            default 42

SMOKE TEST before the real run:
    $env:SIGIL_EPOCHS=2; uv run python apps/trainer/train_temporal.py
proves the loop end-to-end in ~1-2 min. Then unset and run for real.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sigil.intelligence.dataset.jester import SIGIL_V1_CLASSES
from sigil.intelligence.dataset.temporal_storage import read_temporal_arrays
from sigil.intelligence.training.temporal_model import (
    TEMPORAL_MODEL_VERSION,
    TemporalGestureClassifier,
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _build_label_index() -> dict[str, int]:
    """Canonical, deterministic label → index map (alphabetical)."""
    return {name: i for i, name in enumerate(SIGIL_V1_CLASSES)}


def _load_split(
    data_dir: Path, split: str, label_to_idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / f"{split}.parquet"
    sequences, labels, _ids = read_temporal_arrays(path)
    if len(sequences) == 0:
        raise SystemExit(f"No rows in {path}. Run `sigil dataset-v1 build` first.")
    y = np.array([label_to_idx[str(lbl)] for lbl in labels], dtype=np.int64)
    return sequences.astype(np.float32), y


def _augment_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Direction-SAFE augmentation: additive Gaussian noise only.

    IMPORTANT: we deliberately do NOT horizontally mirror sequences.
    The static classifier could mirror freely, but here a horizontal
    flip negates the x-velocity channel, which turns a swipe_left into
    a swipe_right WITHOUT relabelling — corrupting the data. Additive
    noise perturbs magnitude without changing direction, so it's safe.
    """
    if sigma <= 0:
        return x
    return x + torch.randn_like(x) * sigma


def main() -> None:
    data_dir = Path(_env("SIGIL_V1_DATA_DIR", "datasets/processed/v1"))
    output_dir = Path(_env("SIGIL_V1_OUTPUT_DIR", "models/temporal-classifier/v1"))
    epochs = int(_env("SIGIL_EPOCHS", "30"))
    batch_size = int(_env("SIGIL_BATCH_SIZE", "32"))
    lr = float(_env("SIGIL_LR", "5e-4"))
    label_smoothing = float(_env("SIGIL_LABEL_SMOOTHING", "0.05"))
    aug_sigma = float(_env("SIGIL_AUG_NOISE", "0.01"))
    seed = int(_env("SIGIL_SEED", "42"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_to_idx = _build_label_index()
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    print(f"Classes ({len(label_to_idx)}): {list(label_to_idx)}")
    print(f"Loading data from {data_dir} …")
    x_train, y_train = _load_split(data_dir, "train", label_to_idx)
    x_val, y_val = _load_split(data_dir, "val", label_to_idx)
    print(f"  train: {x_train.shape}  val: {x_val.shape}")

    # Class distribution sanity print.
    for name, i in label_to_idx.items():
        print(f"    {name:<22} train={int((y_train == i).sum())} "
              f"val={int((y_val == i).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = TemporalGestureClassifier(num_classes=len(label_to_idx)).to(device)
    print(f"Model params: {model.count_parameters():,} "
          f"(version {TEMPORAL_MODEL_VERSION})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history: list[dict] = []
    best_val_acc = -1.0
    best_path = output_dir / "best.pt"
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in train_dl:
            xb = _augment_noise(xb, aug_sigma).to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total += xb.size(0)

        tr_acc = tr_correct / max(tr_total, 1)
        va_acc = va_correct / max(va_total, 1)
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss / max(tr_total, 1),
            "train_acc": tr_acc,
            "val_loss": va_loss / max(va_total, 1),
            "val_acc": va_acc,
        }
        history.append(rec)
        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_version": TEMPORAL_MODEL_VERSION,
                    "label_map": idx_to_label,
                    "val_acc": va_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            marker = "  <- best (saved)"
        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train_loss {rec['train_loss']:.4f} acc {tr_acc:.4f}  |  "
            f"val_loss {rec['val_loss']:.4f} acc {va_acc:.4f}{marker}"
        )

    elapsed = time.time() - start
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "label_map.json").write_text(
        json.dumps({str(i): n for i, n in idx_to_label.items()}, indent=2)
    )
    print(f"\nDone in {elapsed/60:.1f} min. Best val acc: {best_val_acc:.4f}")
    print(f"Saved: {best_path}")
    print(f"       {output_dir / 'history.json'}")
    print(f"       {output_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
