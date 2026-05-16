# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---
"""Sigil — Static gesture classifier training.

Designed to run in Google Colab on a T4 GPU, but works locally on any
machine with the `training` and `perception` extras installed.

Quick start in Colab:

    !git clone https://github.com/<org>/sigil
    %cd sigil
    !pip install -e .[training,perception]

    # If you've already prepared the parquet cache locally and uploaded it:
    # leave RAW_DIR alone and skip the dataset preparation cell.

    !python apps/trainer/train_static.py

Or as a notebook: open this file in Colab, choose "open as notebook" — the
`# %%` markers split it into runnable cells automatically.
"""

# %% [markdown]
# # Static gesture classifier training
#
# Inputs:  21×2 normalised landmark arrays from train/val/test parquet caches
#          produced by `sigil dataset prepare` (see ADR-0003 for why 2D).
# Outputs: `best.pt` checkpoint + `label_map.json` + `history.json` in the
#          run output directory.
#
# The trained checkpoint goes to Phase 2c (ONNX export + INT8 quantization).

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Detect whether we're in Colab so the few Colab-specific bits (e.g. Drive
# mount, !-shell commands) can be conditional. The rest works anywhere.
IN_COLAB = "google.colab" in sys.modules

# %% [markdown]
# ## Configuration
#
# Edit these paths to taste. Defaults assume the parquet caches were produced
# by `sigil dataset prepare` and live under `datasets/processed/` as
# `train.parquet`, `val.parquet`, `test.parquet`.

# %%
# Directory holding train/val/test parquet caches (output of `sigil dataset prepare`).
LANDMARK_DIR = Path(
    os.environ.get(
        "SIGIL_LANDMARK_DIR",
        "datasets/processed",
    )
)

# Where to dump checkpoints / metrics / label map.
OUTPUT_DIR = Path(
    os.environ.get(
        "SIGIL_OUTPUT_DIR",
        "models/static-classifier/run-001",
    )
)

EPOCHS = int(os.environ.get("SIGIL_EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("SIGIL_BATCH_SIZE", "256"))
LEARNING_RATE = float(os.environ.get("SIGIL_LR", "3e-4"))

# %% [markdown]
# ## (Optional) Prepare the dataset
#
# Skip this cell if you've already produced the parquet caches. If running
# from a fresh Colab and you've checked out the repo, this is how you'd
# fetch HaGRIDv2 annotations and build the caches in one shot.

# %%
# Uncomment to run dataset prep from scratch:
#
# from sigil.cli.dataset import prepare_cmd
# from click.testing import CliRunner
# CliRunner().invoke(
#     prepare_cmd,
#     ["--gestures", "open_palm,thumbs_up,thumbs_down,fist,peace,ok,no_gesture",
#      "--raw-dir", "datasets/raw",
#      "--output-dir", str(LANDMARK_DIR)],
#     catch_exceptions=False,
# )

# %% [markdown]
# ## Train

# %%
from sigil.intelligence.training.train import TrainingConfig, train_static
from sigil.logging import setup_logging

setup_logging(level="INFO", json=False)

cfg = TrainingConfig(
    train_parquet=LANDMARK_DIR / "train.parquet",
    val_parquet=LANDMARK_DIR / "val.parquet",
    test_parquet=LANDMARK_DIR / "test.parquet",
    output_dir=OUTPUT_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    # Augmentation off by default for the *reserved* gestures (thumbs_up vs
    # thumbs_down are NOT mirror-symmetric — mirroring would corrupt labels).
    mirror_prob=0.0,
    noise_std=0.005,
)

result = train_static(cfg)

print("\n=== Training complete ===")
print(f"Best checkpoint:    {result.best_checkpoint}")
print(f"Label map:          {result.label_map_path}")
print(f"History:            {result.history_path}")
print(f"Test accuracy:      {result.test_metrics['accuracy']:.3%}")
print(f"Test loss:          {result.test_metrics['loss']:.4f}")
print(f"Parameters:         {result.num_parameters:,}")
print(f"Model size (FP32):  {result.model_size_mb:.2f} MB")

# %% [markdown]
# ## Inspect the run

# %%
history = json.loads(result.history_path.read_text())
label_map = json.loads(result.label_map_path.read_text())

print("Final-epoch metrics:")
print(json.dumps(history[-1], indent=2))
print("\nLabel index → gesture name:")
print(json.dumps(label_map, indent=2))

# %% [markdown]
# ## (Optional) Plot training curves
#
# Useful for spotting overfit / underfit at a glance. Skipped if matplotlib
# isn't installed.

# %%
try:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed — skipping plot.")

# %% [markdown]
# ## Next step: Phase 2c — ONNX export + INT8 quantization
#
# The `best.pt` checkpoint produced here is consumed by the Phase 2c export
# script (coming next), which produces an INT8 ONNX model the daemon loads
# at runtime. Until that ships, you've still got a working PyTorch model
# you can sanity-check with `torch.load(...)` and a few hand-crafted samples.
