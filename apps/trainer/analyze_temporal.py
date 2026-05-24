"""Analyse the trained V1 temporal classifier + emit report figures.

Runs best.pt over the val set and reports:
  - overall accuracy
  - per-class accuracy (the honest view: is 87.8% strong swipes or just
    an easy negative class?)
  - macro-average accuracy (treats every class equally regardless of
    how many val examples it has)
  - a confusion matrix

Saves two report-quality PNG figures:
  <figures_dir>/v1_training_curves.png    (from history.json)
  <figures_dir>/v1_confusion_matrix.png   (from val predictions)

Confusion matrix is computed with numpy (no sklearn dependency).

Environment variables:
    SIGIL_V1_DATA_DIR     default datasets/processed/v1
    SIGIL_V1_OUTPUT_DIR   default models/temporal-classifier/v1
    SIGIL_FIGURES_DIR     default <output_dir>/figures

Usage:
    uv run python apps/trainer/analyze_temporal.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from sigil.intelligence.dataset.jester import SIGIL_V1_CLASSES
from sigil.intelligence.dataset.temporal_storage import read_temporal_arrays
from sigil.intelligence.training.temporal_model import TemporalGestureClassifier


def main() -> None:
    data_dir = Path(os.environ.get("SIGIL_V1_DATA_DIR", "datasets/processed/v1"))
    output_dir = Path(
        os.environ.get("SIGIL_V1_OUTPUT_DIR", "models/temporal-classifier/v1")
    )
    figures_dir = Path(os.environ.get("SIGIL_FIGURES_DIR", str(output_dir / "figures")))
    figures_dir.mkdir(parents=True, exist_ok=True)

    classes = list(SIGIL_V1_CLASSES)
    n_classes = len(classes)

    # --- load model ---
    ckpt_path = output_dir / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TemporalGestureClassifier(num_classes=n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path}  (val_acc={ckpt.get('val_acc'):.4f}, "
          f"epoch={ckpt.get('epoch')})")

    # --- run on val ---
    x_val, y_lbl, _ids = read_temporal_arrays(data_dir / "val.parquet")
    label_to_idx = {name: i for i, name in enumerate(classes)}
    y_true = np.array([label_to_idx[str(l)] for l in y_lbl], dtype=np.int64)

    with torch.no_grad():
        logits = model(torch.from_numpy(x_val.astype(np.float32)))
        y_pred = logits.argmax(1).numpy()

    overall = float((y_pred == y_true).mean())

    # --- confusion matrix (numpy) ---
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # --- per-class accuracy (recall) ---
    per_class_acc = {}
    for i, name in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc[name] = (correct / total) if total else 0.0

    macro = float(np.mean(list(per_class_acc.values())))
    swipe_only = float(np.mean([
        per_class_acc[c] for c in classes if c.startswith("swipe_")
    ]))

    # --- print report ---
    print(f"\nOverall val accuracy : {overall:.4f}")
    print(f"Macro-avg accuracy   : {macro:.4f}  (all classes weighted equally)")
    print(f"Swipe-only macro acc : {swipe_only:.4f}  (the 4 directional swipes)")
    print("\nPer-class accuracy (recall):")
    for name in classes:
        print(f"  {name:<22} {per_class_acc[name]:.4f}  (n={cm[classes.index(name)].sum()})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "            " + " ".join(f"{c[:8]:>9}" for c in classes)
    print(header)
    for i, name in enumerate(classes):
        row = " ".join(f"{cm[i, j]:>9d}" for j in range(n_classes))
        print(f"  {name[:10]:<10} {row}")

    # --- figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping figures. "
              "Numbers above are the key result.")
        return

    short = [c.replace("swipe_", "→").replace("no_dynamic_gesture", "none")
             for c in classes]

    # Figure 1: training curves from history.json
    hist_path = output_dir / "history.json"
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text())
        epochs = [h["epoch"] for h in hist]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(epochs, [h["train_loss"] for h in hist], label="train", lw=2)
        ax1.plot(epochs, [h["val_loss"] for h in hist], label="val", lw=2)
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs, [h["train_acc"] for h in hist], label="train", lw=2)
        ax2.plot(epochs, [h["val_acc"] for h in hist], label="val", lw=2)
        best_e = max(hist, key=lambda h: h["val_acc"])
        ax2.axvline(best_e["epoch"], color="grey", ls="--", alpha=0.6)
        ax2.scatter([best_e["epoch"]], [best_e["val_acc"]], color="red", zorder=5,
                    label=f"best val {best_e['val_acc']:.3f}")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
        ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.suptitle("V1 temporal classifier — training history")
        fig.tight_layout()
        p1 = figures_dir / "v1_training_curves.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved figure: {p1}")

    # Figure 2: confusion matrix heatmap (row-normalised for readability)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(short)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("V1 confusion matrix (row-normalised)")
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p2 = figures_dir / "v1_confusion_matrix.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {p2}")
    print("\nThese two PNGs are ready to drop into the report's results chapter.")


if __name__ == "__main__":
    main()
