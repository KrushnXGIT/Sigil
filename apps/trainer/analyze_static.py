"""Analyse the trained static gesture classifier + emit report figures.

Runs the trained best.pt over the held-out test split and reports:
  - overall test accuracy
  - per-class accuracy (recall) for all 17 classes
  - macro-average accuracy
  - the SPECIFIC collision pairs we set out to measure (peace vs two_up,
    grabbing vs fist, thumb_index vs ok, little_finger vs rock)
  - a full confusion matrix

Saves report-quality PNGs:
  <figures_dir>/static_confusion_matrix.png
  <figures_dir>/static_training_curves.png   (if history.json present)

Confusion matrix is numpy (no sklearn). Reads the held-out test.parquet
produced by `sigil dataset build`, so this is an honest generalisation
estimate, not a re-score of training data.

Environment variables:
    SIGIL_TEST_PARQUET   default datasets/processed/new-statics/test.parquet
    SIGIL_OUTPUT_DIR     default models/static-classifier/new-statics-17class
    SIGIL_FIGURES_DIR    default <output_dir>/figures

Usage:
    uv run python apps/trainer/analyze_static.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from sigil.intelligence.dataset.storage import read_landmark_arrays
from sigil.intelligence.training.model import StaticGestureClassifier

# Collision pairs we explicitly want to read off the matrix (Sigil names).
WATCH_PAIRS = [
    ("peace", "two_up"),
    ("fist", "grabbing"),
    ("ok", "thumb_index"),
    ("rock", "little_finger"),
]


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main() -> None:
    test_parquet = Path(os.environ.get(
        "SIGIL_TEST_PARQUET", "datasets/processed/new-statics/test.parquet"))
    output_dir = Path(os.environ.get(
        "SIGIL_OUTPUT_DIR", "models/static-classifier/new-statics-17class"))
    figures_dir = Path(os.environ.get("SIGIL_FIGURES_DIR", str(output_dir / "figures")))
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- load checkpoint (carries its own label_names, in training order) ---
    ckpt_path = output_dir / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = list(ckpt["label_names"])      # sorted() order used at train time
    n_classes = len(classes)
    label_to_idx = {name: i for i, name in enumerate(classes)}

    model = StaticGestureClassifier(num_classes=n_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path}  (val_acc={ckpt.get('val_acc'):.4f}, "
          f"epoch={ckpt.get('epoch')}, classes={n_classes})")

    # --- load held-out test split ---
    X, y_lbl, _users = read_landmark_arrays(test_parquet)
    # Keep only rows whose label is one the model knows (defensive).
    keep = np.array([str(l) in label_to_idx for l in y_lbl])
    X, y_lbl = X[keep], y_lbl[keep]
    y_true = np.array([label_to_idx[str(l)] for l in y_lbl], dtype=np.int64)
    print(f"Test samples: {len(y_true)}  from {test_parquet}")

    # --- inference (model expects (B,21,3) float32) ---
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)))
        y_pred = logits.argmax(1).numpy()

    overall = float((y_pred == y_true).mean())
    cm = _confusion(y_true, y_pred, n_classes)

    per_class_acc = {}
    for i, name in enumerate(classes):
        tot = cm[i].sum()
        per_class_acc[name] = (cm[i, i] / tot) if tot else 0.0
    macro = float(np.mean(list(per_class_acc.values())))

    # --- print report ---
    print(f"\nOverall test accuracy : {overall:.4f}")
    print(f"Macro-avg accuracy    : {macro:.4f}")
    print("\nPer-class accuracy (recall):")
    for name in classes:
        print(f"  {name:<16} {per_class_acc[name]:.4f}  (n={cm[classes.index(name)].sum()})")

    # --- the collision pairs we care about ---
    print("\nCollision-pair check (how often true row is called the partner):")
    for a, b in WATCH_PAIRS:
        if a in label_to_idx and b in label_to_idx:
            ia, ib = label_to_idx[a], label_to_idx[b]
            a_tot, b_tot = cm[ia].sum(), cm[ib].sum()
            a_to_b = cm[ia, ib] / a_tot if a_tot else 0.0
            b_to_a = cm[ib, ia] / b_tot if b_tot else 0.0
            verdict = "OK" if max(a_to_b, b_to_a) < 0.05 else (
                "WATCH" if max(a_to_b, b_to_a) < 0.12 else "PROBLEM")
            print(f"  {a:<14}<->{b:<14}  {a}->{b}={a_to_b:.3f}  "
                  f"{b}->{a}={b_to_a:.3f}   [{verdict}]")
        else:
            print(f"  {a} <-> {b}: one of these isn't in the model — skipped")

    # --- figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — numbers above are the result.")
        return

    # confusion matrix (row-normalised, counts in cells)
    cmn = cm.astype(np.float64)
    rs = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, rs, where=rs != 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Static classifier confusion matrix ({n_classes} classes, row-normalised)")
    for i in range(n_classes):
        for j in range(n_classes):
            v = cmn[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p1 = figures_dir / "static_confusion_matrix.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved figure: {p1}")

    hist_path = output_dir / "history.json"
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text())
        ep = [h["epoch"] for h in hist]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.plot(ep, [h["train_loss"] for h in hist], label="train", lw=2)
        a1.plot(ep, [h["val_loss"] for h in hist], label="val", lw=2)
        a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(ep, [h["train_acc"] for h in hist], label="train", lw=2)
        a2.plot(ep, [h["val_acc"] for h in hist], label="val", lw=2)
        be = max(hist, key=lambda h: h["val_acc"])
        a2.scatter([be["epoch"]], [be["val_acc"]], color="red", zorder=5,
                   label=f"best val {be['val_acc']:.3f}")
        a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
        fig.suptitle(f"Static classifier ({n_classes} classes) — training history")
        fig.tight_layout()
        p2 = figures_dir / "static_training_curves.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved figure: {p2}")
    print("\nFigures ready for the report's results chapter.")


if __name__ == "__main__":
    main()
