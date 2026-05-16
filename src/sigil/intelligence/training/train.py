"""Training loop for the static gesture classifier.

Lives entirely in training-time code (gated behind the `training` extra).
Designed to run identically in Colab (via apps/trainer/train_static.py)
and locally on the user's RTX 5050.

HaGRIDv2 ships pre-split by user_id, so training consumes three parquet
files directly (train/val/test) — no runtime splitting. The user-disjoint
splitter in `dataset.splits` is still used for custom data prep.

Public flow:
    cfg = TrainingConfig(
        train_parquet=..., val_parquet=..., test_parquet=..., output_dir=...
    )
    result = train_static(cfg)
    # result.best_checkpoint, result.history, result.test_metrics
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sigil.intelligence.dataset.storage import read_landmark_arrays
from sigil.intelligence.training.model import (
    MODEL_VERSION,
    StaticGestureClassifier,
    count_parameters,
    estimated_size_mb,
)
from sigil.logging import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters + paths for one training run.

    HaGRIDv2 is pre-split by user_id, so we take three parquet files
    directly rather than splitting at train time.
    """

    train_parquet: Path
    val_parquet: Path
    test_parquet: Path
    output_dir: Path

    # Model
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1

    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 40
    label_smoothing: float = 0.05

    # Augmentation (we keep it minimal — see ADR-0002 for why no rotation)
    noise_std: float = 0.005  # additive Gaussian noise on landmarks
    mirror_prob: float = 0.0  # off by default; enable only if your set of
    # gestures is mirror-symmetric (peace, fist, ok
    # are; thumbs_up/down become each other when
    # mirrored — DON'T flip those).

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    num_workers: int = 2


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Outputs of a finished training run."""

    best_checkpoint: Path
    label_map_path: Path
    history_path: Path
    test_metrics: dict[str, float]
    num_parameters: int
    model_size_mb: float


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------


def _build_dataset_class(torch: torch) -> type:
    """Lazy nn.utils.data.Dataset definition — gated behind torch import."""
    from torch.utils.data import Dataset

    class _LandmarkDatasetImpl(Dataset):
        """Wraps (X, y, augmentations) for the static classifier."""

        def __init__(
            self,
            keypoints: np.ndarray,  # (N, 21, 2) float32
            labels: np.ndarray,  # (N,) int64
            *,
            noise_std: float = 0.0,
            mirror_prob: float = 0.0,
            rng_seed: int = 0,
        ) -> None:
            self.keypoints = keypoints
            self.labels = labels
            self.noise_std = noise_std
            self.mirror_prob = mirror_prob
            self._rng = np.random.default_rng(rng_seed)

        def __len__(self) -> int:
            return len(self.keypoints)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.keypoints[idx].copy()
            if self.noise_std > 0:
                x += self._rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            if self.mirror_prob > 0 and self._rng.random() < self.mirror_prob:
                # Mirror horizontally — x ↔ -x. ONLY safe for mirror-symmetric
                # gestures (see TrainingConfig docstring).
                x[:, 0] = -x[:, 0]
            return torch.from_numpy(x), torch.tensor(int(self.labels[idx]), dtype=torch.long)

    return _LandmarkDatasetImpl


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_static(cfg: TrainingConfig) -> TrainingResult:
    """Train the static gesture classifier end-to-end. Returns checkpoint paths."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install with:\n" "    uv sync --extra training"
        ) from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = _pick_device(torch, cfg.device)
    device = torch.device(device_str)
    log.info("training_starting", config=asdict(cfg), device=device_str)

    # 1. Load the three pre-split parquet files (HaGRIDv2 is pre-split by user).
    X_train, y_train_str, _ = read_landmark_arrays(cfg.train_parquet)
    X_val, y_val_str, _ = read_landmark_arrays(cfg.val_parquet)
    X_test, y_test_str, _ = read_landmark_arrays(cfg.test_parquet)
    log.info(
        "dataset_loaded",
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
    )

    # Unified label map across all three splits, so class indices are
    # consistent no matter which split a label first appears in.
    label_names = sorted(
        set(y_train_str.tolist()) | set(y_val_str.tolist()) | set(y_test_str.tolist())
    )
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    def _to_int(y_str: np.ndarray) -> np.ndarray:
        return np.array([label_to_idx[s] for s in y_str], dtype=np.int64)

    y_train = _to_int(y_train_str)
    y_val = _to_int(y_val_str)
    y_test = _to_int(y_test_str)

    # Per-split class balance — a starved test class makes accuracy lie.
    log.info(
        "split_class_balance",
        train=dict(Counter(y_train_str.tolist())),
        val=dict(Counter(y_val_str.tolist())),
        test=dict(Counter(y_test_str.tolist())),
    )
    for name, X_split in (("train", X_train), ("val", X_val), ("test", X_test)):
        if len(X_split) == 0:
            raise RuntimeError(f"Split {name!r} parquet is empty — check dataset preparation.")

    # 2. Build datasets + loaders.
    DatasetCls = _build_dataset_class(torch)
    train_ds = DatasetCls(
        X_train,
        y_train,
        noise_std=cfg.noise_std,
        mirror_prob=cfg.mirror_prob,
        rng_seed=cfg.seed,
    )
    val_ds = DatasetCls(X_val, y_val)
    test_ds = DatasetCls(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device_str == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3. Build model + optimiser.
    model = StaticGestureClassifier(
        num_classes=len(label_names),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)

    n_params = count_parameters(model)
    size_mb = estimated_size_mb(n_params)
    log.info("model_built", num_parameters=n_params, size_fp32_mb=f"{size_mb:.2f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # 4. Training loop.
    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_ckpt = cfg.output_dir / "best.pt"
    epoch_t0 = time.monotonic()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            train=True,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            loss_fn,
            optimizer=None,
            device=device,
            train=False,
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        log.info(
            "epoch_complete",
            epoch=epoch,
            total=cfg.epochs,
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.3%}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.3%}",
            elapsed_s=f"{time.monotonic() - epoch_t0:.1f}",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": label_names,
                    "model_version": MODEL_VERSION,
                    "config": asdict(cfg),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_ckpt,
            )
            log.info("best_checkpoint_saved", path=str(best_ckpt), val_acc=f"{val_acc:.3%}")

    # 5. Evaluate best on test.
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = _run_epoch(
        model,
        test_loader,
        loss_fn,
        optimizer=None,
        device=device,
        train=False,
    )
    test_metrics = {"loss": test_loss, "accuracy": test_acc}

    # 6. Persist label map + history.
    label_map_path = cfg.output_dir / "label_map.json"
    label_map_path.write_text(json.dumps({i: n for i, n in enumerate(label_names)}, indent=2))

    history_path = cfg.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    log.info("training_complete", test_metrics=test_metrics, best_val_acc=best_val_acc)
    return TrainingResult(
        best_checkpoint=best_ckpt,
        label_map_path=label_map_path,
        history_path=history_path,
        test_metrics=test_metrics,
        num_parameters=n_params,
        model_size_mb=size_mb,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pick_device(torch: torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run_epoch(
    model: object,
    loader: object,
    loss_fn: object,
    optimizer: object,
    device: object,
    *,
    train: bool,
) -> tuple[float, float]:
    """One pass over a DataLoader. Returns (avg_loss, accuracy)."""
    import torch

    model = model  # type: ignore[assignment]
    if train:
        model.train()  # type: ignore[attr-defined]
    else:
        model.eval()  # type: ignore[attr-defined]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:  # type: ignore[misc]
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # type: ignore[operator]
            loss = loss_fn(logits, y)  # type: ignore[operator]

            if train:
                optimizer.zero_grad()  # type: ignore[union-attr]
                loss.backward()
                optimizer.step()  # type: ignore[union-attr]

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


__all__ = ["TrainingConfig", "TrainingResult", "train_static"]
