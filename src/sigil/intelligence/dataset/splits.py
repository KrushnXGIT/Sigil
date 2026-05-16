"""Train / val / test splitting with user-disjoint partitioning.

Naive random splits on a gesture dataset *look* fine in metrics but lie:
the same person's hand ends up in train AND test, so the model just
learns to recognise that user's hand shape. Test accuracy looks great,
production accuracy falls off a cliff.

The fix: partition by `user_id` first, then assign each user wholesale to
one split. Two examples from the same user go to the same split, always.

This module is pure-Python + numpy; deterministic given a seed. No torch
dep — the splits are produced once during dataset prep and stored.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Split = Literal["train", "val", "test"]
SPLIT_NAMES: tuple[Split, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Target proportions and reproducibility seed."""

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total} "
                f"({self.train_ratio}, {self.val_ratio}, {self.test_ratio})"
            )
        for name, ratio in [
            ("train", self.train_ratio),
            ("val", self.val_ratio),
            ("test", self.test_ratio),
        ]:
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"{name}_ratio must be in (0, 1), got {ratio}")


def _hash_to_unit(value: str, salt: int) -> float:
    """Deterministic float in [0, 1) from a string + salt. Stable across runs."""
    h = hashlib.blake2b(f"{salt}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def make_user_disjoint_splits(
    user_ids: NDArray[np.str_],
    config: SplitConfig | None = None,
) -> NDArray[np.str_]:
    """Assign each sample to a split such that users are partitioned cleanly.

    Args:
        user_ids: (N,) array of user identifiers (one per sample).
        config: split ratios + seed. Default 80/10/10 with seed 0.

    Returns:
        (N,) array of 'train' / 'val' / 'test' labels.

    Notes:
        - Determinism: same `user_ids` + same `seed` → same assignment.
          We hash per-user, so adding new users won't reshuffle existing ones.
        - Class balance: this routine does NOT balance gesture frequencies
          across splits. If a particular gesture is dominated by one user,
          that user being in test will starve test for that class. Use a
          downstream stratifier or collect more diverse users.
    """
    cfg = config or SplitConfig()
    user_ids_arr = np.asarray(user_ids, dtype=object)

    if user_ids_arr.size == 0:
        return np.array([], dtype=object)

    # Compute a [0,1) hash per *unique* user, then look up per-sample.
    unique = np.unique(user_ids_arr)
    user_to_split: dict[str, Split] = {}
    for u in unique:
        h = _hash_to_unit(str(u), cfg.seed)
        if h < cfg.train_ratio:
            user_to_split[str(u)] = "train"
        elif h < cfg.train_ratio + cfg.val_ratio:
            user_to_split[str(u)] = "val"
        else:
            user_to_split[str(u)] = "test"

    return np.array([user_to_split[str(u)] for u in user_ids_arr], dtype=object)


def summarise_split(
    split_labels: NDArray[np.str_],
    gesture_labels: NDArray[np.str_],
) -> dict[Split, dict[str, int]]:
    """Return a {split: {gesture: count}} summary for sanity-checking.

    A good split has every gesture represented in every split; if you see
    a zero, your test set is starved for that class and accuracy on it
    will be unreliable.
    """
    out: dict[Split, dict[str, int]] = {s: {} for s in SPLIT_NAMES}
    splits = np.asarray(split_labels, dtype=object)
    gestures = np.asarray(gesture_labels, dtype=object)
    for split, gesture in zip(splits, gestures, strict=True):
        out[split][gesture] = out[split].get(gesture, 0) + 1
    return out


__all__ = [
    "SPLIT_NAMES",
    "Split",
    "SplitConfig",
    "make_user_disjoint_splits",
    "summarise_split",
]
