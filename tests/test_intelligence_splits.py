"""Tests for `sigil.intelligence.dataset.splits`."""

from __future__ import annotations

import numpy as np
import pytest

from sigil.intelligence.dataset.splits import (
    SPLIT_NAMES,
    SplitConfig,
    make_user_disjoint_splits,
    summarise_split,
)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


def test_ratios_must_be_in_open_unit_interval() -> None:
    with pytest.raises(ValueError, match="must be in"):
        SplitConfig(train_ratio=0.0, val_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError, match="must be in"):
        SplitConfig(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)


# ---------------------------------------------------------------------------
# User-disjoint property
# ---------------------------------------------------------------------------


def test_users_never_appear_in_two_splits() -> None:
    """The defining invariant: every user lives in exactly one split."""
    rng = np.random.default_rng(7)
    # 50 unique users, each appearing in ~10 samples on average.
    users = rng.choice([f"u{i:03d}" for i in range(50)], size=500)
    splits = make_user_disjoint_splits(users, SplitConfig(seed=42))

    user_to_splits: dict[str, set[str]] = {}
    for u, s in zip(users.tolist(), splits.tolist(), strict=True):
        user_to_splits.setdefault(u, set()).add(s)
    for u, sset in user_to_splits.items():
        assert len(sset) == 1, f"User {u!r} appears in {sset}"


def test_assignment_is_deterministic() -> None:
    users = np.array([f"u{i}" for i in range(100)], dtype=object)
    a = make_user_disjoint_splits(users, SplitConfig(seed=42))
    b = make_user_disjoint_splits(users, SplitConfig(seed=42))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_assignments() -> None:
    users = np.array([f"u{i}" for i in range(100)], dtype=object)
    a = make_user_disjoint_splits(users, SplitConfig(seed=0))
    b = make_user_disjoint_splits(users, SplitConfig(seed=1))
    assert not np.array_equal(a, b)


def test_adding_users_doesnt_reshuffle_existing() -> None:
    """Hashing per-user keeps existing users' splits stable when new users arrive."""
    base_users = np.array([f"u{i}" for i in range(50)], dtype=object)
    extra_users = np.concatenate(
        [
            base_users,
            np.array([f"new{i}" for i in range(20)], dtype=object),
        ]
    )

    a = make_user_disjoint_splits(base_users, SplitConfig(seed=0))
    b_all = make_user_disjoint_splits(extra_users, SplitConfig(seed=0))
    # The first 50 entries (matching base_users) should be unchanged.
    np.testing.assert_array_equal(a, b_all[: len(base_users)])


def test_approximate_ratios() -> None:
    """With many users the empirical split ratios should be close to target."""
    users = np.array([f"u{i:05d}" for i in range(5000)], dtype=object)
    splits = make_user_disjoint_splits(users, SplitConfig(seed=123))
    counts = {s: int((splits == s).sum()) for s in SPLIT_NAMES}
    total = sum(counts.values())
    train = counts["train"] / total
    val = counts["val"] / total
    test = counts["test"] / total
    # Allow ±3 percentage points slack for finite-sample variance.
    assert 0.77 <= train <= 0.83, f"train={train}"
    assert 0.07 <= val <= 0.13, f"val={val}"
    assert 0.07 <= test <= 0.13, f"test={test}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    result = make_user_disjoint_splits(np.array([], dtype=object))
    assert result.size == 0


def test_single_user_all_in_one_split() -> None:
    """One user → all 100 samples end up in one split."""
    users = np.array(["solo"] * 100, dtype=object)
    splits = make_user_disjoint_splits(users)
    assert len(set(splits.tolist())) == 1


# ---------------------------------------------------------------------------
# summarise_split
# ---------------------------------------------------------------------------


def test_summarise_split_counts_correctly() -> None:
    splits = np.array(["train", "train", "val", "test", "train"], dtype=object)
    gestures = np.array(["fist", "peace", "fist", "ok", "ok"], dtype=object)
    summary = summarise_split(splits, gestures)
    assert summary["train"]["fist"] == 1
    assert summary["train"]["peace"] == 1
    assert summary["train"]["ok"] == 1
    assert summary["val"]["fist"] == 1
    assert summary["test"]["ok"] == 1
