"""Tests for the alignment-inspection module.

We don't actually run MediaPipe here — that'd require a real model. We
synthesise InspectionReport / InspectionPair instances with known
properties and assert that:

  - DirectionalAnalysis correctly classifies systematic vs random
    landmarks.
  - build_html_report writes a well-formed self-contained HTML file
    (no external resources, embedded base64 images, contains the
    landmark verdict for every landmark).
  - render_pair_overlay returns PNG bytes whose magic header is valid.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
import pytest

from sigil.intelligence.dataset.inspect import (
    FINGERTIP_INDICES,
    DirectionalAnalysis,
    InspectionPair,
    InspectionReport,
    _verdict_banner,
    build_html_report,
)
from sigil.perception.types import LANDMARK_NAMES, N_COORDS, N_LANDMARKS


def _zero_delta_dir(n_pairs: int = 20) -> DirectionalAnalysis:
    """Random-only disagreement: zero mean, non-zero variance."""
    return DirectionalAnalysis(
        n_pairs=n_pairs,
        per_landmark_mean_dxy=np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float64),
        per_landmark_var_dxy=np.full((N_LANDMARKS, N_COORDS), 0.01, dtype=np.float64),
    )


def _systematic_dir(
    n_pairs: int = 20, mean: float = 0.05, var: float = 0.0001
) -> DirectionalAnalysis:
    """Strong directional bias dominating tiny variance."""
    return DirectionalAnalysis(
        n_pairs=n_pairs,
        per_landmark_mean_dxy=np.full((N_LANDMARKS, N_COORDS), mean, dtype=np.float64),
        per_landmark_var_dxy=np.full((N_LANDMARKS, N_COORDS), var, dtype=np.float64),
    )


# --- DirectionalAnalysis classifier ------------------------------------------


class TestDirectionalClassify:
    def test_zero_mean_high_variance_is_random(self) -> None:
        d = _zero_delta_dir()
        for i in range(N_LANDMARKS):
            assert d.classify(i) == "random"

    def test_high_mean_low_variance_is_systematic(self) -> None:
        d = _systematic_dir()
        for i in range(N_LANDMARKS):
            assert d.classify(i) == "systematic"

    def test_balanced_is_mixed(self) -> None:
        # |mean| ~ noise: ratio close to 1.0 → mixed
        d = DirectionalAnalysis(
            n_pairs=20,
            per_landmark_mean_dxy=np.full((N_LANDMARKS, N_COORDS), 0.05, dtype=np.float64),
            per_landmark_var_dxy=np.full((N_LANDMARKS, N_COORDS), 0.0025, dtype=np.float64),
        )
        # |mean| = 0.05 * sqrt(2) ~ 0.0707
        # noise = sqrt(0.0025 + 0.0025) = 0.0707
        # ratio = 1.0 → "mixed"
        for i in range(N_LANDMARKS):
            assert d.classify(i) == "mixed"

    def test_zero_noise_with_nonzero_mean_is_systematic(self) -> None:
        d = DirectionalAnalysis(
            n_pairs=1,
            per_landmark_mean_dxy=np.full((N_LANDMARKS, N_COORDS), 0.05, dtype=np.float64),
            per_landmark_var_dxy=np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float64),
        )
        for i in range(N_LANDMARKS):
            assert d.classify(i) == "systematic"
            assert d.systematic_ratio(i) == float("inf")

    def test_completely_zero_signal(self) -> None:
        # No mean AND no noise: ratio is 0 (not infinity).
        d = DirectionalAnalysis(
            n_pairs=1,
            per_landmark_mean_dxy=np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float64),
            per_landmark_var_dxy=np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float64),
        )
        for i in range(N_LANDMARKS):
            assert d.classify(i) == "random"


# --- verdict banner ----------------------------------------------------------


class TestVerdictBanner:
    def test_all_systematic_says_systematic(self) -> None:
        banner = _verdict_banner(_systematic_dir())
        assert "SYSTEMATIC" in banner
        assert "verdict-systematic" in banner

    def test_all_random_says_random(self) -> None:
        banner = _verdict_banner(_zero_delta_dir())
        assert "RANDOM" in banner
        assert "verdict-random" in banner

    def test_mixed_says_mixed(self) -> None:
        # Three fingertips systematic, two random: should still trip systematic
        # (we set the bar at >=3 systematic for a clear systematic verdict).
        mean = np.zeros((N_LANDMARKS, N_COORDS))
        var = np.full((N_LANDMARKS, N_COORDS), 0.01)
        # Make first 3 fingertips systematic
        for i in list(FINGERTIP_INDICES)[:3]:
            mean[i] = 0.5
            var[i] = 0.0001
        d = DirectionalAnalysis(n_pairs=20, per_landmark_mean_dxy=mean, per_landmark_var_dxy=var)
        banner = _verdict_banner(d)
        assert "SYSTEMATIC" in banner

    def test_two_systematic_two_random_says_mixed(self) -> None:
        # 2 systematic, 2 random, 1 mixed → no clear majority → MIXED
        mean = np.zeros((N_LANDMARKS, N_COORDS))
        var = np.full((N_LANDMARKS, N_COORDS), 0.01)
        for i in list(FINGERTIP_INDICES)[:2]:
            mean[i] = 0.5
            var[i] = 0.0001
        # One mixed
        mean[list(FINGERTIP_INDICES)[2]] = 0.05
        var[list(FINGERTIP_INDICES)[2]] = 0.0025
        d = DirectionalAnalysis(n_pairs=20, per_landmark_mean_dxy=mean, per_landmark_var_dxy=var)
        banner = _verdict_banner(d)
        assert "MIXED" in banner


# --- HTML report builder -----------------------------------------------------


def _fake_pair(tmp_path: Path, image_key: str = "fake_0001") -> InspectionPair:
    """Build a synthetic InspectionPair with a tiny on-disk image."""
    import cv2  # type: ignore[import-untyped]

    h, w = 240, 320
    bgr = np.full((h, w, 3), 96, dtype=np.uint8)
    img_path = tmp_path / f"{image_key}.png"
    cv2.imwrite(str(img_path), bgr)

    ours_raw = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (N_LANDMARKS, 1))
    theirs_raw = ours_raw.copy() + 0.02  # offset
    ours_norm = np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)
    theirs_norm = np.full((N_LANDMARKS, N_COORDS), 0.05, dtype=np.float32)
    dist = np.linalg.norm(ours_norm - theirs_norm, axis=1)

    return InspectionPair(
        image_key=image_key,
        gesture="fist",
        image_path=img_path,
        image_shape=(h, w),
        ours_raw=ours_raw,
        theirs_raw=theirs_raw,
        ours_norm=ours_norm,
        theirs_norm=theirs_norm,
        per_landmark_dist=dist,
    )


class TestHtmlReport:
    def test_writes_self_contained_html(self, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        pairs = (_fake_pair(tmp_path, "f_1"), _fake_pair(tmp_path, "f_2"))
        report = InspectionReport(
            total_pairs=2,
            total_images=2,
            top_pairs=pairs,
            directional=_zero_delta_dir(n_pairs=2),
            rank_by="fingertip_mean",
            per_gesture_counts={"fist": 2},
        )
        out = tmp_path / "report.html"
        build_html_report(report, out)

        assert out.is_file()
        html = out.read_text(encoding="utf-8")

        # Self-contained: no external script/style refs.
        assert "<script src=" not in html
        assert "<link " not in html
        # Both image overlays are embedded as base64 data URIs.
        b64_images = re.findall(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"', html)
        assert len(b64_images) == 2
        # And the PNG header bytes survive a base64 round-trip.
        for b64 in b64_images:
            png = base64.b64decode(b64)
            assert png[:8] == b"\x89PNG\r\n\x1a\n"

        # Every landmark name appears in the table.
        for name in LANDMARK_NAMES:
            assert name in html

        # Verdict banner present.
        assert "RANDOM" in html  # _zero_delta_dir → random verdict

        # Both image keys appear.
        assert "f_1" in html
        assert "f_2" in html

    def test_systematic_verdict_in_html(self, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        pairs = (_fake_pair(tmp_path, "sys_1"),)
        report = InspectionReport(
            total_pairs=1,
            total_images=1,
            top_pairs=pairs,
            directional=_systematic_dir(n_pairs=1),
            rank_by="fingertip_mean",
            per_gesture_counts={"fist": 1},
        )
        out = tmp_path / "sys_report.html"
        build_html_report(report, out)
        assert "SYSTEMATIC" in out.read_text(encoding="utf-8")


# --- InspectionPair convenience properties -----------------------------------


class TestInspectionPair:
    def test_max_mean_fingertip(self, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        pair = _fake_pair(tmp_path)
        # We seeded uniform distance 0.05*sqrt(2) ~ 0.0707 across all 21
        # landmarks, so max == mean == fingertip_mean.
        assert pair.max_error == pytest.approx(0.0707, abs=1e-3)
        assert pair.mean_error == pytest.approx(0.0707, abs=1e-3)
        assert pair.fingertip_mean_error == pytest.approx(0.0707, abs=1e-3)
