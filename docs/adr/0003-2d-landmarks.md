# ADR-0003: 2D landmarks `(21, 2)` end-to-end

- **Status**: Accepted
- **Date**: 2026-05-14
- **Tags**: architecture, ml, data, perception

## Context

Phase 2a needs training data. The realistic source is HaGRIDv2 — but the
full image dataset is 1.5TB (119GB even for the 512px subset). Downloading
and running MediaPipe over ~1M images ourselves is impractical for this
project's constraints.

HaGRIDv2 ships an `annotations.zip` (JSON, a few GB) that **already contains
MediaPipe-extracted hand landmarks** for ~1.3M hands. This eliminates the
download-and-extract problem entirely.

The catch: HaGRIDv2's pre-computed landmarks are **2D `[x, y]` pairs**, not
3D. Our codebase was built around `(21, 3)` — MediaPipe's live output
includes a `z` (depth) coordinate. This creates a train/serve mismatch:

  - Training data (HaGRIDv2 annotations): `(21, 2)` — x, y
  - Runtime (live MediaPipe HandLandmarker): `(21, 3)` — x, y, z

We must resolve the mismatch before building the classifier on top of it.

## Decision

**We use 2D landmarks `(21, 2)` throughout the entire system** — perception,
the dataset layer, the classifier input, and all data contracts. At runtime,
the perception layer takes MediaPipe's output and **discards the `z`
coordinate** immediately, before smoothing or normalization.

`N_COORDS = 2` becomes a named constant so the shape is `(N_LANDMARKS,
N_COORDS)` everywhere — a single point of change if this is ever revisited.

## Alternatives Considered

- **Pad HaGRID landmarks with `z = 0`, keep runtime `z`** — rejected. The
  classifier would learn "z is always 0" from training data, then real
  nonzero `z` at inference would be out-of-distribution. Textbook train/serve
  skew.
- **Re-extract HaGRID landmarks ourselves to get real `z`** — rejected for
  Phase 2. Requires the 119GB image download plus hours of MediaPipe
  preprocessing. Defeats the entire reason the annotation file is useful.
  May revisit if the alignment-verification step (below) shows a problem.
- **Find a 3D-landmark gesture dataset elsewhere** — investigated. Nothing at
  HaGRID's scale and quality has pre-extracted 3D MediaPipe landmarks with
  our exact Tasks-API version. The candidates were either tiny, older
  (HaGRID v1), unverifiable student projects, or stored landmark *images*
  rather than coordinates.

## Why this is robust, not a compromise

The `z` we're dropping is the *least* reliable part of MediaPipe's output:

- It's a **monocular depth estimate** — inferred, not measured.
- It's the coordinate that **varies most between MediaPipe versions**, so
  it's the biggest train/serve-skew risk even if we *did* have 3D training
  data from a different MediaPipe version.
- HaGRIDv2's own baseline classifiers reach **98%+ F1** on features derived
  from this exact 2D annotation set.

By going 2D we keep the stable, reliable signal (x, y geometry) and drop the
noisy one. As a bonus, every perception stage moves slightly less data.

## Safety net: alignment verification

To convert "2D is fine, trust us" into a measured fact, the dataset pipeline
includes a `verify-alignment` step (see `sigil.intelligence.dataset.verify`):

1. Pull a *small* sample of HaGRID images (hundreds, not millions).
2. Run *our* exact MediaPipe HandLandmarker over them.
3. Compare our `(x, y)` landmarks against HaGRIDv2's annotation landmarks;
   report mean per-landmark error as a fraction of palm span.

If the error is small (target < ~2-3% of palm span), HaGRIDv2's annotations
match our runtime closely enough to train on directly. If it's large, we
have hard evidence — *before* a wasted training run — that we need to
re-extract.

## Consequences

- ✅ Phase 2a needs only the JSON annotation download, not 119GB of images.
- ✅ Train and serve both operate on identical `(21, 2)` normalized data.
- ✅ Every perception stage processes 1/3 less coordinate data.
- ✅ A measurable alignment check guards the "2D is fine" assumption.
- ⚠️ We permanently give up depth information. Gestures that are *only*
     distinguishable via depth (none in our Tier 1 vocabulary) would be
     unrecognisable. Tier 1/2/3 vocab was already chosen to be geometrically
     distinct in the image plane, so this is not expected to bite.
- ⚠️ HaGRIDv2's landmarks were extracted with *some* MediaPipe version that
     may differ from ours. The alignment-verification step exists precisely
     to quantify this; it's a known unknown with a mitigation, not a blind
     spot.
- ❌ If we later add a gesture that genuinely needs depth, this decision
     must be revisited — and that would mean re-extraction (the 119GB path)
     or a depth-capable capture device.

## References

- HaGRIDv2 annotation format: the project README (uploaded), §Annotations
- Supersedes the `(21, 3)` assumption in ADR-0002 (landmark-first
  perception). ADR-0002's core decision — classify on landmarks, not pixels
  — still holds; only the coordinate count changes.
- `sigil.intelligence.dataset.verify` — the alignment-verification
  implementation.
