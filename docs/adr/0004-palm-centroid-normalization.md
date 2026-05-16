# ADR-0004: Palm-centroid normalization (replacing wrist anchor)

- **Status**: Accepted
- **Date**: 2026-05-14
- **Tags**: architecture, perception, ml

## Context

Original Phase 1 normalization (per ADR-0002) translated landmarks to put
the wrist at the origin, then scaled by wrist→middle-MCP distance ("palm
span"). This was the obvious first choice — the wrist is landmark index 0
in MediaPipe's convention.

Running `sigil dataset verify-alignment` against HaGRIDv2's annotations
produced a failing alignment metric (mean error 5.8% palm-span vs. 3%
threshold), with a very informative per-landmark breakdown:

    | Landmark   | Mean error |
    |------------|-----------:|
    | wrist      |     0.108  |   ← worst by a wide margin
    | thumb_cmc  |     0.077  |   ← also a base-of-hand soft point
    | thumb_tip  |     0.064  |
    | ring_tip   |     0.063  |
    | thumb_mcp  |     0.062  |

The wrist alone disagreed by 10.8% palm span between HaGRIDv2's MediaPipe
extraction and ours. **The single point we'd chosen as our coordinate-frame
origin was the noisiest landmark MediaPipe emits** — base-of-palm soft
tissue, no sharp visual feature, version-unstable. Every other landmark
inherited that noise the moment we translated by it.

## Decision

We re-anchor the normalization frame to the **palm centroid**: the
arithmetic mean of the 5 "palm anchor" landmarks (wrist + four MCP knuckle
joints). We scale by the **palm scale**: the mean distance from the
centroid to those same 5 anchors.

In code (`sigil.perception.normalize`):

    PALM_ANCHORS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

    anchors  = landmarks[PALM_ANCHORS]
    centroid = anchors.mean(axis=0)
    centered = landmarks - centroid
    palm_scale = norm(centered[PALM_ANCHORS], axis=1).mean()
    return centered / palm_scale

Bumps `PROTOCOL_VERSION` from 2 → 3 (the coordinate frame's semantic
changed, even though the array shape didn't).

## Why this is better

1. **5-point averaging cancels per-landmark noise.** If MediaPipe wiggles
   any one of the 5 anchors by ±δ between extractions, the centroid moves
   by only ±δ/5. The wrist alone got the full δ.
2. **The 5 palm anchors are rigid relative to each other.** Knuckles and
   wrist sit on bone-anchored skin that doesn't slide when fingers flex.
   Their geometric configuration is the most stable part of the hand.
3. **Standard practice in hand-pose ML.** Most hand-keypoint
   normalization literature anchors to a palm region rather than a single
   point, for exactly these reasons.
4. **Free to fix right now.** No trained models exist yet, no parquet
   caches have been built. After data is committed, this becomes
   significantly more expensive to change.

## Alternatives Considered

- **Stay with wrist anchor and re-extract from images** to get our own
  MediaPipe's wrist landmark. Rejected: doesn't solve the wrist being
  intrinsically unstable; cost is ~119GB and hours of preprocessing.
- **Anchor to middle-MCP alone.** Cleaner than wrist (middle-MCP is more
  stable) but still a single-point anchor — gives up the noise-cancellation
  of averaging. Saved as a possible future tweak if we find a reason to
  prefer it.
- **Procrustes alignment (rotation + scale + translation to a template).**
  Strictly more principled, but introduces rotation invariance which we
  explicitly do *not* want (thumbs-up vs thumbs-down). Defer.
- **Pre-normalize bounding-box first, then landmark-normalize.** Could
  help with HaGRID-vs-runtime ROI differences, but adds a stage and a
  failure mode. The 5-point centroid achieves the same goal more directly.

## Consequences

- ✅ Verify-alignment is expected to drop well under the 3% threshold;
     the wrist-induced error is now distributed across all landmarks
     via the centroid mean, where averaging cancels most of it.
- ✅ Training data and runtime data both pass through the new
     normalization — no train/serve skew introduced.
- ⚠️ `PROTOCOL_VERSION = 3`. Any v2-trained checkpoint (none exist) would
     be incompatible.
- ⚠️ The `HandLandmarks.palm_span` property is removed in favor of
     `palm_scale` and `palm_centroid`. Internal API only — no external
     callers existed.

## References

- The failing run that motivated this: user-reported `verify-alignment`
  output showing wrist at 0.108 palm-span error.
- Builds on ADR-0002 (landmark-first classification) and ADR-0003 (2D
  landmarks). The core decisions in those still hold; this refines the
  normalization step.
- Implementation: `sigil.perception.normalize`,
  `sigil.perception.types.PALM_ANCHORS`.
