# ADR-0008: Tier 2 dynamic gestures — rule-based swipe detector

**Status:** Accepted
**Date:** 2026-05-16

## Context

The vocabulary spec defines four Tier 2 gestures — swipe left, right,
up, down — that map to media navigation and volume verbs. Unlike
Tier 1 (static poses classified from a single frame), these are
motion-based: a swipe is defined by where the hand travels over a
short window of time.

Two design questions had to be answered before any code landed.

**Q1: Learned or rule-based?** A temporal classifier (LSTM/GRU/1D-CNN
over a sliding window of normalised landmarks) would be the
ML-textbook answer. The downside is heavy: no Tier 2 labelled
dataset exists, HaGRID is static-only, and a new training round is a
multi-hour commitment with its own risk surface.

**Q2: How does swipe detection coexist with the static classifier?**
A moving hand makes static classification unreliable (the model was
trained on stationary poses), so a fist held mid-swipe might
spuriously fire play/pause alongside the swipe. The system needs a
mutual-exclusion rule.

## Decisions

### D1: Rule-based trajectory detector

`SwipeDetector` tracks the palm centroid over a 0.5-second sliding
window and classifies a swipe when three geometric conditions hold:

1. Net displacement exceeds a threshold (the hand actually moved).
2. The dominant axis exceeds the perpendicular by ≥ 2.5× (the motion
   is clearly along one axis, not diagonal).
3. Straightness — net distance divided by total path length — exceeds
   0.6 (the path is roughly direct, not zigzag).

Why rule-based: the hard part of swipe detection isn't *what kind of
motion* but *whether the motion is deliberate*. That's naturally
expressed as geometry. The thresholds tune trivially against real
hand motion; a learned model would buy us no accuracy and cost us a
training round, more parameters, and the same threshold-tuning
problem in confidence-space.

Considered alternative: train a small 1D-CNN over normalised
landmark sequences. Rejected for the deadline; can be revisited
in a future phase if Tier 3 needs sequence-aware classification
anyway.

### D2: Swipes take priority over static classification

The daemon's per-frame loop now runs the swipe detector *first*. If
it returns events, those override the static classifier's output for
that frame. The rationale: a moving hand is either a swipe or noise,
not a held pose.

In code:

```python
swipe_events = self.swipe_detector.process(frame)
events = swipe_events if swipe_events else self.classifier.classify(frame)
```

This keeps the interpreter unchanged — it consumes the same
`tuple[GestureEvent, ...]` contract regardless of where the events
originated.

### D3: Sustain-for-debounce

The interpreter requires `DEBOUNCE_FRAMES` (=3) consecutive identical
gesture events before firing. The swipe detector, in contrast,
detects a swipe *once* per gesture — one frame, one detection.
Without further work, the interpreter's debounce buffer would see
`[swipe_right, no_gesture, no_gesture]` and never fire.

Rather than special-case swipes in the interpreter, the detector
sustains its output for `DEFAULT_SUSTAIN_FRAMES` (=`DEBOUNCE_FRAMES`)
frames after detection. The interpreter sees three identical events
in a row, fires once, then never sees the same gesture again because
the detector enters its 800ms cooldown.

This keeps the interpreter ignorant of where events come from — the
"sustained-for-N-frames" semantics live entirely in the detector.

### D4: Cooldown prevents return-motion false positives

A user who swipes right will naturally bring their hand back left
afterwards. Without protection, that return motion would register as
`swipe_left`. The detector enters an 800ms cooldown after each
sustain finishes; during cooldown, all motion is ignored.

### D5: `flip_horizontal` for camera coordinate semantics

Mirror-mode cameras (the desktop default) flip the image
horizontally before MediaPipe sees it. The user's "swipe right"
intention corresponds to the centroid x *decreasing* in image
coordinates. The detector defaults to `flip_horizontal=True` to map
to user intent; users with non-mirrored camera setups can pass
`flip_horizontal=False`.

This is a per-user setting and intentionally not auto-detected — the
mirror-vs-non-mirror choice is a hardware/driver setting we can't
reliably infer.

## Consequences

**Positive:**

- Tier 2 ships with no retraining. The 99.65% static classifier is
  unchanged; the swipe detector is purely additive.
- The interpreter is unchanged. The same FSM handles Tier 1 and
  Tier 2 events identically.
- Geometric thresholds are tunable per-user without retraining.
- Swipes and static gestures coexist via priority rather than fusion,
  so there's no ambiguity about which fires.

**Negative:**

- Rule-based means no per-user calibration of swipe sensitivity. A
  user who gestures broadly vs. tightly may need different
  `min_displacement` values; we don't expose this in the CLI yet.
- The 0.8s cooldown means a user can swipe at most ~75 times/minute,
  which is plenty for media control but might bite an edge use case.

**Neutral:**

- The detector is single-hand-aware (uses the highest-confidence
  hand). Two-handed dynamic compounds will need Tier 3 work that
  extends this with hand-identity tracking.

## References

- `src/sigil/intelligence/swipe_detector.py`
- `src/sigil/intelligence/interpreter.py` (extended `DEFAULT_TIER1_MAPPING`)
- `src/sigil/executor/verbs.py` (added 4 verbs)
- `src/sigil/executor/registry.py` (registered 4 verbs)
- `src/sigil/daemon/runtime.py` (wired SwipeDetector, priority rule)
- `src/sigil/ui/state.py` (caption strings for new actions)
- `docs/gesture-vocabulary-spec.md` (Tier 2 definitions)
- ADR-0004 (palm-centroid normalisation; the centroid the detector tracks)
- ADR-0005 (multi-hand classifier contract; same shape as swipe output)
- ADR-0006 (executor verb registry, dispatcher isolation)
