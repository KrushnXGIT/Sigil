# ADR-0009: Motion-state FSM for swipe detection

**Status:** Accepted
**Date:** 2026-05-17 (supersedes the swipe-detection portion of ADR-0008)

## Context

ADR-0008 shipped Tier 2 dynamic gestures as a continuous-sliding-window
detector. Real-world testing on the user's machine revealed three
compounding failures:

1. **The buffer reset on hand-missing frames.** MediaPipe drops the
   hand for 1–2 frames during the fast portion of a swipe.
   `if not frame.hands: self._buffer.clear()` fired on every drop,
   resetting state mid-swipe. The detector never accumulated a full
   window. Logs showed zero `swipe_detected` AND zero
   `swipe_rejected_*` events during real swipes — the decision point
   was never reached.

2. **Frame-count windows assumed 14 FPS.** Sized to 6–8 frames during
   patch development. The user's actual rate is 27 FPS, so the window
   covered ~220 ms — too short for a full deliberate swipe.

3. **Sustain-for-debounce was a hack.** The previous design held a
   detected swipe for `DEBOUNCE_FRAMES` synthetic frames so the
   interpreter's debounce would see agreement. This leaked
   interpreter internals into the detector and made the state machine
   harder to reason about.

A separate question: ADR-0008 ran the swipe detector alongside the
static classifier with a "swipe wins" priority rule. This composed
correctly but the priority rule was glue — there was no formal
mutual-exclusion invariant. With the FSM, the invariant becomes
explicit (the hand is in STATIONARY when static gestures are
classified, MOVING when swipes are captured, POST_SWIPE when the
detector is emitting).

## Decisions

### D1: Per-hand motion-state FSM with three states

Replace the continuous-sliding-window detector with a finite state
machine:

```
STATIONARY  ── high velocity for N frames ──> MOVING
   ^                                            │
   │           ┌──────────────────────────────┘
   │           ├── low velocity for N frames + valid trajectory ──> POST_SWIPE
   │           ├── low velocity for N frames + invalid trajectory ─┐
   │           └── elapsed > MAX_SWIPE_DURATION ──────────────────┐│
   │                                                              ││
   ├──────── POST_SWIPE window expires ────────────────────────────┘
   ├──────── hand-missing > GRACE_FRAMES ───────────────────────────┘
```

- **STATIONARY**: the hand is at rest. Static classifier should
  produce reliable results during this state. Detector watches for
  motion onset.
- **MOVING**: motion in progress. The detector accumulates trajectory
  data. Static classification is unreliable during this state.
- **POST_SWIPE**: a valid swipe was detected. The detector emits the
  same swipe event each frame for `POST_SWIPE_DURATION_NS` (= 400 ms,
  ≈ 11 frames at 27 FPS, well above the interpreter's
  `DEBOUNCE_FRAMES = 3` requirement). Doubles as a cooldown against
  the natural return motion firing the inverse swipe.

### D2: Time-based thresholds (milliseconds, not frames)

All velocity thresholds are in **normalized image units per second**
and computed from actual timestamp deltas. All duration windows are
in **nanoseconds**. Frame counts only appear for streak detection
(consecutive high-velocity frames before triggering MOVING), where
the frame-count interpretation is correct anyway.

This makes the detector FPS-independent. Tests parametrize across
14 / 20 / 27 / 45 FPS and the same thresholds work at all rates.

### D3: Grace period for hand-missing frames

When `frame.hands` is empty, the detector does **not** immediately
reset. Instead it tolerates up to `GRACE_FRAMES` (= 4) consecutive
missing frames. At 27 FPS that's ~150 ms; at 14 FPS ~285 ms. Both
longer than typical MediaPipe dropout streaks during fast motion,
shorter than a full swipe.

If the dropout exceeds the grace, the detector resets to STATIONARY.
The post-swipe sustain emission also tolerates missing hands — once
a swipe is detected, the event continues to fire for the cooldown
window even if MediaPipe briefly loses the hand.

### D4: Three-gate trajectory validation, each logged

When MOVING ends, the trajectory is checked against three
independent gates:

1. **Displacement gate**: `max(|dx|, |dy|) >= 0.15` (15% of frame).
   Filters small motion.
2. **Peak velocity gate**: `peak_v >= 0.6` norm/s. Filters slow drift
   that accumulates enough displacement to look like a swipe.
3. **Straightness gate**: `net_distance / path_length >= 0.65`.
   Filters zigzag (waving, gesticulation) from intentional swipes.

Each rejection emits a distinct log event
(`swipe_rejected_displacement`, `swipe_rejected_peak_velocity`,
`swipe_rejected_straightness`, `swipe_rejected_axis_gating`) with
the actual values. Future failures will be diagnosable from the
structlog stream in one read.

### D5: Threshold values informed by published literature

Source-by-source provenance for each threshold:

| Threshold | Value | Source |
|---|---|---|
| Min displacement | 0.15 | Deloitte HCI engineering blog (rule-based magnitude threshold convention) |
| Min peak velocity | 0.6 norm/s | Derived: typical deliberate swipe traverses ≥ 0.3 of frame in ≤ 0.5 s ≈ 0.6 norm/s |
| Min straightness | 0.65 | Empirical from synthetic tests; rejects waving (~0.3) and zigzag (~0.4) |
| Dominant axis ratio | 2.0 | Industry convention (most swipe libraries use 2–3×) |
| Max swipe duration | 1200 ms | Li & Hsieh 2025: 40 frames at 30 FPS ≈ 1.3 s; rounded down |
| Post-swipe cooldown | 400 ms | Half of touch-UI default (npm @1ohooks/use-swipe: 500 ms) |
| Grace frames | 4 | Longer than typical MediaPipe dropout (~3 frames per Li & Hsieh) |

### D6: `flip_horizontal` defaults to False

The previous detector defaulted to `True` based on a misread of the
perception pipeline's mirror behaviour. The user's pipeline applies
mirror flip in capture (before MediaPipe), so MediaPipe already sees
mirror-mode coordinates. With `flip_horizontal=False`, x increasing
in the image corresponds to the user's "swipe right" intent.

If a user's setup applies mirroring differently, they pass
`flip_horizontal=True`. Documented in the swipe_detector docstring.

## Consequences

**Positive:**

- Swipes now actually fire on real motion. The smoke tests show all
  four directions detected at 27 FPS with confidence ≥ 0.7.
- Static classification path is untouched (99.65% accuracy preserved).
- FPS independence: works at 14, 27, 45 FPS without retuning.
- Future failures are diagnosable from logs alone — every rejection
  emits a structured event with the failing threshold + actual value.
- Mutual exclusion between static and dynamic gestures becomes a
  formal property of the FSM rather than a glue rule.

**Negative:**

- Three states is more code than the previous one-state continuous
  scanner. ~400 lines vs ~270.
- The motion-onset detection requires `MOTION_ONSET_FRAMES`
  consecutive high-velocity frames, so the detector can't react to
  single-frame motion. This is intentional (filters noise) but
  means a hyper-fast flick under ~100 ms might miss.

**Neutral:**

- Tracks the highest-confidence hand each frame; can produce
  cross-hand identity confusion if both hands swipe simultaneously.
  Tier 3 multi-hand work will replace with spatial-continuity
  tracking.

**Replaces:**

- The whole of ADR-0008 Decision D3 ("sustain-for-debounce"). The new
  POST_SWIPE state achieves the same goal as a natural consequence
  of the FSM rather than as a hack.

## Expected accuracy

Honestly stated: on deliberate swipes with adequate lighting and a
fully-visible hand, expected detection rate is **85–95%** with a
**false-positive rate below 5% per minute** of natural hand activity.

Unlike the static classifier (99.65% test accuracy, learned from
36,000 HaGRID images), this detector is rule-based with zero labeled
training samples. It will not match the static path's accuracy.
Future work: record 200–500 labeled swipes per direction, train a
small temporal classifier (à la Li & Hsieh 2025), expect 95%+.

## References

- `src/sigil/intelligence/swipe_detector.py` (replacement)
- `tests/test_swipe_detector.py` (replacement)
- ADR-0008 (superseded portion: D3 "sustain-for-debounce" hack)
- Li & Hsieh 2025, "Dynamic Hand Gesture Recognition Using MediaPipe
  and Transformer", Eng. Proc. 108(1):22
- Deloitte UK Tech Blog, "How to Control Desktop Apps and Websites
  using Hand Gestures" (rule-based reference)
- Sensors 2022, radar-based gesture trigger-algorithm approach
