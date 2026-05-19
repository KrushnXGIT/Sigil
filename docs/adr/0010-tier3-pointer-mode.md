# ADR-0010: Tier 3 pointer mode (index-tip cursor + pinch click)

**Status:** Accepted
**Date:** 2026-05-17
**Supersedes:** none

## Context

Tier 1 (static gestures) and Tier 2 (dynamic swipes) cover discrete
commands but not continuous spatial input — selecting a button on
screen, opening a menu, dragging a window. The natural way to fill
this gap is a virtual cursor driven by hand position.

Research across 20+ implementations (academic papers, GitHub projects,
practitioner blogs — Li & Hsieh 2025, Deloitte HCI blog, small-cactus
handTrack, maanjk/hand-gesture-virtual-mouse, towardsdatascience
2026, ijrpr 2024, Ghorbel37, heartlessisafk, several others) shows
remarkable convergence on certain design points:

- **Landmark #8 (index fingertip) drives the cursor.** Universal.
- **Pinch click = distance(thumb tip, index tip) below threshold.**
  Universal.
- **Smoothing is essential.** Every project mentions jitter as the
  biggest UX problem; moving averages, EMA, or One Euro Filter all
  appear in the literature.
- **Frame reduction is essential.** Mapping the full camera frame
  1:1 to the screen makes reaching screen corners physically
  impossible. A 10–20% margin on each side is the convention.
- **Mode activation is essential.** If the cursor follows the
  fingertip whenever a hand is visible, it jitters constantly even
  when the user isn't trying to point.

The remaining design space is small and the trade-offs are clear,
so this ADR locks in our choices alongside the rationale.

## Decisions

### D1: Per-hand FSM with two states (NEUTRAL / ACTIVE)

```
NEUTRAL  ── index-up-only pose for POSE_ENTER_FRAMES ──> ACTIVE
   ^                                                       │
   │                                                       │
   └────── pose lost for POSE_EXIT_FRAMES ─────────────────┘
   └────── hand absent for POSE_EXIT_FRAMES ──────────────-┘
```

- **NEUTRAL**: pointer is dormant. Daemon runs static + swipe normally.
  Static gestures (fist, peace, ok, thumbs_down) and dynamic swipes
  (left, right, up, down) can fire as usual.
- **ACTIVE**: every frame, the OS cursor is moved to map(index_tip).
  A pinch fires a single left-click (cooldown enforced).
  **Static + swipe detection are suppressed while ACTIVE** — see D6.

### D2: Activation pose is index-up-only, geometric detection

A finger is "up" when `tip.y < pip.y` in image coordinates. The full
"index-up-only" pose is:

```
index_tip.y    <  index_pip.y      # index raised
middle_tip.y   >= middle_pip.y     # middle curled
ring_tip.y     >= ring_pip.y       # ring curled
pinky_tip.y    >= pinky_pip.y      # pinky curled
```

Thumb position is **deliberately not checked** because the pinch
click intentionally moves the thumb tip toward the index tip.
Constraining thumb position would force the user to choose between
pointing and clicking.

Hysteresis: enter requires `POSE_ENTER_FRAMES = 5` consecutive
positive detections (~333 ms at 15 FPS), exit requires
`POSE_EXIT_FRAMES = 5` consecutive negative ones. This filters
MediaPipe's per-landmark noise, which the fingerpose library
explicitly warns about for single-extended-finger detection.

### D3: Cursor mapping with frame reduction

Index tip image-space position (`x_img`, `y_img`) maps to screen
pixel `(x_scr, y_scr)` via:

```
x_norm = clip( (x_img - margin_x) / (1 - 2*margin_x), 0, 1 )
y_norm = clip( (y_img - margin_y) / (1 - 2*margin_y), 0, 1 )
x_scr = round(x_norm * (screen_width  - 1))
y_scr = round(y_norm * (screen_height - 1))
```

Defaults: `margin_x = margin_y = 0.15` (15% margin per side).
A user pointing at the right edge of the active region (x = 0.85)
reaches the screen's right edge without putting their hand off-camera.

### D4: Pinch click with palm-scale-normalised distance

A pinch is detected when the distance between landmarks #4 (thumb
tip) and #8 (index tip) drops below `PINCH_THRESHOLD = 0.4` in the
already-normalised keypoint space.

The keypoints are pre-divided by `palm_scale` (mean distance from
palm centroid to the 5 palm anchors — ADR-0004). This makes the
pinch distance naturally invariant to how far the hand is from the
camera — addressing the "false touch at distance" failure mode that
small-cactus/handTrack documents in detail (otherwise, when the hand
is far, all fingers look close together and pinch fires accidentally).

A pinch event is debounced two ways:
- **Latched state**: once fired, the next click requires the pinch
  to be released and re-engaged.
- **Time cooldown**: minimum 500 ms between consecutive clicks.

### D5: One Euro Filter on cursor position + dead zone

Cursor position is smoothed by a One Euro Filter (the same
implementation used in the perception layer for landmark smoothing).
Parameters: `min_cutoff = 1.0`, `beta = 0.05` — modestly aggressive
smoothing, since at 15 FPS individual frame jitter is more visible
than at 30 FPS.

After smoothing, a 3-pixel dead zone suppresses moves smaller than
that. Holding the hand still produces a still cursor, not a
twitching one.

### D6: Mutual exclusion with static + swipe via daemon orchestration

The pointer detector's `process(frame)` returns a bool indicating
whether pointer is currently ACTIVE. The daemon's frame loop checks
this return value and skips static classification and swipe detection
when True. The pointer detector runs first in each frame.

This is enforced at the daemon level rather than within each detector
because:
- Detectors should remain unaware of each other's existence.
- Adding mode coordination inside individual detectors would couple
  them.
- The daemon already orchestrates per-frame execution order, so adding
  a single early-exit branch is minimally invasive.

### D7: Mouse control via pynput, lazily imported

The `pynput.mouse.Controller` is constructed once per detector
instance. Lazy import means headless test environments don't fail
to load the module. Tests use a `FakeMouse` that satisfies the
small `MouseBackend` protocol.

Screen size is detected at construction time using:
1. Windows `ctypes` + `GetSystemMetrics(0)`/`(1)`, with
   `SetProcessDPIAware()` for HiDPI displays.
2. Tkinter `winfo_screenwidth()`/`winfo_screenheight()` as fallback.
3. Hardcoded `(1920, 1080)` as a last resort with a logged warning.

## Consequences

**Positive:**

- Adds a continuous spatial input mode without disturbing existing
  Tier 1 or Tier 2 paths. Both still work in their normal states.
- Pose-based activation prevents accidental cursor twitching when the
  hand is doing something else.
- Palm-scale-normalised pinch handles distance-from-camera variation
  for free, no extra configuration.
- Frame reduction makes screen corners reachable.
- One Euro Filter + dead zone produces a stable cursor at 15 FPS.

**Negative:**

- 15 FPS is on the low end for cursor input — perceived latency is
  ~70 ms even with no smoothing. Acceptable for menu navigation but
  not for precision drag-select.
- MediaPipe's hand model has documented difficulty with single
  extended fingers, so the activation pose can flicker on the
  borderline; hysteresis (5+5 frame requirement) mitigates this but
  doesn't eliminate it.
- Adding `image_keypoints` to `HandLandmarks` increases per-frame
  memory by 168 bytes per hand. Trivial in absolute terms but
  worth noting since the perception types are a public contract.

**Neutral:**

- Mode is exclusive — you can't gesture and point at the same time.
  This is a deliberate UX choice; supporting both simultaneously
  would require disambiguation logic the user shouldn't have to
  reason about.

## Expected accuracy

Honest range, on 15 FPS:

- **Cursor smoothness**: ~85–90% smooth tracking. Visible micro-jitter
  is reduced by smoothing + dead zone but not eliminated. Edge cases
  (hand near frame edge, fast motion, hand turning to profile) will
  occasionally produce jumps.
- **Click reliability**: ~95% successful pinch detection. False
  positives are rare because of the palm-scale normalisation.
- **Activation reliability**: ~95% after the hysteresis settles.
  Single-frame MediaPipe misclassifications are filtered. Sustained
  flicker between pose-detected and pose-not-detected would still
  cause activation delay.

Future work (post-deadline): explicit cursor-position smoothing
parameters per user; gesture-recognition-aware filtering to reduce
false-positive activation; learned pose classifier for the activation
gesture.

## References

- `src/sigil/intelligence/pointer_detector.py`
- `tests/test_pointer_detector.py`
- `src/sigil/perception/types.py` (adds `image_keypoints` field)
- `src/sigil/perception/pipeline.py` (populates `image_keypoints`)
- ADR-0004 (palm-centroid normalisation; rationale for the normalised
  keypoint space pinch detection runs in)
- ADR-0007 (perception/intelligence layer contract; this change is
  additive and backward-compatible)
- Research sources: Li & Hsieh 2025 (Eng. Proc. 108:22),
  Deloitte UK HCI blog, small-cactus/handTrack,
  maanjk/hand-gesture-virtual-mouse, towardsdatascience Jan 2026,
  ijrpr 2024 (V5I10:33814), Ghorbel37 hand-gesture-control-suite,
  heartlessisafk/smooth-virtual-mouse, andypotato/fingerpose docs.
