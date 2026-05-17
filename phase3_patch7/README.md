# Patch 7 — Root cause fix: image-space centroid preservation

**The real bug, found.** All those zero-centroid heartbeats weren't a
tuning problem or a math bug. The perception pipeline applies palm-
centroid normalisation (ADR-0004) BEFORE exposing landmarks, which
makes `HandLandmarks.palm_centroid` always (0, 0) by construction —
that's literally the definition of palm-centroid normalisation. The
static classifier doesn't care (it only uses the shape, not the
position). But the swipe detector needs position. It was tracking
the origin not moving.

## The fix

Add a new field `image_palm_centroid` to `HandLandmarks` that
preserves the raw image-space centroid. The pipeline computes it
from `smoothed` (pre-normalisation) keypoints and passes it through.
The swipe detector reads this instead of the normalised property.

## What's in this patch

```
src/sigil/perception/types.py            <- adds image_palm_centroid field
src/sigil/perception/pipeline.py         <- computes & stores it
src/sigil/intelligence/swipe_detector.py <- reads it (with fallback to property)
```

**Three files. Static classifier path completely untouched.** It still
sees the same normalised keypoints it always has.

## Why this didn't show up in tests

My tests construct `HandLandmarks` synthetically with raw coords as
keypoints (not normalised). `palm_centroid` reads those raw coords
and gives sensible values. So tests passed. In production, the
pipeline normalises before exposing → palm_centroid always (0, 0).
A real integration test (with the pipeline in the loop) would have
caught this immediately.

Lesson for the report's future-work section: integration tests with
the pipeline in the loop should run alongside unit tests.

## Apply

```powershell
Copy-Item -Recurse -Force patch\src .
```

Three files replaced. No edits needed to daemon/runtime.py, no
edits to interpreter.py, no edits to anywhere else. Your local
fixes in those files are safe.

## Run

```powershell
uv run sigil daemon run --overlay
```

What you should see in the heartbeat:

```
swipe_heartbeat  state=stationary  current_v=0.05
                 latest_cxy=(0.527,0.412)  prev_cxy=(0.485,0.398)
```

`latest_cxy` should now show real coordinates (something like
0.3-0.7 for a hand in the middle of the frame) AND they should
change frame to frame as you move your hand.

Try a swipe. You should see:

```
swipe_motion_onset  velocity=0.85  ...
swipe_detected  gesture=swipe_right  confidence=0.91  elapsed_ms=420
```

Plus the corresponding `action_dispatched` line as the verb fires.

## Existing tests

The `test_swipe_detector.py` tests still pass — they construct
HandLandmarks without `image_palm_centroid`, the swipe detector
falls back to the property, and the property works correctly in
tests because the keypoints in tests aren't normalised.

If you want to verify:

```powershell
uv run pytest tests/test_swipe_detector.py tests/test_classifier_runtime.py -v
```

Should still pass clean.

## What to send back

Just a few heartbeat lines showing actual centroid values, plus
hopefully a `swipe_detected` event! If anything looks off (centroids
still 0, or detected but wrong direction), the heartbeat will tell
us exactly what.
