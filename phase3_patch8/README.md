# Patch 8 — Patient dropout handling

One file. `src/sigil/intelligence/swipe_detector.py`.

## What I now understand from your log

Patch 7 fixed the centroid issue — coordinates are real now. **The
real problem is MediaPipe.** It silently drops your hand for 5+
consecutive frames during every swipe. At 15 FPS that's ~350 ms of
total blackout DURING the swipe.

My grace period was 4 frames. So we exceeded grace and panic-reset
on every swipe, throwing away the trajectory we'd captured up to that
point.

Look at the one swipe that actually completed in your log:

```
19:15:14.673  swipe_motion_onset velocity=0.83
19:15:15.068  swipe_rejected_displacement abs_dx=0.042 abs_dy=0.093
```

Only 0.042 horizontal displacement — that's the tiny slice MediaPipe
could still see at the start before it lost the hand. The actual
swipe was probably 0.4+ but most of it was invisible.

## What this patch changes

In MOVING state, the detector now **waits patiently through
dropouts** instead of resetting. The buffer was already preserved
across resets, so what was needed was to NOT reset.

When `max_swipe_duration` (now 1.5 s) eventually fires while the hand
is still missing, we analyze whatever trajectory we captured before
the dropout. New log event: `swipe_dropout_timeout_analyzing`.

If the hand comes back mid-MOVING, accumulation continues naturally
across the gap. The velocity between pre-dropout and post-dropout
samples is computed from actual timestamp deltas, so it's correct.

STATIONARY state still uses the normal 4-frame grace period
(unchanged).

## Apply

```powershell
Copy-Item -Recurse -Force patch\src .
uv run sigil daemon run --overlay
```

## What to look for

After a swipe, you should see one of:

```
swipe_motion_onset      velocity=1.21
swipe_detected          gesture=swipe_right  confidence=0.91  elapsed_ms=420
```

OR (if MediaPipe dropped the hand and we hit the duration cap):

```
swipe_motion_onset                 velocity=0.83
swipe_dropout_timeout_analyzing    missing_frames=18  elapsed_ms=1510
swipe_detected_after_dropout       gesture=swipe_right  confidence=0.84
```

Either path ends with an `action_dispatched` line as the verb fires.

If you STILL see `swipe_hand_lost_resetting` from MOVING state, that
would mean the new code isn't deployed — double-check the file copied.
(`swipe_hand_lost_resetting` from STATIONARY is fine, that's the
normal grace path.)

## What to send back

5-15 log lines around a few swipe attempts. Hopefully with
`swipe_detected` or `swipe_detected_after_dropout` lines in there.

This is the real fix for the actual problem in your environment.
The grace period was the wrong knob — the right one was patience
during MOVING.
