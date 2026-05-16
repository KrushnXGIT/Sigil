# ADR-0006: Executor verb registry, dispatcher isolation, daemon loop

**Status:** Accepted
**Date:** 2026-05-16

## Context

Patch 2 of Phase 3 adds the executor and the daemon that wires the
intelligence layer to the OS. Patch 1 left an open question: who
actually runs an `ActionDispatch`, and what protects the daemon from
a single misbehaving action tearing down the whole loop?

Four sub-decisions had to land together:

1. How verbs (the OS-side operations like "press play/pause") are
   registered and looked up.
2. How exceptions from a verb are isolated so they don't crash the
   daemon.
3. How the four layers — perception, classifier runtime,
   interpreter, executor — are wired into a single loop.
4. How the interpreter wakes up before the wake word lands in
   Patch 3.

## Decisions

### D1: Verb registry as a whitelist

`DEFAULT_REGISTRY` is a `dict[str, Verb]` mapping the action strings
the interpreter emits (`"media.play_pause"`, `"media.mute"`,
`"window.maximize"`, `"system.undo"`) to no-argument callables.

The registry is a whitelist, not a dispatcher with name resolution.
An action string the registry doesn't know about is logged at
WARNING and silently dropped. This makes typos in action names
loud-but-recoverable: the interpreter keeps running, but the
dispatch obviously didn't fire.

Considered alternative: dynamic import based on action string
(`"media.play_pause"` → `from sigil.executor.media import play_pause`).
Rejected because (a) it makes the set of reachable actions
non-introspectable, (b) it would let a malicious or buggy mapping
import arbitrary modules, and (c) Tier 1's vocabulary is small
enough that explicit registration is no burden.

### D2: Each verb returns `bool`, dispatcher catches exceptions

Verb contract: `Callable[[], bool]`. Return `True` when the OS
action was attempted successfully; `False` when the verb couldn't
even attempt it (missing dependency, no foreground window, etc.).
Exceptions propagate to the dispatcher.

The dispatcher catches all exceptions from verbs and counts them as
failures. A misbehaving verb can degrade the user experience but
cannot crash the daemon. Without this isolation, a single failing
`window.maximize` call (e.g. on a user's machine where some
unexpected windowing edge case raises) would kill the entire
session.

The dispatcher also tracks `last_dispatched_ns` per action, using
`None` as the "never dispatched" sentinel — not `0`. This is a
direct consequence of the cooldown bug in Patch 1, where
`dict.get(key, 0)` made the first dispatch indistinguishable from
"dispatched at time zero" and got eaten by the cooldown check.
The lesson: when a default sentinel could be a legal value of the
type, use `None` and explicit `is None` checks.

### D3: Daemon loop is single-threaded and synchronous

```
camera → perception → classifier → interpreter → dispatcher → OS
```

Every step runs in the main thread, in order, per frame. The total
budget at 30 FPS is ≈ 33 ms; profiling from Patch 1 work suggests:

  - MediaPipe extraction: ~50 ms on the user's 2-core CPU
    (already known to be the bottleneck)
  - Capture + normalise + smoothing: < 5 ms
  - ONNX inference (140K params on CPU): < 5 ms
  - Interpreter (pure logic): microseconds
  - Verb dispatch: variable but bounded (key sends are sub-ms)

So the loop is bound by MediaPipe, not by anything we added. A
threaded architecture would buy us nothing on this workload and
introduces synchronisation hazards we don't need.

Per-frame exception isolation: each frame's processing is wrapped
in `try/except` so a transient failure (e.g. a 1-in-10000 MediaPipe
hiccup) is logged and the loop continues. Without this, a fragile
mid-demo failure crashes the daemon.

### D4: Auto-activate as the wake-word placeholder

Until Patch 3 lands, the interpreter has no way to wake itself out
of `DORMANT`. The daemon provides two placeholder behaviours:

  - **Auto-activate on startup:** the daemon calls
    `interpreter.activate()` immediately on startup, putting the FSM
    into `LISTENING`. The 60-second LISTENING timeout still applies.
  - **Auto-reactivate on gesture:** when the interpreter is in
    `DORMANT` and the classifier emits at least one event, the
    daemon calls `activate()` and counts it as an auto-activation.

Both behaviours are controlled by a single boolean
`auto_activate_on_gesture` (default `True`, exposed as
`--no-auto-activate`). Patch 3 sets it to `False` and replaces the
auto-activation path with the wake-word listener; no other code in
the daemon needs to change.

This design intentionally lets the demo work today without the wake
word, while making clear that auto-activate is a placeholder. The
session summary distinguishes auto-activations from gesture
dispatches, so a reviewer can see at a glance whether the FSM ever
went DORMANT.

## Consequences

**Positive:**

- Tier 1 is demonstrable today: `sigil daemon run` from a freshly
  trained model.
- Per-verb exception isolation means individual broken verbs
  don't cascade into a daemon crash.
- The wake-word migration in Patch 3 is a small surface: a single
  boolean flips, one component is added in parallel.
- Stats per action surface what worked and what didn't,
  surfacing real demo-time issues (e.g. "we got 12 dispatches but
  3 window.maximize failures") without grepping logs.

**Negative:**

- Single-threaded means a slow verb blocks the next frame's
  classification. For Tier 1 this is fine (verbs are sub-ms key
  sends); for Tier 3 compounds that might issue more complex
  multi-step actions, we might revisit.
- Auto-activate on gesture is leakier than a wake word — any
  gesture, even an accidental fist while reaching for coffee,
  pulls the FSM back to LISTENING. This is exactly the
  false-positive failure mode the wake word is designed to fix
  and is acknowledged as a Patch 2 limitation.

**Neutral:**

- The verb registry is hardcoded today. User-customisable verbs
  (Phase 4 nice-to-have) will reuse the same dict[str, Verb]
  shape; the change will be a config loader, not an architectural
  shift.

## References

- `src/sigil/executor/verbs.py`
- `src/sigil/executor/registry.py`
- `src/sigil/executor/dispatcher.py`
- `src/sigil/daemon/runtime.py`
- `src/sigil/cli/daemon.py`
- ADR-0001 (layered architecture)
- ADR-0005 (interpreter FSM, multi-hand contract)
