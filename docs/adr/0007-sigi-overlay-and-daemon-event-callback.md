# ADR-0007: Sigi overlay and daemon event callback

**Status:** Accepted
**Date:** 2026-05-16

## Context

Patch 2's daemon works end-to-end (perception → classifier → interpreter
→ executor → OS) but is invisible — there's no visual feedback while
it runs. For a demo this is a missed opportunity; for a user it makes
the system feel like a black box.

The desktop OS-mascot pattern (Claude Code's Clawd) is the obvious
reference: a small, friendly character that lives on screen, reacts
to system state, and adds personality without getting in the way.

## Decisions

### D1: Sigi as the mascot, channeling Clawd's vibe

Sigi is an original character (not a fork of Clawd) but adopts the
same design principles: chunky pixel-style face, warm orange palette,
minimal but expressive features. Six moods — sleeping, alert, excited,
uncertain, happy, sad — map 1:1 to the interpreter's FSM state plus
recent activity.

### D2: Tkinter, not Qt

Tkinter ships with standard Python on Windows (no install). PyQt6
(already in the `[ui]` extra) would give more polish but adds a 30 MB
install and asset-loading complexity. For the deadline-driven Patch 3,
zero-install is the right tradeoff.

The character is rendered with Tkinter Canvas primitives (ovals,
arcs, lines, text) — no PNG assets, no resource loader. This means
the patch is one self-contained module.

Magenta `-transparentcolor` (Windows-only) makes the window
chrome-free and the background invisible; on macOS/Linux this is
ignored and the magenta shows, which is acceptable since Sigil is
Windows-only for v1.

### D3: Daemon emits `on_event` callbacks; UI runs on the main thread

Tkinter's mainloop must own the main thread. The daemon's loop also
wants the main thread. Resolution: when `--overlay` is set, the
daemon runs in a background thread, Tkinter on the main thread, and
they communicate via a bounded `queue.Queue`.

The daemon's contribution is minimal: an optional `on_event:
Callable | None` parameter, called once per frame with an
`OverlayEvent` snapshot. When not passed, the daemon behaves
identically to Patch 2 — no threading, no UI dependency, no risk.

The queue is bounded (size 64) and the producer (daemon) drops oldest
on overflow rather than blocking. The consumer (Tkinter) reads only
the latest event each tick. Backpressure is impossible by design.

### D4: State-logic / rendering separation

`ui/state.py` derives moods and captions from events. Pure Python,
no Tkinter, fully unit-testable.

`ui/character.py` defines what Sigi looks like in each mood as a list
of `Shape` primitives. Also pure Python.

`ui/overlay.py` is the only module that touches Tkinter. It reads
state, asks the character module for shapes, paints them.

This split lets us test 100% of the logic without ever opening a
window, and lets us tweak Sigi's design (add a new mood, change
colors) in one place.

## Consequences

**Positive:**

- The default daemon experience (no `--overlay`) is identical to
  Patch 2. Risk-free addition.
- The overlay is genuinely on-brand: orange pixel mascot, sits in the
  corner, reacts in real time. Demo gold.
- Tests cover the state machine; rendering is by inspection in the
  running window, which is the right way to verify visuals anyway.

**Negative:**

- Tkinter's `-transparentcolor` is Windows-only. On other platforms
  the magenta background shows. Acceptable since v1 is Windows-only.
- Two threads instead of one. The bounded queue prevents
  pathological backpressure but adds a small surface for races.
  Mitigated by: queue is the ONLY shared state, daemon stop signal
  is a single bool.

**Neutral:**

- No PNG assets means Sigi is locked to what Canvas primitives can
  draw. Could be swapped for sprite-based rendering later without
  changing the interface.

## References

- `src/sigil/ui/state.py`
- `src/sigil/ui/character.py`
- `src/sigil/ui/overlay.py`
- `src/sigil/daemon/runtime.py` (added `on_event` parameter)
- `src/sigil/cli/daemon.py` (added `--overlay` flag)
- ADR-0006 (executor + daemon loop)
