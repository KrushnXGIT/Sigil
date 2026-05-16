# ADR-0005: Multi-hand classifier contract and interpreter state machine

**Status:** Accepted
**Date:** 2026-05-15

## Context

Patch 1 of Phase 3 adds two new components: the classifier runtime
(loads the exported ONNX, classifies LandmarkFrames into gesture
events) and the interpreter (consumes gesture events, maintains state,
emits action dispatches).

Two design questions had to be resolved before any code landed.

**Q1: How does the classifier runtime handle frames with zero or two
hands?** MediaPipe HandLandmarker is configured with `num_hands=2`,
so any frame can contain 0, 1, or 2 detected hands. The Tier 1
vocabulary spec assumes a single hand; the Tier 3 spec calls for
two-handed modifier gestures. The contract chosen now propagates
through every downstream layer.

**Q2: What are the interpreter state machine's semantics?** The
vocabulary spec describes the states (DORMANT, LISTENING, CONFIRMING,
EXECUTING) and the reserved gestures (open_palm, thumbs_up,
thumbs_down) but doesn't pin down debouncing, cooldown, or
edge-trigger behaviour. Patch 1 has to lock these.

## Decisions

### D1: Multi-hand from the start

`ClassifierRuntime.classify(frame: LandmarkFrame) → tuple[GestureEvent, ...]`

The runtime emits one GestureEvent per detected hand whose detection
confidence and top-1 probability both clear thresholds. Tier 1
consumers (the interpreter) pick the highest-confidence event and
ignore the rest. Tier 3 consumers will compose multiple events into a
single compound action.

Considered alternative: "single-hand for Tier 1; skip frames with 0
or 2+ hands; bump the contract when Tier 3 arrives." Rejected for
three reasons:

1. Silently dropping two-hand frames is a poor user experience: the
   user reaches for a coffee mug with their free hand while gesturing
   with their dominant hand, and the gesture is dropped with no
   feedback. The two-hand world is the real world.

2. Retrofitting from `GestureEvent | None` to
   `tuple[GestureEvent, ...]` later would be a contract break
   touching the runtime, the interpreter's state machine, every test,
   and any future debouncer logic. A full ADR's worth of work to undo
   a decision made for "simplicity."

3. The multi-hand code path is one extra `for hand in frame.hands`
   loop iteration. The complexity isn't in returning a tuple; it's in
   the interpreter's *selection rule*. Tier 1's selection rule
   (highest confidence wins) is three lines; Tier 3's compound rule
   replaces those three lines without touching the data contract.

### D2: Interpreter state machine — locked semantics

**States:** DORMANT → LISTENING → (CONFIRMING ↔ LISTENING) →
EXECUTING (transient) → LISTENING. Wake-word transitions DORMANT to
LISTENING; idle timeout transitions LISTENING to DORMANT.

**Reserved gestures:**
- `open_palm` in CONFIRMING cancels the pending action; in LISTENING
  it is a no-op (nothing to cancel).
- `thumbs_up` in CONFIRMING confirms the pending action; in LISTENING
  it is a no-op (nothing to confirm).
- `thumbs_down` in LISTENING dispatches `system.undo` (Ctrl+Z); in
  CONFIRMING it is ignored (user must use thumbs_up or open_palm).

The interpreter refuses to construct if the user-supplied mapping
overrides any reserved gesture. This makes "I bound thumbs_up to
close the browser" impossible by construction rather than by
convention.

**Debounce (3 frames):** A gesture must appear in the last 3 frames
in a row before the interpreter treats it as the active gesture.
Tuned to filter single-frame misclassifications at ~30 FPS, which is
the dominant failure mode of the classifier at deployment. Three
frames is ~100 ms — well below the human-perceptible response time of
~250 ms.

**Edge-trigger:** An action dispatches on the *transition* into a
gesture, not on every frame the gesture remains stable. Holding a
fist for 30 frames at 30 FPS produces exactly one play/pause, not 30.

**Per-gesture cooldown (1 s):** After a gesture fires an action, the
same gesture cannot fire again until 1 s elapses. Belt-and-braces
with edge-trigger: catches the case where the user actually does want
to fire twice in quick succession but the classifier hasn't seen any
intervening non-gesture frames. Independent per gesture, so fist
firing doesn't suppress a subsequent peace.

**Timeouts:** LISTENING returns to DORMANT after 60 s of no
non-trivial gesture activity. CONFIRMING returns to LISTENING after
4 s without a thumbs_up. Both hardcoded in Patch 1; will move to the
Pydantic config schema when Phase 3 polish lands.

### D3: GestureEvent and ActionDispatch as separate types

The classifier runtime produces `GestureEvent` (one per hand). The
interpreter produces `ActionDispatch` (one per decided-action). They
are not the same thing: a single GestureEvent can produce zero
ActionDispatches (it's in the cooldown), one (normal dispatch), or be
deferred indefinitely (it's destructive and awaiting confirmation).
The two types make the layer boundary explicit in the type system.

## Consequences

**Positive:**

- The data contract is stable through Tier 3. Tier 2's dynamic swipes
  and Tier 3's compounds extend the *selection logic* without
  changing the *data shape* the runtime emits.

- Reserved-gesture rules are unbypassable. A bug in user
  configuration that tries to rebind `thumbs_up` fails at startup,
  not at the moment a confirmation flow gets silently skipped.

- The interpreter is pure: timestamp_ns is caller-supplied,
  there is no I/O, no threading, no real clock. Tests run in
  microseconds rather than seconds.

**Negative:**

- The runtime always allocates a tuple even when there is only one
  hand. The allocation cost (a tuple of one element) is dwarfed by
  the ONNX inference cost; this is a non-issue at our scale.

- Patch 1's hardcoded timing constants will move to config in a
  later patch, which is a small contract change. The constants are
  named exports so the migration path is mechanical (the schema
  will reference the same names).

**Neutral:**

- The interpreter's debounce and cooldown values are guesses. We
  will tune them against real users in Phase 4 user testing. The
  abstractions make the tuning easy; the values are not.

## References

- `src/sigil/intelligence/classifier_runtime.py`
- `src/sigil/intelligence/interpreter.py`
- `src/sigil/intelligence/types.py`
- `docs/gesture-vocabulary-spec.md` (Tier 1 / 2 / 3 vocabularies)
- ADR-0001 (layered architecture and inter-layer contracts)
- ADR-0003 (2D landmarks end-to-end; PROTOCOL_VERSION 2)
- ADR-0004 (palm-centroid normalisation; PROTOCOL_VERSION 3)
