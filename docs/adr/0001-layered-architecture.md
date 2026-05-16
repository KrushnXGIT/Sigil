# ADR-0001: Layered architecture — Perception / Intelligence / Interpreter

- **Status**: Accepted
- **Date**: 2026-05-13
- **Deciders**: Project initial design
- **Tags**: architecture, structure

## Context

Sigil needs to turn a webcam stream into Windows OS actions in <100ms with
production-grade reliability on low-end CPUs. The path from pixels to actions
involves three fundamentally different problems:

1. **Visual perception** — extracting structured information about hands
   from raw frames (computer vision, real-time, hardware-bound).
2. **Pattern classification** — turning that structure into discrete gesture
   events (machine learning, training-vs-inference split).
3. **Intent interpretation** — turning gesture events into the right action
   for the current OS context (rules engine, deterministic, stateful).

Mixing these concerns in one module would mean every change to the ML
classifier risks breaking OS dispatch, every camera tweak risks misfiring
gestures, and we have no clean boundary at which to swap a component
(e.g. moving from MediaPipe to a future SOTA detector).

## Decision

We will structure Sigil as **three independent layers** with well-defined
data contracts between them:

```
camera → [Perception]   produces LandmarkFrame stream
           ↓
         [Intelligence] produces GestureEvent stream
           ↓
         [Interpreter]  produces Intent stream
           ↓
         [Executor]     dispatches Intent as OS action
```

Each layer is its own package under `src/sigil/`. They communicate via
typed records (Pydantic / dataclasses) over in-process queues; the daemon
supervisor connects them. No layer imports from another except via the
shared `sigil.config` and `sigil.ipc` packages.

## Alternatives Considered

- **Single end-to-end model (pixels → action)** — would simplify the code
  but requires gigabytes of labelled training data we don't have, is opaque
  to debug, and forces full retraining on any vocabulary change. Rejected.
- **Pipeline of micro-services over the network** — overkill for a desktop
  app, adds latency, requires running a local broker. Rejected.
- **Plugin-based monolith** (one process, dynamically loaded modules) —
  considered but rejected: a misbehaving plugin can crash the whole daemon.
  Process isolation between workers is worth the IPC cost.

## Consequences

- ✅ Each layer is independently testable with synthetic inputs.
- ✅ Swapping a layer's implementation (e.g. YOLO-pose for MediaPipe) is
     a local change behind a stable contract.
- ✅ Training pipelines (`sigil.intelligence.training`) ship in the same
     repo but are not loaded at runtime.
- ⚠️ Three separate workers mean three serialisation boundaries, three
     places to watch for backpressure, three places to monitor.
- ⚠️ Contracts between layers (`LandmarkFrame`, `GestureEvent`, `Intent`)
     are now public API of each package — changing them needs care.
- ❌ A bug in the contract format can ripple across the stack; integration
     tests must exercise the full pipeline, not just unit tests per layer.

## References

- Spec: `docs/gesture-vocabulary-spec.md` §1 (Design Principles)
- See also: ADR-0002 (Landmark-first perception)
