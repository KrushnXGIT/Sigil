# Sigil — Architecture Overview

> For the *why* behind these choices, see the ADRs in `docs/adr/`.
> For the gesture vocabulary, see `docs/gesture-vocabulary-spec.md`.

## System diagram

```
                     ┌────────────────────────────────────────┐
                     │              Sigil Daemon              │
                     │  (single Windows process, supervisor)  │
                     └────────────────────────────────────────┘

   audio in                                                     OS events out
       │                                                              ▲
       ▼                                                              │
 ┌─────────────┐       ┌─────────────┐       ┌─────────────┐    ┌─────────────┐
 │  wakeword   │──▶──▶│ perception  │──▶──▶│intelligence │──▶▶│ interpreter │
 │  (always-on │   |   │  (camera +  │   |   │  (static +  │   |  │ + executor  │
 │   listener) │   |   │  landmarks) │   |   │  dynamic    │   |  │ + HUD       │
 └─────────────┘   |   └─────────────┘   |   │  classify)  │   |  └─────────────┘
        │          |          │          |   └─────────────┘   |         │
        │          |          │          |          │          |         │
        └──── 1 ───┘          └──── 2 ───┘          └──── 3 ───┘         │
                                                                          │
                              [4] config + telemetry sidecar              │
                                                                          ▼
                                                                       OS APIs

Boundaries:
  1. "wake" event             — wakeword → daemon → start perception
  2. LandmarkFrame stream     — perception → intelligence (shared memory)
  3. GestureEvent stream      — intelligence → interpreter (pub/sub)
  4. Intent + Action records  — interpreter → executor (in-process)
```

## Package layout

```
src/sigil/
├── config/         ✓ Phase 0 — Pydantic schema, YAML loader
├── logging/        ✓ Phase 0 — structlog wiring
├── cli/            ✓ Phase 0 — `sigil` console script
├── perception/     ☐ Phase 1 — camera capture + MediaPipe landmarks
├── intelligence/   ☐ Phase 2/3 — static + dynamic gesture classifiers
├── wakeword/       ☐ Phase 4 — OpenWakeWord listener for "Gesture ON"
├── interpreter/    ☐ Phase 5 — gesture → intent state machine
├── executor/       ☐ Phase 5 — OS action dispatcher + undo
├── ipc/            ☐ Phase 5 — shared memory + ZeroMQ pub/sub
└── daemon/         ☐ Phase 5 — supervisor process, lifecycle
```

## Data contracts (forthcoming)

These are the records that flow between layers. Each will be defined in
the producing layer and re-exported. **Once published they are public API**
and changes are breaking.

- `LandmarkFrame` — single-camera-frame's hand landmarks + timing metadata
  *(defined in `sigil.perception`)*
- `GestureEvent` — a recognised gesture (class, confidence, timing)
  *(defined in `sigil.intelligence`)*
- `Intent` — what the system thinks the user wants given the gesture and
  the current context *(defined in `sigil.interpreter`)*
- `ActionRecord` — log of a dispatched action, used for undo
  *(defined in `sigil.executor`)*

## Performance budget

End-to-end target: **≤100ms** from gesture completion to OS action on a
2-core, no-GPU Windows laptop.

| Stage                       | Budget |
|-----------------------------|--------|
| Camera capture              |   5ms  |
| MediaPipe landmark extract  |  20ms  |
| Smoothing + buffer maintain |   2ms  |
| Static classifier inference |   3ms  |
| Dynamic classifier (when)   |   8ms  |
| Intent resolution           |   5ms  |
| Action dispatch             |   5ms  |
| **Total nominal**           | **~50ms** |
| **Headroom for jitter**     |  50ms  |

If we blow this budget, we re-evaluate. Inference is the most likely
culprit; the mitigation is INT8 quantization (already planned).

## Privacy posture

- Camera is **off** unless LISTENING. Wake word is required to enable it.
- Microphone is on always but only runs the wake detector — no audio is
  recorded, no audio leaves the process.
- No network calls at runtime. Anonymous opt-in telemetry to a local SQLite
  is the only data the daemon produces; exporting it is a manual user step.
- All ML inference is on-device. No cloud dependency.

## Out of scope (for now)

These are explicit non-goals to keep scope contained:

- macOS / Linux support (Phase 0 commits to Windows only)
- Sign-language recognition (different problem domain, different vocab)
- Multi-user / shared sessions
- Cloud-based gesture customisation
- Mobile (Android / iOS) deployment
