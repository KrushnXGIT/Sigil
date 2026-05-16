# ADR-0002: Landmark-first classification (no end-to-end CNN)

- **Status**: Accepted
- **Date**: 2026-05-13
- **Tags**: architecture, ml, performance

## Context

For gesture classification, we can either:

a) **Feed raw RGB frames** (or stacked frames) into a CNN/ViT and learn
   end-to-end from pixels → gesture class.

b) **Extract hand landmarks first** (via MediaPipe HandLandmarker) and
   feed only the 21 (x,y,z) keypoints into a small downstream classifier.

End-to-end is the obvious default in modern computer-vision practice and
typically wins on accuracy when data is plentiful. But Sigil targets
**low-end CPUs on Windows**, where every millisecond matters, and we
expect to ship with a relatively small (sub-million-sample) custom dataset
on top of HaGRIDv2 pretraining.

## Decision

We will use **landmark-first classification**:

```
frame → MediaPipe Hands → 21 × (x,y,z) landmarks → tiny classifier → gesture
```

The classifier never sees pixels. It operates on a 63-float input
(21 landmarks × 3 coords), normalised to wrist-origin and palm-span-scaled.

## Alternatives Considered

- **End-to-end CNN on raw frames** — rejected because: (1) inference cost
  on a 2-core CPU is prohibitive for our latency budget; (2) requires
  full retraining for new gestures; (3) much worse generalisation across
  lighting, skin tone, and background without a far larger dataset.
- **Skeleton + RGB hybrid (two-stream)** — best accuracy in research, but
  doubles the inference cost. Save for a future "high-performance mode"
  if needed.
- **EMG / wearable sensors** — out of scope; vision-only is the product.

## Consequences

- ✅ Classifier is ≤200K params, <1MB ONNX, <3ms CPU inference.
- ✅ Invariant to background, lighting, skin tone (MediaPipe handles
     pixel-level robustness; classifier sees a clean abstraction).
- ✅ Adding a new gesture = collect 50 examples and fine-tune the last
     layer; no full retrain.
- ✅ The same classifier deploys to any device MediaPipe runs on.
- ⚠️ Classifier accuracy is bounded above by MediaPipe's landmark
     accuracy. When MediaPipe fails (very low light, hand partially out of
     frame), we fail too.
- ⚠️ Three-coordinate (x, y, z) landmarks include a depth estimate that
     is monocular and noisy; we will rely primarily on (x, y) and use z
     only as a weak signal.
- ❌ Some gestures that depend on subtle hand-surface details (palm vs.
     back of hand, finger crossings) may be harder to disambiguate than
     with full RGB. Mitigation: pick gestures that are geometrically
     distinct in landmark space (this is already a vocabulary criterion).

## References

- Spec: `docs/gesture-vocabulary-spec.md` §1 (Design Principles)
- HaGRIDv2: https://github.com/hukenovs/hagrid
- MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- See also: ADR-0001 (Layered architecture)
