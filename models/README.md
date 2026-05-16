# models/

Model artefacts (ONNX, PyTorch checkpoints, TFLite) live here.

**Not tracked in git.** Build artefacts of this size belong in a release-asset
pipeline, not the source tree. The Phase 2/3 training scripts will deposit
exports here; the daemon loads them from this directory by default.

Layout (forward-looking):
```
models/
├── static-classifier/
│   ├── v0.1.0/
│   │   ├── model.onnx
│   │   ├── model.fp32.pt
│   │   └── model_card.md
│   └── latest -> v0.1.0
└── dynamic-classifier/
    └── ...
```
