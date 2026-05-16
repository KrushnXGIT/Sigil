# Sigil

> Hand-gesture control for Windows. A sigil is a sign with intent — wave one, your computer responds.

**Status:** Phase 0 — Foundations *(in progress)*. Not yet user-facing.

Sigil is a production-grade hand-gesture control system for Windows. It runs as a low-footprint background daemon, listens for a voice wake word ("Gesture ON"), then watches your webcam and turns hand gestures into OS actions — play/pause media, switch apps, click, scroll, present slides — all on-device, all CPU-only, no cloud.

## Design highlights

- **Layered architecture.** Perception (landmarks), intelligence (classifier), interpreter (state machine), executor (OS actions). Each layer is independently testable and swappable. See [ADR-0001](docs/adr/0001-layered-architecture.md).
- **Landmark-first classification.** MediaPipe extracts 21 hand keypoints; a tiny (<1MB) classifier reads only those — not pixels. Runs on a 2-core CPU in <3ms. See [ADR-0002](docs/adr/0002-landmark-first-perception.md).
- **Three-tier vocabulary.** MVP ships 6 gestures, Standard adds dynamic swipes, Power adds pointer control and modifiers. See [the spec](docs/gesture-vocabulary-spec.md).
- **Voice wake.** Camera stays off until `"Gesture ON"` is spoken. Privacy floor by design.
- **Reserved safety gestures.** Open Palm = cancel, Thumbs Up = confirm, Thumbs Down = undo. Always work. Always.

## Project status

| Phase | Scope | Status |
|------:|-------|--------|
|     0 | Repo, CI, config, logging, ADRs | ✅ this commit |
|     1 | Perception (MediaPipe wrapper, smoothing) | ☐ next |
|     2 | Static classifier — train on HaGRIDv2, ONNX export | ☐ |
|     3 | Dynamic classifier (temporal Transformer) | ☐ |
|     4 | Wake word ("Gesture ON" via OpenWakeWord) | ☐ |
|     5 | Interpreter + executor + state machine + HUD | ☐ |
|     6 | Custom gesture registration UI | ☐ |
|     7 | Packaging, signing, autostart, accessibility | ☐ |

## Dev setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (recommended — much faster than pip).

```powershell
# Install uv if you don't have it
winget install --id astral-sh.uv

# Clone + sync
git clone https://github.com/<org>/sigil
cd sigil
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

### Verify it works

```powershell
uv run sigil version            # prints 0.1.0a0
uv run sigil config path        # prints %APPDATA%\SigilGesture\gestures.yaml
uv run sigil config show -p configs/defaults/mvp.yaml
uv run pytest                   # 17+ tests should pass
```

### Adding a layer's dependencies

Each layer's heavy deps live in optional groups so you can install only what you need:

```powershell
uv sync --extra perception      # camera + MediaPipe
uv sync --extra intelligence    # ONNX Runtime + PyTorch (training)
uv sync --extra executor        # pynput + pywin32
uv sync --extra wakeword        # OpenWakeWord + sounddevice
uv sync --extra all             # everything
```

## Repository layout

```
sigil/
├── src/sigil/         # Python package
│   ├── config/        # ✓ schema + loader
│   ├── logging/       # ✓ structlog wiring
│   ├── cli/           # ✓ `sigil` console script
│   ├── perception/    # camera + landmarks (Phase 1)
│   ├── intelligence/  # classifiers (Phase 2/3)
│   ├── wakeword/      # voice wake (Phase 4)
│   ├── interpreter/   # state machine (Phase 5)
│   ├── executor/      # OS actions (Phase 5)
│   ├── ipc/           # shared mem + pub/sub (Phase 5)
│   └── daemon/        # supervisor (Phase 5)
├── configs/defaults/  # bundled tier configs
├── docs/
│   ├── architecture.md
│   ├── gesture-vocabulary-spec.md
│   └── adr/           # architecture decision records
├── tests/
├── infra/             # packaging, installer scripts
├── datasets/scripts/  # dataset download/preprocess (HaGRIDv2 etc.)
├── models/            # versioned model artefacts (gitignored)
└── pyproject.toml
```

## Documentation

- [Gesture Vocabulary Spec](docs/gesture-vocabulary-spec.md) — what gestures exist and what they do
- [Architecture Overview](docs/architecture.md) — how the pieces fit together
- [ADR-0001](docs/adr/0001-layered-architecture.md) — why three layers
- [ADR-0002](docs/adr/0002-landmark-first-perception.md) — why landmarks not pixels

## License

MIT. See `LICENSE` *(to be added)*.

## Note on the name

There is an unrelated, well-established Windows program called **Sigil** that edits EPUB ebooks. We are not affiliated with it. Sigil-the-gesture-system ships under the PyPI name `sigil-os` and uses `%APPDATA%/SigilGesture/` for user config to avoid any on-disk collision.
