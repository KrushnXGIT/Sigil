# Phase 3, Patch 1 — Classifier runtime + Interpreter FSM

First of three patches that take Sigil from "trained classifier" to
"working Tier 1 daemon." This one adds the two intelligence-layer
components that consume the perception stream:

  - **`ClassifierRuntime`** — loads the `.onnx` + `.meta.json`,
    validates the metadata against runtime expectations (protocol
    version, normalisation scheme, input shape — refuses mismatched
    models at load time), classifies LandmarkFrames into multi-hand
    GestureEvent tuples.

  - **`Interpreter`** — four-state FSM (DORMANT, LISTENING,
    CONFIRMING, EXECUTING) with 3-frame debounce, 1-second per-gesture
    cooldown, edge-triggered dispatch, reserved-gesture handling
    (open_palm = cancel, thumbs_up = confirm, thumbs_down = undo).
    Pure logic — no I/O, no threads. Tests run in microseconds.

See `docs/adr/0005-multi-hand-contract-and-interpreter-fsm.md` for
the multi-hand decision and the state machine semantics.

## What's in this patch

```
src/sigil/intelligence/types.py                    <- NEW
src/sigil/intelligence/classifier_runtime.py       <- NEW
src/sigil/intelligence/interpreter.py              <- NEW
tests/test_classifier_runtime.py                   <- NEW (12 cases)
tests/test_interpreter.py                          <- NEW (24 cases)
docs/adr/0005-multi-hand-contract-and-interpreter-fsm.md   <- NEW
```

No existing file is modified. Drop the patch into the project root and
the existing tree absorbs it.

## How to apply

From the project root on Windows:

```powershell
Copy-Item -Recurse -Force patch\src    .
Copy-Item -Recurse -Force patch\tests  .
Copy-Item -Recurse -Force patch\docs   .
```

Or from Colab against the Drive-mounted project:

```python
import shutil, os
for sub in ("src", "tests", "docs"):
    shutil.copytree(f"/content/patch/{sub}",
                    f"/content/drive/MyDrive/sigil/{sub}",
                    dirs_exist_ok=True)
```

## Run the tests

```powershell
uv run pytest tests/test_interpreter.py tests/test_classifier_runtime.py -v
```

The interpreter tests are pure logic — no ONNX, no torch — and finish
in milliseconds. The runtime tests build a tiny synthetic ONNX via the
real export pipeline you ran last patch, so they need both `torch` and
`onnxruntime` installed (they already are from Patch 2b).

## What this does NOT do yet

Patch 1 produces `ActionDispatch` objects but there's no executor to
*run* them — they're decisions, not actions. That's Patch 2 next.
And there's still no wake word, so the interpreter has to be
manually woken via `interpreter.activate(timestamp_ns)`. That's
Patch 3.

You can't yet run a live preview that demonstrates end-to-end
behaviour. That comes after Patch 2 lands the executor. For now,
the tests are the integration check.

## What to send back

```
uv run pytest tests/test_interpreter.py tests/test_classifier_runtime.py -v
```

Paste the output. Expecting 36 passed. If anything fails, paste the
full failure trace and I'll diagnose.
