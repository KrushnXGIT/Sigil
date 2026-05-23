# V1 Patch 1: Dataset pipeline (corrected)

Reads dataset, runs MediaPipe via HandLandmarkerWrapper, writes parquet.
Patch 1 of 3.

## What's corrected from the first draft

- Uses your real HandLandmarkerWrapper class (not a guessed API).
- Uses .process(CapturedFrame) → LandmarkFrame, reads .primary_hand.keypoints.
- Loads JPGs with cv2 and converts BGR→RGB to match the capture contract.
- Uses normalize_landmarks() directly (no fallback imports).
- Honours is_degenerate() to skip bad hands.

## Files

```
src/sigil/intelligence/dataset/jester.py                   NEW
src/sigil/intelligence/dataset/temporal_preprocessing.py   NEW (corrected)
src/sigil/intelligence/dataset/temporal_storage.py         NEW
src/sigil/cli/dataset_v1.py                                NEW (corrected)
docs/adr/0012-v1-dynamic-classifier.md                     NEW
README_V1.md                                               NEW
```

## Apply

```powershell
Copy-Item -Recurse -Force patch\src .
Copy-Item -Recurse -Force patch\docs .
Copy-Item -Force patch\README_V1.md .
```

Then add 2 lines to `src/sigil/cli/main.py`:

```python
from sigil.cli.dataset_v1 import dataset_v1
cli.add_command(dataset_v1)
```

## Use

```powershell
uv run sigil dataset-v1 instructions      # how to get Jester
# ... download + extract Jester ...
uv run sigil dataset-v1 validate          # sanity-check layout
uv run sigil dataset-v1 build --max-per-class 20   # 5-min smoke test
uv run sigil dataset-v1 build             # full ~8-hour preprocess
```

After preprocessing completes, V1 patch 2 (model + training) ships.
