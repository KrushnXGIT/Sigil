# V1 — Dataset Pipeline (Patch 1 of 3)

This is patch 1 of the V1 (learned dynamic-gesture classifier) work.
It adds the data pipeline: download Jester, run MediaPipe to extract
landmark sequences, save as parquet. The output feeds V1 patch 2
(training) — coming separately.

## What this patch contains

```
src/sigil/intelligence/dataset/jester.py                   # acquisition + label parsing
src/sigil/intelligence/dataset/temporal_preprocessing.py   # MediaPipe extraction
src/sigil/intelligence/dataset/temporal_storage.py         # parquet I/O
src/sigil/cli/dataset_v1.py                                # CLI command group
docs/adr/0012-v1-dynamic-classifier.md                     # design rationale
README_V1.md                                               # this file
```

## Apply

```powershell
Copy-Item -Recurse -Force patch\src .
Copy-Item -Recurse -Force patch\docs .
Copy-Item -Force patch\README_V1.md .
```

Then wire the new CLI group into `src/sigil/cli/main.py` — same
pattern as the V0 training CLI:

```python
from sigil.cli.dataset_v1 import dataset_v1   # near other imports
...
cli.add_command(dataset_v1)                     # near other add_command calls
```

Verify:

```powershell
uv run sigil dataset-v1 -h
```

Should list four subcommands: `instructions`, `validate`, `build`,
`summarise`.

## Steps

### 1. Read the download instructions

```powershell
uv run sigil dataset-v1 instructions
```

Prints the Kaggle CLI and manual browser paths. Pick whichever you
prefer. The dataset is ~5 GB compressed.

### 2. Download Jester

Either:

**Option A — Kaggle CLI:**
```powershell
pip install kaggle
# Place kaggle.json at $env:USERPROFILE\.kaggle\kaggle.json
kaggle datasets download -d toxicmender/20bn-jester -p datasets/raw/jester
cd datasets/raw/jester
# Unzip the downloaded archive (Windows: right-click → Extract All, or use 7-Zip)
```

**Option B — Browser:**
1. Go to https://www.kaggle.com/datasets/toxicmender/20bn-jester
2. Sign in (free account if you don't have one)
3. Download
4. Extract into `datasets/raw/jester/`

After extraction, the layout should be:

```
datasets/raw/jester/
├── annotations/
│   ├── jester-v1-labels.csv
│   ├── jester-v1-train.csv
│   └── jester-v1-validation.csv
└── 20bn-jester-v1/
    ├── 1/
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   └── ...
    ├── 2/
    └── ... (many thousand video folders)
```

### 3. Validate layout

```powershell
uv run sigil dataset-v1 validate
```

Should print `Jester layout OK`. If it complains, the unzip is
incomplete — re-extract and try again.

### 4. Smoke-test the preprocessing pipeline

Before kicking off the 8-hour full preprocessing, validate that
the pipeline runs end-to-end on a tiny subset:

```powershell
uv run sigil dataset-v1 build --max-per-class 20
```

This processes 5 classes × 20 videos × 2 splits = 200 videos =
~5 minutes. Produces `datasets/processed/v1/train.parquet` and
`val.parquet` with ~80 sequences each (after dropouts).

Check the output:

```powershell
uv run sigil dataset-v1 summarise datasets/processed/v1/train.parquet
```

Should show all 5 V1 classes with non-zero counts. If you see any
class at zero, that class's videos may have MediaPipe-undetectable
hands — usually not the case, but worth noting.

### 5. Full preprocessing run

When the smoke test passes:

```powershell
uv run sigil dataset-v1 build
```

This is the long one — ~8 hours on CPU for ~25,000 videos. Best
left running overnight. Progress is logged every 100 videos with a
running ETA, so you can monitor.

The build is idempotent at the split-file level: re-running
overwrites `train.parquet` and `val.parquet`. There's no per-video
checkpoint right now — if you interrupt with Ctrl+C, the current
split is lost and you'd restart it.

### 6. Validate the full output

```powershell
uv run sigil dataset-v1 summarise datasets/processed/v1/train.parquet
uv run sigil dataset-v1 summarise datasets/processed/v1/val.parquet
```

Expected class counts (approximate, before MediaPipe-failure
filtering):

| Class | Train | Val |
|---|---|---|
| swipe_left | ~4500 | ~600 |
| swipe_right | ~4500 | ~600 |
| swipe_up | ~4500 | ~600 |
| swipe_down | ~4500 | ~600 |
| no_dynamic_gesture | ~5000 | ~700 |

If a class is significantly under-represented, the model will be
biased. You can rebalance by passing `--max-per-class <N>` to the
build step. We'd typically want at least 2000/class for a reliable
learned classifier.

## Troubleshooting

**`Could not construct HandLandmarker for preprocessing`** — the
preprocessing module tries a few common constructor signatures.
If yours uses a different API, paste me your
`sigil.perception.landmarks` module and I'll patch the loader.

**`Could not locate the V0 landmark normaliser`** — same issue,
different module. Paste `sigil.perception.normalize` if it appears.

**Preprocessing crashes mid-run with `cannot allocate memory`** —
unlikely given our small per-video memory footprint, but possible
on low-RAM machines. Drop to `--max-per-class 1000` to reduce the
in-memory accumulator size before parquet write.

**`No record produced for class swipe_up` (or any other class)** —
MediaPipe couldn't detect a hand in any frame of those videos, OR
the class label in the Jester CSV doesn't match
`JESTER_TO_SIGIL_V1`. Run with `--max-per-class 5` and inspect the
log warnings.

## Next

After this build completes, V1 patch 2 (model + training + export)
ships. That'll be a separate patch with its own README.
