# apps/trainer

Training scripts for Sigil's models. Designed to run **identically** in Colab
(free T4) and on a local machine with a GPU. The repo's runtime daemon
never imports anything from here.

## Scripts

| Script | Phase | Produces |
|---|---|---|
| `train_static.py` | 2b | `best.pt` static-gesture classifier checkpoint |
| _(to come)_ | 2c | INT8 ONNX export from `best.pt` |
| _(to come)_ | 3 | Dynamic (temporal) classifier |

## Running in Google Colab

```python
# In a fresh Colab notebook cell:
!git clone https://github.com/<org>/sigil
%cd sigil
!pip install -e .[training,perception]

# (Optional) prepare the dataset — slow. Skip if you already have the parquet
# cache uploaded to /content or to your Drive.
!python -m sigil.cli.main dataset prepare \
    --gestures open_palm,thumbs_up,thumbs_down,fist,peace,ok,no_gesture \
    --raw-dir /content/hagrid_raw \
    --output /content/landmarks.parquet

# Train:
!SIGIL_LANDMARK_PARQUET=/content/landmarks.parquet \
 SIGIL_OUTPUT_DIR=/content/sigil_run \
 python apps/trainer/train_static.py
```

Then download `best.pt`, `label_map.json`, and `history.json` from
`/content/sigil_run` to your local machine for ONNX export (Phase 2c).

## Opening `train_static.py` as a notebook

The script uses **jupytext "percent" format** (`# %%` cell markers). You can
either:

- Run it as a regular Python script: `python train_static.py`
- Convert to notebook: `jupytext --to notebook train_static.py`
- Open directly in VS Code / PyCharm — both recognize `# %%` natively.

This keeps the training code under git (no JSON merge conflicts!) while
still being usable as a notebook.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SIGIL_LANDMARK_PARQUET` | `datasets/processed/landmarks.parquet` | Input parquet cache. |
| `SIGIL_OUTPUT_DIR` | `models/static-classifier/run-001` | Where checkpoints go. |
| `SIGIL_EPOCHS` | `40` | Training epochs. |
| `SIGIL_BATCH_SIZE` | `256` | Batch size. |
| `SIGIL_LR` | `3e-4` | Learning rate (AdamW). |

## Targets

For the MVP classifier:

- **Accuracy:** ≥97% on test set (user-disjoint split)
- **Parameters:** ≤200K
- **FP32 size:** ≤1 MB
- **Training time:** ~5–10 minutes on a T4

If you blow past these — especially in the wrong direction — check the dataset
class balance first (`sigil dataset summarise <parquet>`). A starved test
class will tank accuracy without anything else being wrong.
