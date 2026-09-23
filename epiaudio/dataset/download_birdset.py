# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets>=3.3.1,<4.0.0", "huggingface_hub", "soundfile", "librosa"]
# ///
"""One-time download script for BirdSet.

## Why this script exists

BirdSet ships a custom dataset builder (BirdSet.py) that requires datasets<4.0.0.
The rest of this project uses datasets>=5.0.0, which dropped support for such
builders. These two version ranges are mutually exclusive, so downloading must
happen in a separate environment.

## Environment isolation via PEP 723

The `# /// script` block at the top of this file is a PEP 723 inline script
dependency declaration. When you invoke the script with `uv run`, uv reads that
block and automatically creates a throwaway venv containing only
`datasets>=3.3.1,<4.0.0` and `huggingface_hub` — completely separate from the
project's own `.venv`. No manual venv creation or pip install is needed.

    DO:     uv run epiaudio/dataset/download_birdset.py
    DO NOT: python epiaudio/dataset/download_birdset.py   # uses project venv, wrong datasets version
    DO NOT: uv run --no-project ...                       # same as above

The data is saved to disk in Arrow format via `save_to_disk`. Once on disk it
can be loaded with any datasets version using `load_from_disk` (see load_birdset.py).

## Usage

    uv run epiaudio/dataset/download_birdset.py               # downloads HSN (default)
    uv run epiaudio/dataset/download_birdset.py HSN XCL NBP   # downloads multiple subsets

Subsets are saved to data/birdset_<subset_lower>_raw/.
"""

import sys
import os
from datasets import load_dataset

SUBSETS = sys.argv[1:] if len(sys.argv) > 1 else ["HSN"]
OUT_DIR = "data"

for subset in SUBSETS:
    print(f"Downloading BirdSet/{subset}...")
    ds = load_dataset("DBD-research-group/BirdSet", subset, trust_remote_code=True)
    out_path = os.path.join(OUT_DIR, f"birdset_{subset.lower()}_raw")
    os.makedirs(OUT_DIR, exist_ok=True)
    ds.save_to_disk(out_path)
    print(f"Saved to {out_path}")
