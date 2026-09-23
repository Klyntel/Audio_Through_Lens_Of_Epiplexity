# `epiaudio.analysis`

Reads finished sweep outputs from `sweep_outputs/` and produces one table of
epiplexity results, each cell reported at its own recorded inference-set
size -- the token budget a sweep used is the fixed comparison axis, not
duration (team decision, 2026-08-25), so there is no common-duration mode to
opt into. Does not train and does not modify any sweep artefact; the one
exception to touching Comet is an opt-in fetch to diagnose a negative
endpoint (see `pipeline.fetch_endpoint_losses`).

## Usage

```bash
uv run python -m epiaudio.analysis \
  --data-root data \
  --csv sweep_outputs/cells.csv
```

- `--data-root` — lets a sweep's inference-set size be measured from `<dataset>/test.bin` instead of being supplied explicitly
- `--output-dir` — directory holding sweep output bundles (default `sweep_outputs/`)
- `--sweep-prefix` — only analyze sweeps whose name starts with this
- `--csv` — also write the table to this path
- `--keep-negative-points` — include `model_bits < 0` points in each cell's frontier instead of excluding them by default; `negative_points`/`negative_point_fraction` are reported either way

If `--sweep-prefix` (plus `--output-dir`) matches more than one sweep for the
same (dataset, tokenizer) -- e.g. a rerun under an unchanged sweep name --
all of them are skipped with a named explanation rather than one being
guessed at.

Output is three blocks: cell provenance, the results table, and any
comparability warnings (conditions that make comparing specific cells
unsound). A warning fires only on what's actually anomalous for that table,
not on facts that are true of every cross-tokenizer or cross-dataset table by
construction: cells naturally have different native inference-set sizes and
training audio across datasets, and tokenizers naturally have different
bitrates, so none of that alone is flagged. What is flagged: one dataset's
own tokenizer variants disagreeing on inference-set size (they tokenize the
same held-out audio, so they shouldn't), training exposure that varies for
cells sharing a tokenizer, replayed training tokens (with the exact repeat
factor, since the team's decision is to caption this rather than avoid it),
an unsaturated endpoint, or a negative endpoint.

Negative epiplexity values are retained and diagnosed rather than dropped; see
`curves.NegativityMechanism` for what a given diagnosis means.

## Tests

```bash
uv run python -m unittest discover -s tests -p "test_analysis_*.py"
```

## Module layout

| Module | Responsibility |
|---|---|
| `units.py` | durations, widths, parameter counts, compute, repeat factors |
| `estimators.py` | restandardization, both `K(M)` variants, precision arithmetic |
| `curves.py` | loss-curve shape and negativity diagnosis |
| `frontier.py` | negative-safe hull, endpoint, headroom slope |
| `pipeline.py` | fetch sweeps, build one row per cell, print and save the table |

`pipeline.py` is a straight-line script, not an object model: each sweep
becomes a plain dict of facts, then a plain dict row, and `main` collects
rows into a list before handing them to `pd.DataFrame` once at the end.
Nothing holds a cell's facts in an object on the way to becoming a table row.
