# Conditional sweep: Syntheory intervals

This command runs the existing conditional-label pipeline as one independent
sweep per audio tokenizer, using the Experiment 1 model-size grid. Each sweep
expands configurations, isolates checkpoints, records results, and selects the
smallest two-part code on its lower convex hull; it does not introduce another
estimator or training loop.

## 1. Prepare the classification dataset

From the repository root:

```bash
uv run python epiaudio/dataset/prepare_audio.py \
  syntheory_intervals dac,encodec,sqcodec,xcodec midi_program_name 8 8 4
```

Each binary row contains audio tokens followed by one
`midi_program_name` class index. The generated `metadata.yaml` supplies
`model.V` and `model.L`; the sweep maps those values to the conditional
teacher's `vocab_size` and `seq_length`.

The checked-in configuration uses eight DDP ranks. Its training split must
therefore contain a fixed multiple of eight examples; the trainer deliberately
does not duplicate or silently drop examples.

## 2. Inspect the grid

```bash
uv run python -m epiaudio.conditional_epiplexity.sweep \
  epiaudio/conditional_epiplexity/configs/sweeps/syntheory_intervals_4tokenizer.yaml \
  --dry-run
```

This prints all validated run configurations without training.

## 3. Run the sweep

```bash
uv run python -m epiaudio.conditional_epiplexity.sweep \
  epiaudio/conditional_epiplexity/configs/sweeps/syntheory_intervals_4tokenizer.yaml
```

The checked-in sweep enables Comet logging. Set `COMET_ML_API` before running.
Each point is tagged with `experiment-2`, `conditional-epiplexity`, the sweep
name, dataset, and tokenizer. Each tokenizer sweep also uploads a
`sweep-summary` experiment containing its `summary.json` and best-run metrics.

The tokenizer sweeps write separate checkpoints and machine-readable summaries
to:

```text
outputs/conditional_epiplexity/sweeps/syntheory_intervals_4tokenizer/
├── dac/
│   ├── checkpoints/
│   └── summary.json
├── encodec/
├── sqcodec/
└── xcodec/
```

Each `summary.json` contains that tokenizer's reported conditional epiplexity
and two-part-code curves. Selection pools its reported checkpoints, takes the
lower convex hull in training-compute/code-length space, retains the median hull
point from each run, and minimizes `two_part_code_bits` over the reduced
frontier. Each summary also records whether its runs were trained, resumed
locally, or reused from Comet. The production configuration creates four
tokenizer sweeps with four depths, twelve parameter budgets, and one
deterministic seed each. Configurations
whose total parameter budget cannot fit the minimum model width are skipped.

Re-running the command automatically queries Comet for completed matching
points and skips them. If a matching local checkpoint is incomplete, its model,
optimizer, RNG, accounting, and estimator state are restored before training
continues. To resume a differently named historical sweep, pass
`--resume-sweep NAME`; to force every point to restart, pass `--no-resume`.

The teacher remains strictly one-pass, and no teacher-student KL or requential
training is involved.
