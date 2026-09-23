# EpiMixture

EpiMixture tests whether an unconditional model's epiplexity on an audio source
or controlled mixture predicts its next-token performance on a fully unseen
audio distribution. It uses the existing EpiAudio token preparation, PF-grid
sweeper, PyTorch trainer, and Comet logging; it does not create a second
training implementation.

## Code layout

The public module `epimixture.py` is intentionally only the CLI and a small
compatibility surface. The implementation is organized by responsibility:

- `config.py`: validated experiment models and atomic artifact writes.
- `data.py`: registry access, token preparation, planning, and deterministic mixtures.
- `sweeps.py`: sweep YAMLs, Comet-only sweep reuse, fixed-regime generalization,
  and reporting.
- `runner.py`: the resumable end-to-end lifecycle used by the `run` command.

## Design

1. Select three to five source datasets from EpiAudio's merged dataset registry
   (which includes `audio_preprocessing` loaders) and one disjoint held-out
   dataset.
2. Prepare each source and the held-out dataset with one discrete tokenizer.
3. Create named mixtures by sampling complete five-second token clips without
   replacement at declared source weights. Train and test mixtures are sampled
   independently and every mixture records an immutable source manifest.
4. Materialize and run one existing PF-grid sweep per original dataset and per
   mixture. The selected candidate is the point with the lowest two-part code
   length, with its `K(M)` retained as the epiplexity measurement.
5. Train one fixed, pre-declared transformer per original dataset and mixture,
   then evaluate its causal next-token NLL on the unseen target's test split.
   The sweep-selected configuration is never used as this checkpoint.
6. Report source/mixture comparisons, Pearson and Spearman correlations, and a
   leave-one-out linear prediction error where there are enough results.

This measures unconditional distribution transfer, not downstream supervised
classification. Each source and mixture must use the same tokenizer; the
held-out dataset must be absent from `sources`.

Two sources are permitted only for a deliberately small smoke test. Use three
to five sources for any result intended to support mixture rules. Set
`preparation.max_samples_per_split` to bound a smoke test before tokenization;
this selects a deterministic subset after the loader's split normalization.
For a preexisting raw archive, use `raw_roots` in your local config rather than
copying or symlinking the archive into the repository.

## Commands

Run from the repository root:

```bash
# Run the complete experiment: prepare, mix, sweep, transfer, and analyze.
uv run python -m experiments.epimixture.epimixture example.yaml run
```

`run` is resumable: existing complete prepared datasets and mixtures are reused.
Pass `--overwrite` only when intentionally replacing generated datasets,
mixtures, or sweep YAMLs. It records one epiplexity selection per training
dataset, but separately trains a fixed generalization model before writing the
final report below `output_root`.

### Fixed generalization regime

The held-out test uses a transformer declared before inspecting any sweep
result. The default is intentionally simple: 8 blocks, width 512, head width
64, 31.25M training tokens, batch size 256, and no student model. The
tokenizer's vocabulary and sequence length are read from each dataset's
metadata; all other model and optimizer choices are fixed by the design YAML.

Declare the regime explicitly for a publishable experiment, for example:

```yaml
generalization:
  model: {N: 8, D: 512, dh: 64}
  training:
    seed: 20260825
    T: 31250000
    B: 256
    A: 8
    num_evals: 20
    T_eval: 131072
    lr: 2.0
    schedule: const
    warmup_tokens: 16384
    b1: 0.9
    b2: 0.95
    weight_decay: 0.0
    teacher_ema: 50
    compile: false
```

Changing this block invalidates prior generalization results, while existing
sweep selections remain reusable because they only provide the epiplexity axis.

### Reuse an existing Comet sweep

To avoid rerunning completed PF-grid sweeps, record their names per training
dataset and use the flag below. The importer requires every configured point to
exist and have completed in Comet; it fails rather than training a missing
point.

```yaml
comet_sweeps:
  demand: existing_demand_sweep
  urbansound: existing_urbansound_sweep
  balanced: existing_balanced_mixture_sweep
```

```bash
uv run python -m experiments.epimixture.epimixture example.yaml run --reuse-comet-sweeps
```

Use `list-registry`, `plan`, `prepare`, `build-mixtures`, `generate-sweeps`,
`run-sweeps`, `generalize`, or `analyze` only to inspect or recover an
interrupted stage; they are not required for normal execution. `transfer`
remains a compatibility alias for `generalize`.

## Dataset suggestions

For an `audio_preprocessing`-only, cross-domain pilot, start with `arte`,
`demand`, and `eigenscape`, holding out `avspeech`. Inspect dataset sizes first:
the current AVSpeech loader intentionally limits itself to 100 rows, so it is a
small transfer smoke target rather than a final benchmark.

For a larger controlled first study, use the existing SynthTheory registry
entries (`syntheory_chords`, `syntheory_intervals`,
`syntheory_time_signatures`, and `syntheory_simple_progressions`) and hold out
a fifth, separately registered distribution. Those datasets make it easier to
control split size and isolate mixture effects. Record any decision to use the
broader EpiAudio registry rather than only `audio_preprocessing` in the design
YAML and final report.

## Remote host

Before running compute stages on `38.80.81.100`, confirm the checkout, commit,
dataset root, generated sweep YAMLs, `.env`, and GPU availability. Run each
materialized YAML with `--preflight-only` first. Do not overwrite the remote
worktree or data directories as part of this experiment.
