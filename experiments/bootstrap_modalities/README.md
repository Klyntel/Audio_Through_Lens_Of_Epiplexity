# Cross-modal epiplexity replication

This experiment compares the PyTorch prequential epiplexity estimate across
OpenWebText, Lichess puzzles, CIFAR-5M, and FSD50K/EnCodec using a matched
training budget of **31,250,000 tokens per run**. It reuses the reference
tokenization contracts from `epiplexity/picodo/dataset` while writing the
metadata and signed-int16 memmaps expected by EpiAudio's sweep runner.

## Dataset contracts

| Dataset | Output directory | Tokenizer | Vocabulary | Sequence length |
| --- | --- | --- | ---: | ---: |
| OpenWebText | `data/openwebtext_ascii96` | newline + printable ASCII | 96 | 512 |
| Lichess puzzles | `data/lichess_puzzles` | reference chess characters with BOS/EOS | 64 | 512 |
| CIFAR-5M | `data/cifar5m_grayscale` | flattened grayscale 32x32 pixels | 256 | 1024 |
| FSD50K | `data/fsd50k_encodec` | existing EnCodec representation | 1024 | 375 |

OpenWebText keeps only documents entirely representable by the reference
newline-plus-printable-ASCII tokenizer, then appends a newline document
separator. Lichess filters puzzles longer than the reference 512-token
context, preserves the padded representation, and writes target masks for
downstream work, although the unconditional sweep trains on every token.
CIFAR-5M applies the reference `RGB.mean(-1)` grayscale conversion and holds
out 2,000 seeded images.

## Prepare

Run each command from the repository root. Preparation refuses to overwrite
an existing output directory, so an interrupted or changed preparation must
use a new directory or be cleaned up explicitly.

```bash
uv run python -m epiaudio.dataset.prepare_discrete openwebtext
uv run python -m epiaudio.dataset.prepare_discrete lichess
```

CIFAR-5M is public as six NPZ files, but it is intentionally not downloaded by
the preparer. Download the exact shards first, then give every path explicitly:

```bash
uv run python -m epiaudio.dataset.prepare_discrete cifar5m \
  --shard /datasets/cifar5m/part0.npz \
  --shard /datasets/cifar5m/part1.npz \
  --shard /datasets/cifar5m/part2.npz \
  --shard /datasets/cifar5m/part3.npz \
  --shard /datasets/cifar5m/part4.npz \
  --shard /datasets/cifar5m/part5.npz
```

For a bounded smoke preparation, use a separate output and a small Lichess
test split or tiny local CIFAR shard fixture; do not change the production
tokenizer or split parameters.

## Sweep

First inspect all prepared representations:

```bash
uv run python -m epiaudio.sweep epiaudio/sweeps/bootstrap_modalities.yaml --preflight-only
```

Then run the complete 192-run (four modalities x 48 model points) PyTorch
Pareto-front sweep:

```bash
uv run python -m epiaudio.sweep epiaudio/sweeps/bootstrap_modalities.yaml --backend torch
```

The resolved per-modality metadata sets `model.V` and `model.L`; do not add
those values as sweep-wide overrides. FSD50K/EnCodec must already exist at
`data/fsd50k_encodec`. To use a different audio representation, replace that
path in `ds_path` and retain the same `T=31250000` budget.
