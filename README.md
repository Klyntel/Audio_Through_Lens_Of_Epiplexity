# EpiAudio

Based heavily on the work conducted by Finzi et al [https://arxiv.org/pdf/2601.03220]. 
This work focuses on being a harness for testing epiplexity over audio datasets. 



## Install

### Installed Software

Required: UV
Make sure to install uv via https://docs.astral.sh/uv/getting-started/installation/


1. Make a directory called `epi`.
2. Download and extract the [anonymous audio-preprocessing repository](https://anonymous.4open.science/r/epiaudio_preprocessing-5155/README.md) into `epi/epiaudio_preprocessing`.
3. Download and extract this anonymous repository into `epi/Audio_Through_Lens_Of_Epiplexity`.
4. Clone the required Epiplexity source checkout and create a CPU PyTorch environment:

```bash
cd epi/Audio_Through_Lens_Of_Epiplexity
git clone https://github.com/shikaiqiu/epiplexity.git epiplexity
git -C epiplexity checkout 3aa12a1be6a413fe9eaa41374a6a46a4a0d4e100
uv sync --extra pytorch --extra pytorch-cpu
```

The two extracted repositories must remain sibling directories with these names:
the project resolves `audio_preprocessing` from `../epiaudio_preprocessing`.

### Create Env

The installation command above installs CPU PyTorch. To select a different
backend, choose one concrete extra:

```bash
# PyTorch
uv sync --extra pytorch --extra pytorch-cu128
uv sync --extra pytorch --extra pytorch-cu130
uv sync --extra pytorch --extra pytorch-rocm

# JAX
uv sync --extra jax --extra jax-cpu
uv sync --extra jax --extra jax-cu12
uv sync --extra jax --extra jax-cu13
uv sync --extra jax --extra jax-rocm
uv sync --extra jax --extra jax-tpu
```

### Environment Variables

Create a file `.env` and copy the following into it

```
# Make a copy of this file and create .env file
# DO NOT COMMIT THIS FILE UNLESS YOU KNOW WHAT YOU ARE DOING
COMET_ML_API="<INSERT_API_KEY>"
```


## Running Natural Data Experiments

NOTE: This will run a large training run, I'll do some tests
to find a less heavy model to run but below was able to do a single training step on 40gbs ram
before it was killed by OOM. 

```bash
cd epiaudio
CUDA_VISIBLE_DEVICES=0 python main.py -cn chess \
  wandb_mode=online \
  wandb_project=requential \
  tag=test \
  train_student=true \
  train_teacher=true \
  teacher_ema=50 \
  student_ema=50 \
  model.N=3 \
  model.P=5 \
  ds_path=chess \
  opt.lr=2 \
  B=256 \
  model.L=512 \
  max_kl=0.1 \
  A=8 \
  opt.schedule=const \
  opt.warmup_tokens=16384000 \
  T=5000000000 \
  T_eval=1000000 \
  num_evals=50 \
  seed=0 \
  save=false
```

## Running Sweeps

A sweep runs `main.py`'s training loop many times over a grid of
hyperparameters, logs each run to Comet ML (named and tagged with the sweep),
and reports the configuration that minimizes epiplexity (`K(M)`).

Run sweeps from the **repository root** (relative paths like `ds_path=data/fsd50k`
are resolved against the current directory):

```bash
uv run python -m epiaudio.sweep <path-to-sweep.yaml> --backend torch  # PyTorch DDP (default)
uv run python -m epiaudio.sweep <path-to-sweep.yaml> --backend jax    # JAX/Flax
```

Apply extra Hydra overrides to every run with `--override` (repeatable), e.g. to
point at an absolute dataset path or shrink a run for a quick CPU test:

```bash
uv run python -m epiaudio.sweep <path-to-sweep.yaml> \
  --override ds_path=/abs/data/fsd50k \
  --override T=2048000
```

A sweep is described by a wandb-style YAML:
`command` selects the Hydra base config (`-cn <name>`, resolved against
`epiplexity/picodo/configs`) plus fixed overrides, and `parameters` defines the
grid (`value:` for a fixed value, `values: [...]` for a swept axis). When the
sweep finishes it prints all runs ranked by `K(M)` and the best (lowest) one.

Note: the requential estimate `K(M)_req` is only logged when a student is
distilled (`train_student=true`); teacher-only sweeps report the prequential
`K(M)`.

`pf_grid` sweeps also create one Comet summary experiment per dataset/tokenizer
after the Pareto-front report is built. The summary experiment logs the selected
epiplexity, compute and two-part code length, and uploads the Pareto CSVs, plot,
resolved sweep config, and summary JSON. Local files remain in `sweep_outputs/`.
If the automatic upload fails—or for outputs created before this feature—retry
without rerunning training:

```bash
# Inspect matching local bundles first.
uv run python -m epiaudio.upload_sweep_outputs \
  --sweep birdset_hsn \
  --dry-run

# Upload them to the default epi-audio/epiaudio Comet project.
uv run python -m epiaudio.upload_sweep_outputs \
  --sweep birdset_hsn
```

## Cross-modal replication

The OpenWebText, Lichess-puzzle, and CIFAR-5M preparation and matched-token
PyTorch sweep are documented in
[`experiments/bootstrap_modalities/README.md`](experiments/bootstrap_modalities/README.md).


## Dev

For linting, use `uv run python lint.py`. 

Don't run it over the whole repo since it will flag errors in `epiplexity`.
When developing, don't edit files in epiplexity. We should build off the library in `epiaudio`. 


## Basic Demos

### Replicate Epiplexity Experiments for Testing

To test an older syntheic experiment, run

`CUDA_VISIBLE_DEVICES=0 uv run python epiplexity/experiments/soi.py`

soi can be replaced by another experiment in the folder. 

### Running Experiments in Epiplexity.

To test the audio epiplexity experiments, run


`CUDA_VISIBLE_DEVICES=0 uv run python eiaudio/experiments/soi.py`
