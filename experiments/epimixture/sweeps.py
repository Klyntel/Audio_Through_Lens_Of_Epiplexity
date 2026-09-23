"""Sweep materialization, transfer evaluation, and result analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .config import SCHEMA_VERSION, ExperimentSpec, FixedGeneralizationSpec, write_json
from .data import dataset_path, source_dataset_path

GENERALIZATION_DIRNAME = "generalization"


def generate_sweeps(spec: ExperimentSpec, *, overwrite: bool = False) -> list[Path]:
    """Materialize one existing-PF-grid sweep config per source or mixture."""
    output_dir = spec.output_root / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for dataset_id in spec.dataset_ids:
        output_path = output_dir / f"{dataset_id}.yaml"
        cfg = OmegaConf.load(spec.sweep_template)
        cfg.name = f"{spec.name}__{dataset_id}"
        cfg.seed = spec.seed
        cfg.parameters["ds_path"] = {"values": [str(dataset_path(spec, dataset_id))]}
        cfg.parameters["save"] = {"value": False}
        if output_path.exists() and not overwrite:
            if _config_sha256(OmegaConf.load(output_path)) == _config_sha256(cfg):
                generated.append(output_path)
                continue
            print(f"[epimixture] updating changed sweep config: {output_path}")
        OmegaConf.save(config=cfg, f=output_path)
        generated.append(output_path)
    return generated


def _config_sha256(config: Any) -> str:
    """Hash literal sweep YAML so completed work is tied to its exact grid."""
    content = OmegaConf.to_yaml(config, resolve=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _path_sha256(path: Path) -> str:
    return _config_sha256(OmegaConf.load(path))


def _matching_fixed_point(sweeper: Any, selected_point: Mapping[str, Any]) -> dict[str, Any]:
    for fixed, point in sweeper.iter_points():
        if point == selected_point:
            return fixed
    raise RuntimeError("The selected sweep point no longer exists in its materialized sweep config.")


def run_sweeps(spec: ExperimentSpec, *, backend: str = "torch") -> None:
    """Run existing sweeps sequentially and persist the selected transfer point."""
    from epiaudio.sweep import _build_sweeper, load_sweep_configs

    sweep_paths = generate_sweeps(spec)
    selections_dir = spec.output_root / "selections"
    selections_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id, sweep_path in zip(spec.dataset_ids, sweep_paths, strict=True):
        configs = load_sweep_configs(str(sweep_path))
        if len(configs) != 1:
            raise ValueError(f"EpiMixture requires one sweep config per file: {sweep_path}")
        config = configs[0]
        method = str(OmegaConf.select(config, "method", default="pf_grid"))
        sweeper = _build_sweeper(config, method, [], backend=backend)
        selected = sweeper.sweep()
        if not selected:
            raise RuntimeError(f"Sweep did not select a best point: {dataset_id}")
        point = selected["point"]
        write_json(selections_dir / f"{dataset_id}.json", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path(spec, dataset_id)),
            "sweep_path": str(sweep_path),
            "sweep_config_sha256": _path_sha256(sweep_path),
            "sweep_name": sweeper.sweep_name,
            "selection_source": "local_sweep",
            "selected": selected,
            "fixed": _matching_fixed_point(sweeper, point),
        })


def _write_comet_selection(
    spec: ExperimentSpec,
    dataset_id: str,
    sweep_path: Path,
    sweep_name: str,
) -> None:
    """Materialize one complete historical PF-grid selection from Comet only.

    A missing or incomplete point is an error rather than permission to start
    training.  This is deliberately stricter than normal sweep resumption: the
    ``--reuse-comet-sweeps`` mode must never silently rerun a sweep.
    """
    from epiaudio.sweep import _build_sweeper, load_sweep_configs

    sweep_cfgs = load_sweep_configs(str(sweep_path))
    if len(sweep_cfgs) != 1:
        raise ValueError(f"EpiMixture requires one sweep config per file: {sweep_path}")
    sweep_cfg = sweep_cfgs[0]
    method = str(OmegaConf.select(sweep_cfg, "method", default="pf_grid"))
    sweeper = _build_sweeper(sweep_cfg, method, [], backend="torch", sweep_name=sweep_name)
    candidates: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for fixed, point in sweeper.iter_points():
        run_cfg = sweeper.build_run_config(fixed, point, f"{sweep_name}__comet_lookup")
        resumed = sweeper._find_resumed_experiment(  # type: ignore[attr-defined]
            run_cfg,
            bool(OmegaConf.select(run_cfg, "train_student", default=True)),
            sweeper._ema_active(run_cfg, "teacher_ema"),  # type: ignore[attr-defined]
            sweeper._ema_active(run_cfg, "student_ema"),  # type: ignore[attr-defined]
        )
        if resumed is None:
            missing.append(point)
            continue
        experiment_key, _, epiplexity, code_length = resumed
        candidates.append({
            "run_name": f"{sweep_name}__comet:{experiment_key}",
            "point": point,
            "code_length": code_length,
            "epiplexity": epiplexity,
            "comet_experiment_key": experiment_key,
        })
    if missing:
        raise RuntimeError(
            f"Historical Comet sweep {sweep_name!r} is incomplete for {dataset_id!r}: "
            f"{len(missing)} of {len(missing) + len(candidates)} configured point(s) are missing. "
            "Refusing to rerun them in --reuse-comet-sweeps mode."
        )
    selected = min(candidates, key=lambda candidate: float(candidate["code_length"]))
    write_json(spec.output_root / "selections" / f"{dataset_id}.json", {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path(spec, dataset_id)),
        "sweep_path": str(sweep_path),
        "sweep_config_sha256": _path_sha256(sweep_path),
        "sweep_name": sweep_name,
        "selection_source": "comet",
        "selected": selected,
        "fixed": _matching_fixed_point(sweeper, dict(selected["point"])),
    })


def reuse_comet_sweeps(spec: ExperimentSpec) -> None:
    """Select every existing PF-grid result from Comet without training."""
    sweep_paths = generate_sweeps(spec)
    (spec.output_root / "selections").mkdir(parents=True, exist_ok=True)
    for dataset_id, sweep_path in zip(spec.dataset_ids, sweep_paths, strict=True):
        sweep_name = spec.comet_sweeps.get(dataset_id, f"{spec.name}__{dataset_id}")
        _write_comet_selection(spec, dataset_id, sweep_path, sweep_name)


def _load_selection(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read selection {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("selected"), dict):
        raise ValueError(f"Invalid selection record: {path}")
    return value


def _evaluate_checkpoint(
    cfg: DictConfig, checkpoint: Path, heldout_path: Path, eval_tokens: int
) -> dict[str, float | int]:
    """Evaluate a saved unconditional model's causal NLL on unseen token data."""
    import torch  # type: ignore[import-not-found]
    import epiaudio.model_torch as model_lib
    from epiaudio.train_torch import _ce_loss, get_in_out, make_ds_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = np.int16 if int(cfg.model.V) <= 32_768 else np.int32
    sequence_length = int(cfg.model.L)
    raw = np.memmap(heldout_path / "test.bin", dtype=dtype, mode="r")
    if len(raw) < sequence_length:
        raise ValueError(f"Held-out test split is smaller than one sequence: {heldout_path}")
    batch_size = min(int(cfg.B), len(raw) // sequence_length)
    get_batch, _ = make_ds_loader(str(heldout_path), "test", sequence_length, batch_size, dtype=dtype)
    batches = max(1, min(eval_tokens // (batch_size * sequence_length), len(raw) // (batch_size * sequence_length)))
    model = model_lib.create_model(cfg.model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index in range(batches):
            x, y, weights = get_in_out(get_batch(index), device=device)
            losses.append(float(_ce_loss(model(x), y, weights).item()))
    nll_nats = float(np.mean(losses))
    return {
        "batches": batches,
        "evaluated_tokens": batches * batch_size * sequence_length,
        "nll_nats_per_token": nll_nats,
        "nll_bits_per_token": nll_nats / math.log(2),
        "perplexity": math.exp(nll_nats),
    }


def _fixed_regime_manifest(regime: FixedGeneralizationSpec) -> dict[str, int | float | str | bool]:
    """Stable, serializable record of the pre-declared generalization regime."""
    return {
        "model.N": regime.depth,
        "model.D": regime.width,
        "model.dh": regime.head_dim,
        "seed": regime.seed,
        "T": regime.train_tokens,
        "B": regime.batch_size,
        "A": regime.accumulation_steps,
        "num_evals": regime.num_evals,
        "T_eval": regime.eval_tokens,
        "opt.lr": regime.learning_rate,
        "opt.schedule": regime.schedule,
        "opt.warmup_tokens": regime.warmup_tokens,
        "opt.b1": regime.beta1,
        "opt.b2": regime.beta2,
        "opt.wd": regime.weight_decay,
        "teacher_ema": regime.teacher_ema,
        "compile": regime.compile,
    }


def _fixed_regime_sha256(regime: FixedGeneralizationSpec) -> str:
    manifest = _fixed_regime_manifest(regime)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def _fixed_run_config(sweeper: Any, spec: ExperimentSpec, dataset_id: str, checkpoint: Path) -> DictConfig:
    """Compose the fixed generalization config without consulting a sweep point."""
    regime = spec.fixed_generalization
    fixed = {
        "ds_path": str(dataset_path(spec, dataset_id)),
        "seed": regime.seed,
        "T": regime.train_tokens,
        "T_eval": regime.eval_tokens,
        "B": regime.batch_size,
        "A": regime.accumulation_steps,
        "num_evals": regime.num_evals,
        "train_student": False,
        "train_teacher": True,
        "teacher_ema": regime.teacher_ema,
        "student_ema": 0,
        "model.N": regime.depth,
        "model.P": None,
        "model.D": regime.width,
        "model.dh": regime.head_dim,
        "opt.lr": regime.learning_rate,
        "opt.schedule": regime.schedule,
        "opt.warmup_tokens": regime.warmup_tokens,
        "opt.b1": regime.beta1,
        "opt.b2": regime.beta2,
        "+opt.wd": regime.weight_decay,
        "+compile": regime.compile,
    }
    run_name = f"{spec.name}__{dataset_id}__fixed_generalization"
    cfg = sweeper.build_run_config(fixed, {}, run_name)
    cfg.save = str(checkpoint)
    cfg.run_name = run_name
    cfg.sweep_name = f"{spec.name}__fixed_generalization"
    cfg.tag = "epimixture-fixed-generalization"
    return cfg


def run_fixed_generalization(spec: ExperimentSpec) -> None:
    """Train one pre-declared model per dataset and evaluate the unseen split.

    Sweep selections are intentionally not used to choose the architecture,
    optimizer, duration, or checkpoint.  They remain only the epiplexity value
    paired with this independently trained held-out score during analysis.
    """
    from epiaudio.sweep import _build_sweeper, load_sweep_configs

    heldout_path = source_dataset_path(spec, spec.heldout)
    if not (heldout_path / "test.bin").is_file():
        raise ValueError(f"Held-out dataset is not prepared: {heldout_path}")
    # Keep phase-2 artifacts separate from legacy transfer outputs. This makes
    # it impossible to load, overwrite, or mistake a sweep-era checkpoint for
    # a freshly trained fixed-regime model.
    results_dir = spec.output_root / GENERALIZATION_DIRNAME
    checkpoints_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id in spec.dataset_ids:
        selection = _load_selection(spec.output_root / "selections" / f"{dataset_id}.json")
        sweep_path = Path(selection["sweep_path"])
        sweep_cfg = load_sweep_configs(str(sweep_path))[0]
        sweeper = _build_sweeper(sweep_cfg, str(OmegaConf.select(sweep_cfg, "method", default="pf_grid")), [], backend="torch")
        checkpoint = checkpoints_dir / f"{dataset_id}.pt"
        selected_cfg = _fixed_run_config(sweeper, spec, dataset_id, checkpoint)
        from epiaudio.train_torch import train_and_evaluate

        train_and_evaluate(selected_cfg)
        write_json(results_dir / f"{dataset_id}.json", {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "training_dataset": str(dataset_path(spec, dataset_id)),
            "heldout_dataset": spec.heldout,
            "heldout_path": str(heldout_path),
            "checkpoint": str(checkpoint),
            "sweep_config_sha256": selection.get("sweep_config_sha256"),
            "fixed_generalization": _fixed_regime_manifest(spec.fixed_generalization),
            "fixed_generalization_sha256": _fixed_regime_sha256(spec.fixed_generalization),
            "selected": selection["selected"],
            "transfer": _evaluate_checkpoint(selected_cfg, checkpoint, heldout_path, spec.transfer_eval_tokens),
        })


def run_transfer(spec: ExperimentSpec) -> None:
    """Compatibility alias for the fixed-regime generalization stage."""
    run_fixed_generalization(spec)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def analyze(spec: ExperimentSpec) -> Path:
    """Relate selected epiplexity to held-out NLL and report predictive evidence."""
    records: list[dict[str, Any]] = []
    mixture_names = {mixture.name for mixture in spec.mixtures}
    for dataset_id in spec.dataset_ids:
        path = spec.output_root / GENERALIZATION_DIRNAME / f"{dataset_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing transfer result for {dataset_id}: {path}")
        record = _load_selection(path)
        selected = record["selected"]
        transfer = record.get("transfer")
        if not isinstance(transfer, Mapping) or not isinstance(selected.get("epiplexity"), (int, float)):
            raise ValueError(f"{dataset_id} has an invalid transfer result.")
        records.append({
            "dataset_id": dataset_id,
            "kind": "mixture" if dataset_id in mixture_names else "source",
            "epiplexity_bits": float(selected["epiplexity"]),
            "two_part_code_bits": selected.get("code_length"),
            "heldout_nll_bits_per_token": transfer.get("nll_bits_per_token"),
            "heldout_perplexity": transfer.get("perplexity"),
            "evaluated_tokens": transfer.get("evaluated_tokens"),
        })
    output_dir = spec.output_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(records[0])
    (output_dir / "comparison.csv").write_text(
        ",".join(columns) + "\n" + "\n".join(
            ",".join(str(record[column]) for column in columns) for record in records
        ) + "\n", encoding="utf-8"
    )
    x = np.array([record["epiplexity_bits"] for record in records], dtype=float)
    y = np.array([record["heldout_nll_bits_per_token"] for record in records], dtype=float)
    pearson, spearman = _pearson(x, y), _pearson(_rank(x), _rank(y))
    loo_mae: float | None = None
    if len(records) >= 3 and np.std(x) > 0:
        errors = []
        for index in range(len(records)):
            keep = np.arange(len(records)) != index
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            errors.append(abs(float(slope * x[index] + intercept) - y[index]))
        loo_mae = float(np.mean(errors))
    lines = [
        f"# {spec.name}: epiplexity and held-out transfer", "",
        f"Held-out dataset: `{spec.heldout}`; tokenizer: `{spec.tokenizer}`.",
        "Lower held-out NLL is better. Every row uses the same pre-declared transformer and training regime; the sweep contributes only its `K(M)` measurement and never supplies the evaluated checkpoint.", "",
        "| Training dataset | Kind | K(M) (bits) | Held-out NLL (bits/token) | Perplexity |",
        "| --- | --- | ---: | ---: | ---: |",
        *[f"| {record['dataset_id']} | {record['kind']} | {record['epiplexity_bits']:.6g} | {float(record['heldout_nll_bits_per_token']):.6g} | {float(record['heldout_perplexity']):.6g} |" for record in records],
        "",
        f"Pearson correlation, K(M) vs held-out NLL: `{pearson:.6g}`." if pearson is not None else "Pearson correlation is undefined (need variation in at least two results).",
        f"Spearman rank correlation: `{spearman:.6g}`." if spearman is not None else "Spearman correlation is undefined (need variation in at least two results).",
        f"Leave-one-out linear prediction MAE: `{loo_mae:.6g}` bits/token." if loo_mae is not None else "Leave-one-out prediction needs at least three non-identical epiplexity values.",
        "", "A positive correlation means higher epiplexity coincided with worse held-out NLL; a negative correlation means it coincided with better held-out NLL. This is evidence for this configuration, not a universal mixture rule.",
    ]
    report = output_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def selections_complete(spec: ExperimentSpec) -> bool:
    for dataset_id in spec.dataset_ids:
        selection_path = spec.output_root / "selections" / f"{dataset_id}.json"
        sweep_path = spec.output_root / "sweeps" / f"{dataset_id}.yaml"
        if not selection_path.is_file() or not sweep_path.is_file():
            return False
        try:
            selection = _load_selection(selection_path)
        except ValueError:
            return False
        if selection.get("sweep_config_sha256") != _path_sha256(sweep_path):
            return False
    return True


def transfer_complete(spec: ExperimentSpec) -> bool:
    for dataset_id in spec.dataset_ids:
        transfer_path = spec.output_root / GENERALIZATION_DIRNAME / f"{dataset_id}.json"
        selection_path = spec.output_root / "selections" / f"{dataset_id}.json"
        if not transfer_path.is_file() or not selection_path.is_file():
            return False
        try:
            transfer = _load_selection(transfer_path)
            selection = _load_selection(selection_path)
        except ValueError:
            return False
        if transfer.get("sweep_config_sha256") != selection.get("sweep_config_sha256"):
            return False
        if transfer.get("fixed_generalization_sha256") != _fixed_regime_sha256(spec.fixed_generalization):
            return False
    return True
