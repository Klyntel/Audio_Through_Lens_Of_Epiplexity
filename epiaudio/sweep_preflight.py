"""Validate sweep invariants before a run can allocate GPU time."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS, TokenizerSpec


EXPECTED_METHOD = "pf_grid"


@dataclass(frozen=True)
class PreflightIssue:
    level: str
    message: str


@dataclass
class PreflightResult:
    sweep_path: Path
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)

    def error(self, label: str, message: str) -> None:
        self.checks.append((label, False, message))
        self.issues.append(PreflightIssue("ERROR", message))

    def warn(self, label: str, message: str) -> None:
        self.checks.append((label, False, f"WARN: {message}"))
        self.issues.append(PreflightIssue("WARN", message))

    def pass_check(self, label: str, detail: str) -> None:
        self.checks.append((label, True, detail))


def _parameter_values(cfg: DictConfig, key: str) -> list[Any]:
    parameters = OmegaConf.select(cfg, "parameters", default=None)
    node = parameters.get(key) if isinstance(parameters, DictConfig) else None
    if node is None:
        return []
    value = OmegaConf.select(node, "value", default=None)
    if value is not None:
        return [value]
    values = OmegaConf.select(node, "values", default=None)
    if values is None:
        return []
    raw = OmegaConf.to_container(values, resolve=True)
    return list(raw) if isinstance(raw, list) else [raw]


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"override must be KEY=VALUE, got {raw!r}")
    key, value_text = raw.split("=", 1)
    parsed = OmegaConf.create({"value": value_text})
    value: Any = OmegaConf.select(parsed, "value")
    lowered = value_text.lower()
    if lowered == "true":
        value = True
    elif lowered == "false":
        value = False
    elif lowered in {"null", "none"}:
        value = None
    else:
        try:
            value = int(value_text)
        except ValueError:
            try:
                value = float(value_text)
            except ValueError:
                pass
    return key, value


def _effective_values(cfg: DictConfig, key: str, overrides: dict[str, Any]) -> list[Any]:
    if key in overrides:
        return [overrides[key]]
    return _parameter_values(cfg, key)


def _single_value(
    result: PreflightResult,
    cfg: DictConfig,
    key: str,
    overrides: dict[str, Any],
) -> Any | None:
    values = _effective_values(cfg, key, overrides)
    if not values:
        result.error(key, f"parameters.{key} is missing")
        return None
    if len(values) != 1:
        result.error(key, f"parameters.{key} must have one invariant value, got {values}")
        return None
    return values[0]


def _infer_tokenizer(ds_path: Path) -> tuple[str, TokenizerSpec] | None:
    matches = [name for name in TOKENIZER_SPECS if ds_path.name.endswith(f"_{name}")]
    if not matches:
        return None
    name = max(matches, key=len)
    return name, TOKENIZER_SPECS[name]


def _check_expected(
    result: PreflightResult,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual == expected:
        result.pass_check(label, str(actual))
    else:
        result.error(label, f"{label}={actual!r}; expected {expected!r}")


def _check_model_value(
    result: PreflightResult,
    source: str,
    key: str,
    actual: Any,
    expected: int | None,
) -> None:
    label = f"{source} model.{key}"
    if expected is None:
        result.pass_check(label, "not applicable for continuous features")
    elif actual == expected:
        result.pass_check(label, str(actual))
    else:
        result.error(label, f"{label}={actual!r}; expected {expected}")

def _check_model_tokenizer_compatible(
    result: PreflightResult,
    source: str,
    v_value: Any,
    l_value: Any,
    spec: TokenizerSpec,
) -> None:
    """Compare dataset metadata's model values against a tokenizer spec."""
    _check_model_value(result, source, "V", v_value, spec.vocab_size)
    _check_model_value(result, source, "L", l_value, spec.sequence_length)


def validate_sweep(
    cfg: DictConfig,
    sweep_path: str | Path,
    *,
    method: str,
    override_args: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> PreflightResult:
    """Return every preflight error without starting a sweep."""
    path = Path(sweep_path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    result = PreflightResult(sweep_path=path)

    overrides: dict[str, Any] = {}
    for raw in override_args or []:
        try:
            key, value = _parse_override(raw)
            overrides[key] = value
        except ValueError as exc:
            result.error("override", str(exc))

    _check_expected(result, "method", method.lower(), EXPECTED_METHOD)
    metric_name = str(OmegaConf.select(cfg, "metric.name", default=""))
    expected_train_student = {"K(M)": False, "K(M)_req": True}.get(metric_name)
    if expected_train_student is None:
        result.error(
            "metric",
            f"metric.name={metric_name!r}; expected 'K(M)' or 'K(M)_req' so student mode is unambiguous",
        )
    else:
        result.pass_check("metric", metric_name)
        _check_expected(
            result,
            "train_student",
            _single_value(result, cfg, "train_student", overrides),
            expected_train_student,
        )

    ds_values = _effective_values(cfg, "ds_path", overrides)
    if not ds_values:
        result.error("ds_path", "parameters.ds_path is missing")

    tokenizers: set[str] = set()
    for raw_ds_path in ds_values:
        ds_path = Path(str(raw_ds_path))
        if not ds_path.is_absolute():
            ds_path = root / ds_path
        inferred = _infer_tokenizer(ds_path)
        if inferred is None:
            result.error(
                "tokenizer",
                f"cannot infer a known tokenizer from dataset path '{raw_ds_path}'",
            )
            continue
        tokenizer, spec = inferred
        tokenizers.add(tokenizer)

        metadata_path = ds_path / "metadata.yaml"
        if not metadata_path.is_file():
            result.error(
                "metadata",
                f"missing {metadata_path}; model.L/model.V must come from dataset metadata",
            )
        else:
            metadata = OmegaConf.load(metadata_path)
            metadata_v = OmegaConf.select(metadata, "model.V", default=None)
            metadata_l = OmegaConf.select(metadata, "model.L", default=None)
            _check_model_tokenizer_compatible(result, "metadata", metadata_v, metadata_l, spec)

            T_eval = _single_value(result, cfg, "T_eval", overrides)
            batch_size = _single_value(result, cfg, "B", overrides)
            if (
                T_eval is not None
                and batch_size is not None
                and metadata_l is not None
                and T_eval < batch_size * metadata_l
            ):
                result.error(
                    "parameters",
                    "T_eval must be at least B*L or no eval steps will occur"
                )

        for filename in ("train.bin", "test.bin"):
            data_path = ds_path / filename
            if data_path.is_file():
                result.pass_check(filename, str(data_path))
            else:
                dataset = ds_path.name.removesuffix(f"_{tokenizer}")
                result.error(
                    filename,
                    f"missing {data_path}; run: uv run python "
                    f"epiaudio/dataset/prepare_audio.py {dataset} {tokenizer}",
                )

    if len(tokenizers) > 1:
        result.warn("tokenizers", f"ds_path values use multiple tokenizers: {sorted(tokenizers)}")

    env_path = root / ".env"
    if env_path.is_file() and any(
        line.strip().startswith("COMET_ML_API=") for line in env_path.read_text().splitlines()
    ):
        result.pass_check(".env", "COMET_ML_API is configured")
    else:
        result.warn(".env", "COMET_ML_API is missing; results will not log to CometML")

    return result


def print_preflight(result: PreflightResult) -> None:
    print(f"SWEEP PREFLIGHT: {result.sweep_path}")
    for label, passed, detail in result.checks:
        marker = "PASS" if passed else ("WARN" if detail.startswith("WARN:") else "FAIL")
        print(f"  {marker:4} {label:<18} {detail}")
    if result.ready:
        print("\nREADY TO RUN")
    else:
        error_count = sum(issue.level == "ERROR" for issue in result.issues)
        print(f"\nBLOCKED: {error_count} issue(s) must be fixed before running")
