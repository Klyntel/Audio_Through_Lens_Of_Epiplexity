"""Inventory dataset names logged by EpiAudio sweep experiments in Comet.

Only dataset-related parameter values and aggregate counts are written. API
keys, experiment keys, and unrelated parameters are never persisted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Iterable
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from dotenv import load_dotenv


COMET_API = "https://www.comet.com/api/rest/v2"
DEFAULT_WORKSPACE = "epi-audio"
DEFAULT_PROJECTS = (
    "audio-model-sustained",
    "bioacoustics",
    "classification",
    "conditional-epiplexity",
    "epiaudio",
    "general",
    "next-token-prediction",
    "nolan-chai",
    "requential",
)
DATASET_PARAMETERS = (
    "ds_path",
    "dataset",
    "dataset_name",
    "dataset-dir",
    "config|data|dataset_dir",
    "training_config|data|dataset_dir",
    "data|remote",
)
TOKENIZERS = (
    "dac",
    "encodec",
    "sqcodec",
    "xcodec",
    "wavtokenizer",
    "whisper",
    "hubert",
)
PUBLIC_AUDIO_DATASETS = frozenset(
    {
        "aishell1",
        "aishell3",
        "avspeech_smoke",
        "bat",
        "clothoaqa",
        "cochlscene",
        "common_voice",
        "datased",
        "demand",
        "eigenscape",
        "esd",
        "fake_or_real_original",
        "fleurs",
        "fma_small",
        "libritts",
        "macs",
        "meld",
        "mls",
        "multivox",
        "nonspeech7k",
        "ravdess",
        "ravdess_speech",
        "sonyc_ust",
        "spatial_librispeech",
        "tau_nigens21",
        "tau_urban_2022",
        "toyadmos_toycar",
        "toyadmos_toyconveyor",
        "toyadmos_toytrain",
        "tut2016_acoustic_scenes",
        "tut2017_acoustic_scenes",
        "urbansound",
        "vggsound",
        "vocal_sound_16k",
        "vocal_sound_44k",
        "vocalsound_16k",
        "vocalsound_44k",
        "voxpopuli_en",
    }
)
EPIAUDIO_NATIVE_DATASETS = frozenset(
    {
        "birdset_hsn",
        "fsd50k",
        "locata",
        "starss23",
        "syntheory_chords",
        "syntheory_intervals",
        "syntheory_notes",
        "syntheory_scales",
        "syntheory_simple_progressions",
        "syntheory_tempos",
        "syntheory_time_signatures",
    }
)
DERIVED_DATASETS = frozenset(
    {
        "eigenscape_fullwindow5s_native_v1",
        "epimixture_demand_urbansound",
        "epimixture_fsd50k_demand_urbansound",
        "locata_fullwindow5s_native_v1",
        "starss23_fullwindow5s_native_v1",
        "tau_urban_2022_fullwindow5s_native_v1",
        "tut_acoustic_scenes_2016_fullwindow5s_native_v1",
        "vggsound_00",
    }
)
NON_AUDIO_DATASETS = frozenset(
    {
        "cifar5m_grayscale",
        "lichess_puzzles",
        "openwebtext_ascii96",
    }
)
_TOKENIZED_NAME = re.compile(
    rf"^(.*?)_(?:{'|'.join(TOKENIZERS)})(?:_|$)",
    flags=re.IGNORECASE,
)


def classify_dataset_name(name: str) -> str:
    """Classify an inventory name by how its data is reproduced."""
    categories = (
        (PUBLIC_AUDIO_DATASETS, "public_audio_loader"),
        (EPIAUDIO_NATIVE_DATASETS, "epiaudio_native_loader"),
        (DERIVED_DATASETS, "derived_dataset"),
        (NON_AUDIO_DATASETS, "non_audio_dataset"),
    )
    return next((category for names, category in categories if name in names), "unknown")


def normalize_dataset_name(value: str) -> str | None:
    """Return the dataset portion of a logged local, cloud, or token path."""
    text = value.strip().replace("\\", "/").rstrip("/")
    if not text:
        return None
    name = PurePosixPath(text).name
    match = _TOKENIZED_NAME.match(name)
    if match is not None:
        name = match.group(1)
    normalized = name.lower().replace("-", "_")
    return normalized or None


def normalize_dataset_names(value: Any) -> Iterable[str]:
    """Yield names from scalar values and JSON-encoded sweep value lists."""
    if isinstance(value, dict):
        for nested in value.values():
            yield from normalize_dataset_names(nested)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            yield from normalize_dataset_names(nested)
        return
    if not isinstance(value, str):
        return

    text = value.strip()
    if text.startswith("[") or text.startswith("{"):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            yield from normalize_dataset_names(decoded)
            return
    name = normalize_dataset_name(text)
    if name is not None:
        yield name


class CometInventory:
    def __init__(
        self,
        api_key: str,
        *,
        workspace: str,
        batch_size: int,
        sweep_only: bool = True,
    ) -> None:
        self.headers = {"Authorization": api_key}
        self.workspace = workspace
        self.batch_size = batch_size
        self.sweep_only = sweep_only

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(5):
            try:
                response = requests.request(
                    method,
                    f"{COMET_API}{path}",
                    headers=self.headers,
                    timeout=120,
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def experiment_keys(self, project: str) -> list[str]:
        keys = []
        for archived in (False, True):
            payload = self._request(
                "GET",
                "/experiments",
                params={
                    "workspaceName": self.workspace,
                    "projectName": project,
                    "archived": str(archived).lower(),
                },
            )
            for experiment in payload.get("experiments", []):
                keys.append(experiment["experimentKey"])
        return list(dict.fromkeys(keys))

    def parameter_rows(self, keys: list[str]) -> list[tuple[str | None, list[Any]]]:
        payload = self._request(
            "POST",
            "/experiments/multi-metric-chart",
            json={
                "targetedExperiments": keys,
                "metrics": [],
                "params": [*DATASET_PARAMETERS, "sweep_name"],
                "independentMetrics": True,
            },
        )
        rows = []
        for experiment in payload.get("experiments", {}).values():
            params = experiment.get("params") or {}
            sweep_name = params.get("sweep_name")
            if self.sweep_only and sweep_name in (None, "", "None", "null"):
                continue
            values = [
                params[name]
                for name in DATASET_PARAMETERS
                if params.get(name) not in (None, "", "None", "null")
            ]
            rows.append((str(sweep_name) if sweep_name else None, values))
        return rows

    def collect(self, projects: list[str], *, workers: int) -> dict[str, Any]:
        keys_by_project = {
            project: self.experiment_keys(project) for project in projects
        }
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        experiment_counts: Counter[str] = Counter()
        sweep_names: dict[str, set[str]] = defaultdict(set)
        jobs = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for project, keys in keys_by_project.items():
                for start in range(0, len(keys), self.batch_size):
                    jobs.append(
                        (
                            project,
                            pool.submit(
                                self.parameter_rows,
                                keys[start : start + self.batch_size],
                            ),
                        )
                    )
            for project, future in jobs:
                for sweep_name, values in future.result():
                    experiment_counts[project] += 1
                    if sweep_name is not None:
                        sweep_names[project].add(sweep_name)
                    for value in values:
                        for name in normalize_dataset_names(value):
                            counts[project][name] += 1

        combined: Counter[str] = Counter()
        for project_counts in counts.values():
            combined.update(project_counts)
        return {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "workspace": self.workspace,
            "projects": {
                project: {
                    "experiment_count": experiment_counts[project],
                    "sweep_count": len(sweep_names[project]),
                    "datasets": dict(sorted(counts[project].items())),
                }
                for project in projects
            },
            "datasets": dict(sorted(combined.items())),
            "classification": {
                name: classify_dataset_name(name) for name in sorted(combined)
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    parser.add_argument("--output", type=Path, default=Path("comet_datasets.json"))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--include-one-offs",
        action="store_true",
        help="Include experiments without EpiAudio's logged sweep_name parameter.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("COMET_ML_API")
    if not api_key:
        raise SystemExit("COMET_ML_API is not set")
    inventory = CometInventory(
        api_key,
        workspace=args.workspace,
        batch_size=args.batch_size,
        sweep_only=not args.include_one_offs,
    ).collect(args.projects, workers=args.workers)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(inventory['datasets'])} dataset names to {args.output}")


if __name__ == "__main__":
    main()
