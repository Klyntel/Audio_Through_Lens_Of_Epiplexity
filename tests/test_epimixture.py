from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import DictConfig, OmegaConf

from experiments.epimixture.epimixture import (
    _write_mixture_split,
    _path_sha256,
    _selections_complete,
    allocate_samples,
    build_mixtures,
    generate_sweeps,
    load_spec,
    plan,
    run_experiment,
)
from experiments.epimixture.config import ExperimentSpec, FixedGeneralizationSpec
from experiments.epimixture.sweeps import _fixed_regime_manifest, _fixed_run_config
from epiaudio.sweep import load_sweep_configs
from epiaudio.sweep_preflight import validate_sweep


class EpiMixtureTests(unittest.TestCase):
    def test_allocate_samples_is_exact_and_stable(self) -> None:
        self.assertEqual(
            allocate_samples(10, {"a": 1, "b": 1, "c": 1}),
            {"a": 4, "b": 3, "c": 3},
        )
        self.assertEqual(sum(allocate_samples(17, {"a": 1, "b": 3, "c": 1}).values()), 17)

    def test_mixture_interleaves_complete_token_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources: dict[str, Path] = {}
            for name, start in (("a", 10), ("b", 100)):
                path = root / name
                path.mkdir()
                metadata = {"tokenizer": {"dtype": "int16", "sequence_length": 3}}
                (path / "metadata.json").write_text(json.dumps(metadata))
                np.array([start, start + 1, start + 2, start + 3, start + 4, start + 5], dtype=np.int16).tofile(path / "train.bin")
                sources[name] = path
            output = root / "mixture.bin"
            counts = _write_mixture_split(
                sources,
                split="train",
                weights={"a": 1, "b": 1},
                requested_samples=4,
                destination=output,
                seed=7,
                overwrite=False,
            )
            rows = np.fromfile(output, dtype=np.int16).reshape(-1, 3)
            self.assertEqual(counts, {"a": 2, "b": 2})
            self.assertEqual(rows.shape, (4, 3))
            self.assertTrue(all(tuple(row) in {(10, 11, 12), (13, 14, 15), (100, 101, 102), (103, 104, 105)} for row in rows))

    def test_synthetic_pipeline_generates_preflight_ready_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            sources = ("source_a", "source_b")
            for source_index, name in enumerate((*sources, "heldout")):
                dataset = data_root / f"{name}_encodec"
                dataset.mkdir(parents=True)
                (dataset / "metadata.json").write_text(
                    json.dumps({"tokenizer": {"dtype": "int16", "sequence_length": 375}})
                )
                OmegaConf.save(OmegaConf.create({"model": {"V": 1024, "L": 375}}), dataset / "metadata.yaml")
                for split in ("train", "test"):
                    np.full(1125, source_index + 1, dtype=np.int16).tofile(dataset / f"{split}.bin")

            template = root / "template.yaml"
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "method": "pf_grid",
                        "metric": {"name": "K(M)"},
                        "parameters": {
                            "T": {"value": 375},
                            "T_eval": {"value": 131072},
                            "B": {"value": 16},
                            "train_student": {"value": False},
                            "ds_path": {"value": "unused"},
                            "model.V": {"value": 1024},
                            "model.L": {"value": 375},
                        },
                    }
                ),
                template,
            )
            config = root / "experiment.yaml"
            config.write_text(
                "\n".join(
                    [
                        "name: smoke",
                        "seed: 1",
                        "sources: [source_a, source_b]",
                        "heldout: heldout",
                        "tokenizer: encodec",
                        "data_root: data",
                        "output_root: output",
                        "sweep_template: template.yaml",
                        "mixtures:",
                        "  - name: balanced",
                        "    weights: {source_a: 1, source_b: 1}",
                        "    train_samples: 6",
                        "    test_samples: 6",
                    ]
                )
                + "\n"
            )

            spec = load_spec(config)
            self.assertTrue(plan(spec).is_file())
            build_mixtures(spec)
            sweep_paths = generate_sweeps(spec)

            mixture = data_root / "epimixture_smoke_balanced_encodec"
            self.assertEqual(np.fromfile(mixture / "train.bin", dtype=np.int16).size, 6 * 375)
            self.assertTrue((mixture / "mixture_manifest.json").is_file())
            for sweep_path in sweep_paths:
                cfg = cast(DictConfig, OmegaConf.load(sweep_path))
                self.assertIsInstance(cfg, DictConfig)
                result = validate_sweep(cfg, sweep_path, method="pf_grid", repo_root=root)
                self.assertTrue(result.ready, result.issues)
                self.assertEqual(len(load_sweep_configs(str(sweep_path))), 1)

            selections = spec.output_root / "selections"
            selections.mkdir()
            for dataset_id, sweep_path in zip(spec.dataset_ids, sweep_paths, strict=True):
                (selections / f"{dataset_id}.json").write_text(
                    json.dumps({"selected": {}, "sweep_config_sha256": _path_sha256(sweep_path)})
                )
            self.assertTrue(_selections_complete(spec))

            changed_template = OmegaConf.load(template)
            changed_template.parameters.T = {"value": 750}
            OmegaConf.save(changed_template, template)
            generate_sweeps(spec)
            self.assertFalse(_selections_complete(spec))

            fixed_sweeper = Mock()
            fixed_sweeper.build_run_config.return_value = OmegaConf.create({"model": {}, "opt": {}})
            fixed_checkpoint = root / "fixed.pt"
            fixed_cfg = _fixed_run_config(fixed_sweeper, spec, "source_a", fixed_checkpoint)
            fixed_args = fixed_sweeper.build_run_config.call_args.args[0]
            self.assertEqual(fixed_args["model.N"], 8)
            self.assertEqual(fixed_args["model.D"], 512)
            self.assertIsNone(fixed_args["model.P"])
            self.assertEqual(fixed_args["+opt.wd"], 0.0)
            self.assertFalse(fixed_args["+compile"])
            self.assertEqual(fixed_cfg.save, str(fixed_checkpoint))

    def test_fixed_regime_manifest_is_predeclared_and_not_a_sweep_point(self) -> None:
        manifest = _fixed_regime_manifest(FixedGeneralizationSpec())
        self.assertEqual(manifest["model.N"], 8)
        self.assertEqual(manifest["model.D"], 512)
        self.assertNotIn("model.P", manifest)

    def test_run_experiment_calls_every_stage_in_order(self) -> None:
        spec = cast(ExperimentSpec, object())
        report = Path("report.md")
        with (
            patch("experiments.epimixture.runner.plan") as plan_mock,
            patch("experiments.epimixture.runner.prepare_sources") as prepare_mock,
            patch("experiments.epimixture.runner.build_mixtures") as mixtures_mock,
            patch("experiments.epimixture.runner.generate_sweeps") as sweeps_mock,
            patch("experiments.epimixture.runner.selections_complete", return_value=False),
            patch("experiments.epimixture.runner.transfer_complete", return_value=False),
            patch("experiments.epimixture.runner.run_sweeps") as run_sweeps_mock,
            patch("experiments.epimixture.runner.run_transfer") as transfer_mock,
            patch("experiments.epimixture.runner.analyze", return_value=report) as analyze_mock,
        ):
            self.assertEqual(run_experiment(spec, overwrite=True), report)

        plan_mock.assert_called_once_with(spec)
        prepare_mock.assert_called_once_with(spec, overwrite=True)
        mixtures_mock.assert_called_once_with(spec, overwrite=True)
        sweeps_mock.assert_called_once_with(spec, overwrite=True)
        run_sweeps_mock.assert_called_once_with(spec)
        transfer_mock.assert_called_once_with(spec)
        analyze_mock.assert_called_once_with(spec)

    def test_run_experiment_reuses_comet_without_starting_sweeps(self) -> None:
        spec = cast(ExperimentSpec, object())
        report = Path("report.md")
        with (
            patch("experiments.epimixture.runner.plan"),
            patch("experiments.epimixture.runner.prepare_sources"),
            patch("experiments.epimixture.runner.build_mixtures"),
            patch("experiments.epimixture.runner.generate_sweeps"),
            patch("experiments.epimixture.runner.selections_complete", return_value=False),
            patch("experiments.epimixture.runner.transfer_complete", return_value=False),
            patch("experiments.epimixture.runner.run_sweeps") as run_sweeps_mock,
            patch("experiments.epimixture.runner.load_comet_sweeps") as comet_mock,
            patch("experiments.epimixture.runner.run_transfer"),
            patch("experiments.epimixture.runner.analyze", return_value=report),
        ):
            self.assertEqual(run_experiment(spec, reuse_comet_sweeps=True), report)

        comet_mock.assert_called_once_with(spec)
        run_sweeps_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
