from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from epiaudio.analysis import pipeline
from epiaudio.analysis.frontier import CodeLengthPoint
from epiaudio.sweep_summary import (
    COMPUTE_COLUMN,
    DATA_BITS_COLUMN,
    MODEL_BITS_COLUMN,
    TOTAL_BITS_COLUMN,
    bundle_for_sweep,
)

# Every real __sweep_config.yaml saved by sweep.py carries the sweep's whole
# original YAML, command block included -- sweep.py:_save_sweep_config saves
# self.cfg unmodified. Shaped like sweeps/birdset_hsn.yaml so tests using
# this fixture actually exercise that wandb's ${env}/${program}/${args}
# reuse OmegaConf's interpolation syntax, rather than only ever loading a
# minimal parameters-only file no real sweep produces.
SWEEP_CONFIG = """
program: main.py
method: pf_grid
name: {name}
command:
  - ${{env}}
  - python
  - main.py
  - -cn
  - default
parameters:
  "T": {{value: 31250000}}
  "model.V": {{values: [1024]}}
  "model.L": {{value: 375}}
  "ds_path": {{values: [data/{dataset}_encodec]}}
"""


def _write_config(directory: Path, name: str, dataset: str) -> Path:
    path = directory / f"{name}__sweep_config.yaml"
    path.write_text(SWEEP_CONFIG.format(name=name, dataset=dataset))
    return path


def _facts(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "sweep_name": "demo",
        "dataset": "demo_ds",
        "tokenizer": "encodec",
        "inference_tokens": 750_000,
        "token_budget": 31_250_000,
        "vocab_size": 1024,
        "context_length": 375,
        "train_clips": None,
    }
    defaults.update(overrides)
    return defaults


class TokenizerInferenceTests(unittest.TestCase):
    def test_tokenizer_read_from_the_path_suffix(self):
        self.assertEqual(pipeline.infer_tokenizer("data/birdset_hsn_encodec"), "encodec")
        self.assertEqual(pipeline.infer_tokenizer("data/fsd50k_sqcodec"), "sqcodec")
        self.assertIsNone(pipeline.infer_tokenizer("data/fsd50k_opus"))

    def test_dataset_name_strips_the_suffix(self):
        self.assertEqual(
            pipeline.infer_dataset("data/birdset_hsn_encodec", "encodec"), "birdset_hsn"
        )

    def test_token_dtype_switches_above_uint16(self):
        self.assertIs(pipeline.token_dtype(1024), np.uint16)
        self.assertIs(pipeline.token_dtype(117_649), np.uint32)


class SweepFactsDerivedValueTests(unittest.TestCase):
    def test_audio_hours_derived_from_the_token_budget(self):
        self.assertAlmostEqual(pipeline.audio_hours_seen(_facts()), 115.74, places=2)

    def test_repeat_factor_unknown_without_a_clip_count(self):
        facts = _facts()
        self.assertIsNone(pipeline.repeat_factor_for(facts))
        self.assertIsNone(pipeline.prequential_assumption_holds(facts))

    def test_prequential_assumption_tracks_the_repeat_factor(self):
        clears = _facts(train_clips=469_942)
        repeats = _facts(train_clips=6_394)
        self.assertTrue(pipeline.prequential_assumption_holds(clears))
        self.assertFalse(pipeline.prequential_assumption_holds(repeats))

    def test_common_duration_maps_to_different_token_counts(self):
        seconds = 24 * 3600.0
        encodec = _facts(tokenizer="encodec")
        dac = _facts(tokenizer="dac")
        self.assertEqual(pipeline.target_inference_tokens_for(encodec, seconds), 6_480_000)
        self.assertEqual(pipeline.target_inference_tokens_for(dac, seconds), 51_840_000)


class BuildSweepFactsTests(unittest.TestCase):
    def test_config_supplies_everything_but_the_inference_set_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(Path(tmp), "demo", "birdset_hsn")
            facts = pipeline.build_sweep_facts(
                config, sweep_name="demo", inference_tokens=750_000
            )
            self.assertEqual(facts["dataset"], "birdset_hsn")
            self.assertEqual(facts["tokenizer"], "encodec")
            self.assertEqual(facts["token_budget"], 31_250_000)
            self.assertEqual(facts["vocab_size"], 1024)
            self.assertEqual(facts["context_length"], 375)

    def test_missing_inference_size_refuses_rather_than_defaulting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _write_config(Path(tmp), "demo", "birdset_hsn")
            with self.assertRaises(pipeline.MissingProvenanceError) as ctx:
                pipeline.build_sweep_facts(config, sweep_name="demo")
            self.assertIn("entropy rate", str(ctx.exception))

    def test_inference_size_measured_from_test_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _write_config(root, "demo", "birdset_hsn")
            data_dir = root / "birdset_hsn_encodec"
            data_dir.mkdir()
            np.arange(4096, dtype=np.uint16).tofile(data_dir / "test.bin")
            facts = pipeline.build_sweep_facts(config, sweep_name="demo", data_root=root)
            self.assertEqual(facts["inference_tokens"], 4096)

    def test_multi_dataset_sweep_must_be_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi__sweep_config.yaml"
            path.write_text(
                'parameters:\n'
                '  "T": {value: 31250000}\n'
                '  "ds_path": {values: [data/a_encodec, data/a_dac]}\n'
            )
            with self.assertRaises(pipeline.MissingProvenanceError) as ctx:
                pipeline.build_sweep_facts(path, sweep_name="multi")
            self.assertIn("split the outputs", str(ctx.exception))

    def test_non_positive_token_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad__sweep_config.yaml"
            path.write_text(
                'parameters:\n'
                '  "T": {value: 0}\n'
                '  "ds_path": {values: [data/a_encodec]}\n'
            )
            with self.assertRaises(pipeline.MissingProvenanceError):
                pipeline.build_sweep_facts(path, sweep_name="bad", inference_tokens=1000)


class LoadPointsTests(unittest.TestCase):
    def _bundle_with_runs(self, directory: Path, sweep_name: str):
        rows = [
            (0, 1.0e14, -5.0e4, 1.0e7, 1.0e7 - 5.0e4),
            (0, 5.0e14, 2.0e5, 8.0e6, 8.2e6),
            (1, 1.0e16, 9.0e5, 4.0e6, 4.9e6),
        ]
        frame = pd.DataFrame(
            rows,
            columns=[
                pipeline.RUN_INDEX_COLUMN,
                COMPUTE_COLUMN,
                MODEL_BITS_COLUMN,
                DATA_BITS_COLUMN,
                TOTAL_BITS_COLUMN,
            ],
        )
        (directory / f"{sweep_name}__pareto_front_data.csv").write_text("x\n")
        frame.to_csv(directory / f"{sweep_name}__all_run_data.csv", index=False)
        return bundle_for_sweep(directory, sweep_name)

    def test_points_load_unchanged_without_a_target_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_runs(directory, "demo")
            points = pipeline.load_points(bundle, _facts(inference_tokens=1_000_000))
            self.assertEqual(len(points), 3)
            self.assertAlmostEqual(points[0].data_bits, 1.0e7)

    def test_negative_model_bits_survive_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_runs(directory, "demo")
            points = pipeline.load_points(bundle, _facts(inference_tokens=1_000_000))
            self.assertTrue(any(point.model_bits < 0 for point in points))

    def test_restandardization_scales_only_the_data_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_runs(directory, "demo")
            points = pipeline.load_points(
                bundle, _facts(inference_tokens=1_000_000), target_inference_tokens=4_000_000
            )
            self.assertAlmostEqual(points[0].data_bits, 4.0e7)
            self.assertAlmostEqual(points[0].model_bits, -5.0e4)

    def test_missing_all_run_csv_explains_why_it_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "demo__pareto_front_data.csv").write_text("x\n")
            bundle = bundle_for_sweep(directory, "demo")
            with self.assertRaises(FileNotFoundError) as ctx:
                pipeline.load_points(bundle, _facts())
            self.assertIn("rebuilt", str(ctx.exception))

    def test_missing_columns_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "demo__pareto_front_data.csv").write_text("x\n")
            (directory / "demo__all_run_data.csv").write_text("run_idx\n0\n")
            bundle = bundle_for_sweep(directory, "demo")
            with self.assertRaises(ValueError) as ctx:
                pipeline.load_points(bundle, _facts())
            self.assertIn("missing columns", str(ctx.exception))


class SelectLossMetricTests(unittest.TestCase):
    def test_prequential_prefers_teacher_ema_when_active(self):
        self.assertEqual(
            pipeline._select_loss_metric(False, True, False), "ema_teacher_eval_loss"
        )

    def test_prequential_falls_back_without_teacher_ema(self):
        self.assertEqual(
            pipeline._select_loss_metric(False, False, False), "teacher_eval_loss"
        )

    def test_requential_prefers_student_ema_when_active(self):
        self.assertEqual(
            pipeline._select_loss_metric(True, False, True), "ema_student_eval_loss"
        )

    def test_requential_falls_back_without_student_ema(self):
        self.assertEqual(
            pipeline._select_loss_metric(True, False, False), "student_eval_loss"
        )


class EndpointLossesTests(unittest.TestCase):
    def _bundle_with_comet_provenance(
        self, directory: Path, sweep_name: str, *, extra_columns: bool = True
    ):
        rows = [
            (0, 1.0e14, -5.0e4, 1.0e7, 1.0e7 - 5.0e4),
            (1, 1.0e16, 9.0e5, 4.0e6, 4.9e6),
        ]
        columns = [
            pipeline.RUN_INDEX_COLUMN,
            COMPUTE_COLUMN,
            MODEL_BITS_COLUMN,
            DATA_BITS_COLUMN,
            TOTAL_BITS_COLUMN,
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if extra_columns:
            frame["experiment_key"] = ["exp-0", "exp-1"]
            frame["train_student"] = [False, True]
            frame["teacher_ema_active"] = [True, False]
            frame["student_ema_active"] = [False, True]
        (directory / f"{sweep_name}__pareto_front_data.csv").write_text("x\n")
        frame.to_csv(directory / f"{sweep_name}__all_run_data.csv", index=False)
        return bundle_for_sweep(directory, sweep_name)

    def test_provenance_absent_from_an_older_csv_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo", extra_columns=False)
            self.assertIsNone(pipeline.load_run_comet_provenance(bundle))

    def test_provenance_read_correctly_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo")
            provenance = pipeline.load_run_comet_provenance(bundle)
            assert provenance is not None
            self.assertEqual(provenance[0]["experiment_key"], "exp-0")
            self.assertFalse(provenance[0]["train_student"])
            self.assertTrue(provenance[1]["train_student"])

    def test_missing_provenance_gives_none_without_calling_the_fetcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo", extra_columns=False)
            calls: list[tuple[str, str]] = []

            def fake_fetch(experiment_key: str, metric_name: str):
                calls.append((experiment_key, metric_name))
                return []

            result = pipeline.fetch_endpoint_losses(
                bundle, 0, output_dir=directory, fetch_metric=fake_fetch
            )
            self.assertIsNone(result)
            self.assertEqual(calls, [])

    def test_cache_miss_queries_the_fetcher_and_selects_the_right_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo")
            calls: list[tuple[str, str]] = []

            def fake_fetch(experiment_key: str, metric_name: str):
                calls.append((experiment_key, metric_name))
                if metric_name == "K(X)":
                    return [{"timestamp": 1, "metricValue": 2.5}]
                return [
                    {"timestamp": 2, "metricValue": 4.0},
                    {"timestamp": 1, "metricValue": 5.0},
                ]

            # run 0: train_student=False, teacher_ema_active=True -> ema_teacher_eval_loss
            result = pipeline.fetch_endpoint_losses(
                bundle, 0, output_dir=directory, fetch_metric=fake_fetch
            )
            assert result is not None
            losses, online_code_bits = result
            self.assertEqual(losses, [5.0, 4.0])  # sorted by timestamp, not insertion order
            assert online_code_bits is not None
            self.assertAlmostEqual(online_code_bits, 2.5 * pipeline.COMET_BITS_PER_MEGABIT)
            self.assertIn(("exp-0", "ema_teacher_eval_loss"), calls)

    def test_second_call_reads_the_cache_instead_of_refetching(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo")
            call_count = 0

            def fake_fetch(experiment_key: str, metric_name: str):
                nonlocal call_count
                call_count += 1
                return [{"timestamp": 1, "metricValue": 3.0}]

            first = pipeline.fetch_endpoint_losses(
                bundle, 0, output_dir=directory, fetch_metric=fake_fetch
            )
            calls_after_first = call_count
            second = pipeline.fetch_endpoint_losses(
                bundle, 0, output_dir=directory, fetch_metric=fake_fetch
            )
            self.assertEqual(first, second)
            self.assertGreater(calls_after_first, 0)
            self.assertEqual(call_count, calls_after_first)  # unchanged: no second fetch

    def test_cache_file_is_a_readable_json_companion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo")
            pipeline.fetch_endpoint_losses(
                bundle, 1, output_dir=directory,
                fetch_metric=lambda experiment_key, metric_name: [{"timestamp": 1, "metricValue": 1.0}],
            )
            cache_path = directory / "demo__endpoint_losses_cache.json"
            self.assertTrue(cache_path.is_file())
            self.assertEqual(pipeline._read_endpoint_losses_cache(cache_path)[1]["losses"], [1.0])


class ProvenanceRowTests(unittest.TestCase):
    def test_provenance_row_lists_the_facts_figures_depend_on(self):
        row = pipeline.build_provenance_row(_facts(train_clips=6_394))
        self.assertAlmostEqual(row["audio_hours_seen"], 115.74, places=2)
        self.assertGreater(row["repeat_factor"], 13.0)
        self.assertFalse(row["prequential_assumption_holds"])


class BuildRowTests(unittest.TestCase):
    def _points(self) -> list[CodeLengthPoint]:
        # The last point's total (5.1e7) is far below the earlier points'
        # (~1e9), so it survives the hull unambiguously and becomes the
        # endpoint regardless of the geometric details in between -- this
        # tests build_row's flattening, not lower_convex_hull's geometry
        # (already covered by test_analysis_frontier.py).
        return [
            CodeLengthPoint(compute_flops=1.0e14, model_bits=-1.0e4, data_bits=1.0e9, run_index=0),
            CodeLengthPoint(compute_flops=1.0e15, model_bits=2.0e8, data_bits=8.0e8, run_index=1),
            CodeLengthPoint(compute_flops=3.0e16, model_bits=1.0e6, data_bits=5.0e7, run_index=2),
        ]

    def test_entropy_gets_a_rate_and_epiplexity_does_not(self):
        row = pipeline.build_row(_facts(), self._points(), inference_seconds=86_400.0)
        self.assertEqual(row["endpoint_compute_flops"], 3.0e16)
        self.assertAlmostEqual(row["epiplexity_bits"], 1.0e6)
        self.assertAlmostEqual(row["entropy_bits"], 5.0e7)
        self.assertAlmostEqual(row["entropy_bits_per_second"], 5.0e7 / 86_400.0)
        self.assertNotIn("epiplexity_bits_per_second", row)
        # pos_only=True (the default) drops point0 (negative) before the hull
        # is built, leaving point1 and point2; negative_points still counts
        # against all 3 raw points regardless.
        self.assertEqual(row["frontier_points"], 2)
        self.assertAlmostEqual(row["negative_point_fraction"], 0.5)

    def test_empty_points_give_an_invalid_endpoint_not_a_crash(self):
        row = pipeline.build_row(_facts(), [], inference_seconds=86_400.0)
        self.assertEqual(row["frontier_points"], 0)
        self.assertIsNone(row["epiplexity_bits"])
        self.assertIsNone(row["endpoint_valid"])

    def _points_with_a_dominant_negative_point(self) -> list[CodeLengthPoint]:
        # point0 is cheap enough (total = -50) that, if kept, nothing at
        # higher compute ever beats it under tolerance=0 -- it becomes the
        # *entire* hull by itself, and its negative model_bits becomes the
        # reported endpoint. This is exactly the failure pos_only guards
        # against: one broken measurement dominating an otherwise-healthy
        # frontier.
        return [
            CodeLengthPoint(compute_flops=1.0e14, model_bits=-100.0, data_bits=50.0, run_index=0),
            CodeLengthPoint(compute_flops=1.0e15, model_bits=30.0, data_bits=60.0, run_index=1),
            CodeLengthPoint(compute_flops=1.0e16, model_bits=10.0, data_bits=20.0, run_index=2),
        ]

    def test_pos_only_defaults_to_true_and_keeps_a_broken_point_from_taking_over(self):
        row = pipeline.build_row(
            _facts(), self._points_with_a_dominant_negative_point(), inference_seconds=86_400.0
        )
        self.assertEqual(row["frontier_points"], 2)
        self.assertEqual(row["endpoint_compute_flops"], 1.0e16)
        self.assertAlmostEqual(row["epiplexity_bits"], 10.0)
        # negative_points/negative_point_fraction still see the excluded point.
        self.assertEqual(row["negative_points"], 1)
        self.assertAlmostEqual(row["negative_point_fraction"], 0.5)

    def test_pos_only_false_restores_the_unfiltered_frontier(self):
        row = pipeline.build_row(
            _facts(),
            self._points_with_a_dominant_negative_point(),
            inference_seconds=86_400.0,
            pos_only=False,
        )
        # The broken point is cheap enough to be the whole hull by itself.
        self.assertEqual(row["frontier_points"], 1)
        self.assertEqual(row["endpoint_compute_flops"], 1.0e14)
        self.assertAlmostEqual(row["epiplexity_bits"], -100.0)
        self.assertEqual(row["negative_points"], 1)
        self.assertAlmostEqual(row["negative_point_fraction"], 1.0)

    # A late rise: loss climbs at the end. model_bits below is
    # curves.implied_model_bits(_LATE_RISE_LOSSES, 31_250_000) -- derived
    # from the same curve the fake fetcher returns, not hand-picked, so the
    # SCOPE_MISMATCH guard sees a genuinely consistent (losses, model_bits)
    # pair instead of flagging an inconsistency of the test's own making.
    _LATE_RISE_LOSSES = [6.0, 5.0, 4.0, 3.5, 4.5, 5.5]
    _LATE_RISE_MODEL_BITS = -33_813_165.02083508

    def _bundle_with_comet_provenance(self, directory: Path, sweep_name: str):
        rows = [(0, 3.0e16, self._LATE_RISE_MODEL_BITS, 5.0e7, 5.0e7 + self._LATE_RISE_MODEL_BITS)]
        columns = [
            pipeline.RUN_INDEX_COLUMN,
            COMPUTE_COLUMN,
            MODEL_BITS_COLUMN,
            DATA_BITS_COLUMN,
            TOTAL_BITS_COLUMN,
        ]
        frame = pd.DataFrame(rows, columns=columns)
        frame["experiment_key"] = ["exp-0"]
        frame["train_student"] = [False]
        frame["teacher_ema_active"] = [False]
        frame["student_ema_active"] = [False]
        (directory / f"{sweep_name}__pareto_front_data.csv").write_text("x\n")
        frame.to_csv(directory / f"{sweep_name}__all_run_data.csv", index=False)
        return bundle_for_sweep(directory, sweep_name)

    def test_bundle_diagnoses_a_negative_endpoint_end_to_end(self):
        # A single point, kept via pos_only=False so its negative model_bits
        # survives as the endpoint -- exercising the full chain: bundle ->
        # provenance -> fetch (faked) -> cache -> diagnose -> row fields.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo")
            point = CodeLengthPoint(
                compute_flops=3.0e16,
                model_bits=self._LATE_RISE_MODEL_BITS,
                data_bits=5.0e7,
                run_index=0,
            )

            def fake_fetch(experiment_key: str, metric_name: str):
                if metric_name == "K(X)":
                    return [{"timestamp": 1, "metricValue": 100.0}]
                return [
                    {"timestamp": i, "metricValue": v}
                    for i, v in enumerate(self._LATE_RISE_LOSSES)
                ]

            row = pipeline.build_row(
                _facts(),
                [point],
                inference_seconds=86_400.0,
                pos_only=False,
                bundle=bundle,
                output_dir=directory,
                fetch_metric=fake_fetch,
            )
            self.assertAlmostEqual(row["epiplexity_bits"], self._LATE_RISE_MODEL_BITS)
            self.assertEqual(row["negativity_mechanism"], "late_divergence")
            self.assertEqual(row["loss_curve_shape"], "late_rise")

    def test_explicit_endpoint_losses_take_precedence_over_a_bundle_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = self._bundle_with_comet_provenance(directory, "demo")
            point = CodeLengthPoint(compute_flops=3.0e16, model_bits=-3.0e4, data_bits=5.0e7, run_index=0)

            def failing_fetch(experiment_key: str, metric_name: str):
                raise AssertionError("bundle fetch should not run when endpoint_losses is given")

            row = pipeline.build_row(
                _facts(),
                [point],
                inference_seconds=86_400.0,
                pos_only=False,
                endpoint_losses=[6.0, 5.0, 4.0],
                bundle=bundle,
                output_dir=directory,
                fetch_metric=failing_fetch,
            )
            self.assertIsNotNone(row["negativity_mechanism"])


class ResultsFrameTests(unittest.TestCase):
    def test_empty_results_give_an_empty_frame(self):
        self.assertTrue(pipeline.results_frame([]).empty)

    def test_rows_sort_by_tokenizer_then_dataset(self):
        rows = [
            {"tokenizer": "dac", "dataset": "b"},
            {"tokenizer": "dac", "dataset": "a"},
        ]
        frame = pipeline.results_frame(rows)
        self.assertEqual(list(frame["dataset"]), ["a", "b"])


class WarningTests(unittest.TestCase):
    def _row(self, **overrides: object) -> dict[str, object]:
        defaults: dict[str, object] = {
            "sweep_name": "demo",
            "dataset": "demo_ds",
            "tokenizer": "encodec",
            "inference_seconds": 86_400.0,
            "train_audio_hours": 115.74,
            "repeat_factor": 5.64,
            "prequential_assumption_holds": True,
            "is_saturated": True,
            "epiplexity_bits": 1.0e6,
        }
        defaults.update(overrides)
        return defaults

    def _frame(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_clean_comparison_produces_no_warnings(self):
        warnings = pipeline.comparability_warnings(
            self._frame([self._row(), self._row(sweep_name="demo2", dataset="other")])
        )
        self.assertEqual(warnings, [], msg=str(warnings))

    def test_repeating_cells_warned(self):
        warnings = pipeline.comparability_warnings(
            self._frame([self._row(sweep_name="small", prequential_assumption_holds=False)])
        )
        self.assertTrue(any("replay training tokens" in text for text in warnings))

    def test_repeating_cells_warning_states_the_exact_factor(self):
        # Team decision: replay under a fixed token budget is acknowledged in
        # the paper rather than avoided, so the caption needs the actual
        # factor, not just a yes/no flag.
        warnings = pipeline.comparability_warnings(
            self._frame(
                [self._row(sweep_name="small", prequential_assumption_holds=False, repeat_factor=5.64)]
            )
        )
        self.assertTrue(any("small (5.6x)" in text for text in warnings))

    def test_unknown_clip_counts_warned(self):
        warnings = pipeline.comparability_warnings(
            self._frame(
                [self._row(sweep_name="unknown", repeat_factor=None, prequential_assumption_holds=None)]
            )
        )
        self.assertTrue(any("no recorded clip count" in text for text in warnings))

    def test_mismatched_inference_size_within_a_dataset_is_warned(self):
        # Same dataset, two tokenizer variants (encodec/dac) disagreeing on
        # inference_seconds -- prepare_audio.py tokenizes one shared,
        # already-split AudioDataset per dataset, so their test.bin's should
        # encode the same held-out audio; a mismatch means the split or a
        # test.bin itself is inconsistent.
        warnings = pipeline.comparability_warnings(
            self._frame(
                [
                    self._row(sweep_name="a", dataset="demo_ds", tokenizer="encodec", inference_seconds=86_400.0),
                    self._row(sweep_name="b", dataset="demo_ds", tokenizer="dac", inference_seconds=3600.0),
                ]
            )
        )
        self.assertTrue(any("demo_ds" in text and "inference-set sizes" in text for text in warnings))

    def test_different_inference_size_across_datasets_is_not_warned(self):
        # Different datasets naturally have different-sized test splits; the
        # token budget is the fixed comparison axis, not duration, so this is
        # not a defect and should not warn.
        warnings = pipeline.comparability_warnings(
            self._frame(
                [
                    self._row(sweep_name="a", dataset="ds_a", tokenizer="encodec", inference_seconds=86_400.0),
                    self._row(sweep_name="b", dataset="ds_b", tokenizer="encodec", inference_seconds=3600.0),
                ]
            )
        )
        self.assertFalse(any("inference-set sizes" in text for text in warnings))

    def test_mixed_tokenizers_at_matched_training_audio_produce_no_warning(self):
        # A fixed token budget buys different audio under every tokenizer by
        # construction (documented in audio_hours_seen's own docstring), so a
        # table spanning tokenizers should not warn just because it spans
        # tokenizers -- ethanalwaise-del, PR #121: "this will show up whenever
        # we look at measurements from two different tokenizers."
        warnings = pipeline.comparability_warnings(
            self._frame([self._row(), self._row(sweep_name="b", tokenizer="dac", train_audio_hours=14.47)])
        )
        self.assertEqual(warnings, [], msg=str(warnings))

    def test_uneven_training_audio_within_one_tokenizer_is_warned(self):
        # Two cells sharing a tokenizer have different training audio, which
        # isn't explained by tokenizer choice and is worth flagging -- whether
        # or not the cause turns out to be deliberate (see the next test).
        warnings = pipeline.comparability_warnings(
            self._frame(
                [
                    self._row(sweep_name="a", dataset="ds_a", tokenizer="encodec", train_audio_hours=115.74),
                    self._row(sweep_name="b", dataset="ds_b", tokenizer="encodec", train_audio_hours=57.87),
                ]
            )
        )
        self.assertTrue(any("encodec" in text and "different training audio" in text for text in warnings))

    def test_uneven_training_audio_warning_does_not_assume_it_is_accidental(self):
        # sweeps/fsd50k.yaml deliberately sets T to 16x the standard value --
        # a real, checked-in exception, not a hypothetical -- so the message
        # must not assert the mismatch is "likely" a non-standard/accidental
        # budget; it can only state the fact and its consequence.
        warnings = pipeline.comparability_warnings(
            self._frame(
                [
                    self._row(sweep_name="birdset_hsn", dataset="birdset_hsn", tokenizer="encodec"),
                    self._row(sweep_name="fsd50k", dataset="fsd50k", tokenizer="encodec", train_audio_hours=1851.85),
                ]
            )
        )
        matches = [text for text in warnings if "different training audio" in text]
        self.assertEqual(len(matches), 1, msg=str(warnings))
        self.assertNotIn("likely", matches[0])
        self.assertNotIn("non-standard", matches[0])

    def test_uneven_training_audio_across_tokenizers_is_not_conflated_with_within(self):
        # Same scenario as above, but the second, differently-tokenized cell
        # (dac) should not itself trigger the within-tokenizer check, and
        # should not suppress it for encodec either.
        warnings = pipeline.comparability_warnings(
            self._frame(
                [
                    self._row(sweep_name="a", dataset="ds_a", tokenizer="encodec", train_audio_hours=115.74),
                    self._row(sweep_name="b", dataset="ds_b", tokenizer="encodec", train_audio_hours=57.87),
                    self._row(sweep_name="c", dataset="ds_a", tokenizer="dac", train_audio_hours=14.47),
                ]
            )
        )
        matches = [text for text in warnings if "different training audio" in text]
        self.assertEqual(len(matches), 1, msg=str(warnings))
        self.assertIn("encodec", matches[0])
        self.assertNotIn("dac", matches[0])

    def test_negative_endpoint_warned(self):
        warnings = pipeline.comparability_warnings(
            self._frame([self._row(sweep_name="neg", epiplexity_bits=-1.0e5)])
        )
        self.assertTrue(any("negative endpoint" in text for text in warnings))

    def test_unsaturated_endpoint_warned_as_a_lower_bound(self):
        warnings = pipeline.comparability_warnings(
            self._frame([self._row(sweep_name="rising", is_saturated=False)])
        )
        self.assertTrue(any("lower bounds" in text for text in warnings))

    def test_empty_frame_gives_no_warnings(self):
        self.assertEqual(pipeline.comparability_warnings(pd.DataFrame()), [])


class KeepNegativePointsFlagTests(unittest.TestCase):
    def test_defaults_to_excluding_negative_points(self):
        args = pipeline._parse_args([])
        self.assertFalse(args.keep_negative_points)

    def test_flag_opts_back_into_the_unfiltered_view(self):
        args = pipeline._parse_args(["--keep-negative-points"])
        self.assertTrue(args.keep_negative_points)


class InferenceHoursFlagRemovedTests(unittest.TestCase):
    def test_inference_hours_is_no_longer_a_cli_option(self):
        # Team decision (2026-08-25): the token budget is the fixed
        # comparison axis, not duration -- restandardizing every cell to a
        # common inference duration is no longer something the CLI offers.
        with self.assertRaises(SystemExit):
            pipeline._parse_args(["--inference-hours", "24"])

    def test_parsed_args_have_no_inference_hours_attribute(self):
        args = pipeline._parse_args([])
        self.assertFalse(hasattr(args, "inference_hours"))


class ResolveAmbiguousCellsTests(unittest.TestCase):
    def _resolved(self, *facts_overrides: dict[str, object]):
        # bundle content is never read by _resolve_ambiguous_cells; only
        # sweep_facts matters, so a placeholder bundle is enough here.
        placeholder = bundle_for_sweep(Path("/nonexistent"), "placeholder")
        return [(placeholder, _facts(**overrides)) for overrides in facts_overrides]

    def test_unique_cells_produce_no_messages(self):
        resolved = self._resolved(
            {"sweep_name": "a", "dataset": "ds_a", "tokenizer": "encodec"},
            {"sweep_name": "b", "dataset": "ds_b", "tokenizer": "encodec"},
        )
        messages, ambiguous = pipeline._resolve_ambiguous_cells(resolved)
        self.assertEqual(messages, [])
        self.assertEqual(ambiguous, set())

    def test_two_sweeps_for_the_same_dataset_and_tokenizer_are_flagged(self):
        # This is the birdset_hsn.yaml/fsd50k.yaml shape: no run id in the
        # sweep name, so a rerun leaves two __sweep_config.yaml's resolving to
        # the same (dataset, tokenizer) sitting in the same --output-dir.
        resolved = self._resolved(
            {"sweep_name": "birdset_hsn_encodec", "dataset": "birdset_hsn", "tokenizer": "encodec"},
            {"sweep_name": "birdset_hsn_encodec_rerun", "dataset": "birdset_hsn", "tokenizer": "encodec"},
            {"sweep_name": "birdset_hsn_dac", "dataset": "birdset_hsn", "tokenizer": "dac"},
        )
        messages, ambiguous = pipeline._resolve_ambiguous_cells(resolved)
        self.assertEqual(len(messages), 1)
        self.assertIn("birdset_hsn/encodec", messages[0])
        self.assertEqual(ambiguous, {"birdset_hsn_encodec", "birdset_hsn_encodec_rerun"})
        self.assertNotIn("birdset_hsn_dac", ambiguous)


class MainAmbiguousSweepPrefixTests(unittest.TestCase):
    def _write_bundle(self, directory: Path, sweep_name: str, dataset: str, tokenizer: str = "encodec"):
        config_path = directory / f"{sweep_name}__sweep_config.yaml"
        config_path.write_text(
            'parameters:\n'
            '  "T": {value: 31250000}\n'
            f'  "ds_path": {{values: [data/{dataset}_{tokenizer}]}}\n'
        )
        frame = pd.DataFrame(
            [(0, 1.0e14, 2.0e5, 8.0e6, 8.2e6)],
            columns=[
                pipeline.RUN_INDEX_COLUMN,
                COMPUTE_COLUMN,
                MODEL_BITS_COLUMN,
                DATA_BITS_COLUMN,
                TOTAL_BITS_COLUMN,
            ],
        )
        (directory / f"{sweep_name}__pareto_front_data.csv").write_text("x\n")
        frame.to_csv(directory / f"{sweep_name}__all_run_data.csv", index=False)
        data_dir = directory / f"{dataset}_{tokenizer}"
        data_dir.mkdir(exist_ok=True)
        np.arange(4096, dtype=np.uint16).tofile(data_dir / "test.bin")

    def test_colliding_sweep_names_are_skipped_not_silently_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # Same shape as a rerun of birdset_hsn.yaml, whose sweep name
            # carries no run id: two saved configs land on the same
            # (dataset, tokenizer) cell under one --sweep-prefix.
            self._write_bundle(directory, "birdset_hsn_encodec", "birdset_hsn", "encodec")
            self._write_bundle(directory, "birdset_hsn_encodec_rerun", "birdset_hsn", "encodec")
            self._write_bundle(directory, "birdset_hsn_dac", "birdset_hsn", "dac")

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exit_code = pipeline.main(
                    [
                        "--output-dir",
                        str(directory),
                        "--sweep-prefix",
                        "birdset_hsn",
                        "--data-root",
                        str(directory),
                    ]
                )
            printed = out.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("birdset_hsn/encodec", printed)
            self.assertIn("birdset_hsn_encodec", printed)
            self.assertIn("birdset_hsn_encodec_rerun", printed)
            self.assertIn("1 analysed, 1 skipped", printed)

            results_section = printed.rsplit("=== native results ===", 1)[-1]
            results_section = results_section.split("=== comparability", 1)[0]
            self.assertNotIn("encodec", results_section)
            self.assertIn("dac", results_section)


if __name__ == "__main__":
    unittest.main()
