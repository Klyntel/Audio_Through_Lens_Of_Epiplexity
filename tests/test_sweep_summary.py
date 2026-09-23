from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from omegaconf import OmegaConf

from epiaudio.sweep import PFGridSweeper
from epiaudio.sweep_summary import (
    COMPUTE_COLUMN,
    DATA_BITS_COLUMN,
    MODEL_BITS_COLUMN,
    SweepOutputBundle,
    TOTAL_BITS_COLUMN,
    upload_sweep_summary,
)


class SweepSummaryTests(unittest.TestCase):
    def _write_pareto(
        self,
        path: Path,
        rows: list[tuple[float, float, float, float]],
    ) -> None:
        frame = pd.DataFrame(
            rows,
            columns=[
                COMPUTE_COLUMN,
                MODEL_BITS_COLUMN,
                DATA_BITS_COLUMN,
                TOTAL_BITS_COLUMN,
            ],
        )
        frame.to_csv(path, index=False)

    def test_upload_sweep_summary_logs_and_saves_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            sweep_name = "test-sweep"
            pareto = output_dir / f"{sweep_name}__pareto_front_data.csv"
            self._write_pareto(
                pareto,
                [
                    (10.0, 1.0, 29.0, 30.0),
                    (100.0, 4.0, 16.0, 20.0),
                ],
            )
            all_runs = output_dir / f"{sweep_name}__all_run_data.csv"
            all_runs.write_text("run_idx,Compute\n0,10.0\n")
            plot = output_dir / f"{sweep_name}_pareto_front.png"
            plot.write_bytes(b"png")
            config = output_dir / f"{sweep_name}__sweep_config.yaml"
            config.write_text("name: test\n")

            experiment = Mock()
            experiment.get_key.return_value = "summary-key"
            factory = Mock(return_value=experiment)

            key = upload_sweep_summary(
                SweepOutputBundle(
                    sweep_name=sweep_name,
                    pareto_csv=pareto,
                    all_run_csv=all_runs,
                    plot=plot,
                    sweep_config=config,
                ),
                project_name="epiaudio",
                workspace="epi-audio",
                api_key="test-key",
                parameters={"T": 123},
                tags=["birdset_hsn", "encodec"],
                experiment_factory=factory,
            )

            self.assertEqual(key, "summary-key")
            factory.assert_called_once_with(
                api_key="test-key",
                project_name="epiaudio",
                workspace="epi-audio",
            )
            experiment.set_name.assert_called_once_with(f"{sweep_name}__summary")
            self.assertEqual(
                [call.args[0] for call in experiment.add_tag.call_args_list],
                [
                    "sweep-summary",
                    sweep_name,
                    "birdset_hsn",
                    "encodec",
                ],
            )
            experiment.log_parameters.assert_called_once()
            self.assertEqual(
                experiment.log_parameters.call_args.args[0]["source_sweep_name"],
                sweep_name,
            )
            self.assertEqual(
                experiment.log_parameters.call_args.args[0]["T"],
                123,
            )
            logged_metrics = experiment.log_metrics.call_args.args[0]
            self.assertEqual(logged_metrics["pareto_epiplexity_bits"], 4.0)
            self.assertEqual(logged_metrics["epiplexity_valid"], 1)
            self.assertEqual(experiment.log_asset.call_count, 4)
            experiment.log_image.assert_called_once_with(
                image_data=str(plot),
                name=plot.stem,
            )
            experiment.end.assert_called_once()

            summary_path = output_dir / f"{sweep_name}__summary.json"
            self.assertTrue(summary_path.is_file())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["pareto_compute"], 100.0)

    def test_upload_requires_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            pareto = Path(tmp) / "test__pareto_front_data.csv"
            self._write_pareto(pareto, [(100.0, 4.0, 16.0, 20.0)])

            with (
                patch.dict("os.environ", {}, clear=True),
                self.assertRaisesRegex(RuntimeError, "COMET_ML_API"),
            ):
                upload_sweep_summary(
                    SweepOutputBundle("test", pareto),
                    project_name="epiaudio",
                    workspace="epi-audio",
                    api_key=None,
                    experiment_factory=Mock(),
                )

    def test_pf_report_saves_and_automatically_uploads_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            plot = output_dir / "birdset_hsn_dac__time_pareto_front.png"
            plot.write_bytes(b"png")

            sweeper = PFGridSweeper.__new__(PFGridSweeper)
            sweeper.sweep_name = "birdset_hsn_dac__time"
            sweeper.results = [
                {
                    "run_name": "run",
                    "point": {"model.N": 3, "model.P": 1, "seed": 0},
                    "epiplexity": 4.0,
                    "code_length": 20.0,
                }
            ]
            sweeper._curves = [[(10.0, 1.0, 30.0), (100.0, 4.0, 20.0)]]
            sweeper._run_provenance = [
                {
                    "experiment_key": "exp-0",
                    "train_student": False,
                    "teacher_ema_active": True,
                    "student_ema_active": False,
                }
            ]
            sweeper.cfg = OmegaConf.create(
                {
                    "method": "pf_grid",
                    "metric": {"name": "K(M)"},
                    "seed": 0,
                    "num_repeats": 1,
                    "parameters": {
                        "ds_path": {"values": ["data/birdset_hsn_dac"]},
                        "T": {"value": 1000},
                    },
                }
            )
            sweeper.backend = "torch"
            sweeper.base_config_name = "chess"
            sweeper.sweep_seed = None
            sweeper.command_overrides = ["wandb_project=epiaudio"]
            sweeper.extra_overrides = []

            with (
                patch("epiaudio.sweep.SWEEP_OUTPUT_DIR", output_dir),
                patch.object(
                    PFGridSweeper,
                    "_plot_pareto_front",
                    return_value=plot,
                ),
                patch(
                    "epiaudio.sweep.upload_sweep_summary",
                    return_value="summary-key",
                ) as upload,
                patch.dict("os.environ", {"COMET_ML_API": "test-key"}),
            ):
                sweeper._report(sweeper.results[0])

            upload.assert_called_once()
            bundle = upload.call_args.args[0]
            assert bundle.all_run_csv is not None
            assert bundle.sweep_config is not None
            self.assertTrue(bundle.all_run_csv.is_file())
            self.assertTrue(bundle.pareto_csv.is_file())
            self.assertTrue(bundle.sweep_config.is_file())
            self.assertNotIn(
                "Unnamed: 0",
                pd.read_csv(bundle.pareto_csv).columns,
            )
            self.assertEqual(
                upload.call_args.kwargs["tags"],
                ["birdset_hsn", "dac"],
            )
            self.assertEqual(
                upload.call_args.kwargs["parameters"]["tokenizer"],
                "dac",
            )

            all_run_df = pd.read_csv(bundle.all_run_csv)
            self.assertEqual(all_run_df["experiment_key"].tolist(), ["exp-0", "exp-0"])
            self.assertEqual(all_run_df["train_student"].tolist(), [False, False])
            self.assertEqual(all_run_df["teacher_ema_active"].tolist(), [True, True])
            self.assertEqual(all_run_df["student_ema_active"].tolist(), [False, False])
            # The pareto-front CSV stays keyed by hull membership, not by run,
            # so it carries no per-run provenance columns.
            self.assertNotIn("experiment_key", pd.read_csv(bundle.pareto_csv).columns)


class BuildAllRunDfTests(unittest.TestCase):
    def _sweeper_with(self, curves, provenance) -> PFGridSweeper:
        sweeper = PFGridSweeper.__new__(PFGridSweeper)
        sweeper._curves = curves
        sweeper._run_provenance = provenance
        return sweeper

    def test_matched_lengths_build_normally(self):
        sweeper = self._sweeper_with(
            curves=[[(10.0, 1.0, 30.0)]],
            provenance=[
                {
                    "experiment_key": "exp-0",
                    "train_student": False,
                    "teacher_ema_active": False,
                    "student_ema_active": False,
                }
            ],
        )
        df = sweeper._build_all_run_df()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["experiment_key"], "exp-0")

    def test_empty_sweeper_builds_an_empty_frame_not_a_crash(self):
        sweeper = self._sweeper_with(curves=[], provenance=[])
        df = sweeper._build_all_run_df()
        self.assertTrue(df.empty)
        self.assertIn("experiment_key", df.columns)


if __name__ == "__main__":
    unittest.main()
