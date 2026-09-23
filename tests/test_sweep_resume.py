from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from epiaudio.sweep import GridSweeper
from epiaudio.sweep import Sweeper


class SweepResumeTests(unittest.TestCase):
    @staticmethod
    def _run_cfg(ds_path: Path):
        return OmegaConf.create(
            {
                "ds_path": str(ds_path),
                "T": 200,
                "B": 2,
                "model": {"V": 1024, "L": 5},
                "train_student": False,
                "teacher_ema": None,
                "student_ema": None,
            }
        )

    @staticmethod
    def _resume_candidate(ds_path: Path, *, logged_step: int) -> GridSweeper:
        np.arange(10, dtype=np.uint16).tofile(ds_path / "test.bin")
        sweeper = GridSweeper.__new__(GridSweeper)
        sweeper._resume_index = {"old-key": {"seed": "123"}}
        sweeper._expected_comet_params = Mock(return_value={"seed": "123"})
        sweeper._fetch_epiplexity = Mock(return_value=(1.5, 2.5))
        api_experiment = Mock()
        api_experiment.get_metrics.return_value = [
            {"metricValue": str(logged_step)}
        ]
        sweeper._test_api_experiment = api_experiment
        return sweeper

    @staticmethod
    def _constructor_cfg():
        return OmegaConf.create(
            {
                "name": "resume-test",
                "command": ["wandb_project=epiaudio"],
                "parameters": {},
            }
        )

    def test_completed_run_reaches_expected_final_step(self):
        records = [
            {"metricValue": "1"},
            {"metricValue": "10"},
            {"metricValue": "20"},
        ]

        self.assertTrue(Sweeper._reached_final_teacher_step(records, 20))

    def test_partial_run_does_not_reach_expected_final_step(self):
        records = [
            {"metricValue": "1"},
            {"metricValue": "10"},
            {"metricValue": "19"},
        ]

        self.assertFalse(Sweeper._reached_final_teacher_step(records, 20))

    def test_missing_teacher_step_history_is_incomplete(self):
        self.assertFalse(Sweeper._reached_final_teacher_step([], 20))

    def test_configuration_mismatch_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            sweeper = self._resume_candidate(ds_path, logged_step=20)
            sweeper._resume_index = {"old-key": {"seed": "different"}}

            with patch("epiaudio.sweep.comet_ml.APIExperiment") as api_cls:
                resumed = sweeper._find_resumed_experiment(
                    self._run_cfg(ds_path), False, False, False
                )

            self.assertIsNone(resumed)
            api_cls.assert_not_called()
            sweeper._fetch_epiplexity.assert_not_called()

    def test_completed_matching_run_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            sweeper = self._resume_candidate(ds_path, logged_step=20)

            with patch(
                "epiaudio.sweep.comet_ml.APIExperiment",
                return_value=sweeper._test_api_experiment,
            ):
                resumed = sweeper._find_resumed_experiment(
                    self._run_cfg(ds_path), False, False, False
                )

            self.assertEqual(resumed, ("old-key", 10, 1.5, 2.5))
            sweeper._fetch_epiplexity.assert_called_once()

            workflow = GridSweeper.__new__(GridSweeper)
            workflow.start_time = datetime.now()
            workflow.sweep_name = "old-sweep"
            workflow.base_config_name = "test"
            workflow.results = []
            workflow.iter_points = Mock(return_value=[({}, {"seed": 123})])
            workflow.run_name_for = Mock(return_value="run-123")
            workflow.build_run_config = Mock(return_value=self._run_cfg(ds_path))
            workflow._find_resumed_experiment = Mock(return_value=resumed)
            workflow._train_fn = Mock()
            workflow._report = Mock()

            with (
                patch.dict(os.environ, {"COMET_ML_API": "test-key"}),
                patch("epiaudio.sweep.load_dotenv"),
            ):
                workflow.sweep()

            workflow._train_fn.assert_not_called()

    def test_partial_matching_run_is_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            run_cfg = self._run_cfg(ds_path)
            sweeper = self._resume_candidate(ds_path, logged_step=19)

            with patch(
                "epiaudio.sweep.comet_ml.APIExperiment",
                return_value=sweeper._test_api_experiment,
            ):
                resumed = sweeper._find_resumed_experiment(
                    run_cfg, False, False, False
                )

            self.assertIsNone(resumed)
            sweeper._fetch_epiplexity.assert_not_called()

            workflow = GridSweeper.__new__(GridSweeper)
            workflow.start_time = datetime.now()
            workflow.sweep_name = "old-sweep"
            workflow.base_config_name = "test"
            workflow.results = []
            workflow.iter_points = Mock(return_value=[({}, {"seed": 123})])
            workflow.run_name_for = Mock(return_value="run-123")
            workflow.build_run_config = Mock(return_value=run_cfg)
            workflow._find_resumed_experiment = Mock(return_value=None)
            workflow._train_fn = Mock(return_value=("new-key", 10))
            workflow._fetch_epiplexity = Mock(return_value=(1.5, 2.5))
            workflow._report = Mock()

            with (
                patch.dict(os.environ, {"COMET_ML_API": "test-key"}),
                patch("epiaudio.sweep.load_dotenv"),
            ):
                workflow.sweep()

            workflow._train_fn.assert_called_once_with(run_cfg)

    def test_missing_sweep_fails_instead_of_starting_fresh(self):
        api = Mock()
        api.query.return_value = []

        with (
            patch.dict(os.environ, {"COMET_ML_API": "test-key"}),
            patch("epiaudio.sweep.load_dotenv"),
            patch("epiaudio.sweep.comet_ml.API", return_value=api),
        ):
            with self.assertRaisesRegex(RuntimeError, "no matching Comet experiments"):
                GridSweeper(self._constructor_cfg(), sweep_name="missing-sweep")

    def test_comet_failure_aborts_resume(self):
        api = Mock()
        api.query.side_effect = ConnectionError("Comet unavailable")

        with (
            patch.dict(os.environ, {"COMET_ML_API": "test-key"}),
            patch("epiaudio.sweep.load_dotenv"),
            patch("epiaudio.sweep.comet_ml.API", return_value=api),
        ):
            with self.assertRaisesRegex(RuntimeError, "Comet lookup failed"):
                GridSweeper(self._constructor_cfg(), sweep_name="old-sweep")


if __name__ == "__main__":
    unittest.main()
