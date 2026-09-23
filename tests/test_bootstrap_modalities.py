from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from epiaudio.sweep import GridSweeper, load_sweep_configs
from epiaudio.sweep_preflight import validate_sweep


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = REPO_ROOT / "epiaudio" / "sweeps" / "bootstrap_modalities.yaml"


class BootstrapModalitiesSweepTests(unittest.TestCase):
    def test_resolves_one_preflight_ready_48_point_sweep_per_modality(self):
        contracts = {
            "openwebtext_ascii96": (96, 512),
            "lichess_puzzles": (64, 512),
            "cifar5m_grayscale": (256, 1024),
            "fsd50k_encodec": (1024, 375),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset_name, (vocab_size, sequence_length) in contracts.items():
                dataset_path = root / "data" / dataset_name
                dataset_path.mkdir(parents=True)
                (dataset_path / "train.bin").touch()
                (dataset_path / "test.bin").touch()
                OmegaConf.save(
                    OmegaConf.create({"model": {"V": vocab_size, "L": sequence_length}}),
                    dataset_path / "metadata.yaml",
                )
            (root / ".env").write_text("COMET_ML_API=test-only\n", encoding="utf-8")

            configs = load_sweep_configs(str(SWEEP_PATH))

            self.assertEqual(len(configs), 4)
            for config in configs:
                preflight = validate_sweep(
                    config,
                    SWEEP_PATH,
                    method="pf_grid",
                    repo_root=root,
                )
                self.assertTrue(preflight.ready, preflight.issues)
                sweeper = GridSweeper.__new__(GridSweeper)
                sweeper.cfg = config
                sweeper.sweep_seed = None
                self.assertEqual(len(list(sweeper.iter_grid())), 48)
                self.assertEqual(config.parameters.T.value, 31_250_000)
