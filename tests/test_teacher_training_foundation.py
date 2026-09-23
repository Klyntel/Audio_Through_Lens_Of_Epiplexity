import random
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from epiaudio.downstream.audio_classification_dataset import (
    AudioClassificationDataset,
)
from epiaudio.downstream.teacher_training import (
    TeacherTrainingConfig,
    TrainingCounters,
    _restore_rng_state,
    resolve_ema_decay,
)


class TeacherTrainingConfigTest(unittest.TestCase):
    def test_requires_exactly_one_epoch(self) -> None:
        for n_epochs in (0, 2):
            with self.subTest(n_epochs=n_epochs):
                cfg = OmegaConf.create({"ds_path": "unused", "n_epochs": n_epochs})
                with self.assertRaisesRegex(ValueError, "exactly 1"):
                    TeacherTrainingConfig.from_cfg(cfg)

    def test_rejects_the_removed_data_repetition_escape_hatch(self) -> None:
        cfg = OmegaConf.create(
            {
                "ds_path": "unused",
                "n_epochs": 1,
                "allow_data_repetition": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "strictly one-pass"):
            TeacherTrainingConfig.from_cfg(cfg)

    def test_uses_spelled_out_configuration(self) -> None:
        cfg = OmegaConf.create(
            {
                "ds_path": "unused",
                "batch_size": 32,
                "accumulation_steps": 4,
                "num_layers": 2,
                "embed_dim": 16,
                "per_head_dim": 4,
                "vocab_size": 64,
                "seq_length": 8,
                "learning_rate": 0.5,
                "schedule": "linear",
            }
        )
        config = TeacherTrainingConfig.from_cfg(cfg)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.accumulation_steps, 4)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.learning_rate, 0.5)
        self.assertEqual(config.schedule, "linear")

    def test_rejects_legacy_parameter_names(self) -> None:
        for legacy in (
            {"B": 32},
            {"A": 4},
            {"model": {"N": 2}},
            {"opt": {"lr": 0.5}},
        ):
            with self.subTest(legacy=legacy):
                cfg = OmegaConf.create({"ds_path": "unused", **legacy})
                with self.assertRaisesRegex(ValueError, "spelled-out"):
                    TeacherTrainingConfig.from_cfg(cfg)


class TrainingAccountingTest(unittest.TestCase):
    def test_counts_labels_separately_from_conditioning_compute(self) -> None:
        data = cast(
            AudioClassificationDataset,
            SimpleNamespace(
                conditioning_tokens_per_example=4,
                label_tokens_per_example=1,
                compute_tokens_per_example=5,
            ),
        )
        counters = TrainingCounters()
        counters.update(3, data)
        self.assertEqual(counters.examples_seen, 3)
        self.assertEqual(counters.conditioning_tokens_seen, 12)
        self.assertEqual(counters.label_tokens_seen, 3)
        self.assertEqual(counters.compute_tokens_seen, 15)

    def test_ema_window_matches_the_eca_convention(self) -> None:
        self.assertIsNone(resolve_ema_decay(0))
        self.assertAlmostEqual(resolve_ema_decay(0.9), 0.9)
        self.assertAlmostEqual(resolve_ema_decay(50), 0.9801986733)


class RngStateTest(unittest.TestCase):
    def test_restore_moves_serialized_torch_rng_states_to_cpu(self) -> None:
        torch_state = Mock()
        cuda_state = Mock()
        cpu_torch_state = torch.get_rng_state()
        cpu_cuda_state = torch.get_rng_state()
        torch_state.cpu.return_value = cpu_torch_state
        cuda_state.cpu.return_value = cpu_cuda_state
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch_state,
            "cuda": [cuda_state],
        }

        with (
            patch("torch.set_rng_state") as set_rng_state,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_rng_state_all") as set_cuda_rng_state_all,
        ):
            _restore_rng_state(state)

        torch_state.cpu.assert_called_once_with()
        cuda_state.cpu.assert_called_once_with()
        set_rng_state.assert_called_once_with(cpu_torch_state)
        set_cuda_rng_state_all.assert_called_once_with([cpu_cuda_state])


if __name__ == "__main__":
    unittest.main()
