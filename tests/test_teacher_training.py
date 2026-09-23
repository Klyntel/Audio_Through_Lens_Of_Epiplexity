import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from epiaudio.downstream.audio_classification_dataset import (
    AudioClassificationDataset,
)
from epiaudio.downstream.teacher_train_eval import (
    _local_examples_per_epoch,
    train_and_evaluate,
)


def _write_classification_dataset(path: Path) -> None:
    train = np.array(
        [
            [0, 1, 2, 3, 0],
            [1, 2, 3, 4, 1],
            [2, 3, 4, 5, 0],
            [3, 4, 5, 6, 1],
            [4, 5, 6, 7, 0],
            [5, 6, 7, 0, 1],
            [6, 7, 0, 1, 0],
            [7, 0, 1, 2, 1],
        ],
        dtype=np.int16,
    )
    test = np.array(
        [
            [0, 2, 4, 6, 0],
            [1, 3, 5, 7, 1],
            [2, 4, 6, 0, 0],
            [3, 5, 7, 1, 1],
        ],
        dtype=np.int16,
    )
    train.tofile(path / "train.bin")
    test.tofile(path / "test.bin")
    metadata = {
        "schema_version": 1,
        "task_type": "single_label_multiclass",
        "label_column": "label",
        "label_position": -1,
        "label_type": "string",
        "num_classes": 2,
        "class_names": ["zero", "one"],
        "class_to_index": {"zero": 0, "one": 1},
        "tokenizer": {
            "dtype": "int16",
            "sequence_length": 4,
            "token_shape": [4],
        },
        "train": {
            "num_samples": 8,
            "tokens_per_sample": 4,
            "dtype": "int16",
            "class_counts": [4, 4],
        },
        "test": {
            "num_samples": 4,
            "tokens_per_sample": 4,
            "dtype": "int16",
            "class_counts": [2, 2],
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _training_config(path: Path, checkpoint_name: str = "teacher.pt"):
    return OmegaConf.create(
        {
            "ds_path": str(path),
            "checkpoint_path": str(path / checkpoint_name),
            "seed": 17,
            "n_epochs": 1,
            "batch_size": 4,
            "accumulation_steps": 2,
            "num_workers": 0,
            "world_size": 1,
            "num_layers": 1,
            "embed_dim": 4,
            "per_head_dim": 2,
            "vocab_size": 8,
            "seq_length": 4,
            "embed_init_std": 0.1,
            "learning_rate": 0.2,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "embed_lr_mult": 1.0,
            "schedule": "const",
            "decay_frac": 0.0,
            "warmup_steps": 0,
            "teacher_ema": 2,
            "amp_dtype": "float32",
            "eval_batch_size": 2,
        }
    )


class TeacherTrainingRuntimeTest(unittest.TestCase):
    def test_global_batch_must_split_across_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            _write_classification_dataset(path)
            cfg = _training_config(path)
            cfg.batch_size = 3
            with self.assertRaisesRegex(ValueError, "must be divisible"):
                train_and_evaluate(cfg)

    def test_ddp_rejects_a_hardware_dependent_partial_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            _write_classification_dataset(path)
            data = AudioClassificationDataset(
                "train",
                path / "train.bin",
                path / "metadata.json",
            )

            self.assertEqual(_local_examples_per_epoch(data, 2), 4)
            with self.assertRaisesRegex(ValueError, "hardware-dependent"):
                _local_examples_per_epoch(data, 3)

    def test_accumulation_matches_one_full_global_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            _write_classification_dataset(path)
            accumulated_cfg = _training_config(path, "accumulated.pt")
            accumulated_cfg.max_steps = 1
            accumulated_cfg.teacher_ema = 0
            full_batch_cfg = _training_config(path, "full_batch.pt")
            full_batch_cfg.accumulation_steps = 1
            full_batch_cfg.max_steps = 1
            full_batch_cfg.teacher_ema = 0

            train_and_evaluate(accumulated_cfg)
            train_and_evaluate(full_batch_cfg)

            accumulated = torch.load(
                accumulated_cfg.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )["model"]
            full_batch = torch.load(
                full_batch_cfg.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )["model"]
            self.assertEqual(accumulated.keys(), full_batch.keys())
            for name in accumulated:
                torch.testing.assert_close(
                    accumulated[name],
                    full_batch[name],
                    rtol=1e-5,
                    atol=1e-7,
                )


class TeacherTrainingTest(unittest.TestCase):
    def test_trains_saves_and_resumes_with_conditional_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            _write_classification_dataset(path)
            cfg = _training_config(path)

            metrics = train_and_evaluate(cfg)

            self.assertEqual(metrics["optimizer_steps"], 2)
            self.assertEqual(metrics["examples_seen"], 8)
            self.assertEqual(metrics["conditioning_tokens_seen"], 32)
            self.assertEqual(metrics["label_tokens_seen"], 8)
            self.assertEqual(metrics["compute_tokens_seen"], 40)
            for key in ("train_label_nll", "raw_label_nll", "ema_label_nll"):
                self.assertTrue(math.isfinite(metrics[key]))

            checkpoint = torch.load(
                cfg.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["version"], 1)
            self.assertEqual(checkpoint["trainer"]["epoch"], 1)
            self.assertEqual(checkpoint["trainer"]["micro_batches_in_epoch"], 0)
            self.assertIsNotNone(checkpoint["ema_model"])
            self.assertEqual(
                sorted(
                    group["_base_lr"]
                    for group in checkpoint["optimizer"]["param_groups"]
                ),
                [0.05, 0.2],
            )

            resumed_cfg = _training_config(path)
            resumed_cfg.resume_from = cfg.checkpoint_path
            resumed_metrics = train_and_evaluate(resumed_cfg)
            self.assertEqual(resumed_metrics, metrics)

    def test_midstream_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            _write_classification_dataset(path)
            full_cfg = _training_config(path, "full.pt")
            partial_cfg = _training_config(path, "partial.pt")
            partial_cfg.max_steps = 1

            full_metrics = train_and_evaluate(full_cfg)
            partial_metrics = train_and_evaluate(partial_cfg)
            self.assertEqual(partial_metrics["optimizer_steps"], 1)
            self.assertEqual(partial_metrics["examples_seen"], 4)

            resumed_cfg = _training_config(path, "resumed.pt")
            resumed_cfg.resume_from = partial_cfg.checkpoint_path
            resumed_metrics = train_and_evaluate(resumed_cfg)
            self.assertEqual(resumed_metrics, full_metrics)

            full = torch.load(
                full_cfg.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            resumed = torch.load(
                resumed_cfg.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            for state_name in ("model", "ema_model"):
                self.assertEqual(full[state_name].keys(), resumed[state_name].keys())
                for name in full[state_name]:
                    torch.testing.assert_close(
                        full[state_name][name],
                        resumed[state_name][name],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
