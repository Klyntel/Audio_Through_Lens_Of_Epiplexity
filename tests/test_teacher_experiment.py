import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

import epiaudio.downstream.teacher_experiment as experiment_module
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS
from epiaudio.downstream.teacher_training import TeacherTrainingConfig


class TeacherExperimentTest(unittest.TestCase):
    def test_forwards_the_complete_training_configuration(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "conditional",
                "ds_path": "unused",
                "n_epochs": 1,
                "batch_size": 32,
                "accumulation_steps": 4,
                "teacher_ema": 25,
                "learning_rate": 0.5,
                "schedule": "linear",
            }
        )
        with (
            patch.object(experiment_module, "HAS_COMET", False),
            patch.object(
                experiment_module,
                "train_and_evaluate",
                return_value={},
            ) as train,
        ):
            experiment_module.run_experiment(cfg)

        train.assert_called_once_with(cfg)

    def test_does_not_change_classification_training_configs(self) -> None:
        config_dir = Path(experiment_module.__file__).parents[1] / "classification"
        config_names = (
            "syntheory_chords_encodec_classify_midi_program_name.yaml",
            "syntheory_intervals_encodec_classify_midi_program_name.yaml",
        )
        for config_name in config_names:
            with self.subTest(path=config_name):
                path = config_dir / config_name
                cfg = OmegaConf.load(path)
                self.assertEqual(cfg.n_epochs, 50)

    def test_checked_in_teacher_configs_are_runnable_encodec_runs(self) -> None:
        epiaudio_dir = Path(experiment_module.__file__).parents[1]
        config_dir = epiaudio_dir / "conditional_epiplexity" / "configs" / "teacher"
        config_names = (
            "syntheory_chords_encodec_midi_program_name.yaml",
            "syntheory_intervals_encodec_midi_program_name.yaml",
        )
        encodec = TOKENIZER_SPECS["encodec"]

        for config_name in config_names:
            with self.subTest(path=config_name):
                cfg = OmegaConf.load(config_dir / config_name)
                config = TeacherTrainingConfig.from_cfg(cfg)

                self.assertEqual(config.n_epochs, 1)
                self.assertEqual(config.seq_length, encodec.sequence_length)
                self.assertEqual(config.vocab_size, encodec.vocab_size)
                self.assertIn(
                    "outputs/conditional_epiplexity/teachers",
                    config.checkpoint_path,
                )


if __name__ == "__main__":
    unittest.main()
