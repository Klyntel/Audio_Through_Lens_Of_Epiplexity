import json
import unittest
from pathlib import Path
from unittest.mock import patch

from epiaudio.dataset.comet_inventory import (
    CometInventory,
    DEFAULT_PROJECTS,
    classify_dataset_name,
    normalize_dataset_names,
)
from epiaudio.dataset.load_dataloaders import AUDIO_PREPROCESSING_DATASETS


class CometInventoryTest(unittest.TestCase):
    def test_normalizes_paths_tokenizers_and_sweep_lists(self) -> None:
        value = json.dumps(
            [
                "/datasets/DEMAND_encodec_train",
                "s3://bucket/VocalSound-16k_dac",
                "/datasets/fsd50k_wavtokenizer_train",
                "urbansound",
            ]
        )
        self.assertEqual(
            list(normalize_dataset_names(value)),
            ["demand", "vocalsound_16k", "fsd50k", "urbansound"],
        )

    def test_snapshot_covers_every_scanned_project(self) -> None:
        snapshot_path = (
            Path(__file__).parents[1] / "epiaudio/dataset/comet_datasets.json"
        )
        inventory = json.loads(snapshot_path.read_text())
        self.assertEqual(set(inventory["projects"]), set(DEFAULT_PROJECTS))
        self.assertEqual(len(inventory["datasets"]), 60)
        self.assertEqual(set(inventory["classification"]), set(inventory["datasets"]))
        self.assertNotIn("unknown", inventory["classification"].values())

    def test_unknown_dataset_names_fail_closed(self) -> None:
        self.assertEqual(classify_dataset_name("not_a_real_dataset"), "unknown")

    def test_parameter_rows_exclude_experiments_without_sweep_name(self) -> None:
        inventory = CometInventory("unused", workspace="test", batch_size=10)
        payload = {
            "experiments": {
                "sweep": {
                    "params": {"sweep_name": "real-sweep", "ds_path": "demand_dac"}
                },
                "one_off": {"params": {"ds_path": "urbansound_dac"}},
            }
        }
        with patch.object(inventory, "_request", return_value=payload):
            self.assertEqual(
                inventory.parameter_rows(["sweep", "one_off"]),
                [("real-sweep", ["demand_dac"])],
            )

    def test_external_comet_loaders_are_registered(self) -> None:
        expected = {
            "aishell1",
            "aishell3",
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
            "voxpopuli_en",
        }
        self.assertLessEqual(expected, AUDIO_PREPROCESSING_DATASETS.keys())


if __name__ == "__main__":
    unittest.main()
