from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from epiaudio.dataset.prepare_discrete import (
    Cifar5MGrayscaleTokenizer,
    LichessPuzzleTokenizer,
    OpenWebTextTokenizer,
    prepare_cifar5m,
)
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS
from experiments.bootstrap_modalities.prepare import TokenDatasetContract, TokenDatasetWriter


class DiscreteTokenizerTests(unittest.TestCase):
    def test_openwebtext_ascii96_matches_the_reference_character_contract(self):
        tokenizer = OpenWebTextTokenizer()

        token_ids = tokenizer.encode_document("A z")

        self.assertEqual(tokenizer.vocab_size, 96)
        self.assertEqual(token_ids.dtype, np.int16)
        self.assertEqual(tokenizer.decode(token_ids), "A z\n")
        self.assertFalse(tokenizer.supports("snowman: ☃"))

    def test_lichess_representation_is_fixed_length_with_reference_special_tokens(self):
        tokenizer = LichessPuzzleTokenizer()

        token_ids, target_mask = tokenizer.encode_example("1. e4 e5", " Nf3")

        self.assertEqual(tokenizer.vocab_size, 64)
        self.assertEqual(token_ids.shape, (512,))
        self.assertEqual(token_ids[0], tokenizer.bos_id)
        self.assertEqual(token_ids[-1], tokenizer.eos_id)
        self.assertTrue(target_mask.any())
        self.assertEqual(tokenizer.format_moves("1. e4 e5", " Nf3"), "e4,e5;Nf3")
        self.assertFalse(tokenizer.fits_context("a " * 300, ""))

    def test_cifar5m_grayscale_uses_reference_rgb_mean_and_flattening(self):
        tokenizer = Cifar5MGrayscaleTokenizer()
        rgb = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = 1
        rgb[..., 1] = 2
        rgb[..., 2] = 5

        token_ids = tokenizer.encode_batch(rgb)

        self.assertEqual(token_ids.shape, (1, 1024))
        self.assertEqual(token_ids.dtype, np.int16)
        self.assertTrue(np.all(token_ids == 2))
        self.assertEqual(tokenizer.decode(token_ids).shape, (1, 32, 32))


class CifarPreparationTests(unittest.TestCase):
    def test_cifar5m_writer_uses_seeded_image_holdout_and_int16_memmaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = []
            for shard_index, value in enumerate((0, 30)):
                images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
                images[0] = value
                images[1] = value + 3
                shard_path = root / f"part{shard_index}.npz"
                np.savez(shard_path, X=images)
                shards.append(shard_path)

            output = root / "cifar5m_grayscale"
            prepare_cifar5m(output, shards, test_images=1, seed=2357, batch_size=1)

            train = np.memmap(output / "train.bin", dtype=np.int16, mode="r")
            test = np.memmap(output / "test.bin", dtype=np.int16, mode="r")
            self.assertEqual(len(train), 3 * 1024)
            self.assertEqual(len(test), 1024)
            expected_test_index = random.Random(2357).sample(range(4), k=1)[0]
            expected_values = [0, 3, 30, 33]
            self.assertTrue(np.all(test == expected_values[expected_test_index]))
            self.assertEqual(set(np.unique(train)) | set(np.unique(test)), set(expected_values))

            metadata = OmegaConf.load(output / "metadata.yaml")
            self.assertEqual(metadata.model.V, 256)
            self.assertEqual(metadata.model.L, 1024)


class ExperimentTokenWriterTests(unittest.TestCase):
    def test_writer_publishes_token_aligned_sidecars_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "puzzles"
            writer = TokenDatasetWriter(
                output,
                TokenDatasetContract(
                    dataset="fixture",
                    tokenizer="fixture_tokens",
                    vocab_size=8,
                    sequence_length=2,
                    source={"fixture": True},
                ),
            )
            writer.write_split(
                "train",
                [(np.asarray([1, 2]), {"mask": np.asarray([True, False])})],
                num_tokens=2,
                num_samples=1,
                sidecars={"mask": np.bool_},
            )
            writer.write_split("test", [np.asarray([3, 4])], num_tokens=2, num_samples=1)
            writer.finalize()

            self.assertEqual(
                np.memmap(output / "train_mask.bin", dtype=np.bool_, mode="r").tolist(),
                [True, False],
            )
            metadata = OmegaConf.load(output / "metadata.yaml")
            self.assertEqual((metadata.model.V, metadata.model.L), (8, 2))


class DiscreteTokenizerSpecTests(unittest.TestCase):
    def test_reference_non_audio_contracts_are_registered_for_sweep_preflight(self):
        self.assertEqual(TOKENIZER_SPECS["ascii96"].shape, (512,))
        self.assertEqual(TOKENIZER_SPECS["ascii96"].vocab_size, 96)
        self.assertEqual(TOKENIZER_SPECS["puzzles"].vocab_size, 64)
        self.assertEqual(TOKENIZER_SPECS["grayscale"].sequence_length, 1024)
