from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from epiaudio.train_torch import make_ds_loader

N_CODEBOOKS = 12
FRAMES_PER_CLIP = 250
SEQ_LEN = N_CODEBOOKS * FRAMES_PER_CLIP


def _synthetic_raw_audio_codes() -> torch.Tensor:
    """A stand-in for DacModel.encode(...).audio_codes[0]: shape (12, 250),
    i.e. (codebooks, frames) — DAC's native layout, codebooks as rows.

    Every value is unique and encodes its own (codebook, frame) coordinates
    (value = codebook * 1000 + frame) so any reordering bug shows up as a
    wrong number rather than a coincidentally-plausible one.
    """
    codebooks = torch.arange(N_CODEBOOKS).unsqueeze(1)  # (12, 1)
    frames = torch.arange(FRAMES_PER_CLIP).unsqueeze(0)  # (1, 250)
    return codebooks * 1000 + frames  # broadcasts to (12, 250)


class DacTokenOrderingTests(unittest.TestCase):
    def test_transpose_matches_dac_tokenizer_and_is_frame_major(self):
        """Mirrors DACTokenizer.tokenize()'s exact transform: transpose
        audio_codes[0] from (codebooks, frames) to (frames, codebooks) before
        flattening, so consecutive positions in the flattened array are
        consecutive codebooks *within the same frame*, not the same codebook
        across the whole clip.
        """
        raw = _synthetic_raw_audio_codes()  # (12, 250) codebook-major
        tokens = raw.transpose(0, 1).contiguous()  # DACTokenizer's fix

        self.assertEqual(tuple(tokens.shape), (FRAMES_PER_CLIP, N_CODEBOOKS))

        flat = tokens.numpy().reshape(-1)  # mirrors prepare_audio.py's tokens_batch[j].reshape(-1)
        for frame in range(3):  # check the first few frames explicitly
            frame_slice = flat[frame * N_CODEBOOKS : (frame + 1) * N_CODEBOOKS]
            expected = [cb * 1000 + frame for cb in range(N_CODEBOOKS)]
            self.assertEqual(
                list(frame_slice),
                expected,
                f"frame {frame}: expected all {N_CODEBOOKS} codebooks for this "
                "timestep to be contiguous in the flattened sequence",
            )

    def test_train_torch_reads_back_the_same_frame_major_order(self):
        """End-to-end: write the flattened tokens the way prepare_audio.py
        does, then read them back with train_torch.py's real make_ds_loader
        and confirm the sequence it hands to the model still has all
        codebooks for frame 0 first, then frame 1, etc.
        """
        raw = _synthetic_raw_audio_codes()
        tokens = raw.transpose(0, 1).contiguous()
        flat = tokens.numpy().reshape(-1).astype(np.uint16)
        self.assertEqual(flat.shape, (SEQ_LEN,))

        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            memmap = np.memmap(ds_path / "test.bin", dtype=np.uint16, mode="w+", shape=(SEQ_LEN,))
            memmap[:] = flat
            memmap.flush()

            get_batch, n_tokens = make_ds_loader(
                str(ds_path), "test", seq_len=SEQ_LEN, batch_size=1, bos_id=0,
            )
            self.assertEqual(n_tokens, SEQ_LEN)

            read_tokens, _mask = get_batch(0)

        # make_ds_loader prepends a BOS token and drops the final position to
        # keep the sequence length fixed — so read_tokens[0, 1:] should equal
        # the original flat array's first SEQ_LEN - 1 values, unchanged in order.
        self.assertEqual(read_tokens.shape, (1, SEQ_LEN))
        np.testing.assert_array_equal(read_tokens[0, 1:], flat[: SEQ_LEN - 1])

        # Spot-check a few frames directly out of what the model would see.
        for frame in range(3):
            start = 1 + frame * N_CODEBOOKS  # +1 for the BOS shift
            frame_slice = read_tokens[0, start : start + N_CODEBOOKS]
            expected = [cb * 1000 + frame for cb in range(N_CODEBOOKS)]
            self.assertEqual(
                list(frame_slice),
                expected,
                f"frame {frame}: train_torch.py's reader did not preserve "
                "frame-major ordering when reading back the memmap",
            )


if __name__ == "__main__":
    unittest.main()
