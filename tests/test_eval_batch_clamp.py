from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from epiaudio.train_torch import _SplitTooSmallError, _clamped_eval_batch_size, make_ds_loader

SEQ_LEN = 4


def _write_bin(ds_path: Path, split: str, num_samples: int, seq_len: int = SEQ_LEN) -> None:
    tokens = np.arange(num_samples * seq_len, dtype=np.int16)
    memmap = np.memmap(ds_path / f"{split}.bin", dtype=np.int16, mode="w+", shape=tokens.shape)
    memmap[:] = tokens
    memmap.flush()


class ClampedEvalBatchSizeTests(unittest.TestCase):
    def test_shrinks_batch_size_to_fit_a_small_split(self):
        """ARTE-scale split: far fewer samples than the requested batch size."""
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            _write_bin(ds_path, "test", num_samples=2)

            eval_batch_size = _clamped_eval_batch_size(
                str(ds_path), "test", SEQ_LEN, requested_batch_size=16, dtype=np.int16,
            )
            self.assertEqual(eval_batch_size, 2)

    def test_leaves_a_large_enough_split_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            _write_bin(ds_path, "test", num_samples=100)

            eval_batch_size = _clamped_eval_batch_size(
                str(ds_path), "test", SEQ_LEN, requested_batch_size=16, dtype=np.int16,
            )
            self.assertEqual(eval_batch_size, 16)

    def test_raises_instead_of_silently_returning_zero(self):
        """A split with fewer tokens than one full sequence can't be evaluated
        at all; this should fail clearly here rather than as a ZeroDivisionError
        deep inside get_batch.

        Writes 3 raw tokens directly (not via _write_bin, whose num_samples is
        whole sequences): 3 < SEQ_LEN (4), but still a byte count numpy can open
        cleanly, so this actually reaches _clamped_eval_batch_size's own check
        instead of numpy's own memmap-opening errors (see the test below).
        """
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            (ds_path / "test.bin").write_bytes(np.arange(3, dtype=np.int16).tobytes())

            with self.assertRaises(_SplitTooSmallError):
                _clamped_eval_batch_size(
                    str(ds_path), "test", SEQ_LEN, requested_batch_size=16, dtype=np.int16,
                )

    def test_does_not_mask_unrelated_memmap_errors_as_split_too_small(self):
        """A corrupted/truncated .bin (byte count not a multiple of the dtype's
        itemsize) should raise numpy's own ValueError, not _SplitTooSmallError,
        so callers catching only _SplitTooSmallError don't misreport real data
        corruption as 'no usable split, falling back to val'."""
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            # int16 is 2 bytes; 3 bytes can't be read back as any whole number of int16s.
            (ds_path / "test.bin").write_bytes(b"\x00\x00\x00")

            with self.assertRaises(ValueError) as ctx:
                _clamped_eval_batch_size(
                    str(ds_path), "test", SEQ_LEN, requested_batch_size=16, dtype=np.int16,
                )
            self.assertNotIsInstance(ctx.exception, _SplitTooSmallError)

    def test_clamped_batch_size_prevents_zerodivisionerror_in_get_batch(self):
        """End to end: without clamping, get_batch(0) divides by zero on a
        split this small (n_tokens < batch_size * seq_len); with clamping, it
        returns a correctly shaped, smaller batch instead."""
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = Path(tmp)
            _write_bin(ds_path, "test", num_samples=2)

            get_batch_unclamped, _ = make_ds_loader(
                str(ds_path), "test", seq_len=SEQ_LEN, batch_size=16,
            )
            with self.assertRaises(ZeroDivisionError):
                get_batch_unclamped(0)

            eval_batch_size = _clamped_eval_batch_size(
                str(ds_path), "test", SEQ_LEN, requested_batch_size=16, dtype=np.int16,
            )
            get_batch_clamped, n_tokens = make_ds_loader(
                str(ds_path), "test", seq_len=SEQ_LEN, batch_size=eval_batch_size,
            )
            tokens, mask = get_batch_clamped(0)
            self.assertEqual(tokens.shape, (eval_batch_size, SEQ_LEN))
            self.assertEqual(mask.shape, (eval_batch_size, SEQ_LEN))
            self.assertEqual(n_tokens, 2 * SEQ_LEN)


if __name__ == "__main__":
    unittest.main()
